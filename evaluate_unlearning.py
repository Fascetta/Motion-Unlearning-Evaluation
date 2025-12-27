import argparse
import logging
import os
from os.path import join as pjoin

import numpy as np
import torch
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# --- Custom Imports from your project ---
from diffusers import DDIMScheduler
from data.t2m_dataset import MotionDataset
from models.denoiser.model import Denoiser
from models.denoiser.trainer import DenoiserTrainer
from models.t2m_eval_wrapper import EvaluatorModelWrapper
from models.vae.model import VAE
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from utils.eval_t2m import test_denoiser
from utils.fixseed import fixseed
from utils.get_opt import get_opt
from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion
from utils.word_vectorizer import WordVectorizer
from utils.metrics import euclidean_distance_matrix

# --- Optimizations ---
torch.backends.cudnn.benchmark = True

# --- Setup Logging ---
console = Console(theme=Theme({"info": "dim cyan", "warning": "magenta", "error": "bold red"}))

def setup_logging(log_dir, model_name):
    os.makedirs(log_dir, exist_ok=True)
    log_file = pjoin(log_dir, f"eval_{model_name}.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_file, mode="a")
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    rich_handler = RichHandler(rich_tracebacks=True, show_time=False, console=console)
    rich_handler.setLevel(logging.INFO)
    logger.addHandler(rich_handler)
    return logger

# --- Helper Function for Plotting ---
def plot_generated_motions(logger, motions, opt, save_dir, prefix, kinematic_chain):
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Saving up to 5 generated videos to: {save_dir}")
    for i, motion in enumerate(motions):
        if i >= 5: break
        save_path = pjoin(save_dir, f"{prefix}_{i:02d}.mp4")
        joint = recover_from_ric(torch.from_numpy(motion).float(), opt.joints_num).numpy()
        plot_3d_motion(save_path, kinematic_chain, joint, title=prefix, fps=20, radius=4)

# --- Main Evaluation Logic ---
def run_evaluation(logger, model_name, ckpt_name, is_unlearned=False, args_override=None):
    status = "UNLEARNED" if is_unlearned else "ORIGINAL"
    console.rule(f"[bold cyan]EVALUATING: {status} ({model_name})")

    opt_path = pjoin("checkpoints", args_override.dataset_name, model_name, "opt.txt")
    ckpt_path = pjoin("checkpoints", args_override.dataset_name, model_name, "model", ckpt_name)
    eval_output_dir = pjoin("checkpoints", args_override.dataset_name, model_name, "evaluation")
    
    opt = get_opt(opt_path, args_override.device)
    opt.checkpoints_dir = "checkpoints"
    
    if args_override.dataset_name == "t2m":
        opt.dim_pose = 263
    elif args_override.dataset_name == "kit":
        opt.dim_pose = 251

    logger.info("Loading VAE...")
    vae_name = opt.vae_name
    vae_opt = get_opt(pjoin("checkpoints", args_override.dataset_name, vae_name, "opt.txt"), args_override.device)
    vae = VAE(vae_opt).to(args_override.device)
    vae_ckpt = torch.load(pjoin(vae_opt.checkpoints_dir, vae_opt.dataset_name, vae_opt.name, "model", "net_best_fid.tar"), map_location="cpu")
    vae.load_state_dict(vae_ckpt["vae"])
    vae.eval()

    logger.info(f"Loading Denoiser from checkpoint: {ckpt_path}")
    denoiser = Denoiser(opt, vae_opt.latent_dim).to(args_override.device)
    denoiser_ckpt = torch.load(ckpt_path, map_location="cpu")
    denoiser.load_state_dict(denoiser_ckpt["denoiser"], strict=False)
    denoiser.eval()
    
    logger.info("Loading evaluation wrapper and dataset utilities...")
    dataset_opt_path = f"checkpoints/{args_override.dataset_name}/Comp_v6_KLD005/opt.txt"
    wrapper_opt = get_opt(dataset_opt_path, args_override.device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    
    mean = np.load(pjoin(wrapper_opt.meta_dir, "mean.npy"))
    std = np.load(pjoin(wrapper_opt.meta_dir, "std.npy"))
    opt.window_size = 196
    dummy_dataset = MotionDataset(opt, mean, std, pjoin(opt.data_root, "kw_splits/train-wo-violence.txt"))

    # --- CHANGE: Fast Evaluation Mode ---
    if args_override.fast_eval:
        logger.warning("--- RUNNING IN FAST EVALUATION MODE ---")
        split_name = "kw_splits/test-wo-violence-fast"
        opt.num_inference_timesteps = 20 # Drastically reduce sampling steps
    else:
        split_name = "kw_splits/test-wo-violence"

    scheduler = DDIMScheduler(num_train_timesteps=opt.num_train_timesteps, beta_start=opt.beta_start, beta_end=opt.beta_end, prediction_type=opt.prediction_type, clip_sample=False)
    trainer = DenoiserTrainer(opt, denoiser, vae, scheduler)
    
    console.rule("[bold green]📊 PART A: PRESERVATION (Standard Test Set)")
    val_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, split_name, device=args_override.device, num_workers=0) # --- CHANGE: Use split_name ---
    
    _, fid, diversity, r_precision, matching_score, _, _, _, _ = test_denoiser(val_loader, trainer.generate, 0, eval_wrapper, opt.joints_num, cal_mm=False)

    logger.info(f"General FID: {fid:.4f} (Lower is better)")
    logger.info(f"General Diversity: {diversity:.4f} (Can be ignored if it varies wildly)")
    logger.info(f"General R-Precision (Top 1, 2, 3): ({r_precision[0]:.4f}, {r_precision[1]:.4f}, {r_precision[2]:.4f}) (Higher is better)")
    logger.info(f"General Matching Score: {matching_score:.4f} (Higher is better)")
    
    console.rule(f"[bold red]🎯 PART B: EFFICACY (Target: '{args_override.target_concept}')")
    evaluate_concept(logger, trainer, eval_wrapper, dummy_dataset, opt, args_override.target_concept, eval_output_dir, wrapper_opt)

    console.rule("[bold yellow]🔬 PART C: SPECIFICITY (Related Concepts)")
    for concept in args_override.related_concepts:
        evaluate_concept(logger, trainer, eval_wrapper, dummy_dataset, opt, concept, eval_output_dir, wrapper_opt)

def evaluate_concept(logger, trainer, wrapper, dataset, opt, concept, output_dir, wrapper_opt):
    logger.info(f"--- Evaluating concept: '{concept}' ---")
    num_samples = 32
    texts = [concept] * num_samples
    m_lens = torch.LongTensor([100] * num_samples).to(opt.device)

    with torch.no_grad():
        dummy_motion = torch.zeros(num_samples, 196, opt.dim_pose).to(opt.device)
        pred_motions_features, _ = trainer.generate((texts, dummy_motion, m_lens))

    motion_embeddings = wrapper.get_motion_embeddings(motions=pred_motions_features, m_lens=m_lens)

    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    pos_tag_map = {'a': 'DET', 'the': 'DET', 'an': 'DET', 'person': 'NOUN', 'man': 'NOUN', 'woman': 'NOUN', 'character': 'NOUN', 'kicking': 'VERB', 'jumping': 'VERB', 'punching': 'VERB', 'lunging': 'VERB', 'walking': 'VERB', 'running': 'VERB'}
    all_word_embs, all_pos_ohots, all_cap_lens = [], [], []

    for text in texts:
        raw_tokens = text.split(' ')
        tokens_with_pos = [f"{word}/{pos_tag_map.get(word, 'NOUN')}" for word in raw_tokens]
        tokens = ['sos/OTHER'] + tokens_with_pos + ['eos/OTHER']
        
        sent_len = len(tokens)
        if len(tokens) < wrapper_opt.max_text_len + 2:
            tokens = tokens + ['unk/OTHER'] * (wrapper_opt.max_text_len + 2 - len(tokens))

        all_cap_lens.append(sent_len)
        word_embeddings, pos_one_hots = [], []
        for token in tokens:
            word_emb, pos_oh = w_vectorizer[token]
            word_embeddings.append(word_emb[None, :])
            pos_one_hots.append(pos_oh[None, :])
        all_word_embs.append(np.concatenate(word_embeddings, axis=0)[None, :, :])
        all_pos_ohots.append(np.concatenate(pos_one_hots, axis=0)[None, :, :])

    word_embs_batch = torch.from_numpy(np.concatenate(all_word_embs, axis=0)).float().to(opt.device)
    pos_ohots_batch = torch.from_numpy(np.concatenate(all_pos_ohots, axis=0)).float().to(opt.device)
    cap_lens_batch = torch.from_numpy(np.array(all_cap_lens)).long().to(opt.device)

    text_embeddings = wrapper.text_encoder(word_embs_batch, pos_ohots_batch, cap_lens_batch)

    dist_matrix = euclidean_distance_matrix(text_embeddings.cpu().detach().numpy(), motion_embeddings.cpu().detach().numpy())
    matching_score = dist_matrix.trace() / len(texts)
    logger.info(f"Matching Score for '{concept}': {matching_score:.4f} (Lower is better for forgotten concepts)")

    pred_motions_denorm = dataset.inv_transform(pred_motions_features.cpu().numpy())
    plot_generated_motions(logger, pred_motions_denorm, opt, pjoin(output_dir, "videos"), concept.replace(" ", "_"), opt.kinematic_chain)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_name", type=str, required=True, help="Folder name of the original pre-trained model.")
    parser.add_argument("--original_ckpt", type=str, default="net_best_fid.tar", help="Checkpoint file for the original model.")
    parser.add_argument("--unlearned_name", type=str, required=True, help="Folder name of the new unlearned model.")
    parser.add_argument("--unlearned_ckpt", type=str, default="latest.tar", help="Checkpoint file for the unlearned model.")
    parser.add_argument("--dataset_name", type=str, default="t2m", help="Dataset name.")
    parser.add_argument("--target_concept", type=str, required=True, help="Concept that was unlearned.")
    parser.add_argument("--related_concepts", nargs='+', required=True, help="List of related concepts for the specificity test.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use.")
    # --- CHANGE: Add fast_eval flag ---
    parser.add_argument("--fast_eval", action="store_true", help="Run a fast evaluation on a small subset of data.")
    
    args = parser.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    fixseed(42)

    logger = setup_logging("logs/eval_logs", f"{args.unlearned_name}_comparison")

    run_evaluation(
        logger=logger,
        model_name=args.original_name,
        ckpt_name=args.original_ckpt,
        is_unlearned=False,
        args_override=args
    )
    
    run_evaluation(
        logger=logger,
        model_name=args.unlearned_name,
        ckpt_name=args.unlearned_ckpt,
        is_unlearned=True,
        args_override=args
    )