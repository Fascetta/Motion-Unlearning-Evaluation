import argparse
import logging
import os
from os.path import join as pjoin

import numpy as np
import torch
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# --- Optimizations ---
torch.backends.cudnn.benchmark = True

from diffusers import DDIMScheduler
from data.t2m_dataset import MotionDataset
from models.denoiser.model import Denoiser
from models.denoiser.trainer import DenoiserTrainer
from models.t2m_eval_wrapper import EvaluatorModelWrapper
from utils.word_vectorizer import WordVectorizer
from models.vae.model import VAE
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from utils.eval_t2m import test_denoiser
from utils.fixseed import fixseed
from utils.metrics import euclidean_distance_matrix
from utils.get_opt import get_opt
from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion

# (Keep the setup_logging, plot_generated_motions, run_evaluation, and evaluate_concept functions exactly as they were in the last version)

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
    # Access kinematic_chain from the wrapper's opt object, which is more reliable.
    dummy_dataset = MotionDataset(opt, mean, std, pjoin(opt.data_root, "kw_splits/train-wo-violence.txt"))

    scheduler = DDIMScheduler(num_train_timesteps=opt.num_train_timesteps, beta_start=opt.beta_start, beta_end=opt.beta_end, prediction_type=opt.prediction_type, clip_sample=False)
    trainer = DenoiserTrainer(opt, denoiser, vae, scheduler)
    
    console.rule("[bold green]📊 PART A: PRESERVATION (Standard Test Set)")
    val_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, "kw_splits/test-wo-violence", device=args_override.device, num_workers=0)
    
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
    """Generates motions for a concept, calculates metrics, and saves videos."""
    logger.info(f"--- Evaluating concept: '{concept}' ---")
    num_samples = 32
    texts = [concept] * num_samples
    m_lens = torch.LongTensor([100] * num_samples).to(opt.device)

    # Generate motion
    with torch.no_grad():
        dummy_motion = torch.zeros(num_samples, 196, opt.dim_pose).to(opt.device)
        pred_motions_features, _ = trainer.generate((texts, dummy_motion, m_lens))

    # --- METRIC CALCULATION FIX ---
    # The wrapper does not have 'get_sync_metrics'. We must compute embeddings manually.

    # 1. Get motion embeddings for the generated motions
    # The wrapper expects motions with shape (batch_size, frames, num_features)
    # and lengths as a tensor.
    motion_embeddings = wrapper.get_motion_embeddings(
        motions=pred_motions_features,
        m_lens=m_lens
    )

    # 2. Get text embeddings for the prompts
    # This requires creating word embeddings and pos_one_hots, similar to the dataset loader.
    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    text_embeddings_list = []
    for text in texts:
        tokens = text.split(' ')
        tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        word_emb, pos_oh = w_vectorizer[tokens]
        text_embeddings_list.append(word_emb[None, :, :]) # Add batch dimension

    text_embeddings = torch.from_numpy(np.concatenate(text_embeddings_list, axis=0)).float().to(opt.device)
    
    # The text encoder in the wrapper expects word_embeddings, pos_one_hots, and lengths.
    # We can pass dummy values for pos_one_hots and use the length of our text embeddings.
    cap_lens = torch.LongTensor([text_embeddings.shape[1]] * num_samples).to(opt.device)
    dummy_pos = torch.zeros(num_samples, text_embeddings.shape[1], 45).to(opt.device) # 45 is pos_ohot dim
    
    text_embeddings = wrapper.text_encoder(text_embeddings, dummy_pos, cap_lens)

    # 3. Calculate Matching Score
    # We use the euclidean distance, the same way test_denoiser does.
    dist_matrix = euclidean_distance_matrix(text_embeddings.cpu().numpy(), motion_embeddings.cpu().numpy())
    matching_score = dist_matrix.trace() / len(texts) # Average score per sample

    logger.info(f"Matching Score for '{concept}': {matching_score:.4f} (Lower is better for forgotten concepts)")

    # Plot videos
    pred_motions_denorm = dataset.inv_transform(pred_motions_features.cpu().numpy())
    plot_generated_motions(logger, pred_motions_denorm, opt, pjoin(output_dir, "videos"), concept.replace(" ", "_"), wrapper_opt.kinematic_chain)
    
# --- CORRECTED MAIN BLOCK ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Arguments to define which models to compare
    parser.add_argument("--original_name", type=str, required=True, help="Folder name of the original pre-trained model.")
    parser.add_argument("--original_ckpt", type=str, default="net_best_fid.tar", help="Checkpoint file for the original model.")
    parser.add_argument("--unlearned_name", type=str, required=True, help="Folder name of the new unlearned model.")
    parser.add_argument("--unlearned_ckpt", type=str, default="latest.tar", help="Checkpoint file for the unlearned model.")
    
    # Arguments to define the evaluation itself
    parser.add_argument("--dataset_name", type=str, default="t2m", help="Dataset name.")
    parser.add_argument("--target_concept", type=str, required=True, help="Concept that was unlearned.")
    parser.add_argument("--related_concepts", nargs='+', required=True, help="List of related concepts for the specificity test.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use.")
    
    args = parser.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    fixseed(42)

    logger = setup_logging("logs/eval_logs", f"{args.unlearned_name}_comparison")

    # 1. Evaluate the ORIGINAL model to get a baseline
    run_evaluation(
        logger=logger,
        model_name=args.original_name,
        ckpt_name=args.original_ckpt,
        is_unlearned=False,
        args_override=args
    )
    
    # 2. Evaluate the UNLEARNED model and compare
    run_evaluation(
        logger=logger,
        model_name=args.unlearned_name,
        ckpt_name=args.unlearned_ckpt,
        is_unlearned=True,
        args_override=args
    )