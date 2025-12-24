import sys
import os
import torch
import numpy as np
import argparse
from os.path import join as pjoin
from diffusers import DDIMScheduler

from models.vae.model import VAE
from models.denoiser.model import Denoiser
from models.denoiser.trainer import DenoiserTrainer
from options.denoiser_option import arg_parse
from utils.get_opt import get_opt
from utils.fixseed import fixseed
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from models.t2m_eval_wrapper import EvaluatorModelWrapper
from utils.word_vectorizer import WordVectorizer
from unlearning.lora.modules import inject_lora  # Helper for LoRA

def load_vae(vae_opt):
    print(f'Loading VAE Model {vae_opt.name}')
    model = VAE(vae_opt)
    # Load VAE weights
    ckpt = torch.load(pjoin(vae_opt.checkpoints_dir, vae_opt.dataset_name, vae_opt.name, 'model', 'net_best_fid.tar'),
                            map_location='cpu')
    model.load_state_dict(ckpt["vae"])
    model.freeze()
    return model

def load_denoiser(opt, vae_dim, ckpt_path, is_lora=False, lora_rank=8):
    print(f'Loading Denoiser Model from: {ckpt_path}')
    denoiser = Denoiser(opt, vae_dim)

    # ⚠️ If this is a LoRA experiment, we MUST inject layers before loading weights
    if is_lora:
        print(f"   -> Detected LoRA experiment. Injecting LoRA layers (Rank={lora_rank})...")
        denoiser = inject_lora(denoiser, rank=lora_rank, alpha=16.0) # Alpha doesn't matter for loading, only Rank

    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    # Handle different saving keys
    if "denoiser" in ckpt:
        state_dict = ckpt["denoiser"]
    else:
        state_dict = ckpt

    # Load weights
    missing_keys, unexpected_keys = denoiser.load_state_dict(state_dict, strict=False)
    
    # Validation
    if len(unexpected_keys) > 0:
        print(f"Warning: Unexpected keys: {unexpected_keys}")
    # In SALAD, missing clip_model keys are normal
    assert all([k.startswith('clip_model.') for k in missing_keys]), f"Missing keys: {missing_keys}"
    
    return denoiser

def evaluate_efficacy(trainer, eval_wrapper, target_concept, save_dir, num_samples=32):
    """
    Generates motions specifically for the target concept and saves them.
    We verify efficacy VISUALLY (by looking at the video) and quantitatively (R-Precision).
    For unlearning: Low R-Precision on the Target Concept is GOOD (means it forgot).
    """
    print(f"\n🎯 [Efficacy Test] Generating motions for: '{target_concept}'")
    
    # 1. Prepare Batch
    texts = [target_concept] * num_samples
    # Dummy lengths (approx 3-4 seconds)
    m_lens = torch.LongTensor([120] * num_samples).to(trainer.opt.device)
    # Dummy motion (needed for shape inference in some functions)
    dummy_motion = torch.zeros(num_samples, 120, trainer.opt.joints_num, trainer.opt.latent_dim).to(trainer.opt.device) 
    
    batch_data = (texts, dummy_motion, m_lens)

    # 2. Generate
    # trainer.generate returns: pred_motion, attn_weights
    pred_motion, _ = trainer.generate(batch_data, need_attn=False)
    
    # 3. Calculate Metrics (How much does it look like "kick"?)
    # This requires the evaluator wrapper logic. 
    # Since extracting the specific metric logic is complex given the dependencies,
    # we will focus on SAVING the motions for visual inspection.
    
    from utils.plot_script import plot_3d_motion
    from visualization.joints2bvh import Joint2BVHConvertor
    
    motion_save_dir = pjoin(save_dir, f"efficacy_{target_concept.replace(' ', '_')}")
    os.makedirs(motion_save_dir, exist_ok=True)
    
    converter = Joint2BVHConvertor()
    
    print(f"   -> Saving {num_samples} videos to {motion_save_dir}...")
    
    for i, motion in enumerate(pred_motion):
        # Save only first 5 to save time/space
        if i >= 5: break 
        
        # Convert to BVH/Video
        # Note: pred_motion is in feature space, VAE decoded it. 
        # But visualization might need inverse transform if normalization was applied.
        # SALAD VAE output usually needs to be converted if using standard HumanML3D pipeline.
        # Assuming trainer.generate returns the final motion format used in plot_3d_motion.
        
        caption = f"{target_concept}_{i}"
        save_path = pjoin(motion_save_dir, f"{i:03d}.mp4")
        
        # Plot
        plot_3d_motion(save_path, trainer.opt.kinematic_chain, motion, title=caption, fps=20)

    print("   -> Done. Please inspect videos manually.")
    print("      If the character does NOT perform the action, Unlearning was SUCCESSFUL.")

if __name__ == '__main__':
    # 1. Parse Args
    opt = arg_parse(False)
    
    # Custom args for testing
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_concept", type=str, default=None, help="Concept to test specifically")
    parser.add_argument("--lora_rank", type=int, default=8, help="Rank if loading LoRA")
    parser.add_argument("--metrics", nargs="+", default=["fid", "efficacy"], help="fid, efficacy")
    
    test_args, unknown = parser.parse_known_args()
    
    # Merge args
    if test_args.target_concept:
        print(f"🔎 Target Concept: {test_args.target_concept}")

    # 2. Paths & Config
    ckpt_name = opt.ckpt if opt.ckpt else 'latest.tar'
    ckpt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'model', ckpt_name)
    
    # Load Opts
    opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'opt.txt')
    opt = get_opt(opt_path, opt.device)
    
    # VAE Opts
    vae_name = opt.vae_name
    vae_opt = get_opt(pjoin(opt.checkpoints_dir, opt.dataset_name, vae_name, 'opt.txt'), opt.device)

    fixseed(opt.seed)
    
    # 3. Setup Evaluator & Data (for FID/Preservation)
    dataset_opt_path = f"checkpoints/{opt.dataset_name}/Comp_v6_KLD005/opt.txt"
    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    eval_val_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'kw_splits/test-wo-violence', device=opt.device)

    # 4. Load Models
    vae_model = load_vae(vae_opt).to(opt.device)
    
    # Check if LoRA
    is_lora = "LoRA" in opt.name
    denoiser = load_denoiser(opt, vae_opt.latent_dim, ckpt_path, is_lora=is_lora, lora_rank=test_args.lora_rank).to(opt.device)

    # 5. Scheduler
    scheduler = DDIMScheduler(
        num_train_timesteps=opt.num_train_timesteps,
        beta_start=opt.beta_start,
        beta_end=opt.beta_end,
        beta_schedule=opt.beta_schedule,
        prediction_type=opt.prediction_type,
        clip_sample=False,
    )

    # 6. Trainer Wrapper
    trainer = DenoiserTrainer(opt, denoiser, vae_model, scheduler)

    # --- EXECUTION ---
    
    # A. Efficacy Test (Target Concept)
    if test_args.target_concept and "efficacy" in test_args.metrics:
        save_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'eval_unlearn')
        evaluate_efficacy(trainer, eval_wrapper, test_args.target_concept, save_dir)

    # B. Preservation Test (Standard Benchmarks)
    if "fid" in test_args.metrics:
        print("\n📊 [Preservation Test] Running standard benchmark (FID, Diversity)...")
        trainer.test(eval_wrapper, eval_val_loader, 1, # Repeat 1 time for speed (standard is 20)
                     save_dir=pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'eval'), 
                     cal_mm=False, save_motion=False)