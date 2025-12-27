import sys
import os
import argparse
import torch
from os.path import join as pjoin
from diffusers import DDIMScheduler
from rich.console import Console
import numpy as np

# --- 1. SETUP AND IMPORTS ---

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
# --- THIS IS THE FIX ---
if root_dir not in sys.path:
    sys.path.append(root_dir)
# --- END OF FIX ---

from models.vae.model import VAE
from models.denoiser.model import Denoiser
from models.denoiser.trainer import DenoiserTrainer
from utils.get_opt import get_opt
from utils.fixseed import fixseed
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from models.t2m_eval_wrapper import EvaluatorModelWrapper
import utils.metrics as metrics_module
from unlearning.lora.modules import inject_lora

console = Console()

# --- 2. MONKEY-PATCH FOR DIVERSITY METRIC ---

original_calculate_diversity = metrics_module.calculate_diversity
def safe_calculate_diversity(activations, times):
    num_samples = activations.shape[0]
    safe_times = min(times, num_samples - 1) if num_samples > 1 else 1
    if num_samples > 0 and safe_times < 1: safe_times = 1
    return original_calculate_diversity(activations, safe_times)
metrics_module.calculate_diversity = safe_calculate_diversity
print("Applied a safe monkey-patch to utils.metrics.calculate_diversity.")

# --- 3. HELPER FUNCTIONS ---

def load_options(opt_path, device, default_overrides={}):
    if not os.path.exists(opt_path):
        raise FileNotFoundError(f"FATAL: Cannot find options file at {opt_path}")
    opt = get_opt(opt_path, device)
    for key, value in default_overrides.items():
        if not hasattr(opt, key):
            setattr(opt, key, value)
    return opt

def load_vae(opt):
    print(f"Loading VAE Model: {opt.name}")
    model = VAE(opt).to(opt.device)
    ckpt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'model', 'net_best_fid.tar')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt["vae"])
    model.freeze(); model.eval()
    return model

# --- 4. MAIN EXECUTION BLOCK ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Unlearning Evaluation Script")
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--vae_name", type=str, required=True)
    parser.add_argument("--forget_test_file", type=str, required=True)
    parser.add_argument("--retain_test_file", type=str, required=True)
    parser.add_argument("--ckpt_name", type=str, default="latest.tar")
    parser.add_argument("--is_lora_model", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=123)
    opt = parser.parse_args()
    
    device = torch.device(f'cuda:{opt.gpu_id}' if torch.cuda.is_available() else 'cpu')
    fixseed(opt.seed)
    opt.device = device

    denoiser_opt_path = pjoin('./checkpoints', 't2m', opt.name, 'opt.txt')
    denoiser_opt = load_options(denoiser_opt_path, device)
    
    vae_opt_path = pjoin(denoiser_opt.checkpoints_dir, denoiser_opt.dataset_name, opt.vae_name, 'opt.txt')
    vae_opt = load_options(vae_opt_path, device, {'latent_dim': 256, 'activation': 'gelu', 'n_extra_layers': 1})
    
    dataset_opt_path = f"checkpoints/{denoiser_opt.dataset_name}/Comp_v6_KLD005/opt.txt"
    evaluator_opt = load_options(dataset_opt_path, device, {
        'max_text_len': 20,
        'dim_movement_enc_hidden': 512,
        'dim_movement_latent': 512
    })
    
    print("\nLoading models...")
    vae_model = load_vae(vae_opt)

    denoiser_model = Denoiser(denoiser_opt, vae_opt.latent_dim)
    if opt.is_lora_model:
        print(f"Injecting LoRA layers with rank {opt.lora_rank} for evaluation...")
        denoiser_model = inject_lora(denoiser_model, rank=opt.lora_rank)

    ckpt_path = pjoin(denoiser_opt.checkpoints_dir, denoiser_opt.dataset_name, opt.name, 'model', opt.ckpt_name)
    print(f"Loading Denoiser checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    denoiser_model.load_state_dict(ckpt["denoiser"], strict=False)
    denoiser_model.to(device)
    
    scheduler = DDIMScheduler(
        num_train_timesteps=denoiser_opt.num_train_timesteps, beta_start=denoiser_opt.beta_start,
        beta_end=denoiser_opt.beta_end, beta_schedule=denoiser_opt.beta_schedule,
        prediction_type=denoiser_opt.prediction_type, clip_sample=False,
    )
    trainer = DenoiserTrainer(denoiser_opt, denoiser_model, vae_model, scheduler)

    temp_opt_path = pjoin(os.path.dirname(denoiser_opt_path), "temp_dataset_opt_for_eval.txt")
    try:
        with open(temp_opt_path, 'w') as f:
            for k, v in vars(evaluator_opt).items():
                f.write(f'{k}: {v}\n')
        
        eval_wrapper = EvaluatorModelWrapper(evaluator_opt)

        console.rule("[bold red]🎯 Efficacy Evaluation (FORGET set) 🎯")
        forget_loader, _ = get_dataset_motion_loader(temp_opt_path, 32, opt.forget_test_file.replace('.txt', ''), device=device)
        efficacy_save_dir = pjoin(denoiser_opt.checkpoints_dir, denoiser_opt.dataset_name, opt.name, 'eval_efficacy')
        trainer.test(eval_wrapper, forget_loader, 1, save_dir=efficacy_save_dir, cal_mm=False, save_motion=False)
        console.print(f"[green]Efficacy results saved to: {efficacy_save_dir}[/green]\n")

        console.rule("[bold green]📊 Preservation Evaluation (RETAIN set) 📊")
        retain_loader, _ = get_dataset_motion_loader(temp_opt_path, 32, opt.retain_test_file.replace('.txt', ''), device=device)
        preservation_save_dir = pjoin(denoiser_opt.checkpoints_dir, denoiser_opt.dataset_name, opt.name, 'eval_preservation')
        trainer.test(eval_wrapper, retain_loader, 1, save_dir=preservation_save_dir, cal_mm=False, save_motion=False)
        console.print(f"[green]Preservation results saved to: {preservation_save_dir}[/green]")

    finally:
        if os.path.exists(temp_opt_path):
            os.remove(temp_opt_path)
            print(f"\nCleaned up temporary file: {temp_opt_path}")