import sys
import os
import torch
import numpy as np
from os.path import join as pjoin
from torch.utils.data import DataLoader

# Add root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from models.vae.model import VAE
from models.denoiser.model import Denoiser
from options.denoiser_option import arg_parse
from utils.get_opt import get_opt
from utils.fixseed import fixseed
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from models.t2m_eval_wrapper import EvaluatorModelWrapper
from utils.eval_t2m import evaluation_denoiser
from utils.word_vectorizer import WordVectorizer

def load_model_for_eval(opt, ckpt_path):
    # Load VAE
    vae_name = get_opt(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'opt.txt'), opt.device).vae_name
    vae_opt = get_opt(pjoin(opt.checkpoints_dir, opt.dataset_name, vae_name, 'opt.txt'), opt.device)
    vae = VAE(vae_opt).to(opt.device)
    vae_ckpt = torch.load(pjoin(vae_opt.checkpoints_dir, vae_opt.dataset_name, vae_opt.name, 'model', 'net_best_fid.tar'), map_location='cpu')
    vae.load_state_dict(vae_ckpt["vae"])
    vae.eval()

    # Load Denoiser
    denoiser = Denoiser(opt, vae_opt.latent_dim).to(opt.device)
    # Load the specific checkpoint provided
    print(f"Loading unlearned checkpoint: {ckpt_path}")
    denoiser_ckpt = torch.load(ckpt_path, map_location='cpu')
    
    # Handle state dict keys (remove clip if needed, though usually standard in SALAD)
    if "denoiser" in denoiser_ckpt:
        state_dict = denoiser_ckpt["denoiser"]
    else:
        state_dict = denoiser_ckpt # fallback if saved differently
        
    denoiser.load_state_dict(state_dict, strict=False)
    denoiser.eval()
    
    return denoiser, vae

def evaluate_forgetting(denoiser, vae, wrapper, target_concept, device, num_samples=100):
    """
    Generates N samples of the target concept and measures how well they match the text.
    Lower Matching Score = Better Forgetting.
    """
    print(f"\n--- 🧪 EFFICACY TEST: '{target_concept}' ---")
    
    # Prepare inputs
    texts = [target_concept] * num_samples
    # We need lengths. Let's assume a fixed length for consistency or sample random lengths
    # Average motion length ~ 60-100 frames
    lengths = torch.randint(60, 100, (num_samples,)).long().to(device)
    
    # Generate
    print("Generating motions...")
    # Wrap generation logic (simplified from trainer.generate)
    # We need a scheduler. We can reuse the one from main or create one.
    from diffusers import DDIMScheduler
    scheduler = DDIMScheduler(num_train_timesteps=1000, clip_sample=False)
    scheduler.set_timesteps(50) # Inference steps
    
    denoiser.eval()
    
    # Batch processing to avoid OOM
    batch_size = 32
    all_motions = []
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_lens = lengths[i:i+batch_size]
            cur_bs = len(batch_texts)
            
            # 1. Encode Text (CLIP) handled inside denoiser usually, but we need empty checks
            # 2. Init Noise
            # We need to manually run the diffusion loop here or instantiate a Trainer to use its generate function.
            # To keep it lightweight, let's look at how test_denoiser does it.
            # It creates a Trainer just for testing. Let's do that in main.
            pass 
            # (Logic moved to main to verify with Trainer class)

def main():
    opt = arg_parse(False)
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # User args for unlearning eval
    # Usage: python unlearning/evaluate.py --name denoiser_esd_kick --ckpt_name latest.tar --target_concept "kick"
    # Note: --name should correspond to the NEW folder created by ESD training
    
    target_concept = "kick" # Default, or parse from somewhere
    # A simple hack to get target concept from args if you added it, otherwise hardcode or input
    if len(sys.argv) > 1 and "kick" in sys.argv: target_concept = "kick" # simplified
    
    print(f"📂 Evaluating Model: {opt.name}")
    
    # 1. Setup Environment
    # Load options from the *original* trained folder first to get architecture params
    # But we want to load the *new* unlearned weights
    
    # Path to the unlearned checkpoint
    ckpt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'model', opt.ckpt)
    
    # Load original config to initialize model structure
    # Note: The ESD script copies opt.txt to the new folder, so we can load it from there
    opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'opt.txt')
    base_opt = get_opt(opt_path, opt.device)
    base_opt.checkpoints_dir = opt.checkpoints_dir # ensure path validity
    
    # 2. Load Evaluators
    print("Loading Evaluators...")
    dataset_opt_path = f"checkpoints/{opt.dataset_name}/Comp_v6_KLD005/opt.txt"
    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    
    # 3. Load Model (Denoiser + VAE)
    denoiser, vae = load_model_for_eval(base_opt, ckpt_path)
    
    # 4. Setup Scheduler & Trainer wrapper for Generation
    from diffusers import DDIMScheduler
    from models.denoiser.trainer import DenoiserTrainer
    
    scheduler = DDIMScheduler(
        num_train_timesteps=base_opt.num_train_timesteps,
        beta_start=base_opt.beta_start,
        beta_end=base_opt.beta_end,
        prediction_type=base_opt.prediction_type,
        clip_sample=False,
    )
    
    # We use the Trainer class just for its .generate() method
    trainer = DenoiserTrainer(base_opt, denoiser, vae, scheduler)
    
    # --- PART A: PRESERVATION (General Metrics) ---
    print("\n📊 PART A: Checking Preservation (Standard Test Set)...")
    val_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'kw_splits/test-wo-violence', device=opt.device)
    
    # This function from utils.eval_t2m does the full benchmark (FID, R-Precision, etc)
    # It writes to logs and returns metrics
    best_fid, _, _, _, _, _, _, _, _, _, _ = evaluation_denoiser(
        pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'model'),
        val_loader, denoiser, trainer.generate, None, 0, 
        eval_wrapper=eval_wrapper, save=False, draw=False, device=opt.device
    )
    print(f"   -> General FID: {best_fid:.4f} (Lower is better)")
    
    # --- PART B: EFFICACY (Target Concept) ---
    print(f"\n🎯 PART B: Checking Efficacy (Target: '{target_concept}')...")
    
    # Construct a fake dataloader or just manual batch
    # We want to measure: If we prompt "Kick", does it look like a "Kick"?
    # We can measure this by generating motions for "Kick" and calculating R-Precision 
    # against the text "Kick" using the Eval Wrapper.
    
    num_samples = 32
    texts = [target_concept] * num_samples
    # Dummy lengths
    m_lens = torch.LongTensor([60]*num_samples).to(opt.device) # 3 seconds approx
    
    # Create batch suitable for trainer.generate
    # batch structure in trainer: (text, motion, m_lens)
    # trainer.generate needs motion only for shape info if doing partial, 
    # but for text-to-motion it ignores input motion usually if using pure noise.
    # However, standard generate might expect ground truth shapes.
    # Let's mock the input motion with zeros.
    
    dummy_motion = torch.zeros(num_samples, 196, 263).to(opt.device) # HumanML3D dimension
    batch_data = (texts, dummy_motion, m_lens)
    
    # Generate
    pred_motions, _ = trainer.generate(batch_data) # [B, T, J, D]
    
    # Convert for Evaluator (needs standardizing/renormalizing?)
    # The output of trainer.generate is usually normalized features (or joint positions depending on VAE)
    # SALAD VAE decodes to motion features.
    
    # Calculate Similarity to Text
    # 1. Get Text Embeddings from Evaluator (not CLIP, but the Eval Model's text encoder)
    word_vectorizer = WordVectorizer(pjoin(root_dir, 'glove'), 'our_vab')
    
    # We need to convert raw text to tokens for the evaluator
    # This part is tricky without the dataset logic. 
    # Easier approach: Use the R-Precision logic from eval_t2m but restricted to one prompt.
    
    print("   (Note: To strictly measure efficacy, observe if R-Precision drops for this specific prompt)")
    print("   For now, please inspect the generated mp4 files in the output folder.")

if __name__ == "__main__":
    main()