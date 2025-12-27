import sys
import os
import torch
import argparse
import logging
import shutil
from os.path import join as pjoin

# Rich Imports
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme

# Add root directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

from models.denoiser.model import Denoiser
from models.vae.model import VAE
from options.denoiser_option import arg_parse
from utils.get_opt import get_opt

# Setup Rich Console
console = Console(theme=Theme({"info": "dim cyan", "warning": "magenta", "error": "bold red"}))

def setup_logging(log_dir):
    """Sets up Rich logging to console and standard logging to file."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = pjoin(log_dir, 'edit.log')
    
    # 1. Root Logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers if any
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # 2. File Handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # 3. Rich Console Handler
    rich_handler = RichHandler(rich_tracebacks=True, show_time=True, omit_repeated_times=False)
    rich_handler.setLevel(logging.INFO)
    logger.addHandler(rich_handler)
    
    return logging.getLogger(__name__)

def load_denoiser(opt, vae_dim, logger, ckpt_name='net_best_fid.tar'):
    logger.info('Loading Denoiser Model %s', opt.name)
    denoiser = Denoiser(opt, vae_dim)
    ckpt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'model', ckpt_name)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    denoiser.load_state_dict(ckpt["denoiser"], strict=False)
    denoiser.eval()
    return denoiser

def get_concept_embeddings(denoiser, texts, device):
    """
    Extracts the CLIP text embeddings (keys/values input) for the given texts.
    """
    with torch.no_grad():
        word_emb, _, _ = denoiser.clip_model.encode_text(texts)
        word_emb = denoiser.word_emb(word_emb) 
        return word_emb

def apply_uce(denoiser, target_emb, neutral_emb, logger, lambda_param=0.1):
    """
    Applies Closed-Form UCE update to all Cross-Attention layers by finding all
    STTransformerLayer modules. This is robust to architecture changes.
    """
    logger.info("Applying UCE with lambda=%.2f...", lambda_param)
    
    # --- FINAL, ROBUST FIX: Find all STTransformerLayer modules dynamically ---
    all_st_transformer_blocks = []
    for module in denoiser.transformer.modules():
        # The class name is STTransformerLayer in transformer.py
        if module.__class__.__name__ == "STTransformerLayer":
            all_st_transformer_blocks.append(module)
    
    if not all_st_transformer_blocks:
        logger.error("Could not find any STTransformerLayer blocks in the denoiser.transformer!")
        raise ValueError("Architecture mismatch or error in finding transformer blocks.")

    c_target = target_emb.mean(dim=1).squeeze(0) 
    c_neutral = neutral_emb.mean(dim=1).squeeze(0)
    
    count = 0
    # Iterate through the dynamically found blocks
    for block in all_st_transformer_blocks:
        if hasattr(block, 'cross_attn'):
            ca_module = block.cross_attn
            
            W_v = ca_module.Wv.weight.data
            
            diff = torch.mv(W_v, c_target - c_neutral)
            correction = torch.outer(diff, c_target)
            norm_factor = torch.dot(c_target, c_target)
            
            ca_module.Wv.weight.data -= lambda_param * (correction / (norm_factor + 1e-6))
            count += 1
        
    logger.info("✅ Edited %d Cross-Attention layers.", count)
    return denoiser

if __name__ == '__main__':
    # --- 1. HANDLE ARGUMENTS SAFELY ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_concept", type=str, default="kick", help="Concept to erase")
    parser.add_argument("--uce_lambda", type=float, default=1.0, help="Strength of UCE edit")
    
    # Parse known args
    uce_args, remaining_argv = parser.parse_known_args()
    
    # Hack: Modify sys.argv to hide custom args from base parser
    sys.argv = [sys.argv[0]] + remaining_argv
    
    # Now call the base parser
    opt = arg_parse(False)
    
    # Restore variables
    TARGET_CONCEPT = uce_args.target_concept
    LAMBDA = uce_args.uce_lambda
    NEW_EXP_NAME = f"{opt.name}_UCE_{TARGET_CONCEPT.replace(' ', '_')}"
    
    # --- 2. SETUP PATHS & LOGGING ---
    log_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, NEW_EXP_NAME, 'logs')
    save_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, NEW_EXP_NAME, 'model')
    
    logger = setup_logging(log_dir)
    
    console.rule(f"[bold cyan]UCE Editing: {TARGET_CONCEPT}")
    logger.info("Experiment Dir: %s", NEW_EXP_NAME)
    logger.info("Target: %s", TARGET_CONCEPT)
    logger.info("Lambda: %.2f", LAMBDA)
    
    # --- 3. LOAD CONFIG ---
    opt_path_base = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'opt.txt')
    if not os.path.exists(opt_path_base):
        logger.error("Base model options not found at %s", opt_path_base)
        sys.exit(1)

    opt_base = get_opt(opt_path_base, opt.device)
    vae_name = get_opt(opt_path_base, opt.device).vae_name
    vae_opt = get_opt(pjoin(opt.checkpoints_dir, opt.dataset_name, vae_name, 'opt.txt'), opt.device)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # --- 4. EXECUTE UCE ---
    try:
        # Load Model
        denoiser = load_denoiser(opt, vae_opt.latent_dim, logger).to(opt.device)
        
        # Calculate Embeddings
        logger.info("Computing Concept Embeddings...")
        target_emb = get_concept_embeddings(denoiser, [TARGET_CONCEPT], opt.device)
        neutral_emb = get_concept_embeddings(denoiser, [""], opt.device)
        
        # Apply Edit
        denoiser = apply_uce(denoiser, target_emb, neutral_emb, logger, lambda_param=LAMBDA)
        
        # --- 5. SAVE ---
        save_path = pjoin(save_dir, "latest.tar")
        logger.info("Saving edited model to %s...", save_path)
        
        state = {
            "denoiser": denoiser.state_dict_without_clip(),
            "epoch": 0,
            "total_iter": 0,
        }
        torch.save(state, save_path)
        
        # Copy opt.txt
        shutil.copy(opt_path_base, pjoin(opt.checkpoints_dir, opt.dataset_name, NEW_EXP_NAME, 'opt.txt'))
        
        logger.info("Done! Evaluation ready.")
        
    except Exception as e:
        logger.exception("UCE Editing Failed")
        sys.exit(1)