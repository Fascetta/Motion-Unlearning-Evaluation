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
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.theme import Theme

# --- 1. SETUP AND IMPORTS ---

# Correctly add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(pjoin(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.denoiser.model import Denoiser
from options.denoiser_option import arg_parse as base_arg_parse
from utils.get_opt import get_opt

# Setup Rich Console
console = Console(theme=Theme({"info": "dim cyan", "warning": "magenta", "error": "bold red"}))

# --- 2. HELPER FUNCTIONS ---

def setup_logging(log_dir, concept_name):
    """Sets up Rich logging to console and standard logging to file."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = pjoin(log_dir, f'edit_uce_kv_{concept_name}.log')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_file, mode='a')
    file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    rich_handler = RichHandler(rich_tracebacks=True, console=console, markup=True)
    rich_handler.setLevel(logging.INFO)
    logger.addHandler(rich_handler)
    
    return logging.getLogger(__name__)

def load_captions_from_file(filepath):
    """Loads text captions from a file, one per line."""
    with open(filepath, 'r') as f:
        captions = [line.strip() for line in f.readlines() if line.strip()]
    if not captions:
        raise ValueError(f"File {filepath} is empty or contains no valid captions.")
    return captions

def get_average_embedding_from_captions(denoiser, captions, batch_size=64):
    """Computes the single average embedding vector for a list of captions."""
    all_embeddings = []
    with torch.no_grad(), Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        transient=True
    ) as progress:
        task_desc = f"[green]Processing {len(captions)} captions..."
        task = progress.add_task(task_desc, total=len(captions))
        
        for i in range(0, len(captions), batch_size):
            batch = captions[i:i + batch_size]
            word_emb, _, _ = denoiser.clip_model.encode_text(batch)
            word_emb = denoiser.word_emb(word_emb)
            avg_emb_batch = word_emb.mean(dim=1)
            all_embeddings.append(avg_emb_batch.cpu())
            progress.update(task, advance=len(batch))

    final_avg_embedding = torch.cat(all_embeddings, dim=0).mean(dim=0)
    return final_avg_embedding

def apply_contrastive_uce_kv(denoiser, c_forget, c_retain, logger, lambda_param=1.0):
    """Applies Contrastive UCE to both Key (K) and Value (V) projection matrices."""
    logger.info(f"Applying official UCE (Key & Value) with lambda = {lambda_param:.2f}...")
    
    all_st_transformer_blocks = [
        m for m in denoiser.transformer.modules() if m.__class__.__name__ == "STTransformerLayer"
    ]
    if not all_st_transformer_blocks:
        logger.error("Could not find any 'STTransformerLayer' blocks in the denoiser!")
        raise ValueError("Architecture mismatch.")

    model_device = next(denoiser.parameters()).device
    direction_to_erase = (c_forget - c_retain).to(model_device)
    direction_normalized = direction_to_erase / torch.norm(direction_to_erase)

    count = 0
    for block in all_st_transformer_blocks:
        if hasattr(block, 'cross_attn'):
            ca_module = block.cross_attn
            with torch.no_grad():
                W_k = ca_module.Wk.weight
                proj_k = torch.mv(W_k, direction_to_erase)
                correction_k = torch.outer(proj_k, direction_normalized)
                W_k -= lambda_param * correction_k

                W_v = ca_module.Wv.weight
                proj_v = torch.mv(W_v, direction_to_erase)
                correction_v = torch.outer(proj_v, direction_normalized)
                W_v -= lambda_param * correction_v
            count += 1
            
    logger.info(f"✅ Edited Key & Value matrices in {count} Cross-Attention layers.")
    return denoiser

# --- 3. MAIN EXECUTION BLOCK ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="UCE Contrastive Editing Script")
    parser.add_argument("--forget_file", type=str, required=True, help="Path to text file with captions to FORGET.")
    parser.add_argument("--retain_file", type=str, required=True, help="Path to text file with captions to RETAIN.")
    parser.add_argument("--uce_lambda", type=float, default=1.0, help="Strength of the UCE edit.")
    # --- THIS IS THE FIX ---
    # We add vae_name here and set the correct default.
    parser.add_argument("--vae_name", type=str, default="t2m_vae_gelu", help="Name of the VAE model to use.")
    
    uce_args, remaining_argv = parser.parse_known_args()

    original_argv = sys.argv
    sys.argv = [original_argv[0]] + remaining_argv
    
    opt = base_arg_parse(is_train=False)

    sys.argv = original_argv
    
    opt.forget_file = uce_args.forget_file
    opt.retain_file = uce_args.retain_file
    opt.uce_lambda = uce_args.uce_lambda
    opt.vae_name = uce_args.vae_name  # Add the vae_name to the final opt object
    # --- END OF FIX ---

    concept_name = os.path.basename(opt.forget_file).replace('.txt', '').replace('train_val-w-', '')
    NEW_EXP_NAME = f"{opt.name}_UCE_KV_contrast_{concept_name}"
    
    log_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, NEW_EXP_NAME, 'logs')
    save_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, NEW_EXP_NAME, 'model')
    os.makedirs(save_dir, exist_ok=True)
    
    logger = setup_logging(log_dir, concept_name)
    
    console.rule(f"[bold yellow]Official UCE (K/V) Edit: Forgetting '{concept_name}'")
    logger.info(f"Original Model: [cyan]{opt.name}[/cyan]")
    logger.info(f"New Experiment: [cyan]{NEW_EXP_NAME}[/cyan]")
    logger.info(f"Forget Set File: [red]{opt.forget_file}[/red]")
    logger.info(f"Retain Set File: [green]{opt.retain_file}[/green]")
    logger.info(f"Lambda (Strength): {opt.uce_lambda:.2f}")

    try:
        base_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'opt.txt')
        base_opt = get_opt(base_opt_path, opt.device)
        
        # --- THIS IS THE FIX (in action) ---
        # The script now uses the vae_name from our arguments, ignoring the incorrect one in the file.
        vae_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vae_name, 'opt.txt')
        vae_opt = get_opt(vae_opt_path, opt.device)
        logger.info(f"Using VAE model: [cyan]{opt.vae_name}[/cyan]")
        
        ckpt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'model', 'net_best_fid.tar')
        logger.info(f"Loading Denoiser from: [cyan]{ckpt_path}[/cyan]")
        denoiser = Denoiser(base_opt, vae_opt.latent_dim).to(opt.device)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        denoiser.load_state_dict(ckpt.get("denoiser", ckpt), strict=False)
        denoiser.eval()

    except FileNotFoundError as e:
        logger.error(f"Could not find configuration or model file: {e}")
        sys.exit(1)

    try:
        logger.info("Loading captions from files...")
        forget_captions = load_captions_from_file(opt.forget_file)
        retain_captions = load_captions_from_file(opt.retain_file)
        logger.info(f"Loaded {len(forget_captions)} 'forget' and {len(retain_captions)} 'retain' captions.")

        logger.info("Computing average embedding for FORGET set...")
        forget_embedding = get_average_embedding_from_captions(denoiser, forget_captions)
        
        logger.info("Computing average embedding for RETAIN set...")
        retain_embedding = get_average_embedding_from_captions(denoiser, retain_captions)
        
        denoiser = apply_contrastive_uce_kv(denoiser, forget_embedding, retain_embedding, logger, lambda_param=opt.uce_lambda)
        
        save_path = pjoin(save_dir, "latest.tar")
        logger.info(f"Saving edited model to: [green]{save_path}[/green]")
        torch.save({"denoiser": denoiser.state_dict_without_clip(), "epoch": 0}, save_path)
        shutil.copy(base_opt_path, pjoin(opt.checkpoints_dir, opt.dataset_name, NEW_EXP_NAME, 'opt.txt'))
        
        console.rule("[bold green]Official UCE (K/V) Editing Complete!")
        logger.info(f"To evaluate, use the experiment name: {NEW_EXP_NAME}")
        
    except Exception as e:
        logger.exception("Contrastive UCE editing failed due to an unexpected error.")
        sys.exit(1)