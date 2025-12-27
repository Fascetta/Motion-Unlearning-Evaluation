import sys
import os
import torch
import argparse
import logging
from os.path import join as pjoin

from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme

# Add root directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

from models.vae.model import VAE
from models.denoiser.model import Denoiser
from options.denoiser_option import arg_parse
from utils.get_opt import get_opt
from utils.fixseed import fixseed
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from models.t2m_eval_wrapper import EvaluatorModelWrapper
from diffusers import DDIMScheduler

# Import the refactored Trainer
from unlearning.esd.trainer import ESDTrainer

# --- OPTIMIZATION: Enable cuDNN autotuner ---
torch.backends.cudnn.benchmark = True

console = Console(
    theme=Theme({"info": "dim cyan", "warning": "magenta", "error": "bold red"})
)


def setup_logging(log_dir):
    """Sets up Rich logging to console and standard logging to file."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = pjoin(log_dir, "train.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_file, mode="a")
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    rich_handler = RichHandler(
        rich_tracebacks=True, show_time=True, omit_repeated_times=False
    )
    rich_handler.setLevel(logging.INFO)
    logger.addHandler(rich_handler)
    return logging.getLogger(__name__)


def load_vae(vae_opt, logger):
    logger.info("Loading VAE Model: %s", vae_opt.name)
    model = VAE(vae_opt)
    ckpt_path = pjoin(
        vae_opt.checkpoints_dir,
        vae_opt.dataset_name,
        vae_opt.name,
        "model",
        "net_best_fid.tar",
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["vae"])
    model.freeze()
    model.eval()
    return model


def load_denoiser(opt, vae_dim, logger, ckpt_name="net_best_fid.tar"):
    logger.info("Loading Denoiser Model: %s", opt.name)
    denoiser = Denoiser(opt, vae_dim)
    ckpt_path = pjoin(
        opt.checkpoints_dir, opt.dataset_name, opt.name, "model", ckpt_name
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    missing_keys, unexpected_keys = denoiser.load_state_dict(
        ckpt["denoiser"], strict=False
    )
    if not all([k.startswith("clip_model.") for k in missing_keys]):
        logger.warning("Unexpected missing keys: %s", missing_keys)
    return denoiser


if __name__ == "__main__":
    # --- 1. PARSE UNLEARNING AND BASE ARGUMENTS SEPARATELY ---
    parser = argparse.ArgumentParser(description="Unlearning Script Arguments")
    parser.add_argument("--unlearn_lr", type=float, default=1e-5, help="Learning rate for unlearning")
    parser.add_argument("--unlearn_epochs", type=int, default=3, help="Number of epochs to unlearn")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")
    parser.add_argument("--preservation_weight", type=float, default=0.1, help="Weight for the preservation loss (lambda)")
    parser.add_argument("--forget_split_file", type=str, required=True, help="Path to the split file for the concept to forget")
    parser.add_argument("--preserve_split_file", type=str, required=True, help="Path to the split file for the data to preserve")
    parser.add_argument("--vae_name", type=str, required=True, help="Name of the pretrained VAE model directory (e.g., 't2m_vae_gelu')")

    unlearn_args, remaining_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_argv

    # --- 2. CREATE A COMPLETE CONFIGURATION OBJECT ---
    opt = arg_parse(is_train=True)
    
    opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'opt.txt')
    if not os.path.exists(opt_path):
        print(f"FATAL: Cannot find Denoiser options at {opt_path}")
        sys.exit(1)
        
    file_opt = get_opt(opt_path, opt.device)
    
    for k, v in vars(file_opt).items():
        setattr(opt, k, v)
        
    # --- 3. APPLY UNLEARNING-SPECIFIC SETTINGS ---
    opt.lr = unlearn_args.unlearn_lr
    opt.max_epoch = unlearn_args.unlearn_epochs
    opt.batch_size = unlearn_args.batch_size
    opt.num_workers = unlearn_args.num_workers
    opt.is_train = True
    opt.weight_decay = 0.0

    # --- 4. SETUP NEW PATHS, LOGGING, AND SEED ---
    concept_name = os.path.basename(unlearn_args.forget_split_file).split('.')[0].replace('train_val-w-', '')
    NEW_EXP_NAME = f"{opt.name}_ESD_{concept_name}"
    
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, NEW_EXP_NAME)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.eval_dir = pjoin(opt.save_root, 'eval')
    opt.log_dir = pjoin(opt.save_root, 'logs')

    os.makedirs(opt.model_dir, exist_ok=True)
    
    logger = setup_logging(opt.log_dir)
    fixseed(opt.seed)

    console.rule(f"[bold red]ESD Unlearning for: {concept_name}")
    logger.info("Experiment Dir: %s", NEW_EXP_NAME)
    logger.info("Denoiser Model: %s", opt.name)
    logger.info("VAE Model: %s", unlearn_args.vae_name)
    
    with open(pjoin(opt.save_root, "opt.txt"), 'w') as f:
        for k, v in sorted(vars(opt).items()):
            f.write(f'{k}: {v}\n')

    # --- 5. LOAD VAE (with robust patching) ---
    vae_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, unlearn_args.vae_name, 'opt.txt')
    if not os.path.exists(vae_opt_path):
        logger.error("Cannot find VAE options at %s", vae_opt_path)
        sys.exit(1)
    vae_opt = get_opt(vae_opt_path, opt.device)
    
    vae_defaults = {'latent_dim': 256, 'activation': 'gelu', 'n_extra_layers': 1}
    for key, value in vae_defaults.items():
        if not hasattr(vae_opt, key):
            logger.warning(f"VAE Opt file missing '{key}'. Setting to default '{value}'.")
            setattr(vae_opt, key, value)
    
    vae_model = load_vae(vae_opt, logger).to(opt.device)

    # --- 6. LOAD DATA AND EVALUATION WRAPPER ---
    logger.info("Loading Datasets...")
    dataset_opt_path = f"checkpoints/{opt.dataset_name}/Comp_v6_KLD005/opt.txt"
    
    # --- THE DEFINITIVE FIX FOR THE DATALOADER ---
    # 1. Load the sparse dataset options file
    dataset_opt = get_opt(dataset_opt_path, opt.device)
    
    # 2. Patch it with the missing 'max_text_len' parameter
    if not hasattr(dataset_opt, 'max_text_len'):
        logger.warning("Dataset Opt file missing 'max_text_len'. Setting to default 20.")
        dataset_opt.max_text_len = 20
        
    # 3. Save the patched options to a temporary file in the experiment's log directory
    temp_opt_path = pjoin(opt.log_dir, "temp_dataset_opt.txt")
    with open(temp_opt_path, 'w') as f:
        for k, v in vars(dataset_opt).items():
            f.write(f'{k}: {v}\n')
            
    # 4. Use the path to this new, complete temporary file for all data loaders
    logger.info("Using temporary patched dataset options from %s", temp_opt_path)
    
    forget_split_name = unlearn_args.forget_split_file.replace('.txt', '')
    preserve_split_name = unlearn_args.preserve_split_file.replace('.txt', '')
    
    forget_loader, _ = get_dataset_motion_loader(
        temp_opt_path, opt.batch_size, forget_split_name,
        device=opt.device, num_workers=opt.num_workers
    )
    preserve_loader, _ = get_dataset_motion_loader(
        temp_opt_path, opt.batch_size, preserve_split_name,
        device=opt.device, num_workers=opt.num_workers
    )
    
    wrapper_opt = get_opt(dataset_opt_path, torch.device("cuda"))
    eval_defaults = {'dim_movement_enc_hidden': 512, 'dim_movement_latent': 512}
    for key, value in eval_defaults.items():
        if not hasattr(wrapper_opt, key):
            logger.warning(f"Evaluator Opt file missing '{key}'. Setting to default '{value}'.")
            setattr(wrapper_opt, key, value)
            
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    eval_val_loader, _ = get_dataset_motion_loader(
        temp_opt_path, opt.batch_size, "val",
        device=opt.device, num_workers=opt.num_workers
    )
    logger.info("Datasets Loaded.")

    # --- 7. LOAD DENOISER AND SCHEDULER ---
    denoiser = load_denoiser(opt, vae_opt.latent_dim, logger).to(opt.device)
    for param in denoiser.clip_model.parameters():
        param.requires_grad = False

    scheduler = DDIMScheduler(
        num_train_timesteps=opt.num_train_timesteps, beta_start=opt.beta_start,
        beta_end=opt.beta_end, beta_schedule=opt.beta_schedule,
        prediction_type=opt.prediction_type, clip_sample=False,
    )

    # --- 8. INITIALIZE AND RUN TRAINER ---
    trainer = ESDTrainer(
        opt, denoiser, vae_model, scheduler,
        preservation_weight=unlearn_args.preservation_weight
    )

    logger.info("Starting Training Loop...")
    try:
        trainer.train(
            forget_loader=forget_loader, preserve_loader=preserve_loader,
            eval_val_loader=eval_val_loader, eval_wrapper=eval_wrapper,
            plot_eval=lambda *args: None,
        )
        logger.info("Training Finished Successfully.")
    except Exception as e:
        logger.exception("Training failed with error")
        sys.exit(1)