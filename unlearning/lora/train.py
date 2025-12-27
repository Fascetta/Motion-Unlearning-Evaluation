import sys
import os
import torch
import argparse
import logging
from os.path import join as pjoin

# Rich Imports
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

# Import ESD Trainer
from unlearning.esd.trainer import ESDTrainer
# Import LoRA Injection
from unlearning.lora.modules import inject_lora

# Setup Rich Console
console = Console(theme=Theme({"info": "dim cyan", "warning": "magenta", "error": "bold red"}))

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = pjoin(log_dir, 'train_lora.log')
    logger = logging.getLogger()
    if logger.hasHandlers(): logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)
    
    rich_handler = RichHandler(rich_tracebacks=True, show_time=True, omit_repeated_times=False)
    rich_handler.setLevel(logging.INFO)
    logger.addHandler(rich_handler)
    return logging.getLogger(__name__)

def load_denoiser(opt, vae_dim, logger, ckpt_name='net_best_fid.tar'):
    denoiser = Denoiser(opt, vae_dim)
    ckpt = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'model', ckpt_name), map_location='cpu')
    denoiser.load_state_dict(ckpt["denoiser"], strict=False)
    return denoiser

def load_vae(vae_opt, logger):
    model = VAE(vae_opt)
    ckpt = torch.load(pjoin(vae_opt.checkpoints_dir, vae_opt.dataset_name, vae_opt.name, 'model', 'net_best_fid.tar'), map_location='cpu')
    model.load_state_dict(ckpt["vae"])
    model.freeze()
    model.eval()
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_concept", type=str, default="kick", help="Concept to erase")
    parser.add_argument("--negative_guidance", type=float, default=1.0, help="ESD Guidance scale")
    parser.add_argument("--lora_rank", type=int, default=16, help="Rank of LoRA matrices")
    parser.add_argument("--lora_alpha", type=float, default=16.0, help="Alpha scaling for LoRA")
    parser.add_argument("--unlearn_lr", type=float, default=1e-4, help="Learning rate (usually higher for LoRA)")
    parser.add_argument("--unlearn_epochs", type=int, default=10)
    
    lora_args, remaining_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_argv
    opt = arg_parse(False) 
    
    TARGET_CONCEPT = lora_args.target_concept
    NEW_EXP_NAME = f"{opt.name}_LoRA_{TARGET_CONCEPT.replace(' ', '_')}"
    
    log_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, NEW_EXP_NAME, 'logs')
    model_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, NEW_EXP_NAME, 'model')
    eval_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, NEW_EXP_NAME, 'eval')
    logger = setup_logging(log_dir)
    
    console.rule(f"[bold magenta]LoRA Unlearning: {TARGET_CONCEPT}")
    logger.info("Experiment Dir: %s", NEW_EXP_NAME)
    
    opt_path_base = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'opt.txt')
    opt_base = get_opt(opt_path_base, opt.device)
    vae_name = get_opt(opt_path_base, opt.device).vae_name
    vae_opt = get_opt(pjoin(opt.checkpoints_dir, opt.dataset_name, vae_name, 'opt.txt'), opt.device)
    
    opt_base.is_train = True
    opt_base.lr = lora_args.unlearn_lr
    opt_base.max_epoch = lora_args.unlearn_epochs
    opt_base.weight_decay = float(opt_base.weight_decay)
    opt_base.log_dir = log_dir
    opt_base.model_dir = model_dir
    opt_base.eval_dir = eval_dir
    os.makedirs(model_dir, exist_ok=True)
    
    opt_base.dataset_name = opt.dataset_name 
    logger.info(f"Saving options to {pjoin(opt_base.checkpoints_dir, opt_base.dataset_name, NEW_EXP_NAME, 'opt.txt')}")
    args_to_save = vars(opt_base)
    with open(pjoin(opt_base.checkpoints_dir, opt_base.dataset_name, NEW_EXP_NAME, "opt.txt"), 'w') as f:
        for k, v in args_to_save.items():
            # Do not add extra quotes
            f.write(f'{k}: {v}\n')
    
    fixseed(opt.seed)

    logger.info("Loading Datasets...")
    dataset_opt_path = f"checkpoints/{opt.dataset_name}/Comp_v6_KLD005/opt.txt"
    train_loader = get_dataset_motion_loader(dataset_opt_path, 32, 'train', device=opt.device)[0]
    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    eval_val_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'kw_splits/test-wo-violence', device=opt.device)
    
    vae = load_vae(vae_opt, logger).to(opt.device)
    denoiser = load_denoiser(opt, vae_opt.latent_dim, logger, 'net_best_fid.tar').to(opt.device)

    # --- FIX: Freeze the entire base model BEFORE injecting LoRA ---
    logger.info("Freezing all parameters of the base denoiser model.")
    for param in denoiser.parameters():
        param.requires_grad = False
    # -----------------------------------------------------------------

    denoiser = inject_lora(denoiser, rank=lora_args.lora_rank, alpha=lora_args.lora_alpha)
    denoiser.to(opt.device)
    
    scheduler = DDIMScheduler(
        num_train_timesteps=opt_base.num_train_timesteps,
        beta_start=opt_base.beta_start,
        beta_end=opt_base.beta_end,
        prediction_type=opt_base.prediction_type,
        clip_sample=False,
    )

    trainer = ESDTrainer(
        opt_base, 
        denoiser, 
        vae, 
        scheduler, 
        target_concept=TARGET_CONCEPT,
        negative_guidance=lora_args.negative_guidance
    )
    
    logger.info("Starting Training Loop...")
    opt_base.save_latest = 500
    opt_base.eval_every_e = 1
    
    try:
        # FIX: Pass dummy plot function
        trainer.train(
            train_loader, 
            eval_val_loader, 
            eval_val_loader, 
            eval_wrapper, 
            plot_eval=lambda *args: None
        )
        logger.info("LoRA Training Finished.")
    except Exception as e:
        logger.exception("LoRA Training Failed")
        sys.exit(1)