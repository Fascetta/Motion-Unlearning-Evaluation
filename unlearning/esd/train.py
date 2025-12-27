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

# Import our new Trainer
from unlearning.esd.trainer import ESDTrainer

# --- OPTIMIZATION: Enable cuDNN autotuner ---
# This finds the best algorithm for the hardware and input shapes
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
    logger.info("Loading VAE Model %s", vae_opt.name)
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
    logger.info("Loading Denoiser Model %s", opt.name)
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
    # --- 1. HANDLE ARGUMENTS SAFELY ---
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_concept", type=str, default="kick", help="Concept to erase"
    )
    parser.add_argument(
        "--negative_guidance",
        type=float,
        default=1.0,
        help="ESD Guidance scale (1.0=Neutral, >1.0=Push away)",
    )
    parser.add_argument(
        "--unlearn_lr", type=float, default=1e-5, help="Learning rate for unlearning"
    )
    parser.add_argument(
        "--unlearn_epochs", type=int, default=10, help="Number of epochs to unlearn"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of workers for data loading"
    )

    unlearn_args, remaining_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_argv
    opt = arg_parse(False)

    TARGET_CONCEPT = unlearn_args.target_concept
    NEGATIVE_GUIDANCE = unlearn_args.negative_guidance
    NEW_EXP_NAME = f"{opt.name}_ESD_{TARGET_CONCEPT.replace(' ', '_')}"

    # --- 2. SETUP PATHS & LOGGING ---
    log_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, NEW_EXP_NAME, "logs")
    model_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, NEW_EXP_NAME, "model")
    eval_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, NEW_EXP_NAME, "eval")

    logger = setup_logging(log_dir)

    console.rule(f"[bold red]ESD Unlearning: {TARGET_CONCEPT}")
    logger.info("Experiment Dir: %s", NEW_EXP_NAME)
    logger.info("Target: %s", TARGET_CONCEPT)
    logger.info("Guidance: %.2f", NEGATIVE_GUIDANCE)
    logger.info("LR: %e", unlearn_args.unlearn_lr)
    logger.info("Batch Size: %d", unlearn_args.batch_size)
    logger.info("Num Workers: %d", unlearn_args.num_workers)


    # 3. Load Base Options
    opt_path_base = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, "opt.txt")
    if not os.path.exists(opt_path_base):
        logger.error("Cannot find base model options at %s", opt_path_base)
        sys.exit(1)

    vae_name = get_opt(opt_path_base, opt.device).vae_name
    vae_opt = get_opt(
        pjoin(opt.checkpoints_dir, opt.dataset_name, vae_name, "opt.txt"), opt.device
    )
    opt_base = get_opt(opt_path_base, opt.device)

    # 4. Overwrite Options for Unlearning
    opt_base.weight_decay = float(opt_base.weight_decay)
    opt_base.lr = unlearn_args.unlearn_lr
    opt_base.max_epoch = unlearn_args.unlearn_epochs
    opt_base.is_train = True
    opt_base.log_dir = log_dir
    opt_base.model_dir = model_dir
    opt_base.eval_dir = eval_dir
    opt_base.save_latest = 100
    opt_base.eval_every_e = 5

    os.makedirs(opt_base.model_dir, exist_ok=True)
    os.makedirs(opt_base.eval_dir, exist_ok=True)
    
    args_to_save = vars(opt_base)
    logger.info(f"Saving options to {pjoin(opt_base.checkpoints_dir, opt_base.dataset_name, NEW_EXP_NAME, 'opt.txt')}")
    args_to_save = vars(opt_base)
    with open(pjoin(opt_base.checkpoints_dir, opt_base.dataset_name, NEW_EXP_NAME, "opt.txt"), 'w') as f:
        for k, v in args_to_save.items():
            # Do not add extra quotes
            f.write(f'{k}: {v}\n')
    

    fixseed(opt.seed)

    # 5. Data Loaders
    logger.info("Loading Datasets...")
    dataset_opt_path = f"checkpoints/{opt.dataset_name}/Comp_v6_KLD005/opt.txt"
    train_loader, _ = get_dataset_motion_loader(
        dataset_opt_path, unlearn_args.batch_size, "train", device=opt.device, num_workers=unlearn_args.num_workers
    )

    wrapper_opt = get_opt(dataset_opt_path, torch.device("cuda"))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    eval_val_loader, _ = get_dataset_motion_loader(
        dataset_opt_path, unlearn_args.batch_size, "kw_splits/test-wo-violence", device=opt.device, num_workers=unlearn_args.num_workers
    )
    logger.info("Datasets Loaded.")

    # 6. Load Models
    vae_model = load_vae(vae_opt, logger).to(opt.device)
    denoiser = load_denoiser(
        opt, vae_opt.latent_dim, logger, ckpt_name="net_best_fid.tar"
    ).to(opt.device)
    
    # --- `torch.compile` REMOVED as it is not available in PyTorch 1.x ---

    # Freeze CLIP
    for param in denoiser.clip_model.parameters():
        param.requires_grad = False

    # 7. Scheduler
    scheduler = DDIMScheduler(
        num_train_timesteps=opt_base.num_train_timesteps,
        beta_start=opt_base.beta_start,
        beta_end=opt_base.beta_end,
        beta_schedule=opt_base.beta_schedule,
        prediction_type=opt_base.prediction_type,
        clip_sample=False,
    )

    # 8. Initialize Trainer
    trainer = ESDTrainer(
        opt_base,
        denoiser,
        vae_model,
        scheduler,
        target_concept=TARGET_CONCEPT,
        negative_guidance=NEGATIVE_GUIDANCE,
    )

    # 9. Start Training
    logger.info("Starting Training Loop...")
    try:
        trainer.train(
            train_loader,
            eval_val_loader,
            eval_val_loader,
            eval_wrapper,
            plot_eval=lambda *args: None,
        )
        logger.info("Training Finished Successfully.")
    except Exception as e:
        logger.exception("Training failed with error")
        sys.exit(1)