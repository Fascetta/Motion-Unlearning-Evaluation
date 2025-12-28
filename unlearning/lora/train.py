"""
This script trains a LoRA (Low-Rank Adaptation) model for contrastive unlearning
in a diffusion-based motion generation model. The goal is to make the model
"forget" a specific concept (e.g., a type of motion) while retaining its
general knowledge.
"""

import argparse
import itertools
import logging
import os
import shutil
import sys
from os.path import join as pjoin

import torch
from diffusers import DDIMScheduler

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.theme import Theme

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(pjoin(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.denoiser.model import Denoiser
from models.vae.model import VAE
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from options.denoiser_option import arg_parse as base_arg_parse
from unlearning.lora.modules import inject_lora
from utils.fixseed import fixseed
from utils.get_opt import get_opt

console = Console(
    theme=Theme({"info": "dim cyan", "warning": "magenta", "error": "bold red"})
)


def setup_logging(_log_dir):
    """Initializes logging to both file and a rich console."""
    os.makedirs(_log_dir, exist_ok=True)
    log_file = pjoin(_log_dir, "train_lora.log")
    _logger = logging.getLogger()
    _logger.setLevel(logging.INFO)
    if _logger.hasHandlers():
        _logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    _logger.addHandler(file_handler)

    rich_handler = RichHandler(rich_tracebacks=True, console=console)
    rich_handler.setLevel(logging.INFO)
    _logger.addHandler(rich_handler)
    return logging.getLogger(__name__)


def load_options(opt_path, device, default_overrides=None):
    """Loads options from a file and applies defaults if attributes are missing."""
    if default_overrides is None:
        default_overrides = {}
    if not os.path.exists(opt_path):
        raise FileNotFoundError(f"Cannot find options file at {opt_path}")
    _opt = get_opt(opt_path, device)
    for key, value in default_overrides.items():
        if not hasattr(_opt, key):
            setattr(_opt, key, value)
    return _opt


def load_vae(_vae_opt, _logger):
    """Loads and returns a pre-trained VAE model in evaluation mode."""
    _logger.info("Loading VAE model: %s", _vae_opt.name)
    model = VAE(_vae_opt)
    ckpt_path = pjoin(
        _vae_opt.checkpoints_dir,
        _vae_opt.dataset_name,
        _vae_opt.name,
        "model",
        "net_best_fid.tar",
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["vae"])
    model.freeze()
    model.eval()
    return model


def load_denoiser(_opt, vae_dim, _logger):
    """Loads and returns a pre-trained Denoiser model."""
    _logger.info("Loading Denoiser model: %s", _opt.name)
    ckpt_path = pjoin(
        _opt.checkpoints_dir, _opt.dataset_name, _opt.name, "model", "net_best_fid.tar"
    )
    _denoiser = Denoiser(_opt, vae_dim)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    _denoiser.load_state_dict(ckpt["denoiser"], strict=False)
    return _denoiser


class LoRAContrastiveTrainer:
    """
    Manages the training loop for LoRA-based contrastive unlearning.
    """

    def __init__(
        self, _opt, _denoiser, _vae, _scheduler, _retain_loader, _forget_loader, _logger
    ):
        self.opt = _opt
        self.denoiser = _denoiser
        self.vae = _vae
        self.scheduler = _scheduler
        self.retain_loader = _retain_loader
        self.forget_loader = _forget_loader
        self.logger = _logger
        self.device = _opt.device

        trainable_params = [p for p in self.denoiser.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=_opt.unlearn_lr, weight_decay=float(_opt.weight_decay)
        )
        num_params = sum(p.numel() for p in trainable_params)
        self.logger.info(
            "Initialized LoRA Trainer. Trainable parameters: %s", f"{num_params:,}"
        )

    def _compute_loss(self, batch):
        """Computes the MSE loss for a given batch."""
        motions = batch[4].to(self.device, dtype=torch.float32)
        texts = batch[2]
        with torch.no_grad():
            z, _ = self.vae.encode(motions)
        # pylint: disable=no-member
        noise = torch.randn_like(z)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (z.shape[0],),
            device=self.device,
        ).long()
        noisy_z = self.scheduler.add_noise(z, noise, timesteps)
        noise_pred, _ = self.denoiser(noisy_z, timesteps, texts)
        return torch.nn.functional.mse_loss(noise_pred, noise)

    def train(self):
        """Main training loop with progress visualization."""
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            console=console,
        )
        with progress:
            epoch_task = progress.add_task("[cyan]Epochs", total=self.opt.max_epoch)
            for epoch in range(self.opt.max_epoch):
                total_steps = len(self.retain_loader)
                if self.opt.max_steps_per_epoch is not None:
                    total_steps = min(total_steps, self.opt.max_steps_per_epoch)

                step_desc = f"[green]Epoch {epoch+1}"
                step_task = progress.add_task(step_desc, total=total_steps)
                self.denoiser.train()

                for step, retain_batch in enumerate(self.retain_loader):
                    if (
                        self.opt.max_steps_per_epoch is not None
                        and step >= self.opt.max_steps_per_epoch
                    ):
                        break

                    forget_batch = next(iter(itertools.cycle(self.forget_loader)))
                    self.optimizer.zero_grad()
                    loss_retain = self._compute_loss(retain_batch)
                    loss_forget = self._compute_loss(forget_batch)
                    loss = loss_retain - self.opt.lambda_forget * loss_forget
                    loss.backward()
                    self.optimizer.step()
                    progress.update(
                        step_task,
                        advance=1,
                        description=f"{step_desc} | Loss: {loss.item():.4f}",
                    )

                progress.update(
                    step_task,
                    completed=total_steps,
                    description=f"[bold green]Epoch {epoch+1} Complete",
                )

                self.save_checkpoint(epoch)
                progress.update(epoch_task, advance=1)

    def save_checkpoint(self, epoch):
        """Saves the model checkpoint."""
        save_path = pjoin(self.opt.model_dir, f"epoch_{epoch+1}.tar")
        self.logger.info("Saving checkpoint to %s", save_path)
        torch.save({"denoiser": self.denoiser.state_dict()}, save_path)
        shutil.copy(save_path, pjoin(self.opt.model_dir, "latest.tar"))


def main():
    """Main function to parse arguments and run the training process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--forget_file", type=str, required=True)
    parser.add_argument("--retain_file", type=str, required=True)
    parser.add_argument("--lambda_forget", type=float, default=1.0)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--unlearn_lr", type=float, default=1e-4)
    parser.add_argument("--unlearn_epochs", type=int, default=2)
    parser.add_argument("--vae_name", type=str, default="t2m_vae_gelu")
    parser.add_argument(
        "--max_steps_per_epoch",
        type=int,
        default=5,
        help="Limit training steps per epoch for faster iteration.",
    )

    lora_args, remaining_argv = parser.parse_known_args()
    original_argv = sys.argv
    sys.argv = [original_argv[0]] + remaining_argv
    opt = base_arg_parse(is_train=True)
    sys.argv = original_argv

    concept_name = (
        os.path.basename(lora_args.forget_file)
        .replace(".txt", "")
        .replace("train_val-w-", "")
    )
    new_exp_name = f"{opt.name}_LoRA_{concept_name}"

    log_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, new_exp_name, "logs")
    model_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, new_exp_name, "model")
    os.makedirs(model_dir, exist_ok=True)
    logger = setup_logging(log_dir)

    console.rule(
        f"[bold magenta]LoRA Contrastive Unlearning: Forgetting '{concept_name}'"
    )

    opt.unlearn_lr = lora_args.unlearn_lr
    opt.max_epoch = lora_args.unlearn_epochs
    opt.lambda_forget = lora_args.lambda_forget
    opt.model_dir = model_dir
    opt.log_dir = log_dir
    opt.max_steps_per_epoch = lora_args.max_steps_per_epoch
    fixseed(opt.seed)

    vae_opt_path = pjoin(
        opt.checkpoints_dir, opt.dataset_name, lora_args.vae_name, "opt.txt"
    )
    vae_opt = get_opt(vae_opt_path, opt.device)
    vae = load_vae(vae_opt, logger).to(opt.device)

    denoiser_opt_path = pjoin(
        opt.checkpoints_dir, opt.dataset_name, opt.name, "opt.txt"
    )
    denoiser_opt = get_opt(denoiser_opt_path, opt.device)
    denoiser = load_denoiser(denoiser_opt, vae_opt.latent_dim, logger).to(opt.device)

    logger.info("Freezing all parameters of the base denoiser model.")
    for param in denoiser.parameters():
        param.requires_grad = False

    denoiser = inject_lora(
        denoiser, rank=lora_args.lora_rank, alpha=lora_args.lora_alpha
    )
    denoiser.to(opt.device)

    logger.info("Loading Datasets...")
    DATASET_OPT_PATH = f"checkpoints/{opt.dataset_name}/Comp_v6_KLD005/opt.txt"
    dataset_opt = load_options(DATASET_OPT_PATH, opt.device, {"max_text_len": 20})
    temp_opt_path = pjoin(
        os.path.dirname(DATASET_OPT_PATH), "temp_lora_dataset_opt.txt"
    )

    try:
        with open(temp_opt_path, "w", encoding="utf-8") as f:
            for k, v in vars(dataset_opt).items():
                f.write(f"{k}: {v}\n")

        logger.info("Loading RETAIN set from: %s", lora_args.retain_file)
        retain_loader, _ = get_dataset_motion_loader(
            temp_opt_path,
            32,
            lora_args.retain_file.replace(".txt", ""),
            device=opt.device,
        )

        logger.info("Loading FORGET set from: %s", lora_args.forget_file)
        forget_loader, _ = get_dataset_motion_loader(
            temp_opt_path,
            32,
            lora_args.forget_file.replace(".txt", ""),
            device=opt.device,
        )
    finally:
        if os.path.exists(temp_opt_path):
            os.remove(temp_opt_path)

    scheduler = DDIMScheduler(
        num_train_timesteps=opt.num_train_timesteps,
        beta_start=opt.beta_start,
        beta_end=opt.beta_end,
        prediction_type=opt.prediction_type,
        clip_sample=False,
    )

    trainer = LoRAContrastiveTrainer(
        opt, denoiser, vae, scheduler, retain_loader, forget_loader, logger
    )

    logger.info("Starting LoRA Contrastive Training Loop...")
    try:
        trainer.train()

        base_opt_path = pjoin(
            opt.checkpoints_dir, opt.dataset_name, opt.name, "opt.txt"
        )
        new_opt_path = pjoin(
            opt.checkpoints_dir, opt.dataset_name, new_exp_name, "opt.txt"
        )
        shutil.copy(base_opt_path, new_opt_path)
        logger.info("Copied config file to %s", new_opt_path)

        console.rule("[bold green]LoRA Unlearning Finished.")
        logger.info("To evaluate, use experiment name: %s", new_exp_name)

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        sys.exit(0)
    except Exception:
        logger.exception("LoRA Training Failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
