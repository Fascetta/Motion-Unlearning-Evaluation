"""
Implements the ESDTrainer for Erased State Finetuning (ESD), a technique
for unlearning specific concepts from a diffusion model.
"""

import copy
import logging
import time
from collections import OrderedDict, defaultdict
from itertools import cycle
from os.path import join as pjoin

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from models.denoiser.trainer import DenoiserTrainer, lengths_to_mask
from utils.utils import print_current_loss

logger = logging.getLogger(__name__)


class ESDTrainer(DenoiserTrainer):
    """
    Trainer for the Erased State Finetuning (ESD) method.

    This trainer modifies a pre-trained denoiser model to "unlearn" a specific
    concept (the "forget set") while retaining its knowledge of other concepts
    (the "preserve set").
    """

    def __init__(self, opt, denoiser, vae, scheduler, preservation_weight=1.0):
        super().__init__(opt, denoiser, vae, scheduler)
        self.preservation_weight = preservation_weight
        self.scheduler = scheduler
        self.scaler = GradScaler()

        logger.info("Creating a frozen copy of the denoiser for ESD reference.")
        self.denoiser_frozen = copy.deepcopy(denoiser).to(self.opt.device)
        self.denoiser_frozen.eval()
        for param in self.denoiser_frozen.parameters():
            param.requires_grad = False

        # Initialize attributes that will be defined in the train method
        self.optim = None
        self.lr_scheduler = None
        self.start_epoch = 0

        logger.info("   ESD Trainer Initialized.")
        logger.info("   Preservation Weight (lambda): %.2f", self.preservation_weight)

    # pylint: disable=arguments-differ
    def train_forward(self, forget_batch, preserve_batch):
        """
        Performs a forward pass for one training step of ESD.

        This involves calculating the erasure loss on the "forget" batch and
        the preservation loss on the "preserve" batch.
        """
        # --- Erasure Loss Calculation ---
        forget_motion = forget_batch[4].to(self.opt.device, dtype=torch.float)
        forget_text = list(forget_batch[2])
        forget_m_lens = forget_batch[5].to(self.opt.device, dtype=torch.long)
        batch_size = forget_motion.shape[0]

        with torch.no_grad():
            mu, _ = self.vae.encode(forget_motion[..., :263])
            z = mu.detach()

        noise = torch.randn_like(z)  # pylint: disable=no-member
        timesteps = torch.randint(  # pylint: disable=no-member
            0, self.scheduler.config.num_train_timesteps, (batch_size,), device=z.device
        ).long()
        noisy_z = self.scheduler.add_noise(z, noise, timesteps)
        len_mask = lengths_to_mask(forget_m_lens // 4)

        pred_noise_trainable, _ = self.denoiser(
            noisy_z, timesteps, forget_text, len_mask=len_mask
        )

        with torch.no_grad():
            null_text = [""] * batch_size
            pred_noise_uncond_frozen, _ = self.denoiser_frozen(
                noisy_z, timesteps, null_text, len_mask=len_mask
            )

        loss_erase = F.mse_loss(pred_noise_trainable, pred_noise_uncond_frozen)

        # --- Preservation Loss Calculation ---
        preserve_motion = preserve_batch[4].to(self.opt.device, dtype=torch.float)
        preserve_text = list(preserve_batch[2])
        preserve_m_lens = preserve_batch[5].to(self.opt.device, dtype=torch.long)

        with torch.no_grad():
            mu_preserve, _ = self.vae.encode(preserve_motion[..., :263])
            z_preserve = mu_preserve.detach()

        noise_preserve = torch.randn_like(z_preserve)  # pylint: disable=no-member
        timesteps_preserve = torch.randint(  # pylint: disable=no-member
            0,
            self.scheduler.config.num_train_timesteps,
            (z_preserve.shape[0],),
            device=z_preserve.device,
        ).long()
        noisy_z_preserve = self.scheduler.add_noise(
            z_preserve, noise_preserve, timesteps_preserve
        )
        len_mask_preserve = lengths_to_mask(preserve_m_lens // 4)

        predicted_noise_preserve, _ = self.denoiser(
            noisy_z_preserve,
            timesteps_preserve,
            preserve_text,
            len_mask=len_mask_preserve,
        )
        loss_preserve = F.mse_loss(predicted_noise_preserve, noise_preserve)

        total_loss = loss_erase + self.preservation_weight * loss_preserve

        return (
            total_loss,
            None,
            {
                "loss_erase": loss_erase.detach(),
                "loss_preserve": loss_preserve.detach(),
                "loss_total": total_loss.detach(),
            },
        )

    # pylint: disable=arguments-renamed
    def train(
        self, forget_loader, preserve_loader, eval_val_loader, eval_wrapper, plot_eval
    ):
        """Main training loop for ESD."""
        self.denoiser.to(self.opt.device)
        self.vae.to(self.opt.device)
        self.optim = torch.optim.AdamW(
            self.denoiser.parameters_without_clip(),
            lr=self.opt.lr,
            betas=(0.9, 0.99),
            weight_decay=self.opt.weight_decay,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optim, milestones=self.opt.milestones, gamma=self.opt.gamma
        )

        self.start_epoch, it = 0, 0
        start_time = time.time()
        total_iters = self.opt.max_epoch * len(forget_loader)

        logger.info(
            "Total Epochs: %d, Total Iters: %d", self.opt.max_epoch, total_iters
        )
        logs = defaultdict(lambda: 0.0, OrderedDict())

        for epoch in range(self.start_epoch, self.opt.max_epoch):
            pbar = tqdm(
                zip(forget_loader, cycle(preserve_loader)),
                desc=f"Epoch {epoch+1}/{self.opt.max_epoch}",
                total=len(forget_loader),
            )
            self.denoiser.train()

            for i, (forget_batch, preserve_batch) in enumerate(pbar):
                it += 1
                self.optim.zero_grad()

                with autocast(dtype=torch.bfloat16):
                    loss, _, loss_dict = self.train_forward(
                        forget_batch, preserve_batch
                    )

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()

                for tag, value in loss_dict.items():
                    logs[tag] += value.item()

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        self.logger.add_scalar(
                            f"Train/{tag}", value / self.opt.log_every, it
                        )
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(lambda: 0.0, OrderedDict())
                    print_current_loss(
                        start_time,
                        it,
                        total_iters,
                        mean_loss,
                        epoch=epoch,
                        inner_iter=i,
                    )

            self.lr_scheduler.step()
            self.save(pjoin(self.opt.model_dir, "latest.tar"), epoch, it)
