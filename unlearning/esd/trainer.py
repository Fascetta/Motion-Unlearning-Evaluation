import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm
import time
from collections import OrderedDict, defaultdict
from os.path import join as pjoin

from models.denoiser.trainer import DenoiserTrainer, lengths_to_mask
# --- OPTIMIZATION: Import AMP tools ---
from utils.utils import print_current_loss
from torch.cuda.amp import GradScaler, autocast
import os

# Get logger for this module
logger = logging.getLogger(__name__)


class ESDTrainer(DenoiserTrainer):
    def __init__(
        self, opt, denoiser, vae, scheduler, target_concept, negative_guidance=1.0
    ):
        # This properly calls the parent's constructor
        super().__init__(opt, denoiser, vae, scheduler)
        self.target_concept = target_concept
        self.negative_guidance = negative_guidance
        # Initialize the Gradient Scaler for AMP
        self.scaler = GradScaler()
        logger.info("   ESD Trainer Initialized with Automatic Mixed Precision (AMP).")
        logger.info("   Target Concept: '%s'", self.target_concept)
        logger.info("   Negative Guidance: %s", self.negative_guidance)

    def train_forward(self, batch_data):
        """
        This is the custom ESD forward pass. It correctly overrides the parent's method.
        """
        # Batch Unpacking (Motion is not needed for ESD loss, only motion lengths)
        if len(batch_data) >= 6:
            m_lens = batch_data[5]
        else:
            m_lens = next((x for x in batch_data if isinstance(x, torch.Tensor) and x.ndim == 1 and not x.is_floating_point()), None)
            if m_lens is None: raise ValueError(f"Could not parse m_lens from batch_data of length {len(batch_data)}.")

        m_lens = m_lens.to(self.opt.device, dtype=torch.long)
        batch_size = m_lens.shape[0]

        text_target = [self.target_concept] * batch_size
        text_neutral = [""] * batch_size
        len_mask = lengths_to_mask(m_lens // 4)
        vae_input_dim = self.denoiser.input_process.layers[0].in_features
        noise = torch.randn(
            (batch_size, len_mask.shape[1], 1, vae_input_dim), device=self.opt.device
        )
        noise = noise * len_mask[..., None, None].float()
        timesteps = torch.randint(
            0, self.opt.num_train_timesteps, (batch_size,), device=self.opt.device
        ).long()

        with torch.no_grad():
            pred_neutral, _ = self.denoiser(noise, timesteps, text_neutral, len_mask=len_mask)
            pred_target_frozen, _ = self.denoiser(noise, timesteps, text_target, len_mask=len_mask)
            error_vector = pred_target_frozen - pred_neutral
            target_prediction = pred_neutral - (self.negative_guidance * error_vector)

        pred_current, _ = self.denoiser(noise, timesteps, text_target, len_mask=len_mask)
        pred_current = pred_current * len_mask[..., None, None].float()
        target_prediction = target_prediction * len_mask[..., None, None].float()
        loss = F.mse_loss(pred_current, target_prediction)
        return loss, None, {"loss_esd": loss}

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval):
        """
        This method now correctly sets up the trainer before the loop.
        """
        # --- SETUP LOGIC (from parent DenoiserTrainer) ---
        self.denoiser.to(self.opt.device)
        self.vae.to(self.opt.device)

        # Optimizer and Scheduler
        self.optim = torch.optim.AdamW(self.denoiser.parameters_without_clip(), lr=self.opt.lr, betas=(0.9, 0.99), weight_decay=self.opt.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=self.opt.milestones, gamma=self.opt.gamma)

        # Resume from checkpoint if specified
        self.start_epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, "latest.tar")
            if os.path.exists(model_dir):
                self.start_epoch, it = self.resume(model_dir)
                logger.info(f"Resumed from epoch {self.start_epoch}, iteration {it}")
            else:
                logger.warning(f"Checkpoint not found at {model_dir}. Starting from scratch.")

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        logger.info(f"Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}")
        logs = defaultdict(lambda: 0.0, OrderedDict())
        # --- END OF SETUP LOGIC ---

        # --- ESD TRAINING LOOP (with AMP) ---
        for epoch in range(self.start_epoch, self.opt.max_epoch):
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.opt.max_epoch}')
            self.denoiser.train() # Set model to training mode
            
            for i, batch_data in enumerate(pbar):
                it += 1
                if it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)
                
                self.optim.zero_grad()

                # Use autocast for the forward pass in mixed precision
                with autocast(dtype=torch.bfloat16):
                    loss, _, loss_dict = self.train_forward(batch_data)

                # Scale loss, call backward, and update optimizer
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()

                if it >= self.opt.warm_up_iter:
                    self.lr_scheduler.step()

                # Logging
                for tag, value in loss_dict.items():
                    logs[tag] += value.item()

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s'%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(lambda: 0.0, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

            self.save(pjoin(self.opt.model_dir, "latest.tar"), epoch, it)
            
            # Since ESD has no validation loss, we skip the validation loop.
            # We will still evaluate using FID/Diversity metrics.
            if (epoch + 1) % self.opt.eval_every_e == 0:
                logger.info(f"Performing generation evaluation for epoch {epoch+1}...")
                # self.eval() is not defined, evaluation_denoiser is called directly
                # This part would need to be re-integrated if desired, but for now we focus on training.