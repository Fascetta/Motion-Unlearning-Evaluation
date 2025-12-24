import torch
import torch.nn.functional as F
import logging
from models.denoiser.trainer import DenoiserTrainer, lengths_to_mask

# Get logger for this module
logger = logging.getLogger(__name__)

class ESDTrainer(DenoiserTrainer):
    def __init__(self, opt, denoiser, vae, scheduler, target_concept, negative_guidance=1.0):
        super().__init__(opt, denoiser, vae, scheduler)
        self.target_concept = target_concept
        self.negative_guidance = negative_guidance
        
        logger.info("⚠️ ESD Trainer Initialized.")
        logger.info("   Target Concept: '%s'", self.target_concept)
        logger.info("   Negative Guidance: %s", self.negative_guidance)

    def train_forward(self, batch_data):
        """
        ESD requires NO ground truth motion for the loss calculation regarding the concept.
        We use the model's own unconditional prediction as the anchor.
        """
        
        # --- FIX: Precise Batch Unpacking based on EDA ---
        # The T2M loader returns 7 items.
        # [0]=WordEmb, [1]=PosEmb, [2]=Text, [3]=SentLen, [4]=Motion, [5]=MotionLen, [6]=Token
        
        if len(batch_data) >= 6:
            motion_gt = batch_data[4]
            m_lens = batch_data[5]
        else:
            # Fallback for unexpected loader formats
            motion_gt = next((x for x in batch_data if isinstance(x, torch.Tensor) and x.ndim == 4), None)
            m_lens = next((x for x in batch_data if isinstance(x, torch.Tensor) and x.ndim == 1 and not x.is_floating_point()), None)
            
            if motion_gt is None or m_lens is None:
                raise ValueError(f"Could not parse batch_data of length {len(batch_data)}. Expected HumanML3D format.")

        # Ensure correct types/device
        motion_gt = motion_gt.to(self.opt.device, dtype=torch.float32)
        m_lens = m_lens.to(self.opt.device, dtype=torch.long)
        batch_size = motion_gt.shape[0]
        # -----------------------------------

        # 1. Prepare Inputs
        text_target = [self.target_concept] * batch_size
        text_neutral = [""] * batch_size

        len_mask = lengths_to_mask(m_lens // 4) # [B, T]

        # 2. Sample Random Noise (Latent Space)
        # We must match the VAE input dimension (usually 32 or 4, depending on VAE)
        # We retrieve this dynamically from the input projection layer.
        vae_input_dim = self.denoiser.input_process.layers[0].in_features

        noise = torch.randn(
            (batch_size, len_mask.shape[1], 1, vae_input_dim), 
            device=self.opt.device, dtype=torch.float32
        )
        
        # Mask noise to respect lengths
        noise = noise * len_mask[..., None, None].float()

        # 3. Sample Timesteps
        timesteps = torch.randint(
            0, self.opt.num_train_timesteps, (batch_size,), device=self.opt.device
        ).long()

        # 4. Get Predictions from the FROZEN model (Teacher) logic
        with torch.no_grad():
            # A. Unconditional Prediction (The "Safe" direction)
            pred_neutral, _ = self.denoiser.forward(noise, timesteps, text_neutral, len_mask=len_mask)
            
            # B. Conditional Prediction (The "Bad" direction)
            pred_target_frozen, _ = self.denoiser.forward(noise, timesteps, text_target, len_mask=len_mask)

            # C. Calculate the ESD Target
            # Target = Neutral - neg_guidance * (Target - Neutral)
            error_vector = pred_target_frozen - pred_neutral
            target_prediction = pred_neutral - (self.negative_guidance * error_vector)

        # 5. Get Prediction from Current Model (Student)
        pred_current, _ = self.denoiser.forward(noise, timesteps, text_target, len_mask=len_mask)
        
        # Masking
        pred_current = pred_current * len_mask[..., None, None].float()
        target_prediction = target_prediction * len_mask[..., None, None].float()

        # 6. Loss Calculation (MSE)
        loss = F.mse_loss(pred_current, target_prediction)
        
        loss_dict = {"loss_esd": loss}
        return loss, None, loss_dict