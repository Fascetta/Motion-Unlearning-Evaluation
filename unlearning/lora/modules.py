import torch
import torch.nn as nn
import math
import logging

logger = logging.getLogger(__name__)

class LoRALinear(nn.Module):
    """
    Wraps a standard nn.Linear layer with Low-Rank Adapters.
    Formula: h = Wx + (BA)x * scale
    """
    def __init__(self, original_linear, rank=4, alpha=1.0):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Freeze the original linear layer
        self.original_linear = original_linear
        for param in self.original_linear.parameters():
            param.requires_grad = False

        # LoRA Matrices
        # A: [Rank, In] (Gaussian Init)
        # B: [Out, Rank] (Zero Init)
        self.lora_a = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_b = nn.Parameter(torch.zeros(self.out_features, rank))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize A with random gaussian, B with zeros
        # This ensures the LoRA starts as an "Identity" function (no change to output)
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, x):
        # Original output
        out_orig = self.original_linear(x)
        
        # LoRA path: (x @ A.T) @ B.T
        out_lora = (x @ self.lora_a.t()) @ self.lora_b.t()
        
        return out_orig + (out_lora * self.scaling)

def inject_lora(model, rank=4, alpha=1.0):
    """
    Iterates over the model, finds MultiheadAttention layers, and replaces
    Linear projections (Wq, Wk, Wv, Wo) with LoRALinear.
    """
    logger.info("💉 Injecting LoRA (Rank=%d, Alpha=%.1f)...", rank, alpha)
    
    params_before = sum(p.numel() for p in model.parameters())
    
    count = 0
    # We look for the MultiheadAttention class by checking attributes
    for name, module in model.named_modules():
        if hasattr(module, 'Wq') and hasattr(module, 'Wk') and hasattr(module, 'Wv') and isinstance(module.Wq, nn.Linear):
            # Replace Projections
            module.Wq = LoRALinear(module.Wq, rank, alpha)
            module.Wk = LoRALinear(module.Wk, rank, alpha)
            module.Wv = LoRALinear(module.Wv, rank, alpha)
            module.Wo = LoRALinear(module.Wo, rank, alpha)
            count += 1

    params_after = sum(p.numel() for p in model.parameters())
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = (trainable_after / params_after) * 100

    logger.info("✅ Injected LoRA into %d attention modules.", count)
    logger.info("   Original Params: %s", f"{params_before:,}")
    logger.info("   Trainable Params Now: %s (%.2f%%)", f"{trainable_after:,}", ratio)
    
    return model