# LoRA for Motion Unlearning

This module implements **LoRA (Low-Rank Adaptation)** for unlearning specific motion concepts. 
Instead of fine-tuning the entire model (which risks "catastrophic forgetting" of other motions), we inject small, trainable rank-decomposition matrices into the Attention layers and train *only* those matrices.

## 🧠 How it Works
LoRA freezes the pre-trained model weights ($W_0$) and injects trainable low-rank matrices ($A$ and $B$) into the linear layers of the Transformer.

$$ h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} (B A) x $$

*   **Training:** We use the same negative guidance objective as ESD, but gradients are only computed for $A$ and $B$.
*   **Inference:** The adapters "steer" the internal representations away from the target concept while leaving the original backbone untouched.

## ✨ Why use LoRA?
1.  **Safety:** The original model weights are frozen. You can revert to the original behavior simply by disabling the LoRA layers.
2.  **Efficiency:** We only train ~0.5% - 4% of the parameters, making it faster and less memory-intensive.
3.  **Modularity:** You can train different LoRAs for different concepts (e.g., one for "kick", one for "wave") and potentially switch them at runtime.

## 🚀 Usage

### 1. Training (Unlearning)
Run the training script. Note that LoRA often requires a **higher learning rate** than full fine-tuning.

```bash
python unlearning/lora/train.py \
  --name t2m_denoiser_vpred_vaegelu \
  --dataset_name t2m \
  --target_concept "kick" \
  --lora_rank 16 \
  --unlearn_lr 1e-4 \
  --unlearn_epochs 10
```

### 2. Arguments
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--target_concept` | "kick" | The concept to erase. |
| `--lora_rank` | `16` | The dimension of the low-rank matrices. Lower (4-8) is subtler; Higher (32-64) allows complex changes. |
| `--lora_alpha` | `16.0` | Scaling factor. Usually set equal to rank (scale = 1x) or higher (scale > 1x). |
| `--unlearn_lr` | `1e-4` | Learning rate. LoRA needs higher LR than standard fine-tuning (usually `1e-4` vs `1e-5`). |
| `--negative_guidance` | `1.0` | Strength of erasure (ESD loss scale). |

## 📂 Output & Logging
The script creates a new experiment folder.

**Directory Structure:**
```
checkpoints/t2m/
└── {ORIGINAL_NAME}_LoRA_{TARGET_CONCEPT}/
    ├── model/
    │   └── latest.tar      <-- Contains full model state (Weights + LoRA)
    ├── logs/
    │   └── train_lora.log  <-- Full training logs
    └── eval/               <-- Evaluation results
```

## ⏭️ Next Steps: Evaluation
To evaluate a LoRA model, the testing script must know to **inject the LoRA layers** before loading the weights.

```bash
python test_unlearn.py \
  --name t2m_denoiser_vpred_vaegelu_LoRA_kick \
  --dataset_name t2m \
  --ckpt latest.tar \
  --target_concept "kick" \
  --lora_rank 16
```
*> Note: Ensure `--lora_rank` matches the value used during training.*