# Motion Unlearning with Low-Rank Adaptation (LoRA)

This module implements **LoRA (Low-Rank Adaptation)** as an efficient method for unlearning specific motion concepts from a pre-trained diffusion model. Instead of fine-tuning the entire model (which is computationally expensive and risks "catastrophic forgetting"), we inject small, trainable rank-decomposition matrices into the model's attention layers and train *only* those matrices.

## 🧠 How it Works

LoRA freezes the pre-trained model weights ($W_0$) and injects a pair of trainable low-rank matrices ($A$ and $B$) into the linear layers of the Transformer architecture. The forward pass is modified to include the output of these new matrices.

The update to a given layer is represented by:
$$ h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} (B A) x $$

Where:
*   $r$ is the rank of the LoRA matrices.
*   $\alpha$ is a scaling factor.
*   **Training:** We use a contrastive objective that encourages the model to forget motions from a "forget set" while retaining knowledge from a "retain set." The backpropagated gradients only update the LoRA matrices $A$ and $B$.
*   **Inference:** During generation, the trained adapters "steer" the model's internal representations away from the target concept while leaving the original backbone untouched.

## ✨ Why use LoRA for Unlearning?

1.  **Safety & Reversibility:** The original model weights are frozen. You can revert to the original model's behavior simply by disabling or removing the LoRA layers.
2.  **Efficiency:** We only train a tiny fraction of the parameters (typically 0.5% - 4%), making the unlearning process significantly faster and less memory-intensive than full fine-tuning.
3.  **Modularity:** You can train different LoRA adapters for different concepts (e.g., one for "kicking," another for "waving") and switch between them at runtime.

## 🚀 Usage

### 1. Training (Unlearning)

Run the training script to begin the unlearning process. This script uses two text files: one listing the data to forget and another listing the data to retain. Note that LoRA often requires a **higher learning rate** (e.g., `1e-4`) than full fine-tuning.

```bash
python ./unlearning/lora/train.py \
  --forget_file "kw_splits/train_val-w-kick.txt" \
  --retain_file "kw_splits/train_val-wo-kick.txt"
```

### 2. Key Arguments

| Argument | Description |
| :--- | :--- |
| `--forget_file` | **Required.** Path to the split file listing data samples to be forgotten. |
| `--retain_file` | **Required.** Path to the split file listing data samples to be retained. |
| `--lora_rank` | The rank (dimension) of the low-rank matrices. Lower (4-8) is subtler; higher (32-64) is stronger. Default: `16`. |
| `--lora_alpha` | The scaling factor for the LoRA output. Often set equal to the rank. Default: `16.0`. |
| `--unlearn_lr` | Learning rate for training the LoRA adapters. LoRA benefits from a higher LR than standard fine-tuning. Default: `1e-4`. |
| `--lambda_forget` | The weight for the forget term in the contrastive loss function (`loss = loss_retain - λ * loss_forget`). Default: `1.0`. |

## 📂 Output & Logging

The script generates a new experiment folder to store the unlearned model and logs, keeping your original model weights safe. The folder is automatically named based on the forget file.

**Directory Structure:**
```
checkpoints/t2m/
└── {ORIGINAL_NAME}_LoRA_contrast_kick/
    ├── model/
    │   └── latest.tar      <-- State dict containing original weights + LoRA adapters
    ├── logs/
    │   └── train_lora.log  <-- Full, detailed training logs
    └── eval/               <-- Directory for evaluation results
```

## ⏭️ Next Steps: Evaluation

To evaluate a LoRA-unlearned model, the testing script must be instructed to **inject the LoRA layers** before loading the checkpoint. This is done by providing the same rank used during training. The evaluation also uses corresponding "forget" and "retain" test sets.

```bash
python test_unlearn.py \
  python test_unlearn.py   \
  --name "t2m_denoiser_vpred_vaegelu_LoRA_kick" \ 
  --forget_test_file "kw_splits/test-w-kick.txt"  \
  --retain_test_file "kw_splits/test-wo-kick.txt"
```
**Note:** Ensure the `--lora_rank` argument here matches the value used during the training step.