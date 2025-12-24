# Erasing Stable Diffusion (ESD) for Motion

This module implements **ESD (Erasing Stable Diffusion)** adapted for 3D Motion Generation. 
ESD is a fine-tuning technique that modifies the model weights to "forget" a specific concept by guiding the model's predictions away from the target concept and towards a neutral or unconditional prediction.

## 🧠 How it Works
Unlike standard training which minimizes the distance to a ground truth motion, ESD minimizes the likelihood of generating the specific target concept.

The loss function steers the noise prediction $\epsilon_\theta$ for the target prompt $c_{target}$ (e.g., "Kick") towards the neutral prediction (empty prompt $\emptyset$).

$$ \mathcal{L}_{ESD} = || \epsilon_\theta(x_t, c_{target}) - \underbrace{[\epsilon_\theta(x_t, \emptyset) - \eta (\epsilon_\theta(x_t, c_{target}) - \epsilon_\theta(x_t, \emptyset))]}_{\text{Guided Target}} ||^2 $$

Where:
- $c_{target}$: The text prompt to erase (e.g., "kick").
- $\emptyset$: The null/empty prompt.
- $\eta$: **Negative Guidance** scale (controlled by `--negative_guidance`).

## 🚀 Usage

### 1. Training (Unlearning)
Run the training script to erase a concept. The script uses **Rich** logging for better visualization.

```bash
python unlearning/esd/train.py \
  --name t2m_denoiser_vpred_vaegelu \
  --dataset_name t2m \
  --target_concept "kick" \
  --negative_guidance 1.0 \
  --unlearn_epochs 10 \
  --unlearn_lr 1e-5
```

### 2. Arguments
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--target_concept` | "kick" | The text description of the motion to remove. |
| `--negative_guidance` | `1.0` | Strength of erasure. `1.0` moves the concept to neutral. Higher values (`>1.0`) actively push it to the opposite direction. |
| `--unlearn_lr` | `1e-5` | Learning rate. ESD usually requires a low LR (`1e-5`) to avoid destroying general knowledge. |
| `--unlearn_epochs` | `10` | How long to train. Usually 500-1000 iterations (approx 1-3 epochs) are enough. |

## 📂 Output & Logging
The script creates a new experiment folder to preserve your original weights.

**Directory Structure:**
```
checkpoints/t2m/
└── {ORIGINAL_NAME}_ESD_{TARGET_CONCEPT}/  <-- New Folder
    ├── model/
    │   └── latest.tar      <-- The unlearned weights
    ├── logs/
    │   └── train.log       <-- Full training logs
    └── eval/               <-- Evaluation results (FID, etc.)
```

## ⏭️ Next Steps: Evaluation
After training, verify that the concept is gone using the test script:

```bash
python test_unlearn.py \
  --name t2m_denoiser_vpred_vaegelu_ESD_kick \
  --dataset_name t2m \
  --target_concept "kick" \
  --ckpt latest.tar
```

## 📚 References
Based on the paper: **"Erasing Concepts from Diffusion Models"** (Gandikota et al., 2023).