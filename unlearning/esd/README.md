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
Run the training script to erase a concept from a pre-trained model. The script is optimized for modern GPUs and uses **Rich** for better logging.

```bash
python unlearning/esd/train.py \
  --name t2m_denoiser_vpred_vaegelu \
  --dataset_name t2m \
  --target_concept "a person kicking" \
  --negative_guidance 1.5 \
  --unlearn_epochs 10 \
  --unlearn_lr 1e-5 \
  --batch_size 128 \
  --num_workers 8
```

### 2. Key Arguments
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--target_concept` | "kick" | The text description of the motion to remove. |
| `--negative_guidance`| `1.0` | Strength of erasure. `1.0` moves the concept to neutral. `>1.0` actively pushes it away. `1.5` to `2.5` often works well. |
| `--unlearn_lr` | `1e-5` | Learning rate. ESD requires a low LR (`1e-5` to `2e-5`) to avoid destroying general knowledge. |
| `--unlearn_epochs` | `10` | How long to train. Usually 500-1000 iterations (approx 1-3 epochs on HumanML3D) are enough. |
| `--batch_size` | `128` | Training batch size. Adjust based on your GPU VRAM. |
| `--num_workers` | `8` | Number of CPU workers for data loading. Adjust based on your CPU cores. |

## 📂 Output
The script creates a new experiment folder to preserve your original weights.

**Directory Structure:**
```
checkpoints/t2m/
└── {ORIGINAL_NAME}_ESD_{TARGET_CONCEPT}/  <-- New Folder
    ├── model/
    │   ├── latest.tar      <-- Unlearned weights from the last epoch
    │   └── net_e_XX.tar    <-- Weights from a specific epoch
    ├── logs/
    │   └── train.log       <-- Full training logs
    └── evaluation/         <-- Folder for evaluation results
```

## ⏭️ Next Steps: Comprehensive Evaluation

Evaluating unlearning requires a structured approach. We need to verify three things:
1.  **Efficacy:** Did the model forget the target concept?
2.  **Preservation:** Does the model still work well for other concepts?
3.  **Specificity:** Did the model accidentally forget related concepts?

Our comprehensive evaluation script (`unlearning/esd/evaluate.py`) automates this by comparing the original model to the unlearned one.

### Step 1: Evaluate the ORIGINAL Model (Establish a Baseline)

First, run the evaluation on your original, pre-trained model to see how it performs before unlearning.

```bash
python unlearning/esd/evaluate.py \
  --name "t2m_denoiser_vpred_vaegelu" \
  --ckpt "net_best_fid.tar" \
  --target_concept "a person kicking" \
  --related_concepts "a person jumping" "a person punching" "a person lunging"
```

### Step 2: Evaluate the UNLEARNED Model

Now, run the same evaluation, but point it to the new model folder and the unlearned checkpoint (`latest.tar` or a specific epoch like `net_e_10.tar`).

```bash
python unlearning/esd/evaluate.py \
  --name "t2m_denoiser_vpred_vaegelu_ESD_a_person_kicking" \
  --ckpt "latest.tar" \
  --target_concept "a person kicking" \
  --related_concepts "a person jumping" "a person punching" "a person lunging"
```

### Step 3: Interpret the Results

Compare the output from Step 1 and Step 2. A successful unlearning experiment will show:
*   ✅ **Efficacy:** The **Matching Score** for the `--target_concept` ("a person kicking") is **much lower** on the unlearned model. The generated videos should no longer show kicking.
*   ✅ **Preservation:** The **General FID**, **R-Precision**, and **Matching Score** on the standard test set are **very close** to the original model's scores.
*   ✅ **Specificity:** The **Matching Scores** for the `--related_concepts` ("jumping", "punching") have **not decreased significantly**.

## 📚 References
Based on the paper: **"Erasing Concepts from Diffusion Models"** (Gandikota et al., 2023).