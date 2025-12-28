# Erasing Stable Diffusion (ESD) for Motion

This project implements **ESD (Erasing Stable Diffusion)**, a fine-tuning technique adapted for 3D Motion Generation. ESD modifies a pre-trained model to "forget" a specific concept by guiding its predictions away from the target concept and towards a neutral or unconditional output. This is particularly useful for removing undesirable or copyrighted motions from a generative model without having to retrain it from scratch.

## 🧠 How it Works

Unlike standard training that minimizes the distance to a ground truth motion, ESD minimizes the likelihood of generating a specific target concept. The core idea is to steer the noise prediction for a target prompt (e.g., "a person kicking") towards the prediction for a neutral (empty) prompt. This is achieved by fine-tuning the model weights to move away from the concept to be erased.

The loss function guides the noise prediction for the target prompt, c_target, towards a modified unconditional prediction. This process uses the model's own knowledge to steer the diffusion process away from the undesired concept.

## 🚀 Usage

### 1. Training (Unlearning)

To erase a concept from a pre-trained model, run the training script. This script is optimized for modern GPUs and uses **Rich** for enhanced logging.

```bash
python unlearning/esd/train.py \
  --forget_split_file "kw_splits/train_val-w-kick.txt"  \
  --preserve_split_file "kw_splits/train_val-wo-kick.txt"
```

### 2. Key Arguments

| Argument | Description |
| :--- | :--- |
| `--name` | The name of the original pre-trained model directory. |
| `--dataset_name` | The dataset the model was trained on (e.g., `t2m`). |
| `--forget_split_file` | Path to a file containing the names of motions to be forgotten. |
| `--preserve_split_file`| Path to a file containing the names of motions to be preserved. |
| `--unlearn_epochs` | The number of epochs for the unlearning process. Typically, 500-1000 iterations are sufficient. |
| `--unlearn_lr` | The learning rate for unlearning. A low learning rate (e.g., 1e-5 to 2e-5) is crucial to avoid damaging the model's general knowledge. |
| `--preservation_weight` | A weight to balance the preservation of other concepts while erasing the target one. |
| `--batch_size` | The training batch size. Adjust this based on your available GPU VRAM. |
| `--num_workers` | The number of CPU workers for data loading. Adjust based on your CPU cores. |

## 📂 Output

The script creates a new experiment folder, preserving your original model weights.

**Directory Structure:**
```
checkpoints/t2m/
└── {ORIGINAL_NAME}_ESD_{TARGET_CONCEPT}/  <-- New Folder
    ├── eval_efficacy/      <-- Folder for evaluation results on forget set
    ├── eval_preservation/  <-- Folder for evaluation results on retain set
    |
    ├── logs/
    │   └── train.log       <-- Full training logs
    |
    ├── model/
    │   └── latest.tar      <-- Unlearned weights from the last epoch
    └── opt.txt
```

## ⏭️ Next Steps: Comprehensive Evaluation

A structured evaluation is necessary to ensure the unlearning process was successful. This involves verifying three key aspects:
1.  **Efficacy:** Did the model successfully forget the target concept?
2.  **Preservation:** Does the model still perform well on other, unrelated concepts?

Our comprehensive evaluation script (`unlearning/test_unlearn.py`) automates this by comparing the original and unlearned models.

### Step 1: Evaluate the ORIGINAL Model (Establish a Baseline)

First, evaluate your original, pre-trained model to establish a baseline performance.

```bash
python unlearning/test_unlearn.py  \
  --name "t2m_denoiser_vpred_vaegelu"  
  --forget_test_file "kw_splits/test-w-kick.txt"  \
  --retain_test_file "kw_splits/test-wo-kick.txt"
```

### Step 2: Evaluate the UNLEARNED Model

Next, run the same evaluation, but point it to the new model folder and the unlearned checkpoint (`latest.tar` or a specific epoch's weights).

```bash
python unlearning/test_unlearn.py  \
  --name "t2m_denoiser_vpred_vaegelu_ESD_kick"  
  --forget_test_file "kw_splits/test-w-kick.txt"  \
  --retain_test_file "kw_splits/test-wo-kick.txt"
```

### Step 3: Interpret the Results

Compare the results from both evaluations. A successful unlearning experiment should demonstrate:
*   ✅ **Efficacy:** An high FID on the forget set means that the model correctly forgot about the concept.
*   ✅ **Preservation:** A low FID on the retain set means that the model has not been damaged by the unlearning procedure.

## 📚 References
Based on the paper: **"Erasing Concepts from Diffusion Models"** (Gandikota et al., 2023).