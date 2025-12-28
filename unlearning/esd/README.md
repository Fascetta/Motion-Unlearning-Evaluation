# Erasing Stable Diffusion (ESD) for 3D Motion

This project implements **Erasing Stable Diffusion (ESD)**, a pioneering fine-tuning technique adapted for 3D Motion Generation. ESD surgically removes concepts from a pre-trained generative model, offering a powerful solution for eliminating undesirable, copyrighted, or unsafe motions without the need for full retraining.

This method is a key approach in the rapidly evolving field of machine unlearning. It modifies the model's weights directly, making the erasure robust and difficult to circumvent—a critical feature for deploying safe and ethical generative models.

## 🧠 How it Works

ESD operates by steering the model's predictions away from a target concept during the diffusion process. Instead of standard training that guides the model *towards* a ground truth, ESD fine-tunes the model to minimize the probability of generating a specific, unwanted motion.

The core mechanism involves guiding the noise prediction for a "forget" prompt (e.g., "a person kicking") to align more closely with the prediction for a neutral or unconditional prompt. This is achieved through a specialized loss function that effectively teaches the model to suppress the target concept, leveraging the model's own internal knowledge to guide the erasure process.

## 🚀 Usage

### 1. Training (Unlearning)

To erase a motion concept from a pre-trained model, run the training script below. The script is optimized for modern GPUs and uses the **Rich** library for clear, detailed logging.

```bash
python unlearning/esd/train.py \
  --forget_file "kw_splits/train_val-w-kick.txt"  \
  --preserve_file "kw_splits/train_val-wo-kick.txt"
```

### 2. Key Arguments

| Argument | Description |
| :--- | :--- |
| `--name` | Name of the original pre-trained model directory. |
| `--dataset_name` | The dataset used for training (e.g., `t2m`). |
| `--forget_split_file` | Path to a file listing motions to be forgotten. |
| `--preserve_split_file`| Path to a file listing motions to be preserved. |
| `--unlearn_epochs` | Number of unlearning epochs. Typically, 500-1000 iterations are sufficient. |
| `--unlearn_lr` | Learning rate for unlearning. A low value (e.g., 1e-5) is crucial to avoid catastrophic forgetting. |
| `--preservation_weight` | Balances concept erasure with knowledge preservation. |
| `--batch_size` | Training batch size, adjust based on available VRAM. |
| `--num_workers` | Number of CPU workers for data loading. |

## 📂 Output

The script generates a new experiment folder, ensuring your original model weights are preserved.

**Directory Structure:**
```
checkpoints/t2m/
└── {ORIGINAL_NAME}_ESD_{TARGET_CONCEPT}/  <-- New Folder
    ├── eval_efficacy/      <-- Evaluation results on the forget set
    ├── eval_preservation/  <-- Evaluation results on the retain set
    ├── logs/
    │   └── train.log       <-- Full training logs
    ├── model/
    │   └── latest.tar      <-- Unlearned model weights
    └── opt.txt
```

## ⏭️ Next Steps: Comprehensive Evaluation

A rigorous evaluation is essential to confirm successful unlearning. This process validates two critical outcomes:
1.  **Efficacy:** The model has successfully forgotten the target concept.
2.  **Preservation:** The model's performance on other, unrelated concepts remains intact.

Our unified evaluation script (`unlearning/test_unlearn.py`) automates this by comparing the performance of the original and unlearned models.

### Step 1: Establish a Baseline (Evaluate Original Model)

First, benchmark the performance of your original, pre-trained model.

```bash
python unlearning/test_unlearn.py  \
  --name "t2m_denoiser_vpred_vaegelu"  
  --forget_test_file "kw_splits/test-w-kick.txt"  \
  --retain_test_file "kw_splits/test-wo-kick.txt"
```

### Step 2: Evaluate the Unlearned Model

Next, run the same evaluation, pointing to the new model folder and the unlearned checkpoint.

```bash
python unlearning/test_unlearn.py  \
  --name "t2m_denoiser_vpred_vaegelu_ESD_kick"  
  --forget_test_file "kw_splits/test-w-kick.txt"  \
  --retain_test_file "kw_splits/test-wo-kick.txt"
```

### Step 3: Interpret the Results

A successful unlearning experiment is characterized by:
*   ✅ **High FID (Efficacy):** A high FID score on the "forget" set indicates the model can no longer generate the target motion effectively.
*   ✅ **Low FID (Preservation):** A low FID score on the "retain" set confirms that the model's general motion generation quality has not been compromised.

## 📚 References
This implementation is based on the foundational paper:
*   **Gandikota, R., Materzyńska, J., Fiotto-Kaufman, J., & Bau, D. (2023). "Erasing Concepts from Diffusion Models". In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*.**