# Unified Concept Editing (UCE) for Motion

This module implements **UCE (Unified Concept Editing)**, a **training-free** method to erase concepts from a pre-trained motion generation model. Unlike iterative methods like ESD or LoRA, UCE calculates a closed-form mathematical update to the model weights, making the editing process nearly instantaneous.

This implementation uses a **contrastive** approach, where the model is edited to move away from a "forget" concept and towards a "retain" concept. The official UCE update is applied to both the **Key (K) and Value (V)** projection matrices for a more robust edit.

## 🧠 How it Works

Instead of training, we precisely identify the direction in the feature space that distinguishes the concept we want to forget (e.g., "kicking") from a concept we want to retain (e.g., general locomotion). We then mathematically update the **Cross-Attention Key ($W_k$) and Value ($W_v$) matrices** to nullify the forget concept's direction.

The update for a weight matrix $W$ is:
$$ W_{new} = W - \lambda \cdot \text{outer}(\ W(c_{forget} - c_{retain}), \frac{c_{forget} - c_{retain}}{||c_{forget} - c_{retain}||} ) $$

This effectively "short-circuits" the attention mechanism. When the model encounters text related to the "forget" set, the attention layers now produce a representation closer to that of the "retain" set, while leaving other concepts mostly untouched.

## 🚀 Usage

### 1. Editing (Instant)

Run the edit script with paths to your "forget" and "retain" caption files. Since there is no training loop, this process completes in seconds.

```bash
python unlearning/uce/edit.py \
  --name t2m_denoiser_vpred_vaegelu \
  --forget_file "kw_splits/train_val-w-kick.txt" \
  --retain_file "kw_splits/train_val-wo-kick.txt" \
```

### 2. Arguments

| Argument | Description |
| :--- | :--- |
| `--forget_file` | **Required.** Path to a text file with captions describing the motions to FORGET. |
| `--retain_file` | **Required.** Path to a text file with captions describing motions to RETAIN. |
| `--uce_lambda` | The strength of the edit. `1.0` fully projects the forget concept onto the retain concept. Higher values can "invert" the concept. Default: `1.0`. |
| `--name` | The name of the original pre-trained model directory. |

## 📂 Output & Logging

The script creates a new experiment folder containing the edited model weights and logs.

**Directory Structure:**
```
checkpoints/t2m/
└── {ORIGINAL_NAME}_UCE_kick/
    ├── model/
    │   └── latest.tar      <-- The edited model weights
    ├── logs/
    │   └── edit_uce_kick.log <-- Execution logs
    └── opt.txt             <-- A copy of the original model's configuration
```

## 📊 Comparison: UCE vs ESD vs LoRA

| Feature | ESD (Fine-tuning) | LoRA (Adapters) | UCE (Editing) |
| :--- | :--- | :--- | :--- |
| **Speed** | Slow (~30-60 mins) | Medium (~20 mins) | **Instant** (< 1 min) |
| **Method** | Optimization (Loss) | Optimization (Loss) | Linear Algebra (Closed-form) |
| **Type** | Destructive | Additive (Weights frozen) | Destructive |
| **Efficacy** | Very High | High | High |
| **Preservation**| Can drift | Excellent | Very Good |

## ⏭️ Next Steps: Evaluation

Verify the success of the erasure by running the evaluation script on the new model, using corresponding test sets.

```bash
python test_unlearn.py \
  --name "t2m_denoiser_vpred_vaegelu_UCE_kick" \
  --forget_test_file "kw_splits/test-w-kick.txt" \
  --retain_test_file "kw_splits/test-wo-kick.txt"
```

## 📚 References

Based on the paper: **"Editing Large Language Models: A Closed-form Solution" (Unofficial)** and its principles, adapted for motion. The K/V editing technique is inspired by the official UCE paper: **"Unified Concept Editing for Diffusion Models"** (Gandikota et al., 2024).