# Unified Concept Editing (UCE) for Motion

This module implements **UCE (Unified Concept Editing)**, a **training-free** method to erase concepts from the SALAD model. 
Unlike ESD or LoRA, UCE calculates a closed-form mathematical update to the model weights, making it instantaneous.

## 🧠 How it Works
We identify the direction in the feature space that represents the target concept (e.g., "kick") and mathematically project the **Cross-Attention Value Matrices ($W_v$)** to map that concept onto a neutral output.

$$ W_v^{new} = W_v - \lambda \frac{(W_v c_{target} - W_v c_{neutral}) c_{target}^T}{||c_{target}||^2} $$

This effectively "short-circuits" the attention mechanism. When the model sees the "kick" token, the attention layer outputs the vector for "" (neutral/nothing), while leaving other concepts mostly untouched.

## 🚀 Usage

### 1. Editing (Instant)
Run the edit script. Since there is no training loop, this runs in seconds.

```bash
python unlearning/uce/edit.py \
  --name t2m_denoiser_vpred_vaegelu \
  --dataset_name t2m \
  --target_concept "kick" \
  --uce_lambda 1.0
```

### 2. Arguments
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--target_concept` | "kick" | The text description of the motion to remove. |
| `--uce_lambda` | `1.0` | Strength of the edit. `1.0` maps the concept exactly to neutral. Higher values can "invert" the concept, lower values (`0.5`) reduce it partially. |

## 📂 Output & Logging
The script creates a new experiment folder with the edited weights.

**Directory Structure:**
```
checkpoints/t2m/
└── {ORIGINAL_NAME}_UCE_{TARGET_CONCEPT}/
    ├── model/
    │   └── latest.tar      <-- The edited weights (saved immediately)
    ├── logs/
    │   └── edit.log        <-- Execution logs
    └── opt.txt             <-- Copied configuration
```

## 📊 Comparison: UCE vs ESD vs LoRA
| Feature | ESD (Fine-tuning) | LoRA (Adapters) | UCE (Editing) |
| :--- | :--- | :--- | :--- |
| **Speed** | Slow (~30-60 mins) | Medium (~20 mins) | **Instant** (< 1 min) |
| **Method** | Optimization (Loss) | Optimization (Loss) | Linear Algebra (Closed-form) |
| **Type** | Destructive (Weights changed) | Additive (Weights frozen) | Destructive (Weights changed) |
| **Efficacy** | Very High | High | High |
| **Preservation**| Can drift | Excellent | Very Good |

## ⏭️ Next Steps: Evaluation
Verify the erasure using the evaluation script:

```bash
python test_unlearn.py \
  --name t2m_denoiser_vpred_vaegelu_UCE_kick \
  --dataset_name t2m \
  --ckpt latest.tar \
  --target_concept "kick"
```

## 📚 References
Based on the paper: **"Unified Concept Editing for Diffusion Models"** (Gandikota et al., 2024).