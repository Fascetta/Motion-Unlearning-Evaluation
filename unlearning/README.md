# 🧠 Motion Unlearning Suite

This module implements state-of-the-art Machine Unlearning techniques adapted for the SALAD 3D Motion Diffusion model.

Our goal is to remove specific motion concepts (e.g., "kick", "jump", "wave") from the model's knowledge while preserving its ability to generate other motions.

## 📂 Implemented Techniques

| Method | Full Name | Type | Characteristics | Link |
| :--- | :--- | :--- | :--- | :--- |
| **ESD** | Erasing Stable Diffusion | Fine-Tuning | **Robust**. Optimizes the model to negate the target concept. | [Docs](esd/README.md) |
| **LoRA** | Low-Rank Adaptation | Adapters | **Safe**. Injects and trains small layers; original weights are frozen. | [Docs](lora/README.md) |
| **UCE** | Unified Concept Editing | Editing | **Instant**. Closed-form update to attention matrices. Training-free. | [Docs](uce/README.md) |

---

## 🚦 Quick Start

### 1. Run ESD (Fine-Tuning)
Destructive but effective. Fine-tunes the main weights.
```bash
python unlearning/esd/train.py \
  --name t2m_denoiser_vpred_vaegelu \
  --dataset_name t2m \
  --target_concept "kick" \
  --negative_guidance 1.0 \
  --unlearn_epochs 10
```

### 2. Run LoRA (Adapters)
Non-destructive. Adds small trainable layers.
```bash
python unlearning/lora/train.py \
  --name t2m_denoiser_vpred_vaegelu \
  --dataset_name t2m \
  --target_concept "kick" \
  --lora_rank 16 \
  --unlearn_lr 1e-4
```

### 3. Run UCE (Instant Edit)
Fastest method. Edits attention keys mathematically.
```bash
python unlearning/uce/edit.py \
  --name t2m_denoiser_vpred_vaegelu \
  --dataset_name t2m \
  --target_concept "kick" \
  --uce_lambda 1.0
```

---

## 📊 Evaluation
Use the universal `test_unlearn.py` script to measure **Efficacy** (did it forget?) and **Preservation** (is general quality intact?).

### For ESD & UCE Models:
```bash
python test_unlearn.py \
  --name t2m_denoiser_vpred_vaegelu_ESD_kick \
  --dataset_name t2m \
  --ckpt latest.tar \
  --target_concept "kick"
```

### For LoRA Models:
*⚠️ Note: You must specify `--lora_rank` so the evaluator knows how to load the adapters.*
```bash
python test_unlearn.py \
  --name t2m_denoiser_vpred_vaegelu_LoRA_kick \
  --dataset_name t2m \
  --ckpt latest.tar \
  --target_concept "kick" \
  --lora_rank 16
```