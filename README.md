# Motion Unlearning: Evaluation on SALAD
### LoRA, ESD, and UCE for 3D Motion Unlearning

This repository extends the official implementation of **[SALAD: Skeleton-Aware Latent Diffusion Model (CVPR 2025)](https://seokhyeonhong.github.io/projects/salad/)** to investigate **Machine Unlearning** in 3D motion generation.

We provide a modular framework to **erase specific motion concepts** (e.g., "kick", "jump") by defining "forget" and "retain/preserve" datasets, while preserving the model's general generation capabilities.

---

## 🚀 Implemented Techniques
We have adapted three state-of-the-art unlearning methods for the SALAD architecture. All implementations are located in the `unlearning/` directory.

| Method | Type | Description | Status |
| :--- | :--- | :--- | :--- |
| **[ESD](unlearning/esd/README.md)** | Fine-Tuning | **Erased State Finetuning**. Optimizes model weights to negate a specific concept using score guidance. | ✅ Ready |
| **[LoRA](unlearning/lora/README.md)** | Adapters | **Low-Rank Adaptation**. Injects and trains small adapter layers while freezing the original model. Safe and efficient. | ✅ Ready |
| **[UCE](unlearning/uce/README.md)** | Editing | **Unified Concept Editing**. A training-free, closed-form update to the Cross-Attention layers. Instant execution. | ✅ Ready |

---

## ⚙️ Environment Setup
The environment follows the original SALAD requirements.

```bash
conda create -n salad-unlearn python=3.9 -y
conda activate salad-unlearn
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
pip install rich  # Required for improved logging
```

## 📖 Dataset & Weights
1.  **Dataset:** Place `humanml3d` and `kit-ml` in `dataset/`.
2.  **Weights:** Download pre-trained SALAD models:
    ```bash
    bash prepare/download_t2m.sh
    bash prepare/download_glove.sh
    ```3.  **Keyword Splits:** Pre-generated data splits for concepts like "kick" are located in `kw_splits/`.

---

## 🧹 Usage: Unlearning Workflow

### 1. ESD (Fine-Tuning)
Destructive but robust. It creates a copy of the model and fine-tunes it using forget/preserve sets.
```bash
python unlearning/esd/train.py \
  --forget_split_file "kw_splits/train_val-w-kick.txt" \
  --preserve_split_file "kw_splits/train_val-wo-kick.txt"
```

### 2. LoRA (Adapters)
Non-destructive. Trains small adapters (~0.5-4% params) using a contrastive loss.
```bash
python unlearning/lora/train.py \
  --forget_file "kw_splits/train_val-w-kick.txt" \
  --retain_file "kw_splits/train_val-wo-kick.txt"
```

### 3. UCE (Instant Edit)
Training-free mathematical edit based on the average embeddings of forget/retain sets.
```bash
python unlearning/uce/edit.py \
  --forget_file "kw_splits/train_val-w-kick.txt" \
  --retain_file "kw_splits/train_val-wo-kick.txt"
```

---

## 📊 Evaluation
We provide a unified testing script `test_unlearn.py` that measures:
1.  **Efficacy:** Does the model still generate the forbidden concept?
2.  **Preservation:** Is the general generation quality (FID, R-Precision) intact?

**Evaluate ESD or UCE Models:**
```bash
python test_unlearn.py \
  --name "t2m_denoiser_vpred_vaegelu_ESD_kick" \
  --ckpt "latest.tar" \
  --forget_test_file "kw_splits/test-w-kick.txt" \
  --retain_test_file "kw_splits/test-wo-kick.txt"
```

**Evaluate LoRA Models:**
*Note: You must specify `--lora_rank` to load the adapters correctly.*
```bash
python test_unlearn.py \
  --name "t2m_denoiser_vpred_vaegelu_LoRA_contrast_kick" \
  --ckpt "latest.tar" \
  --lora_rank 16 \
  --forget_test_file "kw_splits/test-w-kick.txt" \
  --retain_test_file "kw_splits/test-wo-kick.txt"
```

---

## 📂 Repository Structure
```text
salad
├── unlearning/                 <-- NEW: Unlearning Core
│   ├── esd/                    <-- ESD Implementation
│   │   ├── train.py
│   │   └── trainer.py
│   ├── lora/                   <-- LoRA Implementation
│   │   ├── train.py
│   │   └── modules.py
│   └── uce/                    <-- UCE Implementation
│       └── edit.py
├── test_unlearn.py             <-- Main Unlearning Evaluation Script
├── kw_splits/                  <-- Data splits for forget/retain sets
├── train_denoiser.py           <-- Original SALAD training
└── ...
```

## 📚 Citation
**SALAD (CVPR 2025):**
```bibtex
@inproceedings{hong2025salad,
  title={SALAD: Skeleton-Aware Latent Diffusion Model for Text-driven Motion Generation and Editing},
  author={Hong, Seokhyeon and Kim, Chaelin and Yoon, Serin and Nam, Junghyun and Cha, Sihun and Noh, Junyong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```