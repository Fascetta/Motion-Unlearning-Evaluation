#!/bin/bash

# ==========================================
# 🧠 Motion Unlearning Benchmark Script
# Runs UCE, ESD, and LoRA + Evaluation
# ==========================================

# --- Configuration ---
MODEL_NAME="t2m_denoiser_vpred_vaegelu"
DATASET="t2m"
CONCEPT="kick"

# Training Hyperparameters
EPOCHS=5
LORA_RANK=16
LORA_ALPHA=16.0
LR_ESD=1e-5
LR_LORA=1e-4

# Handle spaces in concept name for folder naming (e.g. "wave hello" -> "wave_hello")
CONCEPT_SANITIZED=${CONCEPT// /_}

echo "======================================================"
echo "🚀 STARTING BENCHMARK FOR CONCEPT: '$CONCEPT'"
echo "   Model: $MODEL_NAME"
echo "   Epochs: $EPOCHS"
echo "======================================================"

# ---------------------------------------------------------
# 1. UCE (Unified Concept Editing)
# ---------------------------------------------------------
echo ""
echo "------------------------------------------------------"
echo "🔵 [1/3] Running UCE (Unified Concept Editing)..."
echo "------------------------------------------------------"

# A. Edit
python unlearning/uce/edit.py \
  --name $MODEL_NAME \
  --dataset_name $DATASET \
  --target_concept "$CONCEPT" \
  --uce_lambda 1.0

# B. Evaluate
echo "   -> Evaluating UCE..."
python test_unlearn.py \
  --name "${MODEL_NAME}_UCE_${CONCEPT_SANITIZED}" \
  --dataset_name $DATASET \
  --ckpt latest.tar \
  --target_concept "$CONCEPT"

# ---------------------------------------------------------
# 2. ESD (Erasing Stable Diffusion)
# ---------------------------------------------------------
echo ""
echo "------------------------------------------------------"
echo "🔴 [2/3] Running ESD (Erasing Stable Diffusion)..."
echo "------------------------------------------------------"

# A. Train
python unlearning/esd/train.py \
  --name $MODEL_NAME \
  --dataset_name $DATASET \
  --target_concept "$CONCEPT" \
  --negative_guidance 1.0 \
  --unlearn_epochs $EPOCHS \
  --unlearn_lr $LR_ESD

# B. Evaluate
echo "   -> Evaluating ESD..."
python test_unlearn.py \
  --name "${MODEL_NAME}_ESD_${CONCEPT_SANITIZED}" \
  --dataset_name $DATASET \
  --ckpt latest.tar \
  --target_concept "$CONCEPT"

# ---------------------------------------------------------
# 3. LoRA (Low-Rank Adaptation)
# ---------------------------------------------------------
echo ""
echo "------------------------------------------------------"
echo "🟣 [3/3] Running LoRA (Low-Rank Adaptation)..."
echo "------------------------------------------------------"

# A. Train
python unlearning/lora/train.py \
  --name $MODEL_NAME \
  --dataset_name $DATASET \
  --target_concept "$CONCEPT" \
  --lora_rank $LORA_RANK \
  --lora_alpha $LORA_ALPHA \
  --unlearn_epochs $EPOCHS \
  --unlearn_lr $LR_LORA

# B. Evaluate (Must pass rank!)
echo "   -> Evaluating LoRA..."
python test_unlearn.py \
  --name "${MODEL_NAME}_LoRA_${CONCEPT_SANITIZED}" \
  --dataset_name $DATASET \
  --ckpt latest.tar \
  --target_concept "$CONCEPT" \
  --lora_rank $LORA_RANK

echo ""
echo "======================================================"
echo "✅ BENCHMARK COMPLETE"
echo "   Check 'efficacy_...' folders in each checkpoint dir"
echo "   for visual results."
echo "======================================================"