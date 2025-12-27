#!/bin/bash

# =========================================================
# 🧠 Motion Unlearning Benchmark Script
# Runs and evaluates ESD, UCE, and LoRA+ESD methods.
# =========================================================

# --- Configuration ---
MODEL_NAME="t2m_denoiser_vpred_vaegelu"
DATASET="t2m"
CONCEPT="a person kicking"

# Concepts for Specificity Test (should NOT be forgotten)
RELATED_CONCEPTS="\"a person jumping\" \"a person punching\""

# Training Hyperparameters (tuned from our experiments)
EPOCHS_ESD=10
EPOCHS_LORA=10
LR_ESD=1e-5
LR_LORA=1e-4
LORA_RANK=16
NEG_GUIDANCE=2.5  # A stronger guidance is often needed
UCE_LAMBDA=1.5    # Strength for UCE edit

# Handle spaces in concept name for folder naming (e.g. "a person kicking" -> "a_person_kicking")
CONCEPT_SANITIZED=${CONCEPT// /_}

# --- SCRIPT START ---
echo "======================================================"
echo "🚀 STARTING BENCHMARK FOR CONCEPT: '$CONCEPT'"
echo "   Base Model: $MODEL_NAME"
echo "======================================================"


# ---------------------------------------------------------
# 1. BASELINE EVALUATION (CRITICAL STEP)
# ---------------------------------------------------------
echo ""
echo "------------------------------------------------------"
echo "📊 [1/4] Evaluating ORIGINAL model (Baseline)..."
echo "------------------------------------------------------"
python evaluate_unlearning.py \
  --original_name "$MODEL_NAME" \
  --unlearned_name "$MODEL_NAME" \
  --target_concept "$CONCEPT" \
  --related_concepts $RELATED_CONCEPTS \
  --fast_eval # Use --fast_eval for speed, remove for final paper-quality results


# ---------------------------------------------------------
# 2. ESD (Full Fine-Tuning)
# ---------------------------------------------------------
UNLEARNED_MODEL_ESD="${MODEL_NAME}_ESD_${CONCEPT_SANITIZED}"
echo ""
echo "------------------------------------------------------"
echo "🔴 [2/4] Running ESD (Full Fine-Tuning)..."
echo "------------------------------------------------------"

# A. Train
python unlearning/esd/train.py \
  --name $MODEL_NAME \
  --dataset_name $DATASET \
  --target_concept "$CONCEPT" \
  --negative_guidance $NEG_GUIDANCE \
  --unlearn_epochs $EPOCHS_ESD \
  --unlearn_lr $LR_ESD

# B. Evaluate
echo "   -> Evaluating ESD..."
python evaluate_unlearning.py \
  --original_name "$MODEL_NAME" \
  --unlearned_name "$UNLEARNED_MODEL_ESD" \
  --target_concept "$CONCEPT" \
  --related_concepts $RELATED_CONCEPTS \
  --fast_eval


# ---------------------------------------------------------
# 3. UCE (One-Shot Edit)
# ---------------------------------------------------------
UNLEARNED_MODEL_UCE="${MODEL_NAME}_UCE_${CONCEPT_SANITIZED}"
echo ""
echo "------------------------------------------------------"
echo "🔵 [3/4] Running UCE (One-Shot Edit)..."
echo "------------------------------------------------------"

# A. Edit
python unlearning/uce/edit.py \
  --name $MODEL_NAME \
  --dataset_name $DATASET \
  --target_concept "$CONCEPT" \
  --uce_lambda $UCE_LAMBDA

# B. Evaluate
echo "   -> Evaluating UCE..."
python evaluate_unlearning.py \
  --original_name "$MODEL_NAME" \
  --unlearned_name "$UNLEARNED_MODEL_UCE" \
  --target_concept "$CONCEPT" \
  --related_concepts $RELATED_CONCEPTS \
  --fast_eval


# ---------------------------------------------------------
# 4. LoRA + ESD (Parameter-Efficient Fine-Tuning)
# ---------------------------------------------------------
UNLEARNED_MODEL_LORA="${MODEL_NAME}_LoRA_${CONCEPT_SANITIZED}"
echo ""
echo "------------------------------------------------------"
echo "🟣 [4/4] Running LoRA + ESD..."
echo "------------------------------------------------------"

# A. Train
python unlearning/lora/train.py \
  --name $MODEL_NAME \
  --dataset_name $DATASET \
  --target_concept "$CONCEPT" \
  --negative_guidance $NEG_GUIDANCE \
  --lora_rank $LORA_RANK \
  --unlearn_epochs $EPOCHS_LORA \
  --unlearn_lr $LR_LORA

# B. Evaluate
echo "   -> Evaluating LoRA..."
python evaluate_unlearning.py \
  --original_name "$MODEL_NAME" \
  --unlearned_name "$UNLEARNED_MODEL_LORA" \
  --target_concept "$CONCEPT" \
  --related_concepts $RELATED_CONCEPTS \
  --fast_eval

echo ""
echo "======================================================"
echo "✅ BENCHMARK COMPLETE"
echo "   Check 'logs/eval_logs/' for detailed comparison logs."
echo "   Check 'checkpoints/.../evaluation/videos/' for visual results."
echo "======================================================"