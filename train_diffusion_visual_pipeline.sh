#!/bin/bash
# Complete pipeline for training diffusion on visual embeddings
#
# This script runs the full pipeline:
# 1. Extract adapted text embeddings (for text mean calculation)
# 2. Extract visual embeddings from training set
# 3. Train diffusion model on visual embeddings
# 4. Test diffusion bridge on sample images
#
# Prerequisites:
# - Trained EZ-HOI checkpoint
# - HICO-DET dataset
# - CLIP model checkpoint
# - diffusion-bridge submodule installed

set -e  # Exit on error

echo "========================================================================"
echo "Diffusion Training on Visual Embeddings - Full Pipeline"
echo "========================================================================"
echo ""

# ============================================================================
# Configuration
# ============================================================================

# Model checkpoint (CHANGE THIS to your trained EZ-HOI checkpoint)
CHECKPOINT="checkpoints/hico_HO_pt_default_vitbase/best.pth"

# CLIP model path (CHANGE THIS to match your checkpoint)
CLIP_MODEL_PATH="checkpoints/pretrained_CLIP/ViT-B-16.pt"
CLIP_MODEL_NAME="vitB"  # vitB or vitL

# Dataset configuration (CHANGE THIS if using different zero-shot setting)
NUM_CLASSES=117
ZS_TYPE="unseen_verb"

# Data paths
DATA_ROOT="hicodet"
OUTPUT_DIR="hicodet_pkl_files"

# Diffusion training parameters
BATCH_SIZE=64
TRAIN_STEPS=500000
LEARNING_RATE=8e-5
INFERENCE_STEPS=600  # For final testing
SCALE_FACTOR=5.0

# Derived paths
ADAPTED_TEXT_PKL="${OUTPUT_DIR}/hoi_adapted_text_embeddings_${CLIP_MODEL_NAME}_${NUM_CLASSES}_212classes.pkl"
TEXT_MEAN_PKL="${OUTPUT_DIR}/hoi_text_mean_adapted_${ZS_TYPE}_${CLIP_MODEL_NAME}_${NUM_CLASSES}_212classes.pkl"
VISUAL_EMB_PKL="${OUTPUT_DIR}/hoi_visual_embeddings_normalized_${CLIP_MODEL_NAME}_train.pkl"
DIFFUSION_RESULTS="hoi_diffusion_results"

echo "Configuration:"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  CLIP model: ${CLIP_MODEL_PATH}"
echo "  Num classes: ${NUM_CLASSES}"
echo "  ZS type: ${ZS_TYPE}"
echo "  Output dir: ${OUTPUT_DIR}"
echo ""

# ============================================================================
# Step 1: Extract adapted text embeddings
# ============================================================================

echo "========================================================================"
echo "Step 1/4: Extracting adapted text embeddings..."
echo "========================================================================"
echo ""
echo "This extracts text embeddings through learnable prompts and computes"
echo "the text mean for normalization. Required for diffusion normalization."
echo ""

if [ ! -f "${ADAPTED_TEXT_PKL}" ] || [ ! -f "${TEXT_MEAN_PKL}" ]; then
    python extract_adapted_text_embeddings.py \
        --checkpoint ${CHECKPOINT} \
        --num_classes ${NUM_CLASSES} \
        --zs_type ${ZS_TYPE} \
        --clip_model_path ${CLIP_MODEL_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --scale_factor ${SCALE_FACTOR} \
        --device cuda

    echo ""
    echo "✓ Adapted text embeddings extracted successfully"
    echo "  Embeddings: ${ADAPTED_TEXT_PKL}"
    echo "  Text mean: ${TEXT_MEAN_PKL}"
else
    echo "✓ Adapted text embeddings already exist, skipping extraction"
    echo "  Embeddings: ${ADAPTED_TEXT_PKL}"
    echo "  Text mean: ${TEXT_MEAN_PKL}"
fi

echo ""
read -p "Press Enter to continue to Step 2..."
echo ""

# ============================================================================
# Step 2: Extract visual embeddings from training set
# ============================================================================

echo "========================================================================"
echo "Step 2/4: Extracting visual embeddings from training set..."
echo "========================================================================"
echo ""
echo "This extracts visual features (adapter_feat) from all HICO-DET training"
echo "images. This will take 1-2 hours depending on your GPU."
echo ""

if [ ! -f "${VISUAL_EMB_PKL}" ]; then
    python extract_visual_embeddings_for_diffusion.py \
        --checkpoint ${CHECKPOINT} \
        --data_root ${DATA_ROOT} \
        --num_classes ${NUM_CLASSES} \
        --zs_type ${ZS_TYPE} \
        --text_mean_path ${TEXT_MEAN_PKL} \
        --scale_factor ${SCALE_FACTOR} \
        --output_path ${VISUAL_EMB_PKL} \
        --batch_size 1 \
        --num_workers 4 \
        --device cuda \
        --clip_model_path ${CLIP_MODEL_PATH}

    echo ""
    echo "✓ Visual embeddings extracted successfully"
    echo "  Output: ${VISUAL_EMB_PKL}"
else
    echo "✓ Visual embeddings already exist, skipping extraction"
    echo "  File: ${VISUAL_EMB_PKL}"
fi

echo ""
read -p "Press Enter to continue to Step 3 (this will take 8-15 hours)..."
echo ""

# ============================================================================
# Step 3: Train diffusion model
# ============================================================================

echo "========================================================================"
echo "Step 3/4: Training diffusion model on visual embeddings..."
echo "========================================================================"
echo ""
echo "This trains a diffusion model on the extracted visual embeddings."
echo "Training will take approximately 8-15 hours on a single GPU."
echo ""
echo "You can monitor progress in: ${DIFFUSION_RESULTS}/"
echo "Checkpoints are saved every 10k steps."
echo ""

python train_hoi_diffusion.py \
    --data_path ${VISUAL_EMB_PKL} \
    --init_dim 32 \
    --dim_mults 1 2 4 8 \
    --timesteps 1000 \
    --objective pred_x0 \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --train_steps ${TRAIN_STEPS} \
    --gradient_accumulate 1 \
    --ema_decay 0.995 \
    --use_amp \
    --results_folder ${DIFFUSION_RESULTS}

echo ""
echo "✓ Diffusion training completed!"
echo "  Model saved to: ${DIFFUSION_RESULTS}/model-${TRAIN_STEPS}.pt"
echo ""
read -p "Press Enter to continue to Step 4 (testing)..."
echo ""

# ============================================================================
# Step 4: Test diffusion bridge
# ============================================================================

echo "========================================================================"
echo "Step 4/4: Testing diffusion bridge on sample images..."
echo "========================================================================"
echo ""
echo "This tests the trained diffusion bridge on test images and generates"
echo "visualizations showing the effect of diffusion on visual features."
echo ""

# Use the final trained model (or specify a different checkpoint)
DIFFUSION_MODEL="${DIFFUSION_RESULTS}/model-${TRAIN_STEPS}.pt"

# If final model doesn't exist, use latest checkpoint
if [ ! -f "${DIFFUSION_MODEL}" ]; then
    echo "Final model not found, searching for latest checkpoint..."
    DIFFUSION_MODEL=$(ls -t ${DIFFUSION_RESULTS}/model-*.pt | head -1)
    echo "Using: ${DIFFUSION_MODEL}"
fi

python test_visual_diffusion.py \
    --diffusion_model ${DIFFUSION_MODEL} \
    --checkpoint ${CHECKPOINT} \
    --text_mean_path ${TEXT_MEAN_PKL} \
    --adapted_text_pkl ${ADAPTED_TEXT_PKL} \
    --data_root ${DATA_ROOT} \
    --num_test_images 20 \
    --inference_steps ${INFERENCE_STEPS} \
    --output_dir test_diffusion_results \
    --device cuda

echo ""
echo "========================================================================"
echo "Pipeline completed successfully!"
echo "========================================================================"
echo ""
echo "Results:"
echo "  1. Adapted text embeddings: ${ADAPTED_TEXT_PKL}"
echo "  2. Text mean: ${TEXT_MEAN_PKL}"
echo "  3. Visual embeddings: ${VISUAL_EMB_PKL}"
echo "  4. Trained diffusion model: ${DIFFUSION_MODEL}"
echo "  5. Test results: test_diffusion_results/"
echo ""
echo "Next steps:"
echo "  1. Review test results: test_diffusion_results/diffusion_effect.png"
echo "  2. If diffusion improves alignment, integrate into EZ-HOI inference:"
echo "     - Add --diffusion_bridge argument to inference script"
echo "     - Load DiffusionBridgeHOI module with trained model"
echo "  3. Run full evaluation on HICO-DET test set with diffusion enabled"
echo ""
echo "========================================================================"
