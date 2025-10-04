#!/bin/bash
# Test script for extract_adapted_text_embeddings.py
# This will extract adapted text embeddings from trained EZ-HOI checkpoint

echo "Testing extract_adapted_text_embeddings.py..."
echo ""

# Update these paths to match your checkpoint and CLIP model locations
CHECKPOINT_PATH="checkpoints/hico_HO_pt_default_vitbase/best.pth"
CLIP_MODEL_PATH="checkpoints/pretrained_CLIP/ViT-B-16.pt"

python extract_adapted_text_embeddings.py \
  --checkpoint ${CHECKPOINT_PATH} \
  --num_classes 117 \
  --zs_type unseen_verb \
  --clip_model_path ${CLIP_MODEL_PATH} \
  --output_dir hicodet_pkl_files \
  --scale_factor 5.0 \
  --device cuda

echo ""
echo "Test completed!"
