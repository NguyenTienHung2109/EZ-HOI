#!/bin/bash
# Inference script with diffusion bridge enabled
# This script tests the trained EZ-HOI model with diffusion bridge for modality alignment
#
# IMPORTANT: Use ADAPTED embeddings for diffusion paths!
# The diffusion model should be trained on adapted text embeddings (after txtmem_adapter),
# not raw CLIP embeddings, to ensure vision-text alignment consistency.

CUDA_VISIBLE_DEVICES=0 python main_tip_finetune.py --world-size 1 \
 --pretrained "checkpoints/detr-r50-hicodet.pth" \
 --output-dir checkpoints/hico_HO_pt_default_vitbase/ \
 --epochs 12  --use_insadapter  --num_classes 117 --use_multi_hot \
 --file1 hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p \
 --clip_dir_vit checkpoints/pretrained_CLIP/ViT-B-16.pt \
 --batch-size 8  --logits_type "HO"  --port 1236 \
 --txtcls_pt   --img_align  --unseen_pt_inj  --img_clip_pt  \
 --zs --zs_type "unseen_verb" \
 --clip_img_file   hicodet_pkl_files/clipbase_img_hicodet_test \
 --eval --resume checkpoints/hico_HO_pt_default_vitbase/best.pth \
 --use_diffusion_bridge \
 --diffusion_model_path hoi_diffusion_results_adapted/model-300.pt \
 --diffusion_text_mean hicodet_pkl_files/hoi_text_mean_adapted_unseen_verb_vitB_117.pkl \
 --diffusion_inference_steps 600

# Notes:
# - --use_diffusion_bridge enables the diffusion bridge module
# - --diffusion_model_path points to the trained diffusion model (trained on ADAPTED embeddings)
# - --diffusion_text_mean points to the HOI text mean from adapted embeddings
# - --diffusion_inference_steps=600 is the default (can reduce to 300 for faster inference)
#
# CRITICAL: Ensure diffusion paths match the output from extract_adapted_text_embeddings.py:
#   --diffusion_model_path: hoi_diffusion_results_adapted/model-300.pt
#   --diffusion_text_mean: hicodet_pkl_files/hoi_text_mean_adapted_unseen_verb_vitB_117.pkl
#
# Pipeline order:
#   1. Train EZ-HOI (bash scripts/hico_train_vitB_zs.sh)
#   2. Extract adapted embeddings (python extract_adapted_text_embeddings.py ...)
#   3. Train diffusion on adapted embeddings (python train_hoi_diffusion.py ...)
#   4. Run this script for inference with diffusion bridge
#
# For comparison without diffusion, simply remove the diffusion-related flags or use:
#   bash scripts/hico_test_vitB_zs.sh
