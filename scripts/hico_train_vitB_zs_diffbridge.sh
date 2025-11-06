#!/bin/bash
# HICO-DET training with ViT-B/16 + Diffusion Bridge on HOI ROI Features
# Zero-shot unseen verb setting
#
# This script applies diffusion bridge to fused HOI features (vis_feat)
# to align them with text distribution before similarity computation.
#
# Key differences from baseline:
#   --use_diffusion_bridge: Enable diffusion on HOI features
#   --diffusion_model_path: Path to trained diffusion model
#   --vision_mean_path: Vision mean for normalization
#   --inference_steps 600: Timestep range for DDIM (600→0, with 5 iterations)
#                         Note: Actual denoising iterations = 5 (hardcoded in model)
#                               inference_steps controls timestep RANGE, not iteration count

CUDA_VISIBLE_DEVICES=0,1,2,3 python main_tip_finetune.py --world-size 4 \
 --pretrained "checkpoints/detr-r50-hicodet.pth" \
 --output-dir checkpoints/hico_HO_pt_default_vitbase_diffbridge/ \
 --epochs 12  --use_insadapter  --num_classes 117 --use_multi_hot \
 --file1 hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p \
 --clip_dir_vit checkpoints/pretrained_CLIP/ViT-B-16.pt \
 --batch-size 8  --logits_type "HO"  --port 1236 \
 --txtcls_pt   --img_align  --unseen_pt_inj  --img_clip_pt  \
 --zs --zs_type "unseen_verb" \
 --clip_img_file   hicodet_pkl_files/clipbase_img_hicodet_train \
 --use_diffusion_bridge \
 --diffusion_model_path checkpoints/diffusion_bridge/model-300.pt \
 --vision_mean_path checkpoints/diffusion_bridge/vision_mean.pkl \
 --inference_steps 600
 --clip_test

