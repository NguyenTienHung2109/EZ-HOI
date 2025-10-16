#!/bin/bash
# HICO-DET training with ViT-L/14@336px + Diffusion Bridge (Zero-shot unseen verb setting)
#
# This script trains EZ-HOI with frozen MS-COCO diffusion bridge for vision-text alignment.
#
# IMPORTANT:
# - Do NOT use --txt_align flag (text adapter corrupts CLIP distribution)
# - Requires: configs/diffusion_bridge_config.yaml with correct paths
# - Your COCO diffusion must have been trained with ViT-L/14 CLIP (not ViT-B/16)!
# - Training will be 2-3x slower due to diffusion sampling (100 steps per batch)
# - Update diffusion_bridge_config.yaml with your COCO checkpoint paths before running

CUDA_VISIBLE_DEVICES=0,1,2,3 python main_tip_finetune.py --world-size 4 \
 --pretrained "checkpoints/detr-r50-hicodet.pth" \
 --output-dir checkpoints/hico_diffusion_vitL/ \
 --epochs 12  --use_insadapter  --num_classes 117 --use_multi_hot \
 --file1 hicodet_pkl_files/hicodet_union_embeddings_cachemodel_crop_padding_zeros_vit336.p \
 --clip_dir_vit checkpoints/pretrained_CLIP/ViT-L-14-336px.pt \
 --batch-size 4  --logits_type "HO"  --port 1231 \
 --txtcls_pt   --img_align  --unseen_pt_inj  --img_clip_pt \
 --clip_img_file hicodet_pkl_files/clip336_img_hicodet_train \
 --zs --zs_type "unseen_verb" \
 --use_diffusion_bridge \
 --diffusion_config configs/diffusion_bridge_config.yaml

