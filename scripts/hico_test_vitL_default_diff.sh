#!/bin/bash
# HICO-DET testing with ViT-L/14@336px + Diffusion Bridge (Default rare/non-rare setting)
#
# This script evaluates EZ-HOI with diffusion bridge for vision-text alignment.
#
# IMPORTANT:
# - Do NOT use --txt_align flag (text adapter corrupts CLIP distribution)
# - Ensure diffusion model (model_59.pt) and text mean (normalized_text_embed_mean.pkl) exist
# - Your COCO diffusion should have been trained with matching CLIP architecture
# - Replace <path to the model file> with your trained checkpoint path

CUDA_VISIBLE_DEVICES=0 python main_tip_finetune.py --world-size 1 \
 --pretrained "checkpoints/detr-r50-hicodet.pth" \
 --output-dir checkpoints/hico_diffusion_vitL_default/ \
 --epochs 12  --use_insadapter  --num_classes 117 --use_multi_hot \
 --file1 hicodet_pkl_files/hicodet_union_embeddings_cachemodel_crop_padding_zeros_vit336.p \
 --clip_dir_vit checkpoints/pretrained_CLIP/ViT-L-14-336px.pt \
 --batch-size 4  --logits_type "HO"  --port 1231 \
 --txtcls_pt   --img_align  --unseen_pt_inj  --img_clip_pt \
 --clip_img_file hicodet_pkl_files/clip336_img_hicodet_test \
 --use_diffusion_bridge \
 --diffusion_model_path diffusion-bridge/ddpm/results/model_59.pt \
 --diffusion_text_mean diffusion-bridge/ddpm/data/coco/normalized_text_embed_mean.pkl \
 --diffusion_inference_steps 600 \
 --eval --resume <path to the model file>
