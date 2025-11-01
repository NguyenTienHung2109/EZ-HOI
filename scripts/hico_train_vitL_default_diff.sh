#!/bin/bash
# HICO-DET training with ViT-L/14@336px + Pre-Bridged Embeddings (Default rare/non-rare setting)
#
# This script trains EZ-HOI with pre-computed diffusion-bridged vision embeddings.
#
# PREREQUISITES:
# 1. Run precompute_bridged_vision_embeddings.py FIRST to generate bridged embeddings:
#    python precompute_bridged_vision_embeddings.py \
#        --input_dir hicodet_pkl_files/clip336_img_hicodet_train \
#        --output_dir hicodet_pkl_files/clip336_img_hicodet_train_bridged \
#        --diffusion_model hoi_diffusion_results/model-vitL-300.pt \
#        --text_mean hicodet_pkl_files/hoi_text_mean_vitL_600.pkl \
#        --embed_dim 768 \
#        --inference_steps 600

CUDA_VISIBLE_DEVICES=0,1,2,3 python main_tip_finetune.py --world-size 4 \
 --pretrained "checkpoints/detr-r50-hicodet.pth" \
 --output-dir checkpoints/hico_diffusion_vitL_default/ \
 --epochs 12  --use_insadapter  --num_classes 117 --use_multi_hot \
 --file1 hicodet_pkl_files/hicodet_union_embeddings_cachemodel_crop_padding_zeros_vit336.p \
 --clip_dir_vit checkpoints/pretrained_CLIP/ViT-L-14-336px.pt \
 --batch-size 4  --logits_type "HO"  --port 1231 \
 --txtcls_pt   --img_align  --unseen_pt_inj  --img_clip_pt \
 --clip_img_file hicodet_pkl_files/clip336_img_hicodet_train_bridged
