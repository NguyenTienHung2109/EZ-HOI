#!/bin/bash
# HICO-DET testing with ViT-B/16 + Pre-Bridged Embeddings (Zero-shot unseen verb setting)
#
# This script evaluates EZ-HOI with pre-computed diffusion-bridged vision embeddings.
#
# PREREQUISITES:
# 1. Run precompute_bridged_vision_embeddings.py FIRST for TEST set:
#    python precompute_bridged_vision_embeddings.py \
#        --input_dir hicodet_pkl_files/clipbase_img_hicodet_test \
#        --output_dir hicodet_pkl_files/clipbase_img_hicodet_test_bridged \
#        --diffusion_model hoi_diffusion_results/model-300.pt \
#        --text_mean hicodet_pkl_files/hoi_text_mean_vitB_600.pkl \
#        --embed_dim 512 \
#        --inference_steps 600
#
# IMPORTANT:
# - Replace <path to the model file> with your trained checkpoint path

CUDA_VISIBLE_DEVICES=0 python main_tip_finetune.py --world-size 1 \
 --pretrained "checkpoints/detr-r50-hicodet.pth" \
 --output-dir checkpoints/hico_diffusion_vitB/ \
 --epochs 12  --use_insadapter  --num_classes 117 --use_multi_hot \
 --file1 hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p \
 --clip_dir_vit checkpoints/pretrained_CLIP/ViT-B-16.pt \
 --batch-size 8  --logits_type "HO"  --port 1236 \
 --txtcls_pt   --img_align  --unseen_pt_inj  --img_clip_pt  \
 --zs --zs_type "unseen_verb" \
 --clip_img_file hicodet_pkl_files/clipbase_img_hicodet_test_bridged \
 --eval --resume <path to the model file>
