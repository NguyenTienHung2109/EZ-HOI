#!/bin/bash

# Create log directory if it doesn't exist
mkdir -p logs

TIMESTAMP=$(TZ="Asia/Ho_Chi_Minh" date +"%Y%m%d_%H%M%S")

LOG_FILE="logs/training_${TIMESTAMP}.log"

echo "Starting training ... Log will be saved to $LOG_FILE"

START_TIME=$(date +%s)

CUDA_VISIBLE_DEVICES=1,2 python main_tip_finetune.py --world-size 2 \
 --pretrained "checkpoints/detr-r50-hicodet.pth" \
 --output-dir checkpoints/hico_training_ckpt/ \
 --epochs 12  --use_insadapter  --num_classes 117 --use_multi_hot \
 --file1 checkpoints/hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p \
 --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt \
 --batch-size 12  --logits_type "HO"  --port 1236 \
 --txtcls_pt   --img_align  --unseen_pt_inj  --img_clip_pt  \
 --zs --zs_type "unseen_verb" \
 --clip_img_file  /app/hicodet/hico_20160224_det/clipbase_img_hicodet_train \
 --use_diffusion_bridge \
 --diffusion_model_path checkpoints/diffusion/model-100.pt \
 --vision_mean_path checkpoints/hicodet_pkl_files/hoi_vision_mean_vitB_train.pkl \
 --inference_steps 600 \
 --print-interval 200 2>&1 | tee "$LOG_FILE"

END_TIME=$(date +%s)
if [[ -n "$START_TIME" && "$START_TIME" =~ ^[0-9]+$ ]]; then
  ELAPSED_TIME=$((END_TIME - START_TIME))
  echo "Training time: $ELAPSED_TIME seconds" >> "$LOG_FILE"
  echo "Training time in minutes: $((ELAPSED_TIME / 60)) minutes" >> "$LOG_FILE"
  echo "Training time in hours: $((ELAPSED_TIME / 3600)) hours" >> "$LOG_FILE"
else
  echo "Error: START_TIME is not properly initialized." >> "$LOG_FILE"
fi

echo "Training completed. Log saved to $LOG_FILE"