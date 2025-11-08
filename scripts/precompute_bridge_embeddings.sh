# # Train
# python precompute_bridged_vision_embeddings.py \
#     --input_dir /mnt/sdc/duong.quang.minh/lab_data/hico_20160224_det/clipbase_img_hicodet_train \
#     --output_dir /mnt/sdc/duong.quang.minh/lab_data/hico_20160224_det/clipbase_img_hicodet_train_bridged \
#     --diffusion_model /mnt/sdc/duong.quang.minh/lab_data/mscoco/results/model-100.pt \
#     --vision_mean /mnt/sdc/duong.quang.minh/lab_data/ezhoi/checkpoints/hicodet_pkl_files/hoi_vision_mean_vitB_train.pkl \
#     --embed_dim 512 \
#     --inference_steps 600 \
#     --num_gpus 2


# Test
python precompute_bridged_vision_embeddings.py \
      --input_dir /mnt/sdc/duong.quang.minh/lab_data/hico_20160224_det/clipbase_img_hicodet_test \
      --output_dir /mnt/sdc/duong.quang.minh/lab_data/hico_20160224_det/clipbase_img_hicodet_test_bridged \
      --diffusion_model /mnt/sdc/duong.quang.minh/lab_data/mscoco/results/model-100.pt \
      --vision_mean /mnt/sdc/duong.quang.minh/lab_data/ezhoi/checkpoints/hicodet_pkl_files/hoi_vision_mean_vitB_test.pkl \
      --embed_dim 512 \
      --inference_steps 600 \
      --num_gpus 2