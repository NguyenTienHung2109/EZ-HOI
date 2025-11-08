#!/bin/bash

# Make sure gdown is installed
pip install gdown

mkdir -p ./checkpoints

# Download pretrained weights DETR hicodet

DETR_HICO_FILE=detr-r50-hicodet.pth
DETR_HICO_ID=1BQ-0tbSH7UC6QMIMMgdbNpRw2NcO8yAD

if [ -f "./checkpoints/$DETR_HICO_FILE" ]; then
  echo "$DETR_HICO_FILE already exists."
else
  echo "Downloading $DETR_HICO_FILE ..."

    gdown --id "$DETR_HICO_ID" -O "./checkpoints/$DETR_HICO_FILE"

  echo "Downloaded $DETR_HICO_FILE."
fi


# Download pretrained weights DETR coco

DETR_COCO_FILE=detr-r50-vcoco.pth
DETR_COCO_ID=1AIqc2LBkucBAAb_ebK9RjyNS5WmnA4HV

if [ -f "./checkpoints/$DETR_COCO_FILE" ]; then
  echo "$DETR_COCO_FILE already exists."
else
  echo "Downloading $DETR_COCO_FILE ..."

    gdown --id "$DETR_COCO_ID" -O "./checkpoints/$DETR_COCO_FILE"

  echo "Downloaded $DETR_COCO_FILE."
fi


# Download Pre-extracted HICO-DET features
EXTRACTED_FEATURES_FILE=ADA-CM-FEATS.tar.gz
EXTRACTED_FEATURES_ID=1lUnUQD3XcWyQdwDHMi74oXBcivibGIWN

EXTRACTED_BBOXES=bbox_files.tar.gz
EXTRACTED_BBOXES_ID=19Mo1d4J6xX9jDNvDJHEWDpaiPKxQHQsT

mkdir -p ./checkpoints/hicodet_pkl_files

mkdir -p ./checkpoints/vcoco_pkl_files

if [ -f "./checkpoints/$EXTRACTED_FEATURES_FILE" ]; then
  echo "$EXTRACTED_FEATURES_FILE already exists."
else
  echo "Downloading $EXTRACTED_FEATURES_FILE ..."

    gdown --id "$EXTRACTED_FEATURES_ID" -O "./checkpoints/$EXTRACTED_FEATURES_FILE"

  echo "Downloaded $EXTRACTED_FEATURES_FILE."
fi

if [ -f "./checkpoints/$EXTRACTED_BBOXES" ]; then
  echo "$EXTRACTED_BBOXES already exists."
else
  echo "Downloading $EXTRACTED_BBOXES ..."

    gdown --id "$EXTRACTED_BBOXES_ID" -O "./checkpoints/$EXTRACTED_BBOXES"

  echo "Downloaded $EXTRACTED_BBOXES."
fi

# Extract pre-extracted features
echo "Extracting $EXTRACTED_FEATURES_FILE ..."
tar -xvf "./checkpoints/$EXTRACTED_FEATURES_FILE" -C ./checkpoints/
rm "./checkpoints/$EXTRACTED_FEATURES_FILE"
echo "Extracted $EXTRACTED_FEATURES_FILE."

echo "Extracting $EXTRACTED_BBOXES ..."
tar -xvf "./checkpoints/$EXTRACTED_BBOXES" -C ./checkpoints/
rm "./checkpoints/$EXTRACTED_BBOXES"
echo "Extracted $EXTRACTED_BBOXES."
echo "Setup done."

mv ./checkpoints/ADA-CM-FEATS/hicodet_union_embeddings_cachemodel_crop_padding_zeros_vit336.p ./checkpoints/hicodet_pkl_files/
mv ./checkpoints/ADA-CM-FEATS/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p ./checkpoints/hicodet_pkl_files/
mv ./checkpoints/vcoco_union_embeddings_cachemodel_crop_padding_zeros_vit16.p ./checkpoints/vcoco_pkl_files/
mv ./checkpoints/vcoco_union_embeddings_cachemodel_crop_padding_zeros_vit336.p ./checkpoints/vcoco_pkl_files/


mv ./checkpoints/bbox_files/hicodet_train_bbox_R50.p ./checkpoints/hicodet_pkl_files/
mv ./checkpoints/bbox_files/hicodet_test_bbox_R50.p ./checkpoints/hicodet_pkl_files/
mv ./checkpoints/bbox_files/vcoco_train_bbox_R50.p ./checkpoints/vcoco_pkl_files/
mv ./checkpoints/bbox_files/vcoco_test_bbox_R50.p ./checkpoints/vcoco_pkl_files/

rm -rf ./checkpoints/ADA-CM-FEATS
rm -rf ./checkpoints/bbox_files

# The extracted folder structure is as follows:
# checkpoints
# |   |- hicodet_pkl_files
# |   |   |- union_embeddings_cachemodel_crop_padding_zeros_vitb16.p
# |   |   |- hicodet_union_embeddings_cachemodel_crop_padding_zeros_vit336.p
# |   |   |- hicodet_train_bbox_R50.p
# |   |   |- hicodet_test_bbox_R50.p
# |   |- vcoco_pkl_files
# |   |   |- vcoco_union_embeddings_cachemodel_crop_padding_zeros_vit16.p
# |   |   |- vcoco_union_embeddings_cachemodel_crop_padding_zeros_vit336.p
# |   |   |- vcoco_train_bbox_R50.p
# |   |   |- vcoco_test_bbox_R50.p


# Download pre-trained CLIP
CLIP_DIR="./checkpoints/pretrained_clip"

CLIP_BASE_URL="https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"
CLIP_FILE="ViT-B-16.pt"

CLIP_L_URL="https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt"
CLIP_L_FILE="ViT-L-14-336px.pt"

mkdir -p $CLIP_DIR

if [ -f "$CLIP_DIR/$CLIP_FILE" ]; then
  echo "$CLIP_FILE already exists."
else
  echo "Downloading $CLIP_FILE ..."
  wget -O "$CLIP_DIR/$CLIP_FILE" "$CLIP_BASE_URL"
  echo "Downloaded $CLIP_FILE."
fi

if [ -f "$CLIP_DIR/$CLIP_L_FILE" ]; then
  echo "$CLIP_L_FILE already exists."
else
  echo "Downloading $CLIP_L_FILE ..."
  wget -O "$CLIP_DIR/$CLIP_L_FILE" "$CLIP_L_URL"
  echo "Downloaded $CLIP_L_FILE."
fi


# Download Diffusion checkpoints
DIFFUSION_DIR="./checkpoints/diffusion"

DIFFUSION_FILE=model-100.pt
DIFFUSION_ID="1-CbASsfBmw-SGf_3kQ8Iqz-i_i-DzQaC"
mkdir -p $DIFFUSION_DIR

if [ -f "$DIFFUSION_DIR/$DIFFUSION_FILE" ]; then
  echo "$DIFFUSION_FILE already exists."
else
  echo "Downloading $DIFFUSION_FILE ..."

    gdown --id "$DIFFUSION_ID" -O "$DIFFUSION_DIR/$DIFFUSION_FILE"

  echo "Downloaded $DIFFUSION_FILE."
fi

