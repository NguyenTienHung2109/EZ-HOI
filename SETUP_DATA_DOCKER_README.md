# EZ-HOI Setup Guide: Data, Checkpoints & Docker

This guide provides step-by-step instructions for setting up the EZ-HOI project, including downloading required datasets, pre-trained models, and running the project using Docker.

## Prerequisites

- CUDA-compatible GPU (for training/inference)
- Docker and Docker Compose (for containerized setup)
- Internet connection for downloading datasets and checkpoints

## 1. Download Pre-trained Models and Features

The setup script automatically downloads all required components:

```bash
bash scripts/setup.sh
```

### What gets downloaded:

- **DETR Pre-trained Weights**:
  - `detr-r50-hicodet.pth` - DETR ResNet-50 trained on HICO-DET
  - `detr-r50-vcoco.pth` - DETR ResNet-50 trained on V-COCO

- **Pre-extracted Features**:
  - HICO-DET and V-COCO union embeddings for ViT-B/16 and ViT-L/14@336px
  - Bounding box files for train/test splits

- **CLIP Models**:
  - `ViT-B-16.pt` - CLIP ViT-Base/16 model
  - `ViT-L-14-336px.pt` - CLIP ViT-Large/14@336px model

- **Diffusion Model** (~400MB):
  - `model-100.pt` - Pre-trained diffusion bridge model

### Final checkpoint structure:
```
checkpoints/
├── detr-r50-hicodet.pth
├── detr-r50-vcoco.pth
├── pretrained_clip/
│   ├── ViT-B-16.pt
│   └── ViT-L-14-336px.pt
├── diffusion/
│   └── model-100.pt
├── hicodet_pkl_files/
│   ├── union_embeddings_cachemodel_crop_padding_zeros_vitb16.p
│   ├── hicodet_union_embeddings_cachemodel_crop_padding_zeros_vit336.p
│   ├── hicodet_train_bbox_R50.p
│   └── hicodet_test_bbox_R50.p
└── vcoco_pkl_files/
    ├── vcoco_union_embeddings_cachemodel_crop_padding_zeros_vit16.p
    ├── vcoco_union_embeddings_cachemodel_crop_padding_zeros_vit336.p
    ├── vcoco_train_bbox_R50.p
    └── vcoco_test_bbox_R50.p
```

## 2. Download HICO-DET Dataset

Navigate to the hicodet directory and run the download script:

```bash
cd hicodet/
bash download.sh
```

### What gets downloaded:
- **HICO-DET Dataset**: Human-Object Interaction Detection dataset
- **Directory**: `hicodet/hico_20160224_det/`
- **Contains**: Train/test images and annotations

### Dataset structure:
```
hicodet/hico_20160224_det/
├── images/
│   ├── train2015/     # Training images
│   └── test2015/      # Test images
└── annotations/       # HOI annotations
```

## 3. Extract CLIP Features (Optional)

If you need to re-extract CLIP features or extract features for custom data:

### Prerequisites:
- Complete steps 1 and 2 above
- Ensure the dataset path in `CLIP_hicodet_extract.py` matches your setup

### Run extraction:
```bash
python CLIP_hicodet_extract.py
```

### What this does:
- Extracts image features using both ViT-B/16 and ViT-L/14@336px CLIP models
- Processes all training and test images from HICO-DET
- Saves individual `.pkl` files for each image in:
  - `hicodet_pkl_files/clipbase_img_hicodet_train/`
  - `hicodet_pkl_files/clipbase_img_hicodet_test/`
  - `hicodet_pkl_files/clip336_img_hicodet_train/`
  - `hicodet_pkl_files/clip336_img_hicodet_test/`

Other steps: vision mean, ...

## 4. Docker Setup

### Build and start the container:
```bash
docker compose up --build -d
```

### Container specifications:
- **Base**: CUDA-enabled Python environment
- **GPU**: All available NVIDIA GPUs
- **Shared Memory**: 4GB (configurable for large batch sizes)
- **Volumes**:
  - `./checkpoints` → `/app/checkpoints`
  - `./scripts` → `/app/scripts`
  - External dataset path → `/app/hicodet/hico_20160224_det`

### Access interactive shell:
```bash
docker exec -it ez-hoi-container /bin/bash
```

### Run training inside container:
Install tmux if needed

```bash
bash scripts/train_docker.sh
```