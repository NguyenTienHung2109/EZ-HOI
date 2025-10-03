# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EZ-HOI is a zero-shot Human-Object Interaction (HOI) detection framework using Vision-Language Model (VLM) adaptation via guided prompt learning. Published at NeurIPS 2024, it builds upon UPT and ADA-CM.

**Key Architecture:**
- **Base Detector**: DETR with ResNet-50 backbone for object detection
- **VLM Backbone**: CLIP (ViT-B/16 or ViT-L/14@336px) with learnable prompt adapters
- **Main Components**:
  - `upt_tip_cache_model_free_finetune_distillself.py`: Core model architecture with UPT detector and CLIP integration
  - `CLIP_models_adapter_prior2.py`: Modified CLIP model with adapter layers for prompt learning
  - `main_tip_finetune.py`: Training entry point with distributed training support
  - `utils_tip_cache_and_union_finetune.py`: Custom dataset loaders, training engine (CustomisedDLE), and utilities
  - `inference.py`: Inference and visualization utilities

**Datasets Supported:**
- HICO-DET: 600 HOI classes (or 117 verbs for zero-shot verb setting)
- V-COCO: 24/236 classes

## Environment Setup

1. Follow environment setup from [UPT](https://github.com/fredzzhang/upt) and [ADA-CM](https://github.com/ltttpku/ADA-CM/tree/main)

2. **Critical**: Use local CLIP directory instead of pip-installed clip package:
   ```bash
   export PYTHONPATH=$PYTHONPATH:"$PWD/CLIP"
   ```
   This ensures the modified CLIP implementation with adapter support is used.

3. Install modified [pocket library](https://github.com/fredzzhang/pocket) with changes mentioned in [issue #2](https://github.com/ChelsieLei/EZ-HOI/issues/2)

## Dataset Structure

```
EZ-HOI/
├── hicodet/
│   └── hico_20160224_det/
│       ├── annotations/
│       └── images/
│           ├── train2015/
│           └── test2015/
├── vcoco/
│   └── mscoco2014/
│       ├── train2014/
│       └── val2014/
├── hicodet_pkl_files/          # Pre-extracted CLIP features
│   ├── clip336_img_hicodet_train/
│   ├── clip336_img_hicodet_test/
│   ├── clipbase_img_hicodet_train/
│   └── clipbase_img_hicodet_test/
└── checkpoints/
    ├── detr-r50-hicodet.pth    # Pretrained DETR
    └── pretrained_CLIP/
        ├── ViT-B-16.pt
        └── ViT-L-14-336px.pt
```

## Pre-processing

Extract CLIP image features before training:
```bash
python CLIP_hicodet_extract.py
```
Ensure paths in the script match your dataset locations. This generates `.pkl` files in `hicodet_pkl_files/`.

## Training Commands

### HICO-DET Zero-Shot Settings

**Unseen Verb (UV) with ViT-B/16:**
```bash
bash scripts/hico_train_vitB_zs.sh
```

**Unseen Verb with ViT-L:**
```bash
bash scripts/hico_train_vitL_zs.sh
```

**Default (Rare/Non-Rare) with ViT-L:**
```bash
bash scripts/hico_train_vitL_default.sh
```

**Key Training Arguments:**
- `--world-size`: Number of GPUs for distributed training
- `--pretrained`: Path to pretrained DETR checkpoint
- `--clip_dir_vit`: Path to CLIP checkpoint (ViT-B-16.pt or ViT-L-14-336px.pt)
- `--file1`: Path to union embeddings pickle file
- `--clip_img_file`: Path to pre-extracted CLIP image features directory
- `--zs`: Enable zero-shot mode
- `--zs_type`: Zero-shot split type (unseen_verb, unseen_object, rare_first, non_rare_first, default)
- `--num_classes`: 117 for verb prediction, 600 for full HOI
- `--logits_type`: "HO" for human-object pair prediction
- `--txtcls_pt`, `--img_align`, `--unseen_pt_inj`, `--img_clip_pt`: Enable various prompt learning components
- `--use_insadapter`: Enable instance adapters
- `--use_multi_hot`: Use multi-hot encoding for labels

## Testing Commands

**Test with ViT-B:**
```bash
bash scripts/hico_test_vitB_zs.sh
```

**Test with ViT-L:**
```bash
bash scripts/hico_test_vitL_zs.sh
```

Add `--eval --resume <path_to_checkpoint>` to test script.

## Inference on Custom Images

```bash
bash scripts/inference.sh
```

Modify script to include `--self_image_path <path_to_image>` argument.

## Model Architecture Notes

**Training Flow:**
1. `main_tip_finetune.py` creates `DataFactory` datasets with CLIP preprocessing
2. `build_detector()` constructs:
   - DETR backbone from detr/models
   - Modified CLIP model with prompt adapters from `CLIP_models_adapter_prior2.py`
   - UPT (Unary-Pairwise Transformer) HOI detection head
3. `CustomisedDLE` (Distributed Learning Engine) handles training loop with gradient clipping
4. Loss computed from interaction predictions using binary focal loss

**Key Design Patterns:**
- Prompt learning with MaPLe-style vision and language prompts
- Instance-level adapters in CLIP visual encoder
- Zero-shot transfer via guided prompt injection from seen to unseen classes
- CLIP image features pre-extracted and cached for efficiency

## Important File Mappings

- `hico_text_label.py`, `vcoco_text_label.py`: Class definitions and zero-shot splits
- `hico_list.py`, `vcoco_list.py`: Verb-object pair definitions
- `transformer_module.py`: Custom transformer layers for adapters
- `ops.py`: Loss functions (binary_focal_loss_with_logits) and box operations
- `detr/`: Official DETR implementation as submodule
- `hicodet/`, `vcoco/`: Dataset utilities from Pocket library integration

## Common Development Patterns

**Multi-GPU Training:**
Uses `torch.multiprocessing.spawn` with `DistributedSampler`. Single GPU training uses `world-size=1` to skip spawn.

**Zero-Shot Evaluation:**
Model filters training data to exclude unseen classes based on `hico_unseen_index[zs_type]`, then evaluates on full test set.

**Checkpoint Loading:**
- DETR weights: `args.pretrained` (detr-r50-hicodet.pth)
- CLIP weights: `args.clip_dir_vit` (ViT weights)
- Resume training: `args.resume` (full model checkpoint)
