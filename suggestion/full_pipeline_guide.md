# Complete EZ-HOI + Diffusion Bridge Pipeline Guide

This guide provides step-by-step instructions for training and testing EZ-HOI with diffusion bridge integration for improved vision-text alignment.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Phase 0: Environment Setup](#phase-0-environment-setup)
4. [Phase 1: Extract Visual Features](#phase-1-extract-visual-features)
5. [Phase 2: Train EZ-HOI](#phase-2-train-ez-hoi)
6. [Phase 3: Extract Adapted Text Embeddings](#phase-3-extract-adapted-text-embeddings)
7. [Phase 4: Train Diffusion Model](#phase-4-train-diffusion-model)
8. [Phase 5: Test with Diffusion Bridge](#phase-5-test-with-diffusion-bridge)
9. [Advanced: Training with Different Configurations](#advanced-training-with-different-configurations)
10. [FAQ](#faq)

---

## Overview

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: Extract Visual Features (ONE-TIME, ~2-4 hours)    │
│  Purpose: Pre-cache CLIP image features for faster training │
│  Output: hicodet_pkl_files/clipbase_img_hicodet_*/*.pkl    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 2: Train EZ-HOI (NO diffusion, ~1-2 days)            │
│  Purpose: Train HOI detector with learnable text adapters   │
│  What happens: txtmem_adapter, act_descriptor_attn learn    │
│                to modify text embeddings                     │
│  Output: checkpoints/hico_HO_pt_default_vitbase/best.pth   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 3: Extract ADAPTED Text Embeddings (~5 min)          │
│  Purpose: Get text embeddings AFTER learned adapters        │
│  Key insight: Text embeddings are MODIFIED during training  │
│  Output: hoi_text_embeddings_adapted_*_normalized.pkl      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 4: Train Diffusion on ADAPTED Embeddings (~6-12h)    │
│  Purpose: Learn the distribution of ADAPTED text space      │
│  Why: Diffusion must align vision to the SAME space that    │
│       EZ-HOI uses (adapted space, not raw CLIP space)       │
│  Output: hoi_diffusion_results_adapted/model-300.pt        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 5: Inference with Diffusion Bridge                   │
│  Vision features → Diffusion (align) → Compare adapted text │
│  Result: Improved HOI detection performance                 │
└─────────────────────────────────────────────────────────────┘
```

### Why This Order Matters

**❌ WRONG Order (Original Approach):**
```
Extract raw text → Train Diffusion → Train EZ-HOI (modifies text)
                    ↓                                ↓
              Learn RAW space                  Use ADAPTED space
                    └──────── MISMATCH ────────────┘
```

**✅ CORRECT Order (This Pipeline):**
```
Train EZ-HOI → Extract ADAPTED text → Train Diffusion
      ↓                ↓                     ↓
  Learn adapters   Get final space    Learn ADAPTED space
                                             ↓
                            Inference: Align to SAME space
                                    ✅ CONSISTENT!
```

See `technical_explanation.md` for detailed theoretical explanation.

---

## Prerequisites

### 1. Hardware Requirements
- **GPU:** NVIDIA GPU with at least 16GB VRAM (24GB recommended)
- **RAM:** 32GB+ system RAM
- **Storage:** ~200GB for dataset + checkpoints + features

### 2. Software Requirements
- Python 3.8+
- PyTorch 1.9+
- CUDA 11.1+
- All dependencies from `requirements.txt`

### 3. Dataset Setup

Ensure your directory structure looks like this:

```
EZ-HOI/
├── hicodet/
│   └── hico_20160224_det/
│       ├── annotations/
│       │   ├── trainval_hico.json
│       │   └── test_hico.json
│       └── images/
│           ├── train2015/
│           └── test2015/
├── checkpoints/
│   ├── detr-r50-hicodet.pth          # Pretrained DETR
│   └── pretrained_CLIP/
│       ├── ViT-B-16.pt
│       └── ViT-L-14-336px.pt
└── hicodet_pkl_files/
    └── union_embeddings_cachemodel_crop_padding_zeros_vitb16.p
```

### 4. Environment Variables

**CRITICAL:** Always set PYTHONPATH before running any scripts:

```bash
export PYTHONPATH=$PYTHONPATH:"$PWD/CLIP"
```

Or add to your `~/.bashrc`:

```bash
echo 'export PYTHONPATH=$PYTHONPATH:"'$(pwd)'/CLIP"' >> ~/.bashrc
source ~/.bashrc
```

---

## Phase 0: Environment Setup

### Step 0.1: Clone Repository and Install Dependencies

```bash
# Clone repository
git clone https://github.com/ChelsieLei/EZ-HOI.git
cd EZ-HOI

# Install dependencies
pip install -r requirements.txt

# Install modified pocket library (see issue #2)
# Follow instructions at: https://github.com/ChelsieLei/EZ-HOI/issues/2
```

### Step 0.2: Download Datasets

```bash
# Download HICO-DET dataset
# Follow instructions at: http://www-personal.umich.edu/~ywchao/hico/

# Extract to:
# - hicodet/hico_20160224_det/images/train2015/
# - hicodet/hico_20160224_det/images/test2015/
```

### Step 0.3: Download Pretrained Checkpoints

```bash
# Download DETR checkpoint
# Follow instructions from EZ-HOI README

# Download CLIP checkpoints
# ViT-B/16 and ViT-L/14@336px from OpenAI CLIP repository
```

### Step 0.4: Verify Setup

```bash
# Check CLIP path
echo $PYTHONPATH  # Should include /path/to/EZ-HOI/CLIP

# Test imports
python -c "import clip; print('CLIP OK')"
python -c "import pocket; print('Pocket OK')"

# Check GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

---

## Phase 1: Extract Visual Features

### Purpose
Pre-extract and cache CLIP visual features from all HICO-DET images. This is a **ONE-TIME** operation that speeds up all subsequent training runs.

### Duration
~2-4 hours depending on GPU

### Command

```bash
cd EZ-HOI
export PYTHONPATH=$PYTHONPATH:"$PWD/CLIP"

python CLIP_hicodet_extract.py
```

### What Happens

The script will:
1. Load CLIP model (ViT-B/16 and ViT-L/14@336px)
2. Process all images in `train2015/` and `test2015/`
3. Extract CLIP features for each image
4. Save features as `.pkl` files

### Output Files

```
hicodet_pkl_files/
├── clipbase_img_hicodet_train/
│   ├── HICO_train2015_00000001_clip.pkl
│   ├── HICO_train2015_00000002_clip.pkl
│   └── ... (~38k files)
├── clipbase_img_hicodet_test/
│   └── ... (~10k files)
├── clip336_img_hicodet_train/
│   └── ... (ViT-L features)
└── clip336_img_hicodet_test/
    └── ... (ViT-L features)
```

### Progress Monitoring

```bash
# Watch progress
tail -f /path/to/log.txt  # If logging to file

# Check number of processed files
ls hicodet_pkl_files/clipbase_img_hicodet_train/*.pkl | wc -l
# Should reach ~38,581 for train set
```

### Verification

```bash
# Check a sample file
python -c "
import pickle
data = pickle.load(open('hicodet_pkl_files/clipbase_img_hicodet_train/HICO_train2015_00000001_clip.pkl', 'rb'))
print(f'Shape: {data.shape}')  # Should be [196, 512] for ViT-B
print(f'Type: {data.dtype}')   # Should be torch.float32
"
```

---

## Phase 2: Train EZ-HOI

### Purpose
Train the main HOI detection model with learnable text adapters. The diffusion bridge is **NOT** used during training.

### Duration
~1-2 days depending on GPU and configuration

### Important Notes
- Text embeddings will be **modified** by learnable modules:
  - `txtmem_adapter`: Transforms text embeddings
  - `act_descriptor_attn`: Refines embeddings with action descriptors
  - Text prompts: Learnable context vectors
- These modifications are learned through training
- We will extract these ADAPTED embeddings in Phase 3

### Command for Unseen Verb Setting (ViT-B)

```bash
cd EZ-HOI
export PYTHONPATH=$PYTHONPATH:"$PWD/CLIP"

bash scripts/hico_train_vitB_zs.sh
```

### Script Configuration

The training script `scripts/hico_train_vitB_zs.sh` contains:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_tip_finetune.py --world-size 4 \
 --pretrained "checkpoints/detr-r50-hicodet.pth" \
 --output-dir checkpoints/hico_HO_pt_default_vitbase/ \
 --epochs 12 \
 --use_insadapter \
 --num_classes 117 \
 --use_multi_hot \
 --file1 hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p \
 --clip_dir_vit checkpoints/pretrained_CLIP/ViT-B-16.pt \
 --batch-size 8 \
 --logits_type "HO" \
 --port 1236 \
 --txtcls_pt \      # ← Learnable text prompts
 --img_align \      # ← Image adapters
 --unseen_pt_inj \  # ← Unseen prompt injection
 --img_clip_pt \    # ← Use pre-extracted image features
 --zs \
 --zs_type "unseen_verb" \
 --clip_img_file hicodet_pkl_files/clipbase_img_hicodet_train
```

### Key Arguments Explained

| Argument | Purpose |
|----------|---------|
| `--txtcls_pt` | Enable learnable text prompts (CRITICAL for adaptation) |
| `--img_align` | Enable image adapter modules |
| `--txt_align` | Enable TEXT adapter (txtmem_adapter) - **modifies text embeddings** |
| `--unseen_pt_inj` | Enable unseen prompt injection with LLaVA descriptions |
| `--img_clip_pt` | Use pre-extracted CLIP image features |
| `--zs` | Enable zero-shot setting |
| `--zs_type unseen_verb` | Use unseen verb split (117 verb classes) |
| `--num_classes 117` | Predict verb-level interactions |

**NOTE:** If `--txt_align` is not in your script, **add it** to enable text adaptation!

### Multi-GPU Training

For 4 GPUs:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_tip_finetune.py --world-size 4 ...
```

For 1 GPU:
```bash
CUDA_VISIBLE_DEVICES=0 python main_tip_finetune.py --world-size 1 ...
```

### Monitoring Training

```bash
# Watch training log
tail -f checkpoints/hico_HO_pt_default_vitbase/log.txt

# Check GPU usage
nvidia-smi -l 1

# Monitor checkpoints
ls -lh checkpoints/hico_HO_pt_default_vitbase/*.pth
```

### Output Files

```
checkpoints/hico_HO_pt_default_vitbase/
├── checkpoint0000.pth    # Epoch 0
├── checkpoint0001.pth    # Epoch 1
├── ...
├── checkpoint0011.pth    # Epoch 11
├── best.pth              # Best model (highest mAP)
└── log.txt               # Training log
```

### When Training is Complete

```bash
# Check best checkpoint
ls -lh checkpoints/hico_HO_pt_default_vitbase/best.pth

# View training log summary
tail -n 50 checkpoints/hico_HO_pt_default_vitbase/log.txt
```

**IMPORTANT:** Before proceeding to Phase 3, ensure training has converged and you have a good checkpoint (e.g., `best.pth`).

---

## Phase 3: Extract Adapted Text Embeddings

### Purpose
Extract text embeddings **AFTER** they have been modified by the learned adapters from Phase 2. These adapted embeddings will be used to train the diffusion model.

### Duration
~5-10 minutes

### Why This Step is Critical

During EZ-HOI training, text embeddings are modified by:
1. **txtmem_adapter:** Learnable adapter that transforms text embeddings
2. **act_descriptor_attn:** Attention mechanism that refines embeddings
3. **Text prompts:** Learnable context vectors

The diffusion model **MUST** be trained on these ADAPTED embeddings, not the raw CLIP embeddings, to ensure vision-text alignment consistency at inference time.

### Command

```bash
cd EZ-HOI
export PYTHONPATH=$PYTHONPATH:"$PWD/CLIP"

python extract_adapted_text_embeddings.py \
  --checkpoint checkpoints/hico_HO_pt_default_vitbase/best.pth \
  --num_classes 117 \
  --zs_type unseen_verb \
  --output_dir hicodet_pkl_files \
  --scale_factor 5.0 \
  --device cuda
```

### Arguments Explained

| Argument | Description | Required |
|----------|-------------|----------|
| `--checkpoint` | Path to trained EZ-HOI checkpoint | Yes |
| `--num_classes` | Number of HOI classes (117 or 600) | Optional* |
| `--zs_type` | Zero-shot type (unseen_verb, etc.) | Optional* |
| `--output_dir` | Where to save embeddings | No (default: hicodet_pkl_files) |
| `--scale_factor` | Diffusion normalization scale | No (default: 5.0) |
| `--device` | cuda or cpu | No (default: cuda) |

*If not provided, will be read from checkpoint

### What Happens

The script will:
1. Load the trained EZ-HOI checkpoint
2. Recreate the exact model structure (including all adapters)
3. Load trained weights
4. Forward pass text through learned adapters:
   ```
   hoicls_txt → txtmem_adapter → act_descriptor_attn → adapted_embeddings
   ```
5. Apply diffusion-bridge normalization
6. Save embeddings for diffusion training

### Output Files

```
hicodet_pkl_files/
├── hoi_text_embeddings_adapted_unseen_verb_vitB_117_raw.pkl
│   └── Raw adapted embeddings (after adapters, before normalization)
├── hoi_text_embeddings_adapted_unseen_verb_vitB_117_normalized.pkl
│   └── Normalized embeddings (ready for diffusion training)
├── hoi_text_mean_adapted_unseen_verb_vitB_117.pkl
│   └── Text modality mean (for diffusion bridge inference)
└── hoi_text_embeddings_adapted_unseen_verb_vitB_117_summary.txt
    └── Summary statistics
```

### Verification

```bash
# Check the summary file
cat hicodet_pkl_files/hoi_text_embeddings_adapted_unseen_verb_vitB_117_summary.txt

# Expected output:
# Adapted HOI Text Embeddings Summary
# ============================================================
#
# Checkpoint: checkpoints/hico_HO_pt_default_vitbase/best.pth
# Number of classes: 117
# Selected classes: 100 (or similar)
# Embedding dimension: 512
# Scale factor: 5.0
#
# Adaptation modules applied:
#   - Text align (txtmem_adapter): True
#   - Text class prompts: True
#   - Action descriptor: True
#   - VLM text: llava (or None)
#
# Text mean norm: 0.xxxx
# Normalized embedding norms (mean ± std): 5.0000 ± 0.xxxx
```

### Troubleshooting

**Issue:** "Cannot extract args from checkpoint"
- **Solution:** Your checkpoint might not have saved training arguments. You need to provide them manually:
  ```bash
  python extract_adapted_text_embeddings.py \
    --checkpoint path/to/checkpoint.pth \
    --num_classes 117 \
    --zs_type unseen_verb
  ```

**Issue:** "KeyError: 'txtmem_adapter.weight'"
- **Solution:** Your training didn't enable `--txt_align`. The checkpoint doesn't have text adapter weights. You may need to retrain or use raw embeddings (less optimal).

**Issue:** CUDA out of memory
- **Solution:** Use CPU instead:
  ```bash
  python extract_adapted_text_embeddings.py ... --device cpu
  ```

---

## Phase 4: Train Diffusion Model

### Purpose
Train a 1D UNet diffusion model on the ADAPTED text embeddings from Phase 3. The diffusion model learns the distribution of the adapted text space.

### Duration
~6-12 hours depending on GPU

### Command

```bash
cd EZ-HOI
export PYTHONPATH=$PYTHONPATH:"$PWD/CLIP"

python train_hoi_diffusion.py \
  --data_path hicodet_pkl_files/hoi_text_embeddings_adapted_unseen_verb_vitB_117_normalized.pkl \
  --train_steps 300000 \
  --batch_size 64 \
  --learning_rate 8e-5 \
  --results_folder hoi_diffusion_results_adapted \
  --timesteps 1000 \
  --objective pred_x0
```

### Arguments Explained

| Argument | Description | Recommended Value |
|----------|-------------|-------------------|
| `--data_path` | Path to ADAPTED normalized embeddings | Output from Phase 3 |
| `--train_steps` | Total training steps | 300000 (300k) |
| `--batch_size` | Training batch size | 64 |
| `--learning_rate` | Learning rate | 8e-5 |
| `--results_folder` | Where to save checkpoints | hoi_diffusion_results_adapted |
| `--timesteps` | Number of diffusion steps | 1000 |
| `--objective` | Diffusion objective | pred_x0 (recommended) |

### What Happens

The script will:
1. Load adapted text embeddings [num_classes, 512]
2. Create 1D UNet model
3. Train diffusion model to denoise embeddings
4. Save checkpoints every 10k steps
5. Final model at step 300k

### Monitoring Training

```bash
# Watch training progress
tail -f hoi_diffusion_results_adapted/log.txt  # If logging enabled

# Check checkpoints
ls -lh hoi_diffusion_results_adapted/*.pt
# Output:
# model-10.pt   (10k steps)
# model-20.pt   (20k steps)
# ...
# model-300.pt  (300k steps - final)

# Monitor GPU usage
nvidia-smi -l 1
```

### Training Progress Expected

```
Step      Loss      Time
--------------------------------
0         ~1.5      -
10k       ~0.8      ~10 min
50k       ~0.5      ~50 min
100k      ~0.3      ~1.5 hours
200k      ~0.2      ~3 hours
300k      ~0.15     ~6 hours
```

(Times are approximate for a single A100 GPU)

### Output Files

```
hoi_diffusion_results_adapted/
├── model-10.pt          # Checkpoint at 10k steps
├── model-20.pt
├── ...
├── model-300.pt         # Final model (USE THIS for inference)
└── training_config.txt  # Training configuration
```

### Early Stopping (Optional)

If you want to stop early for testing:
```bash
# Stop training with Ctrl+C

# Use the latest checkpoint
ls -lt hoi_diffusion_results_adapted/*.pt | head -1
```

You can use intermediate checkpoints like `model-100.pt` for testing, but `model-300.pt` will have the best performance.

### Verification

```bash
# Check model file
python -c "
import torch
model = torch.load('hoi_diffusion_results_adapted/model-300.pt')
print(f'Model keys: {model.keys()}')
print(f'Model loaded successfully!')
"
```

---

## Phase 5: Test with Diffusion Bridge

### Purpose
Run inference on HICO-DET test set using the trained EZ-HOI model with diffusion bridge enabled. The diffusion bridge aligns vision features to the adapted text space.

### Duration
~1-2 hours depending on test set size

### Command

```bash
cd EZ-HOI
export PYTHONPATH=$PYTHONPATH:"$PWD/CLIP"

bash scripts/hico_test_vitB_zs_diffusion.sh
```

### Script Configuration

The script `scripts/hico_test_vitB_zs_diffusion.sh` should contain:

```bash
CUDA_VISIBLE_DEVICES=0 python main_tip_finetune.py --world-size 1 \
 --pretrained "checkpoints/detr-r50-hicodet.pth" \
 --output-dir checkpoints/hico_HO_pt_default_vitbase/ \
 --epochs 12 \
 --use_insadapter \
 --num_classes 117 \
 --use_multi_hot \
 --file1 hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p \
 --clip_dir_vit checkpoints/pretrained_CLIP/ViT-B-16.pt \
 --batch-size 8 \
 --logits_type "HO" \
 --port 1236 \
 --txtcls_pt --img_align --unseen_pt_inj --img_clip_pt \
 --zs --zs_type "unseen_verb" \
 --clip_img_file hicodet_pkl_files/clipbase_img_hicodet_test \
 --eval \
 --resume checkpoints/hico_HO_pt_default_vitbase/best.pth \
 --use_diffusion_bridge \
 --diffusion_model_path hoi_diffusion_results_adapted/model-300.pt \
 --diffusion_text_mean hicodet_pkl_files/hoi_text_mean_adapted_unseen_verb_vitB_117.pkl \
 --diffusion_inference_steps 600
```

### Key Diffusion Arguments

| Argument | Description | Value |
|----------|-------------|-------|
| `--use_diffusion_bridge` | Enable diffusion bridge | (flag, no value) |
| `--diffusion_model_path` | Path to trained diffusion model | `hoi_diffusion_results_adapted/model-300.pt` |
| `--diffusion_text_mean` | Path to text mean | `hicodet_pkl_files/hoi_text_mean_adapted_*.pkl` |
| `--diffusion_inference_steps` | DDIM sampling steps | 600 (can use 300 for faster) |

**CRITICAL:** Ensure paths match the output from Phase 3 and Phase 4!

### Inference Flow

```
Test Image
    ↓
DETR (detect human & object boxes)
    ↓
Extract union features
    ↓
Vision encoder (with adapters)
    ↓
Vision features [batch, 512]
    ↓
Diffusion Bridge (DDIM sampling, 600 steps)
    ↓ (align vision features to adapted text space)
Aligned features [batch, 512]
    ↓
Compare with adapted text embeddings [117, 512]
    ↓
HOI predictions
```

### Monitoring Inference

```bash
# Watch progress
tail -f checkpoints/hico_HO_pt_default_vitbase/test_log.txt

# Check GPU usage
nvidia-smi -l 1
```

### Expected Output

```
Loading checkpoint from: checkpoints/hico_HO_pt_default_vitbase/best.pth
✓ Loaded model weights

============================================================
Initializing Diffusion Bridge for HOI Detection
============================================================
Diffusion model: hoi_diffusion_results_adapted/model-300.pt
Text mean: hicodet_pkl_files/hoi_text_mean_adapted_unseen_verb_vitB_117.pkl
Inference steps: 600
✓ Diffusion bridge initialized and ready
  Mode: Inference-only (frozen weights)
  Integration point: After image adapter, before classification
============================================================

Running inference on HICO-DET test set...
[1/9658] Processing HICO_test2015_00000001.jpg...
[2/9658] Processing HICO_test2015_00000002.jpg...
...

Computing mAP...
============================================================
Results (Unseen Verb Setting):
============================================================
Unseen mAP: XX.XX%
Seen mAP: XX.XX%
Overall mAP: XX.XX%
============================================================
```

### Results Comparison

Compare with baseline (without diffusion):

```bash
# Test WITHOUT diffusion
bash scripts/hico_test_vitB_zs.sh

# Compare mAP scores
# With diffusion should show improvement, especially on unseen classes
```

### Adjusting Inference Speed

If inference is too slow, reduce DDIM steps:

```bash
# In scripts/hico_test_vitB_zs_diffusion.sh, change:
--diffusion_inference_steps 300  # Faster, slight quality drop
# or
--diffusion_inference_steps 100  # Very fast, more quality drop
```

---

## Advanced: Training with Different Configurations

### Full 600 HOI Classes (Instead of 117 Verbs)

```bash
# Phase 2: Train with 600 classes
# Edit scripts/hico_train_vitB_default.sh:
# --num_classes 600
bash scripts/hico_train_vitB_default.sh

# Phase 3: Extract embeddings
python extract_adapted_text_embeddings.py \
  --checkpoint checkpoints/.../best.pth \
  --num_classes 600 \
  --output_dir hicodet_pkl_files

# Phase 4: Train diffusion
python train_hoi_diffusion.py \
  --data_path hicodet_pkl_files/hoi_text_embeddings_adapted_vitB_600_normalized.pkl \
  --results_folder hoi_diffusion_results_adapted_600

# Phase 5: Test
# Edit test script to use 600-class paths
```

### ViT-L Backbone (Larger Model)

```bash
# Phase 1: Extract ViT-L features (already done by CLIP_hicodet_extract.py)

# Phase 2: Train with ViT-L
bash scripts/hico_train_vitL_zs.sh

# Phase 3: Extract embeddings
python extract_adapted_text_embeddings.py \
  --checkpoint checkpoints/.../best.pth \
  --num_classes 117

# Note: Embedding dimension will be 768 for ViT-L instead of 512

# Phase 4: Train diffusion
python train_hoi_diffusion.py \
  --data_path hicodet_pkl_files/hoi_text_embeddings_adapted_unseen_verb_vitL_117_normalized.pkl \
  --results_folder hoi_diffusion_results_adapted_vitL
```

### Different Zero-Shot Splits

Available splits:
- `unseen_verb` (default)
- `unseen_object`
- `rare_first`
- `non_rare_first`

```bash
# Phase 2: Train with different split
# Edit script: --zs_type unseen_object

# Phase 3: Extract embeddings
python extract_adapted_text_embeddings.py \
  --checkpoint ... \
  --zs_type unseen_object

# Phase 4-5: Use corresponding paths
```

---

## FAQ

### Q1: Do I need to run Phase 1 every time?

**No.** Phase 1 (visual feature extraction) is a **ONE-TIME** operation. Once you have the `.pkl` files in `hicodet_pkl_files/`, you can reuse them for all experiments.

### Q2: Can I skip the diffusion pipeline and just use EZ-HOI?

**Yes.** EZ-HOI works fine without diffusion. Simply use the training and testing scripts without diffusion flags. The diffusion bridge is an optional enhancement for better vision-text alignment.

### Q3: Why must I extract embeddings AFTER training EZ-HOI?

Because EZ-HOI has learnable modules (`txtmem_adapter`, `act_descriptor_attn`) that **modify text embeddings during training**. If you train diffusion on raw embeddings, there will be a mismatch at inference time. See `technical_explanation.md` for details.

### Q4: Can I use a different diffusion model architecture?

The current implementation uses 1D UNet from the diffusion-bridge paper. You can experiment with other architectures, but ensure they:
1. Support 1D sequence inputs (embeddings)
2. Implement DDIM sampling for fast inference
3. Are compatible with the normalization strategy

### Q5: How long does the entire pipeline take?

Approximate times on a single A100 GPU:
- Phase 1: 2-4 hours (ONE-TIME)
- Phase 2: 1-2 days
- Phase 3: 5-10 minutes
- Phase 4: 6-12 hours
- Phase 5: 1-2 hours

**Total:** ~2-3 days for first run (Phase 1 only needed once)

### Q6: What if I don't have 4 GPUs?

Adjust `--world-size` and `CUDA_VISIBLE_DEVICES`:
```bash
# 1 GPU
CUDA_VISIBLE_DEVICES=0 python main_tip_finetune.py --world-size 1 ...

# 2 GPUs
CUDA_VISIBLE_DEVICES=0,1 python main_tip_finetune.py --world-size 2 ...
```

You may need to reduce batch size to fit in memory.

### Q7: Can I resume training if it's interrupted?

**EZ-HOI training:** Add `--resume path/to/checkpoint.pth`

**Diffusion training:** Current implementation doesn't support resume. You'd need to modify `train_hoi_diffusion.py` to add checkpoint loading.

### Q8: How do I know if diffusion is helping?

Compare mAP scores:
```bash
# Without diffusion
bash scripts/hico_test_vitB_zs.sh

# With diffusion
bash scripts/hico_test_vitB_zs_diffusion.sh

# Check "Unseen mAP" - diffusion should improve this metric
```

### Q9: What if Phase 3 fails with "txtmem_adapter not found"?

Your training didn't enable `--txt_align`. Either:
1. **Option A (Recommended):** Retrain EZ-HOI with `--txt_align` flag
2. **Option B (Suboptimal):** Skip Phase 3-4 and use EZ-HOI without diffusion

### Q10: Can I use this pipeline for V-COCO dataset?

Yes, but you need to:
1. Change `--dataset vcoco` in all commands
2. Adjust paths for V-COCO data
3. Use V-COCO-specific scripts (e.g., `vcoco_train.sh`)

---

## Next Steps

After completing this pipeline:

1. **Experiment with different configurations** (ViT-L, 600 classes, etc.)
2. **Read `technical_explanation.md`** to understand why this order is important
3. **Check `troubleshooting.md`** if you encounter issues
4. **Try inference on your own images** using `scripts/inference.sh`

---

## References

- [EZ-HOI Paper](https://arxiv.org/abs/2409.06083) (NeurIPS 2024)
- [Diffusion-Bridge Paper](https://arxiv.org/abs/2305.16954) (ICCV 2023)
- [CLIP Paper](https://arxiv.org/abs/2103.00020) (ICML 2021)

---

*Generated by Claude Code as part of the EZ-HOI + Diffusion Bridge integration project.*
