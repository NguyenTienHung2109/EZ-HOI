# Diffusion Training for EZ-HOI

This guide explains how to train a diffusion model on **visual embeddings** to improve vision-text alignment in EZ-HOI.

## Table of Contents

- [Background](#background)
- [Why Visual Embeddings Instead of Text?](#why-visual-embeddings-instead-of-text)
- [Pipeline Overview](#pipeline-overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Step-by-Step Guide](#step-by-step-guide)
- [Integration with EZ-HOI](#integration-with-ez-hoi)
- [Troubleshooting](#troubleshooting)

## Background

The **diffusion-bridge** approach normalizes embeddings to improve cross-modal alignment between vision and text. The key idea:

1. **Normalization chain**: `normalize(normalize(x) - mean) * scale_factor`
2. **Diffusion refinement**: Use a trained diffusion model to transform visual features toward text distribution
3. **Inference**: Apply the diffusion bridge to visual features before computing cosine similarity with text

## Why Visual Embeddings Instead of Text?

| Aspect | Text Embeddings | Visual Embeddings (RECOMMENDED) |
|--------|----------------|--------------------------------|
| **Data size** | 212 classes (limited) | 50k-100k pairs (abundant) |
| **Diffusion quality** | May overfit | Stable training |
| **What it learns** | Text distribution | Vision→text bridging |
| **Training time** | ~6-12 hours | ~8-15 hours |
| **Expected improvement** | Limited (insufficient data) | +1-3% mAP on unseen classes |

**Recommendation**: Train diffusion on visual embeddings extracted from HICO-DET training set.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Extract Adapted Text Embeddings                        │
│  - Run inference through learnable prompts                      │
│  - Compute text mean for normalization                          │
│  - Output: 212 adapted text embeddings + text mean             │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: Extract Visual Embeddings (1-2 hours)                  │
│  - Iterate through HICO-DET training images                     │
│  - Extract adapter_feat (visual features after mem_adapter)     │
│  - Apply normalization chain (subtract text mean, scale)        │
│  - Output: 50k-100k normalized visual embeddings               │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: Train Diffusion Model (8-15 hours)                     │
│  - Train 1D UNet on normalized visual embeddings                │
│  - 500k training steps with batch size 64                       │
│  - Checkpoints saved every 10k steps                            │
│  - Output: Trained diffusion model                             │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 4: Test Diffusion Bridge                                  │
│  - Load trained diffusion model                                 │
│  - Apply to test images                                         │
│  - Compare predictions with/without diffusion                   │
│  - Output: Visualizations + performance metrics                │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. **Trained EZ-HOI checkpoint**
   ```bash
   checkpoints/hico_HO_pt_default_vitbase/best.pth
   ```

2. **HICO-DET dataset**
   ```
   hicodet/
   ├── hico_20160224_det/
   │   ├── annotations/
   │   └── images/
   │       ├── train2015/  # Required for visual extraction
   │       └── test2015/   # Required for testing
   ```

3. **CLIP model checkpoint**
   ```bash
   checkpoints/pretrained_CLIP/ViT-B-16.pt
   # OR
   checkpoints/pretrained_CLIP/ViT-L-14-336px.pt
   ```

4. **diffusion-bridge submodule**
   ```bash
   cd diffusion-bridge/ddpm
   pip install -e .
   ```

## Quick Start

**Run the full pipeline with one command:**

```bash
bash train_diffusion_visual_pipeline.sh
```

This script runs all 4 steps automatically. **Total time: ~10-17 hours**

**Or run steps individually** (see below).

## Step-by-Step Guide

### Step 1: Extract Adapted Text Embeddings

Extract text embeddings through learnable prompts and compute text mean:

```bash
python extract_adapted_text_embeddings.py \
  --checkpoint checkpoints/hico_HO_pt_default_vitbase/best.pth \
  --num_classes 117 \
  --zs_type unseen_verb \
  --clip_model_path checkpoints/pretrained_CLIP/ViT-B-16.pt \
  --output_dir hicodet_pkl_files \
  --scale_factor 5.0 \
  --device cuda
```

**Output:**
- `hicodet_pkl_files/hoi_adapted_text_embeddings_vitB_117_212classes.pkl`
- `hicodet_pkl_files/hoi_text_mean_adapted_unseen_verb_vitB_117_212classes.pkl` ✓ **This is the adapted text mean**

**Time:** ~5 minutes

### Step 2: Extract Visual Embeddings

Extract visual features from all HICO-DET training images:

```bash
python extract_visual_embeddings_for_diffusion.py \
  --checkpoint checkpoints/hico_HO_pt_default_vitbase/best.pth \
  --data_root hicodet \
  --num_classes 117 \
  --zs_type unseen_verb \
  --text_mean_path hicodet_pkl_files/hoi_text_mean_adapted_unseen_verb_vitB_117_212classes.pkl \
  --scale_factor 5.0 \
  --output_path hicodet_pkl_files/hoi_visual_embeddings_normalized_vitB_train.pkl \
  --batch_size 1 \
  --num_workers 4 \
  --device cuda \
  --clip_model_path checkpoints/pretrained_CLIP/ViT-B-16.pt
```

**Output:**
- `hicodet_pkl_files/hoi_visual_embeddings_normalized_vitB_train.pkl`
  - Contains: 50k-100k normalized visual embeddings [N, 512]
  - Metadata: image IDs, object classes, etc.

**Time:** 1-2 hours (38,118 images)

**Progress monitoring:**
```
Processing images: 5234/38118 [13%]  ETA: 01:23:45
```

### Step 3: Train Diffusion Model

Train diffusion on extracted visual embeddings:

```bash
python train_hoi_diffusion.py \
  --data_path hicodet_pkl_files/hoi_visual_embeddings_normalized_vitB_train.pkl \
  --init_dim 32 \
  --dim_mults 1 2 4 8 \
  --timesteps 1000 \
  --objective pred_x0 \
  --batch_size 64 \
  --learning_rate 8e-5 \
  --train_steps 500000 \
  --gradient_accumulate 1 \
  --ema_decay 0.995 \
  --use_amp \
  --results_folder hoi_diffusion_results
```

**Output:**
- `hoi_diffusion_results/model-10.pt` (checkpoint at 10k steps)
- `hoi_diffusion_results/model-20.pt`
- ...
- `hoi_diffusion_results/model-500.pt` (final model)
- `hoi_diffusion_results/training_config.txt`

**Time:** 8-15 hours (500k steps)

**Checkpoints saved every 10k steps**

**Early stopping:** You can stop training early (e.g., at 300k steps) if validation loss plateaus.

### Step 4: Test Diffusion Bridge

Test the trained diffusion on sample images:

```bash
python test_visual_diffusion.py \
  --diffusion_model hoi_diffusion_results/model-500.pt \
  --checkpoint checkpoints/hico_HO_pt_default_vitbase/best.pth \
  --text_mean_path hicodet_pkl_files/hoi_text_mean_adapted_unseen_verb_vitB_117_212classes.pkl \
  --adapted_text_pkl hicodet_pkl_files/hoi_adapted_text_embeddings_vitB_117_212classes.pkl \
  --data_root hicodet \
  --num_test_images 20 \
  --inference_steps 600 \
  --output_dir test_diffusion_results \
  --device cuda
```

**Output:**
- `test_diffusion_results/diffusion_effect.png`
  - Similarity distributions (before/after diffusion)
  - Feature norm changes
  - Top-1 prediction changes
  - Scatter plot showing improvement per pair

**Interpreting results:**

✓ **Good diffusion model:**
- Similarity to text increases (red histogram shifts right)
- Prediction changes are meaningful (better alignment)
- Few catastrophic failures

✗ **Bad diffusion model:**
- Similarity decreases or stays the same
- Random prediction changes
- Feature norms explode or collapse

**Time:** ~5 minutes

## Integration with EZ-HOI

Once you have a trained diffusion model, integrate it into EZ-HOI inference:

### Option 1: Use DiffusionBridgeHOI module

```python
from diffusion_bridge_module import DiffusionBridgeHOI

# In model initialization
diffusion_bridge = DiffusionBridgeHOI(
    diffusion_path='hoi_diffusion_results/model-500.pt',
    text_mean_path='hicodet_pkl_files/hoi_text_mean_adapted_unseen_verb_vitB_117_212classes.pkl',
    inference_steps=600,  # Trade-off: lower=faster, higher=better quality
    scale_factor=5.0,
    verbose=False
)

# In forward pass (after mem_adapter, before cosine similarity)
if self.diffusion_bridge is not None and not self.training:
    adapter_feat = self.diffusion_bridge(adapter_feat)
```

This is already implemented in `upt_tip_cache_model_free_finetune_distillself.py` at lines 1273-1275.

### Option 2: Modify inference script

Add diffusion bridge argument to `inference.py`:

```bash
python inference.py \
  --resume checkpoints/hico_HO_pt_default_vitbase/best.pth \
  --diffusion_bridge hoi_diffusion_results/model-500.pt \
  --diffusion_text_mean hicodet_pkl_files/hoi_text_mean_adapted_unseen_verb_vitB_117_212classes.pkl \
  --diffusion_steps 600 \
  --self_image_path path/to/test/image.jpg
```

### Performance Tuning

**Inference steps trade-off:**
- `inference_steps=100`: Fastest (~0.1s per image), slight quality loss
- `inference_steps=600`: Recommended balance (~0.3s per image)
- `inference_steps=1000`: Best quality (~0.5s per image), diminishing returns

**Batch inference:**
If processing multiple images, batch the diffusion forward pass for efficiency.

## Troubleshooting

### Issue: OOM (Out of Memory) during visual extraction

**Solution:**
```bash
# Reduce batch size (already at 1, so reduce num_workers)
python extract_visual_embeddings_for_diffusion.py \
  --batch_size 1 \
  --num_workers 0  # Disable multiprocessing
```

### Issue: Diffusion training loss not decreasing

**Check:**
1. Data distribution: Are visual embeddings normalized correctly?
   ```python
   import pickle
   with open('hicodet_pkl_files/hoi_visual_embeddings_normalized_vitB_train.pkl', 'rb') as f:
       data = pickle.load(f)
   print(data['embeddings'].mean(), data['embeddings'].std())
   # Should see reasonable values (not all zeros or NaNs)
   ```

2. Learning rate too high/low:
   - Try `--learning_rate 5e-5` (lower) or `1e-4` (higher)

3. Batch size too small:
   - Try `--batch_size 128` if you have enough VRAM

### Issue: Test results show no improvement

**Possible causes:**

1. **Diffusion model undertrained:** Train longer (e.g., 700k steps)
2. **Inference steps too low:** Increase `--inference_steps` to 1000
3. **Visual embeddings don't match inference:** Check that text mean is from **adapted** embeddings, not raw CLIP
4. **Model architecture mismatch:** Ensure UNet config matches training

### Issue: "Diffusion model not found"

**Solution:**
```bash
# Check if diffusion-bridge is installed
cd diffusion-bridge/ddpm
pip install -e .

# Verify installation
python -c "from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d import Unet1D"
```

### Issue: Adapted text mean has wrong norm

Expected norm for adapted text mean: ~0.01-0.05 (much smaller than raw CLIP mean ~1.0)

**Fix:**
Re-run `extract_adapted_text_embeddings.py` and verify the mean is computed from **normalized** embeddings:

```python
# In apply_diffusion_normalization function:
for emb in embeddings:
    normalized_emb = F.normalize(emb.unsqueeze(0), dim=-1)  # ← Must normalize first
    text_mean += normalized_emb
```

## Advanced Configuration

### Training on a subset of data (for testing)

To quickly test the pipeline on a smaller dataset:

```bash
# Modify extract_visual_embeddings_for_diffusion.py:
# Add --max_images argument to limit extraction to first N images

python extract_visual_embeddings_for_diffusion.py \
  --max_images 1000 \  # Only extract from 1000 images
  ... other args ...

# Then train with reduced steps:
python train_hoi_diffusion.py \
  --train_steps 50000 \  # Reduce to 50k for quick testing
  ... other args ...
```

### Multi-GPU training

The diffusion trainer supports distributed training:

```bash
# Modify train_hoi_diffusion.py to use DataParallel or DistributedDataParallel
# (Currently single-GPU only, requires modification)
```

### Hyperparameter tuning

Key hyperparameters to tune:

1. **Scale factor** (default: 5.0)
   - Higher = stronger normalization (may lose information)
   - Lower = weaker normalization (may not align well)
   - Try: 3.0, 5.0, 7.0

2. **Inference steps** (default: 600)
   - More steps = better quality but slower
   - Try: 100, 300, 600, 1000

3. **UNet architecture** (default: dim_mults=[1,2,4,8])
   - Deeper = more capacity but slower
   - Try: [1,2,4] (shallower), [1,2,4,8,16] (deeper)

## File Structure

After running the full pipeline:

```
EZ-HOI/
├── extract_adapted_text_embeddings.py       # Step 1
├── extract_visual_embeddings_for_diffusion.py  # Step 2
├── train_hoi_diffusion.py                   # Step 3
├── test_visual_diffusion.py                 # Step 4
├── diffusion_bridge_module.py               # Inference module
├── train_diffusion_visual_pipeline.sh       # Full pipeline script
├── DIFFUSION_TRAINING.md                    # This file
├── hicodet_pkl_files/
│   ├── hoi_adapted_text_embeddings_vitB_117_212classes.pkl
│   ├── hoi_text_mean_adapted_unseen_verb_vitB_117_212classes.pkl  ✓ Use this!
│   └── hoi_visual_embeddings_normalized_vitB_train.pkl
├── hoi_diffusion_results/
│   ├── model-10.pt
│   ├── model-20.pt
│   ├── ...
│   ├── model-500.pt  ✓ Final model
│   └── training_config.txt
└── test_diffusion_results/
    └── diffusion_effect.png  ✓ Review this visualization
```

## References

- **Diffusion-Bridge paper:** [Add citation]
- **EZ-HOI paper:** NeurIPS 2024
- **DDPM tutorial:** https://github.com/lucidrains/denoising-diffusion-pytorch

## FAQ

**Q: Can I use text embeddings instead of visual embeddings?**

A: Yes, but not recommended. With only 212 text embeddings, diffusion training will likely overfit. Visual embeddings (50k+ samples) provide much better training signal.

**Q: How much VRAM do I need?**

A:
- Visual extraction: 8GB minimum (batch_size=1)
- Diffusion training: 12GB recommended (batch_size=64), 8GB possible (batch_size=32)
- Inference: 6GB

**Q: Can I use a pre-trained diffusion model from another dataset?**

A: Not recommended. The diffusion model must be trained on HOI-specific embeddings with the same normalization strategy. Cross-dataset transfer will likely fail.

**Q: Should I train separate diffusion models for different zero-shot splits?**

A: No. Train one diffusion model on the full training set (all seen classes). It will work for all zero-shot splits since it learns the general vision→text alignment, not class-specific transformations.

**Q: How do I know if diffusion is helping?**

A: Run full evaluation on HICO-DET test set with and without diffusion:
- Baseline (no diffusion): `python main_tip_finetune.py --eval ...`
- With diffusion: Same command but model loaded with diffusion_bridge enabled

Compare mAP, especially on unseen classes. Expect +0.5-2% improvement.

---

**Happy training!** If you encounter issues, check the [troubleshooting section](#troubleshooting) or open an issue on GitHub.
