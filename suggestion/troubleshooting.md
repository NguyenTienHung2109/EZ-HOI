# Troubleshooting Guide for EZ-HOI + Diffusion Pipeline

This guide covers common issues, error messages, and solutions for the complete pipeline.

---

## Table of Contents

1. [Environment and Setup Issues](#environment-and-setup-issues)
2. [Phase 1: Visual Feature Extraction](#phase-1-visual-feature-extraction)
3. [Phase 2: EZ-HOI Training](#phase-2-ez-hoi-training)
4. [Phase 3: Adapted Embedding Extraction](#phase-3-adapted-embedding-extraction)
5. [Phase 4: Diffusion Training](#phase-4-diffusion-training)
6. [Phase 5: Inference](#phase-5-inference)
7. [Performance Issues](#performance-issues)
8. [General Debugging Tips](#general-debugging-tips)

---

## Environment and Setup Issues

### Issue: `ModuleNotFoundError: No module named 'clip'`

**Cause:** CLIP is not in Python path

**Solution 1 - Set PYTHONPATH (Recommended):**
```bash
export PYTHONPATH=$PYTHONPATH:"$PWD/CLIP"
# Verify
echo $PYTHONPATH
python -c "import clip; print('CLIP OK')"
```

**Solution 2 - Add to ~/.bashrc (Permanent):**
```bash
cd /path/to/EZ-HOI
echo 'export PYTHONPATH=$PYTHONPATH:"'$(pwd)'/CLIP"' >> ~/.bashrc
source ~/.bashrc
```

**Solution 3 - Install CLIP package (Not Recommended):**
```bash
# This installs official CLIP, but you need the MODIFIED version from EZ-HOI/CLIP
pip install git+https://github.com/openai/CLIP.git
# ⚠️ May not work due to custom modifications in EZ-HOI/CLIP
```

### Issue: `ImportError: No module named 'pocket'`

**Cause:** Pocket library not installed or outdated

**Solution:**
```bash
# Clone modified pocket library
git clone https://github.com/fredzzhang/pocket.git
cd pocket

# Apply modifications from EZ-HOI issue #2
# See: https://github.com/ChelsieLei/EZ-HOI/issues/2

# Install
pip install -e .

# Verify
python -c "import pocket; print('Pocket OK')"
```

### Issue: `RuntimeError: CUDA out of memory`

**Cause:** Batch size too large for your GPU

**Solution:**
```bash
# Reduce batch size in training script
# Edit scripts/hico_train_vitB_zs.sh:
--batch-size 4  # Reduce from 8

# Or use gradient accumulation
--gradient-accumulate-every 2  # Effective batch size = 4 * 2 = 8
```

### Issue: `FileNotFoundError: No such file or directory: 'checkpoints/detr-r50-hicodet.pth'`

**Cause:** Pretrained DETR checkpoint missing

**Solution:**
```bash
# Download from EZ-HOI repository or official DETR
wget <URL_to_detr_checkpoint> -O checkpoints/detr-r50-hicodet.pth

# Or adjust path in script
--pretrained "path/to/your/detr/checkpoint.pth"
```

---

## Phase 1: Visual Feature Extraction

### Issue: `KeyError: 'file_name'` in CLIP_hicodet_extract.py

**Cause:** Annotation JSON format doesn't match expected structure

**Solution:**
```bash
# Check annotation file
python -c "
import json
data = json.load(open('hicodet/trainval_hico.json'))
print('Keys:', data[0].keys())
print('Sample:', data[0])
"

# Expected format:
# [{'file_name': 'HICO_train2015_00000001.jpg', ...}, ...]
```

If format is different, modify `CLIP_hicodet_extract.py` lines 31-32:
```python
filename = info_hoii['file_name']  # Adjust key name if needed
```

### Issue: Extraction is very slow

**Cause:** Running on CPU instead of GPU

**Solution:**
```python
# Add GPU check in CLIP_hicodet_extract.py
import torch
print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# If showing CPU, check:
nvidia-smi  # Verify GPU is visible
echo $CUDA_VISIBLE_DEVICES  # Should show GPU IDs
```

### Issue: `.pkl` files are 0 bytes or corrupted

**Cause:** Disk full or write permissions issue

**Solution:**
```bash
# Check disk space
df -h .

# Check permissions
ls -la hicodet_pkl_files/

# Re-run extraction (will overwrite corrupt files)
python CLIP_hicodet_extract.py
```

---

## Phase 2: EZ-HOI Training

### Issue: `RuntimeError: Expected all tensors to be on the same device`

**Cause:** Model parts on different devices (CPU/GPU)

**Solution:**
```bash
# Ensure all data is moved to GPU
# Check in upt_tip_cache_model_free_finetune_distillself.py

# Quick fix: Use single GPU
CUDA_VISIBLE_DEVICES=0 python main_tip_finetune.py --world-size 1 ...
```

### Issue: Training loss is NaN

**Cause:** Learning rate too high or gradient explosion

**Solution:**
```bash
# Reduce learning rate
--learning-rate 1e-5  # Reduce from default

# Enable gradient clipping
# Check if enabled in CustomisedDLE (should be by default)

# Check for bad data
# Inspect first batch to ensure no inf/nan values
```

### Issue: Multi-GPU training crashes with "Address already in use"

**Cause:** Port conflict for distributed training

**Solution:**
```bash
# Change port in training script
--port 1237  # Change from 1236

# Or kill process using the port
lsof -ti:1236 | xargs kill -9
```

### Issue: `FileNotFoundError: union_embeddings_cachemodel_crop_padding_zeros_vitb16.p`

**Cause:** Union embeddings file not generated

**Solution:**
```bash
# This file should be provided with the repository or generated beforehand
# Check if file exists
ls -lh hicodet_pkl_files/*.p

# If missing, you may need to generate it (check EZ-HOI README)
# Or adjust path in script if file has different name
```

### Issue: Training completes but no `best.pth` saved

**Cause:** Checkpoint saving logic issue or disk full

**Solution:**
```bash
# Check available space
df -h checkpoints/

# Manually identify best checkpoint
ls -lt checkpoints/hico_HO_pt_default_vitbase/*.pth
# Use the one with highest mAP from log

# Rename to best.pth if needed
cp checkpoints/.../checkpoint0011.pth checkpoints/.../best.pth
```

### Issue: `ValueError: --txt_align` not recognized

**Cause:** Older version of code without this argument

**Solution:**
```bash
# Update to latest EZ-HOI code
git pull origin main

# Or remove --txt_align from script if you want to skip text adaptation
# (But this is not recommended for diffusion pipeline)
```

---

## Phase 3: Adapted Embedding Extraction

### Issue: `ValueError: Cannot extract args from checkpoint`

**Cause:** Checkpoint doesn't contain training arguments

**Solution:**
```python
# Provide arguments manually
python extract_adapted_text_embeddings.py \
  --checkpoint path/to/checkpoint.pth \
  --num_classes 117 \
  --zs_type unseen_verb
```

### Issue: `KeyError: 'txtmem_adapter.weight'` during load_state_dict

**Cause:** Model was trained without `--txt_align` flag

**Solution:**
```bash
# Option 1 (Recommended): Retrain with --txt_align
# Edit training script to add: --txt_align

# Option 2: Skip this phase and use raw embeddings (suboptimal)
# Extract raw embeddings instead:
python extract_hoi_text_embeddings.py \
  --clip_model ViT-B/16 \
  --num_classes 117
```

### Issue: `RuntimeError: CUDA out of memory` during extraction

**Cause:** Model is large and doesn't fit in memory

**Solution:**
```bash
# Use CPU (extraction is fast anyway)
python extract_adapted_text_embeddings.py \
  --checkpoint ... \
  --device cpu
```

### Issue: Extracted embeddings have all zeros

**Cause:** Model weights not loaded correctly

**Solution:**
```python
# Debug: Check if weights were loaded
python extract_adapted_text_embeddings.py ... 2>&1 | grep "Loaded"
# Should see: "✓ Loaded model_state_dict"

# Check checkpoint file
python -c "
import torch
ckpt = torch.load('path/to/checkpoint.pth')
print('Keys:', ckpt.keys())
# Should contain 'model_state_dict' or 'state_dict'
"
```

### Issue: Different embed_dim than expected (768 vs. 512)

**Cause:** Used ViT-L during training, not ViT-B

**Solution:**
```bash
# This is fine! Just ensure consistency throughout:
# - Use ViT-L checkpoint for extraction
# - Train diffusion on ViT-L embeddings (dim=768)
# - Use ViT-L paths during inference

# File names will be different:
# hoi_text_embeddings_adapted_unseen_verb_vitL_117_normalized.pkl
```

---

## Phase 4: Diffusion Training

### Issue: `ModuleNotFoundError: No module named 'denoising_diffusion_pytorch'`

**Cause:** Diffusion library not installed

**Solution:**
```bash
cd diffusion-bridge/ddpm
pip install -e .

# Verify installation
python -c "from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d import Unet1D; print('OK')"
```

### Issue: Training loss doesn't decrease

**Cause:** Learning rate too high or wrong objective

**Solution:**
```bash
# Try different objective
--objective pred_x0  # Instead of pred_noise

# Reduce learning rate
--learning_rate 5e-5  # Reduce from 8e-5

# Check data loading
# Print first batch to ensure embeddings are valid
```

### Issue: Checkpoints are very large (>1GB)

**Cause:** Saving full training state including optimizer

**Solution:**
```python
# This is normal for diffusion models
# Each checkpoint contains:
# - Model weights (~100-200 MB)
# - EMA weights (~100-200 MB)
# - Optimizer state (~500 MB)

# To save space, only keep model-300.pt (final model)
rm hoi_diffusion_results_adapted/model-{10..290..10}.pt
```

### Issue: Training is very slow

**Cause:** Dataset loading or small batch size

**Solution:**
```bash
# Increase batch size
--batch_size 128  # If GPU memory allows

# Reduce timesteps for faster iteration (less accurate)
--timesteps 500  # Reduce from 1000

# Use mixed precision (if not already enabled)
--use_amp True
```

---

## Phase 5: Inference

### Issue: `FileNotFoundError: diffusion_model_path`

**Cause:** Path mismatch or diffusion model not trained

**Solution:**
```bash
# Check if file exists
ls -lh hoi_diffusion_results_adapted/model-300.pt

# If missing, train diffusion first (Phase 4)
# Or update path in test script
```

### Issue: `RuntimeError: shapes mismatch` during diffusion forward pass

**Cause:** Vision features dim doesn't match text embedding dim

**Solution:**
```bash
# Debug: Print dimensions
# In diffusion_bridge_module.py, add:
print(f"Vision features shape: {vision_feat.shape}")
print(f"Expected embedding dim: {self.embedding_dim}")

# Common causes:
# 1. Trained with ViT-B (512) but testing with ViT-L (768)
# 2. Used wrong diffusion checkpoint

# Solution: Ensure consistency
# ViT-B → 512-dim → diffusion trained on 512-dim
# ViT-L → 768-dim → diffusion trained on 768-dim
```

### Issue: Inference is very slow (5+ minutes per image)

**Cause:** Too many diffusion inference steps

**Solution:**
```bash
# Reduce DDIM steps
--diffusion_inference_steps 100  # Reduce from 600

# Trade-off: Speed vs. quality
# 600 steps: Slow, best quality
# 300 steps: Moderate, good quality
# 100 steps: Fast, acceptable quality
```

### Issue: mAP with diffusion is LOWER than without

**Cause 1:** Diffusion trained on wrong embeddings (raw vs. adapted)

**Solution:**
```bash
# Verify you used ADAPTED embeddings
ls -lh hicodet_pkl_files/*adapted*.pkl

# If you used raw embeddings, retrain diffusion on adapted ones
```

**Cause 2:** Path mismatch in test script

**Solution:**
```bash
# Ensure paths match Phase 3 output
grep "diffusion_text_mean" scripts/hico_test_vitB_zs_diffusion.sh
# Should be: hoi_text_mean_adapted_*.pkl, not hoi_text_mean_vitB_*.pkl
```

**Cause 3:** Diffusion model undertrained

**Solution:**
```bash
# Use model-300.pt (300k steps), not model-10.pt
ls -lt hoi_diffusion_results_adapted/*.pt | head -1
```

### Issue: Predictions are all zeros or same class

**Cause:** Vision-text similarity computation is broken

**Solution:**
```python
# Debug: Print similarities before softmax
# In upt_tip_cache_model_free_finetune_distillself.py line 1288:
print(f"phi_union_HO min: {phi_union_HO.min()}, max: {phi_union_HO.max()}")

# Should see reasonable range like [-0.5, 0.5]
# If all zeros or extreme values, check:
# 1. Diffusion output is normalized
# 2. Text embeddings are normalized
```

---

## Performance Issues

### Issue: Training takes longer than expected

**Possible causes and solutions:**

| Cause | Check | Solution |
|-------|-------|----------|
| Slow data loading | `nvidia-smi` shows low GPU utilization | Increase `--num-workers` |
| Small batch size | Check `--batch-size` value | Increase to 16-32 if memory allows |
| Too many logging | Check log frequency | Reduce logging frequency |
| CPU bottleneck | Check CPU usage (htop) | Use more workers, faster storage |

### Issue: Model overfitting quickly

**Symptoms:** Training loss decreases but validation mAP doesn't improve

**Solutions:**
```bash
# Reduce learning rate
--learning-rate 5e-6

# Add weight decay
--weight-decay 1e-4

# Reduce number of epochs
--epochs 8  # Instead of 12

# Use early stopping
# Monitor best.pth creation
```

### Issue: Zero-shot performance is poor

**Possible causes:**

1. **Not using LLaVA descriptions:**
   ```bash
   # Add to training script:
   --vlmtxt llava
   ```

2. **Not using unseen prompt injection:**
   ```bash
   # Ensure this is in script:
   --unseen_pt_inj
   ```

3. **Text adapters not enabled:**
   ```bash
   # Add:
   --txt_align
   ```

4. **Diffusion trained on wrong distribution:**
   - Verify you used adapted embeddings from Phase 3
   - Check file names contain "adapted"

---

## General Debugging Tips

### Enable Verbose Logging

```bash
# Add to any Python script
import logging
logging.basicConfig(level=logging.DEBUG)

# For diffusion
python train_hoi_diffusion.py ... --verbose
```

### Check GPU Memory Usage

```bash
# Real-time monitoring
nvidia-smi -l 1

# Check memory usage of your process
nvidia-smi | grep python
```

### Validate Data Loading

```python
# Quick test script
python -c "
from utils_tip_cache_and_union_finetune import DataFactory
import argparse

# Create args (adjust as needed)
args = argparse.Namespace(...)

# Load dataset
trainset = DataFactory(...)
print(f'Dataset size: {len(trainset)}')

# Load one batch
sample = trainset[0]
print(f'Sample keys: {sample.keys()}')
print(f'Image shape: {sample['image'].shape}')
"
```

### Test Individual Components

```python
# Test diffusion module
python -c "
from diffusion_bridge_module import DiffusionBridgeHOI
import torch

bridge = DiffusionBridgeHOI(
    diffusion_path='hoi_diffusion_results_adapted/model-300.pt',
    text_mean_path='hicodet_pkl_files/hoi_text_mean_adapted_*.pkl',
    inference_steps=100
)

# Test forward pass
vision_feat = torch.randn(4, 512).cuda()
aligned = bridge(vision_feat)
print(f'Input shape: {vision_feat.shape}')
print(f'Output shape: {aligned.shape}')
print(f'Output norm: {aligned.norm(dim=-1).mean()}')
"
```

### Compare with Baseline

```bash
# Always test WITHOUT diffusion first
bash scripts/hico_test_vitB_zs.sh > baseline.log

# Then test WITH diffusion
bash scripts/hico_test_vitB_zs_diffusion.sh > diffusion.log

# Compare
diff baseline.log diffusion.log
```

### Use Checkpointing for Long Training

```python
# Add to training script
import signal
import sys

def signal_handler(sig, frame):
    print('Saving checkpoint before exit...')
    # Save checkpoint
    torch.save(model.state_dict(), 'checkpoint_interrupted.pth')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
```

---

## Getting Help

If you're still stuck after trying these solutions:

1. **Check the GitHub Issues:**
   - [EZ-HOI Issues](https://github.com/ChelsieLei/EZ-HOI/issues)
   - Search for similar problems

2. **Provide debugging information:**
   ```bash
   # System info
   python --version
   torch.__version__
   nvidia-smi

   # Environment
   echo $PYTHONPATH
   pip list | grep -E "torch|clip|pocket"

   # Error message
   # Full error traceback (not just last line)
   ```

3. **Minimal reproducible example:**
   - Isolate the issue to smallest possible code
   - Provide exact command that fails
   - Include sample data if possible

4. **Check related papers:**
   - EZ-HOI paper for model architecture
   - Diffusion-bridge paper for alignment theory
   - CLIP paper for embedding properties

---

## Prevention: Best Practices

1. **Always set PYTHONPATH** before running any command
2. **Verify file existence** before running scripts (checkpoints, data, etc.)
3. **Start with single GPU** to debug, then scale to multi-GPU
4. **Save checkpoints frequently** during long training runs
5. **Monitor GPU memory** - start with small batch size and increase gradually
6. **Test on small subset first** before full training (use `--debug` or limit dataset size)
7. **Keep logs** of all commands and outputs for debugging
8. **Document changes** if you modify any scripts or code

---

*If you encounter an issue not covered here, please open an issue on the EZ-HOI GitHub repository with detailed information.*
