# Testing Diffusion Bridge Integration

Quick guide to test the diffusion bridge modules without GPU or full dataset.

---

## Setup (Required)

### On Colab

```python
# Clone the test branch
!git clone -b test https://github.com/NguyenTienHung2109/EZ-HOI.git
%cd EZ-HOI

# Initialize the diffusion-bridge submodule (REQUIRED for full tests)
!git submodule update --init --recursive
```

### On Local Machine

```bash
# Clone the test branch
git clone -b test https://github.com/NguyenTienHung2109/EZ-HOI.git
cd EZ-HOI

# Initialize the diffusion-bridge submodule (REQUIRED for full tests)
git submodule update --init --recursive
```

**Note:** The `diffusion-bridge` submodule is required to load and run the actual diffusion model. Without it, tests will skip the diffusion sampling part but still verify the geometric transformation logic.

---

## Test Files Created

| File | Purpose | Run Time |
|------|---------|----------|
| `upt_tip_cache_model_free_finetune_distillself.py` | Test `DiffusionGeometricTransform` class | ~1 second |
| `diffusion_bridge_module.py` | Test diffusion bridge loading and methods | ~1-5 seconds |
| `test_diffusion_integration.py` | Test full integration (vision + text paths) | ~1 second |

All tests run on **CPU only** with **dummy data** (no GPU or dataset required).

---

## Quick Test Commands

### Test 1: DiffusionGeometricTransform Module

```bash
python upt_tip_cache_model_free_finetune_distillself.py
```

**Tests:**
- Geometric transformation logic (L2 norm → subtract mean → L2 norm)
- Vision and text feature transformation
- Coordinate space alignment
- Gradient flow (backpropagation)

**Expected output:**
```
======================================================================
Testing DiffusionGeometricTransform Module
======================================================================

1. Creating dummy text mean...
2. Creating DiffusionGeometricTransform...
3. Testing with vision features...
   ✓ Vision transformation correct
4. Testing with text features...
   ✓ Text transformation correct
5. Verifying transformation logic...
   ✓ Transformation logic verified
6. Testing coordinate space alignment...
7. Testing gradient flow (differentiability)...
   ✓ Gradients flow correctly

======================================================================
✅ ALL TESTS PASSED!
======================================================================
```

---

### Test 2: Diffusion Bridge Module

```bash
python diffusion_bridge_module.py
```

**Tests:**
- Diffusion bridge loading (with/without checkpoint)
- `forward()` method (full transform + diffusion)
- `apply_diffusion_only()` method (diffusion sampling only)
- Different inference step configurations

**Expected output (without checkpoint):**
```
======================================================================
Testing Diffusion Bridge Module (CPU Mode)
======================================================================

⚠️  Diffusion checkpoint not found.
   Running basic sanity tests only (without diffusion sampling)...

Test 1: Geometric transformation logic
  ✓ Geometric transformation works correctly

======================================================================
✅ Basic sanity tests passed!
======================================================================
```

**Expected output (with checkpoint):**
```
======================================================================
Testing Diffusion Bridge Module (CPU Mode)
======================================================================

Found checkpoint files:
  Diffusion: dummy_diffusion_files/dummy_diffusion_vitB.pt
  Text mean: dummy_diffusion_files/dummy_text_mean_vitB.pkl

Test 1: Loading diffusion bridge module...
  ✓ Module loaded successfully

Test 2: Testing full forward() method...
  ✓ forward() method works

Test 3: Testing apply_diffusion_only() method...
  ✓ apply_diffusion_only() method works

Test 4: Testing with different inference steps...
  ✓ Different step counts work

======================================================================
✅ ALL TESTS PASSED!
======================================================================
```

---

### Test 3: Full Integration

```bash
python test_diffusion_integration.py
```

**Tests:**
- Vision path: raw → geometric transform → [diffusion] → normalize
- Text path: raw → geometric transform → normalize (NO diffusion)
- Cosine similarity computation
- Gradient flow through both paths
- End-to-end classification pipeline

**Expected output:**
```
======================================================================
Diffusion Bridge Integration Test (CPU Mode)
======================================================================

1. Importing modules...
   ✓ Imported DiffusionGeometricTransform
   ✓ Imported DiffusionBridgeHOI

======================================================================
TEST 1: Geometric Transformation
======================================================================
    ✓ Vision transformation correct
    ✓ Text transformation correct

======================================================================
TEST 2: Coordinate Space Alignment
======================================================================
  ✓ Both modalities in same coordinate space

======================================================================
TEST 3: Vision and Text Paths (Simulated)
======================================================================
  ✓ Vision path works
  ✓ Text path works

======================================================================
TEST 4: Cosine Similarity Computation
======================================================================
  ✓ Logits computed correctly

======================================================================
TEST 5: Gradient Flow (Backpropagation)
======================================================================
    ✓ Gradients flow correctly through both paths

======================================================================
TEST 6: End-to-End Classification
======================================================================
  ✓ End-to-end pipeline works

======================================================================
✅ ALL INTEGRATION TESTS PASSED!
======================================================================
```

---

## Optional: Create Dummy Diffusion Files

To test with a dummy diffusion checkpoint (random weights, for testing only):

```bash
# This requires PyTorch
python create_dummy_diffusion_files.py
```

This creates:
- `dummy_diffusion_files/dummy_diffusion_vitB.pt`
- `dummy_diffusion_files/dummy_text_mean_vitB.pkl`
- `dummy_diffusion_files/dummy_diffusion_vitL.pt`
- `dummy_diffusion_files/dummy_text_mean_vitL.pkl`

Then re-run the tests above - they will use the dummy checkpoint.

**WARNING:** Dummy files use random weights. Do NOT use for actual experiments!

---

## Test All Modules at Once

```bash
# Run all tests sequentially
echo "Test 1: DiffusionGeometricTransform"
python upt_tip_cache_model_free_finetune_distillself.py

echo "\nTest 2: DiffusionBridgeHOI"
python diffusion_bridge_module.py

echo "\nTest 3: Integration"
python test_diffusion_integration.py
```

---

## What Each Test Verifies

### DiffusionGeometricTransform (`upt_tip_cache_model_free_finetune_distillself.py`)
- ✅ Correct transformation pipeline
- ✅ Output is L2-normalized
- ✅ Works for both vision and text
- ✅ Gradients flow correctly

### DiffusionBridgeHOI (`diffusion_bridge_module.py`)
- ✅ Module loads correctly (with/without checkpoint)
- ✅ `forward()` applies full pipeline
- ✅ `apply_diffusion_only()` works separately
- ✅ Configurable inference steps

### Integration (`test_diffusion_integration.py`)
- ✅ Vision and text paths work together
- ✅ Coordinate space alignment verified
- ✅ Cosine similarity computation correct
- ✅ End-to-end pipeline functional
- ✅ Backpropagation works

---

## Troubleshooting

### ImportError: No module named 'torch'

Make sure you're in a Python environment with PyTorch installed:

```bash
pip install torch
```

Or activate your conda environment:

```bash
conda activate your_env_name
```

### All tests pass, what's next?

If all CPU tests pass, you're ready for:

1. **Create/obtain real diffusion checkpoint:**
   - Train diffusion on MS-COCO
   - Or obtain pretrained checkpoint

2. **Update config:**
   - Edit `configs/diffusion_bridge_config.yaml`
   - Point to real checkpoint paths

3. **Run training:**
   ```bash
   bash scripts/hico_train_vitB_zs_diff.sh
   ```

---

## Summary

These tests verify that:
- ✅ All new modules are correctly implemented
- ✅ Geometric transformation works for both modalities
- ✅ Diffusion bridge integrates properly
- ✅ Gradients flow through the entire pipeline
- ✅ Code runs without GPU or dataset

**All tests should complete in < 10 seconds total.**

Next step: Run actual training with real COCO diffusion checkpoint!
