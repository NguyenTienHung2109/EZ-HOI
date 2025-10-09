# Embedding Visualization Pipeline

Complete guide for visualizing adapted visual and text embeddings before/after diffusion transformation.

---

## 📋 Quick Start (3 Steps)

### Step 1: Modify Model Code (5 minutes)

Add **4 lines** to `upt_tip_cache_model_free_finetune_distillself.py`:

See **[MODIFICATION_GUIDE.md](MODIFICATION_GUIDE.md)** for exact locations.

**Summary:**
- Add 2 lines in `compute_roi_embeddings()` (around line 1269)
- Add 2 lines in `forward()` after text encoding (around line 1100-1200)

### Step 2: Test Modifications (1 minute)

```bash
python test_extraction.py --checkpoint checkpoints/your_model/best.pth
```

**Expected output:**
```
✓ Model loaded successfully
✓ Extraction mode enabled
✓ Visual embeddings extracted: torch.Size([N, 512])
✓ Text embeddings extracted: torch.Size([212, 512])
✅ ALL CHECKS PASSED!
```

### Step 3: Extract & Visualize (20-30 minutes)

```bash
# Extract embeddings from 1000 test images
python extract_adapted_embeddings.py \
    --checkpoint checkpoints/your_model/best.pth \
    --num_samples 1000 \
    --output embeddings_for_viz.pkl

# Generate visualizations
python visualize_embedding_distributions.py \
    --embeddings embeddings_for_viz.pkl \
    --output_dir visualization_results/
```

**Output:**
- `visualization_results/tsne_comparison.png` - 2D embedding space
- `visualization_results/distribution_metrics.png` - Quantitative comparison
- `visualization_results/summary_report.txt` - Text summary

---

## 📁 Files Created

| File | Purpose |
|------|---------|
| `MODIFICATION_GUIDE.md` | Exact code changes needed (4 lines) |
| `test_extraction.py` | Quick test to verify modifications work |
| `extract_adapted_embeddings.py` | Extract embeddings from trained model |
| `visualize_embedding_distributions.py` | Create visualizations |
| `VISUALIZATION_README.md` | This file |

---

## 🎨 What You'll Get

### Visualization 1: t-SNE Embedding Space

![t-SNE Example](https://via.placeholder.com/800x600.png?text=t-SNE+Visualization)

**What it shows:**
- **Blue points**: Visual embeddings BEFORE diffusion
- **Red points**: Visual embeddings AFTER diffusion
- **Green stars**: Text embeddings (target distribution)
- **Gray arrows**: Movement from before→after

**What to look for:**
- ✅ Red points closer to green stars = Diffusion working
- ✅ Tighter red clusters = Better alignment
- ❌ Red scattered far from green = Diffusion not helping

---

### Visualization 2: Distribution Metrics

![Metrics Example](https://via.placeholder.com/800x600.png?text=Distribution+Metrics)

**What it shows:**
- Cosine similarity to text (higher = better)
- Feature norms (should be similar to text)
- Distribution distances (lower = better)
- Overall improvement (positive = good)

**What to look for:**
- ✅ Similarity increases after diffusion
- ✅ Distance to text decreases
- ✅ Norms match text distribution

---

### Visualization 3: Summary Report

```
EMBEDDING DISTRIBUTION ANALYSIS REPORT

Metrics:
  Cosine Similarity (Visual→Text):
    Before: 0.3542 ± 0.1234
    After:  0.4821 ± 0.0987
    Improvement: +0.1279 (+36.12%)

Conclusions:
  ✓ Diffusion bridge IMPROVES visual→text alignment
  ✓ Similarity increased by 0.1279
  ✓ This suggests diffusion successfully reduces modality gap
```

---

## 🔧 Advanced Usage

### Extract with Diffusion Transformation

If you have a trained diffusion model:

```bash
python extract_adapted_embeddings.py \
    --checkpoint checkpoints/your_model/best.pth \
    --num_samples 1000 \
    --apply_diffusion \
    --diffusion_model hoi_diffusion_results/model-500.pt \
    --text_mean_path hicodet_pkl_files/hoi_text_mean_vitB_adapted_212.pkl \
    --inference_steps 100 \
    --output embeddings_with_diffusion.pkl
```

### Customize Visualization

```bash
python visualize_embedding_distributions.py \
    --embeddings embeddings_for_viz.pkl \
    --output_dir my_visualizations/ \
    --tsne_samples 5000 \          # More samples (slower but prettier)
    --tsne_perplexity 50 \          # Different perplexity
    --skip_tsne                     # Skip t-SNE (if too slow)
```

### Extract from Training Set

```bash
python extract_adapted_embeddings.py \
    --checkpoint checkpoints/your_model/best.pth \
    --partition train2015 \         # Use training set
    --num_samples 2000 \
    --output embeddings_train.pkl
```

---

## 🐛 Troubleshooting

### Issue: "Model missing '_extracted_visual_feat' attribute"

**Cause:** Model modifications not applied

**Solution:**
1. Check `MODIFICATION_GUIDE.md`
2. Add the 2 lines in `compute_roi_embeddings()`
3. Re-run `python test_extraction.py`

---

### Issue: "No visual embeddings extracted"

**Cause:** No human-object pairs detected in images

**Solution:**
- Try different images (test set has more pairs than train set)
- Check if DETR detector is working
- Verify `model.human_idx = 0` is correct

---

### Issue: "t-SNE takes too long"

**Cause:** Too many samples for t-SNE

**Solution:**
```bash
# Reduce samples
python visualize_embedding_distributions.py \
    --embeddings embeddings_for_viz.pkl \
    --tsne_samples 1000 \
    --output_dir visualization_results/

# Or skip t-SNE entirely
python visualize_embedding_distributions.py \
    --embeddings embeddings_for_viz.pkl \
    --skip_tsne \
    --output_dir visualization_results/
```

---

### Issue: "CUDA out of memory"

**Cause:** Too many embeddings or diffusion batch too large

**Solution:**
```bash
# Extract fewer samples
python extract_adapted_embeddings.py \
    --checkpoint checkpoints/your_model/best.pth \
    --num_samples 500 \
    --output embeddings_for_viz.pkl
```

Or edit `extract_adapted_embeddings.py` line ~355 to reduce batch_size:
```python
batch_size = 64  # Change from 128 to 64 or 32
```

---

## 📊 Interpreting Results

### Good Results (Diffusion Working)

```
Cosine Similarity (Visual→Text):
  Before: 0.35 ± 0.12
  After:  0.48 ± 0.09
  Improvement: +0.13 (+37%)
```

**Interpretation:**
- ✅ Large positive improvement (+37%)
- ✅ Lower std after diffusion (more consistent)
- ✅ Moving closer to text distribution

**Recommendation:** Integrate diffusion into full pipeline

---

### Poor Results (Diffusion Not Helping)

```
Cosine Similarity (Visual→Text):
  Before: 0.35 ± 0.12
  After:  0.33 ± 0.14
  Improvement: -0.02 (-5%)
```

**Interpretation:**
- ❌ Negative improvement (-5%)
- ❌ Higher std after diffusion (more scattered)
- ❌ Moving away from text distribution

**Recommendations:**
1. Check if diffusion model was trained correctly
2. Verify text_mean matches the adapted text embeddings (not raw CLIP)
3. Try different inference steps (50, 100, 200, 500)
4. Try different scale_factor (1.0, 2.0, 5.0, 10.0)
5. Retrain diffusion with more steps or data

---

### Mixed Results (Marginal Improvement)

```
Cosine Similarity (Visual→Text):
  Before: 0.35 ± 0.12
  After:  0.37 ± 0.11
  Improvement: +0.02 (+5%)
```

**Interpretation:**
- 🤔 Small positive improvement (+5%)
- 🤔 Slight reduction in std
- 🤔 Moving slightly closer to text

**Recommendations:**
1. Run on full test set to see if improvement is consistent
2. Check if improvement correlates with mAP gains
3. Consider ensemble: use diffusion only for uncertain predictions

---

## 🎯 Next Steps After Visualization

### If Diffusion Improves Alignment:

1. **Integrate into inference pipeline:**
   - Add diffusion_bridge to `compute_roi_embeddings()`
   - Apply before cosine similarity computation
   - Measure mAP improvement

2. **Optimize inference speed:**
   - Reduce inference_steps (100→50)
   - Use GPU batching
   - Cache diffusion model

3. **Analyze which classes benefit most:**
   - Compare rare vs common classes
   - Check seen vs unseen verbs
   - Identify failure cases

---

### If Diffusion Doesn't Help:

1. **Debug diffusion training:**
   - Check loss curves
   - Verify data preprocessing
   - Compare visual vs text statistics

2. **Try alternative approaches:**
   - Simple linear transformation
   - Learnable projection layer
   - Cross-modal adapter

3. **Analyze failure modes:**
   - Which classes get worse?
   - Over-smoothing?
   - Mode collapse?

---

## 📚 Related Documentation

- **[MODIFICATION_GUIDE.md](MODIFICATION_GUIDE.md)** - Exact code changes
- **[diffusion_bridge_module.py](diffusion_bridge_module.py)** - Diffusion inference
- **[train_hoi_diffusion.py](train_hoi_diffusion.py)** - Train diffusion model
- **[test_visual_diffusion.py](test_visual_diffusion.py)** - End-to-end diffusion test

---

## 💡 Tips

1. **Start small:** Test with 100 samples before running 1000
2. **Save often:** Pickle files are large, don't lose progress
3. **Compare checkpoints:** Extract from multiple checkpoints to track progress
4. **Document everything:** Save configs and metrics for reproducibility
5. **Share results:** Screenshots of t-SNE are great for papers/presentations

---

## 🙋 FAQ

**Q: Do I need to retrain the model?**

A: No! This is purely for analysis. The modifications only enable extraction.

**Q: Will this affect my existing checkpoints?**

A: No! Existing checkpoints work unchanged. Just load and set `extraction_mode=True`.

**Q: Can I use this with V-COCO or custom datasets?**

A: Yes, but you may need to adjust class numbers in the code.

**Q: How long does extraction take?**

A: ~10-30 minutes for 1000 images (depending on GPU and dataset).

**Q: Do I need the diffusion model?**

A: No! You can visualize without diffusion (just visual vs text comparison).

---

## ✅ Checklist

- [ ] Read MODIFICATION_GUIDE.md
- [ ] Add 4 lines to model code
- [ ] Run test_extraction.py
- [ ] Extract embeddings (1000 samples)
- [ ] Generate visualizations
- [ ] Review summary_report.txt
- [ ] Interpret results
- [ ] Decide next steps

---

**Questions? Issues?** Check the troubleshooting section or open an issue.

**Good luck with your visualizations! 🎨📊**
