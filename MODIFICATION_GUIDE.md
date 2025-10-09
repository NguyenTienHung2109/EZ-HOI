# Modification Guide: Enable Embedding Extraction

This guide shows the **exact 4 lines** you need to add to `upt_tip_cache_model_free_finetune_distillself.py` to enable embedding extraction for visualization.

---

## 🎯 Modification 1: Extract Visual Embeddings

**File:** `upt_tip_cache_model_free_finetune_distillself.py`

**Location:** In `compute_roi_embeddings()` method, around **line 1269**

**Find this code:**
```python
        if self.img_align:
            adapter_feat = self.mem_adapter(vis_feat.unsqueeze(1)).squeeze(1)
        else:
            adapter_feat = vis_feat
```

**Add these 2 lines immediately after:**
```python
        if self.img_align:
            adapter_feat = self.mem_adapter(vis_feat.unsqueeze(1)).squeeze(1)
        else:
            adapter_feat = vis_feat

        # ========== ADD THESE 2 LINES ========== ↓
        if hasattr(self, 'extraction_mode') and self.extraction_mode:
            self._extracted_visual_feat = adapter_feat.detach().cpu()
        # ======================================== ↑
```

**What this does:**
- When `extraction_mode=True`, stores visual features after mem_adapter
- Uses `.detach().cpu()` to avoid memory leaks and move to CPU
- This is the embedding BEFORE diffusion transformation

---

## 🎯 Modification 2: Extract Text Embeddings

**File:** `upt_tip_cache_model_free_finetune_distillself.py`

**Location:** In `forward()` method, after text encoding, around **line 1100-1200**

**Find this code block:**
```python
        hoitxt_features, origin_txt_features = self.clip_head.text_encoder(
            prompts, tokenized_prompts, deep_compound_prompts_text,
            txtcls_feat, txtcls_pt_list, origin_ctx
        )
```

**Add these 2 lines immediately after:**
```python
        hoitxt_features, origin_txt_features = self.clip_head.text_encoder(
            prompts, tokenized_prompts, deep_compound_prompts_text,
            txtcls_feat, txtcls_pt_list, origin_ctx
        )

        # ========== ADD THESE 2 LINES ========== ↓
        if hasattr(self, 'extraction_mode') and self.extraction_mode:
            self._extracted_text_feat = hoitxt_features.detach().cpu()
        # ======================================== ↑
```

**Alternative location (if using fixed text):**

If your model uses `fix_txt_pt=True`, find this line instead:
```python
        hoitxt_features = self.hoicls_txt[self.select_HOI_index].to(device)
```

And add after it:
```python
        hoitxt_features = self.hoicls_txt[self.select_HOI_index].to(device)

        # ========== ADD THESE 2 LINES ========== ↓
        if hasattr(self, 'extraction_mode') and self.extraction_mode:
            self._extracted_text_feat = hoitxt_features.detach().cpu()
        # ======================================== ↑
```

**What this does:**
- When `extraction_mode=True`, stores text features after all prompt learning
- This is the adapted text embedding used in similarity computation
- Only captured once (same for all images)

---

## ✅ How to Verify Modifications

After adding the 4 lines, test with:

```bash
python test_extraction.py
```

You should see:
```
✓ Model loaded successfully
✓ Extraction mode enabled
✓ Visual embeddings extracted: torch.Size([N, 512])
✓ Text embeddings extracted: torch.Size([212, 512])
✓ All checks passed!
```

---

## 🔧 Troubleshooting

### Error: "AttributeError: 'UPT' object has no attribute '_extracted_visual_feat'"

**Cause:** Model forward pass didn't reach the extraction point

**Solution:** Make sure you:
1. Set `model.extraction_mode = True` before forward pass
2. Ran at least one forward pass with valid data
3. The model has `img_align=True` (check model config)

### Error: "AttributeError: 'UPT' object has no attribute '_extracted_text_feat'"

**Cause:** Text encoding path not taken

**Solution:** Check if your model uses:
- `fix_txt_pt=False` (normal path) → Add modification at text_encoder call
- `fix_txt_pt=True` (fixed text) → Add modification at hoicls_txt assignment

### Visual embeddings shape is wrong

**Expected:** `[N_pairs, embed_dim]` where N_pairs = number of human-object pairs
- For ViT-B/16: `[N, 512]`
- For ViT-L/14: `[N, 768]`

**If you get:** `[N, 1, 512]` → You forgot `.squeeze(1)` in the extraction line

---

## 🎉 Next Steps

After modifications:

1. **Run extraction:**
   ```bash
   python extract_adapted_embeddings.py \
       --checkpoint checkpoints/your_model/best.pth \
       --num_samples 1000
   ```

2. **Generate visualizations:**
   ```bash
   python visualize_embedding_distributions.py \
       --embeddings embeddings_for_viz.pkl
   ```

3. **View results:**
   - `visualization_results/tsne_comparison.png`
   - `visualization_results/distribution_metrics.png`
   - `visualization_results/summary_report.txt`

---

## 📝 Notes

- These modifications are **read-only** - they don't affect training or inference
- The `extraction_mode` flag is **off by default** - normal usage unchanged
- You can remove these lines anytime without breaking anything
- Total code added: **4 lines** (2 + 2)

---

## 🔗 Related Files

- `extract_adapted_embeddings.py` - Uses these hooks to extract embeddings
- `visualize_embedding_distributions.py` - Visualizes the extracted embeddings
- `test_extraction.py` - Quick test to verify modifications work
