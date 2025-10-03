# Text Embedding Approaches: Diffusion-Bridge vs EZ-HOI

This document compares how the diffusion-bridge and EZ-HOI repositories handle text embedding generation, focusing on CLIP model usage, normalization strategies, and architectural differences.

---

## Quick Comparison Table

| Aspect | **Diffusion-Bridge** | **EZ-HOI** |
|--------|---------------------|-----------|
| **CLIP Model** | ViT-B/32 (default) | ViT-B/16 or ViT-L/14@336px |
| **CLIP Source** | Standard OpenAI CLIP | Modified CLIP with adapters |
| **Text Input** | Natural captions (MSCOCO) | Structured HOI descriptions |
| **Normalization** | Double normalize + mean subtraction | Single L2 normalize |
| **Embedding Space** | Modality-centered (mean-subtracted) | Standard CLIP space |
| **Scale Factor** | 5× amplification | 1× (no scaling) |
| **Learnable Components** | None (static embeddings) | Prompt learners + adapters |
| **Goal** | Reduce modality gap for captioning | Zero-shot HOI detection |

---

## Part 1: CLIP Model Selection

### Diffusion-Bridge

**File**: `parse_coco.py:34`
```python
clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
```

**Default Model**: `ViT-B/32`
- Patch size: 32×32
- Input resolution: 224×224
- Embedding dimension: 512
- Grid size: 7×7 = 49 patches

**Supported Models**:
```python
choices=("RN50", "RN101", "RN50x4", "ViT-B/32", "ViT-L/14")
```

**CLIP Usage**: **Vanilla OpenAI CLIP**
- No modifications to the architecture
- Pretrained weights used as-is
- Frozen during diffusion training

---

### EZ-HOI

**File**: `upt_tip_cache_model_free_finetune_distillself.py:1901-1915`
```python
clip_state_dict = torch.load(clip_model_path, map_location="cpu").state_dict()

design_details = {
    "trainer": 'MaPLe',
    "vision_depth": 0,
    "language_depth": 0,
    "vision_ctx": 0,
    "language_ctx": 0,
    "maple_length": args.N_CTX,
    "init_txtcls_pt": args.init_txtcls_pt,
    "pt_begin_layer": args.pt_begin_layer
}

clip_model = CLIP_models_adapter_prior2.build_model(
    state_dict=clip_state_dict,
    use_adapter=args.use_insadapter,
    adapter_pos=args.adapter_pos,
    adapter_num_layers=args.adapter_num_layers,
    multi_cross=args.multi_cross,
    design_details=design_details
)
```

**Default Models**: `ViT-B/16` or `ViT-L/14@336px`
- **ViT-B/16**: 16×16 patches, 224×224 input, 512-dim, 14×14 = 196 patches
- **ViT-L/14@336px**: 14×14 patches, 336×336 input, 768-dim, 24×24 = 576 patches

**CLIP Usage**: **Modified CLIP with Custom Adapters**
- **Vision Transformer**: Adapter layers injected (if `--use_insadapter`)
- **Text Transformer**: MaPLe-style prompt learning enabled
- **Learnable Components**:
  - Instance adapters in vision encoder
  - Multi-modal prompt learners
  - Text classification prompts
  - Image-guided prompts

**Key Difference**: EZ-HOI uses **higher-resolution models** (ViT-B/16 vs ViT-B/32) for better spatial features, crucial for HOI detection.

---

## Part 2: Text Encoding Process

### Diffusion-Bridge

**File**: `parse_coco.py:64-65`
```python
with torch.no_grad():
    text = clip.tokenize(caption_data["raw"]).to(device)
    text_embed = clip_model.encode_text(text).cpu()
```

**Input Example**:
```python
caption_data["raw"] = "A woman sitting on a bench with a dog"
```

**Process**:
1. **Tokenize**: Convert natural language caption to token IDs
2. **Encode**: Pass through CLIP text transformer
3. **Extract**: Get embedding at [EOS] token position
4. **Store**: Save raw CLIP embedding (no normalization yet)

**Output Shape**: `[1, 512]` (batch=1, ViT-B/32 embed_dim)

**No Learning**: Embeddings are **static**, extracted once and cached.

---

### EZ-HOI

**File**: `upt_tip_cache_model_free_finetune_distillself.py:1862-1877`
```python
@torch.no_grad()
def get_origin_text_emb(args, clip_model, tgt_class_names, obj_class_names):
    if use_templates == False:
        text_inputs = torch.cat([
            clip.tokenize(classname, context_length=77, truncate=True)
            for classname in tgt_class_names
        ])
    elif use_templates:
        text_inputs = get_multi_prompts(tgt_class_names)
        bs_t, nums, c = text_inputs.shape
        text_inputs = text_inputs.view(-1, c)

    with torch.no_grad():
        text_inputs = text_inputs.to("cuda")
        origin_text_embedding = clip_model.encode_text(text_inputs)
    if use_templates:
        origin_text_embedding = origin_text_embedding.view(bs_t, nums, -1).mean(0)

    origin_text_embedding = origin_text_embedding / origin_text_embedding.norm(
        dim=-1, keepdim=True
    )
```

**Input Example**:
```python
tgt_class_names = [
    "a photo of a person riding a bicycle",
    "a photo of a person carrying a bottle",
    ...
]
```

**Process**:
1. **Tokenize**: Convert HOI descriptions to tokens
2. **(Optional) Multi-template**: Generate 5 template variations and average
3. **Encode**: CLIP text transformer (potentially with learnable prompts)
4. **Normalize**: L2 normalization to unit sphere
5. **Store**: Used as initialization for prompt learning

**Output Shape**: `[117, 512]` or `[600, 512/768]` depending on setting

**Learning**: These embeddings serve as **anchors** for learnable prompts.

---

## Part 3: Normalization Strategies

### Diffusion-Bridge: Double Normalization + Mean Subtraction

#### Step 1: Compute Modality Means

**File**: `compute_embed_means.py:24-36`
```python
text_mean = torch.zeros(1, 512)
img_mean = torch.zeros(1, 512)

# Compute text mean
for cap_id in captions:
    cap_embed = captions[cap_id]["embed"]
    text_mean += cap_embed / cap_embed.norm()  # Normalize THEN sum

text_mean = text_mean / len(captions)

# Compute image mean (same process)
for img_id in images:
    img_embed = images[img_id]["embed"]
    img_mean += img_embed / img_embed.norm()

img_mean = img_mean / len(images)
```

**Critical Detail**: Mean is computed from **normalized embeddings**, not raw embeddings.

**Mathematical Formulation**:
```
text_mean = (1/N) * Σ(normalize(text_i))
image_mean = (1/N) * Σ(normalize(image_i))
```

This gives the **centroid** of the modality in normalized CLIP space.

---

#### Step 2: Normalization Chain for Training

**File**: `train_norm.py:20`
```python
caption_embeddings = [
    F.normalize(F.normalize(captions[cap_id]["embed"]) - text_mean)
    for cap_id in captions
]
return scale * torch.tensor(np.array(caption_embeddings)).squeeze().type(torch.float32)
# scale = 5.0
```

**Step-by-step Breakdown**:

```python
# Example with actual values
raw_embed = captions[cap_id]["embed"]  # [1, 512], ||x|| ≈ 25.4

# Step 1: First L2 normalization
x1 = F.normalize(raw_embed)  # [1, 512], ||x1|| = 1.0

# Step 2: Subtract modality mean
x2 = x1 - text_mean  # [1, 512], ||x2|| ≈ 0.15 (varies)

# Step 3: Second L2 normalization
x3 = F.normalize(x2)  # [1, 512], ||x3|| = 1.0

# Step 4: Scale by 5
x4 = x3 * 5.0  # [1, 512], ||x4|| = 5.0
```

**Why This Works**:

1. **First normalize**: Map to unit hypersphere
2. **Subtract mean**: Shift origin to modality centroid
   - Text embeddings center at text_mean
   - Image embeddings center at image_mean
   - This **reduces modality gap** (they're closer after centering)
3. **Second normalize**: Re-project to unit hypersphere
   - Now both modalities are centered at origin
   - Directions are preserved, magnitudes equalized
4. **Scale by 5**: Amplify for diffusion process
   - Larger norm → better signal for UNet
   - Still maintains unit direction information

**Geometric Interpretation**:
```
Original CLIP space:     [text cluster]  ←gap→  [image cluster]
After mean subtraction:        [both clusters overlap more]
After renormalization:           [perfectly aligned on sphere]
```

---

#### Step 3: Reverse Process (Diffusion Bridging)

**File**: `make_dataset_with_diffusion_vision.py:49-60`
```python
# Input: Raw CLIP vision embedding
vision_embeddings = test_tokens['images'][idx]['embed'].cuda()

# Apply same normalization chain
vision_embeddings = F.normalize(vision_embeddings)
vision_embeddings -= image_mean
vision_embeddings = F.normalize(vision_embeddings) * 5
vision_embeddings = vision_embeddings.unsqueeze(1).to(torch.float32)

# Diffusion bridging
generated_data = model.ddim_sample_with_img(generated_data, inference_step=600)

# Reverse normalization
generated_data = generated_data.squeeze(1)
generated_data = F.normalize(generated_data)  # Back to unit sphere
```

**Process**:
1. Vision embed → normalize → center → normalize → scale
2. Diffusion model refines it toward text distribution
3. Output → normalize (ready for caption model)

---

### EZ-HOI: Simple L2 Normalization

**File**: `upt_tip_cache_model_free_finetune_distillself.py:1877`
```python
origin_text_embedding = origin_text_embedding / origin_text_embedding.norm(
    dim=-1, keepdim=True
)
```

**That's it.** Single L2 normalization, no mean subtraction.

**Why Simpler?**

1. **No modality gap problem**: EZ-HOI doesn't need vision→text bridging
   - Text embeddings are used for classification (dot product)
   - Vision embeddings are used for matching (dot product)
   - Both already in CLIP space, designed for alignment

2. **Learnable prompts handle alignment**:
   - `MultiModalPromptLearner` adapts embeddings during training
   - Adapters can learn to bridge any remaining gaps
   - No need for geometric recentering

3. **HOI-specific structure**:
   - Text: "a photo of a person [verb] [object]"
   - Vision: Union region features from detection
   - Structure is fixed, not free-form captions

---

## Part 4: Text Input Differences

### Diffusion-Bridge: Natural Captions

**Source**: MSCOCO dataset
```python
# Examples from parse_coco.py
captions = {
    "sentid": 12345,
    "caption": "A woman sitting on a bench with a dog",
    "img_id": 67890,
    "embed": tensor([...])
}
```

**Characteristics**:
- **Free-form** natural language
- **Variable length** (5-20 words typical)
- **Descriptive** (multiple objects, actions, attributes)
- **High diversity** (~400k unique captions in MSCOCO)

**Challenge**: Text and image embeddings are **far apart** in CLIP space
- Text describes scene holistically
- Image contains visual details CLIP emphasizes differently
- **Modality gap** is significant

---

### EZ-HOI: Structured HOI Descriptions

**Source**: `hico_text_label.py`
```python
hico_text_label = {
    (76, 1): 'a photo of a person riding a bicycle',
    (8, 39): 'a photo of a person carrying a bottle',
    (36, 16): 'a photo of a person holding a dog',
    ...
}
```

**Characteristics**:
- **Structured template**: "a photo of a person [VERB] [OBJECT]"
- **Fixed length** (8-10 tokens)
- **Action-centric** (focuses on interaction)
- **Limited diversity** (117 verbs × ~50 objects = controlled vocabulary)

**Advantage**: Text and image embeddings are **naturally aligned**
- Template matches CLIP's training data
- Action verbs have clear visual correlates
- No significant modality gap for EZ-HOI's task

---

## Part 5: Embedding Space Philosophy

### Diffusion-Bridge: Modality-Centered Space

**Goal**: Make text and image embeddings **indistinguishable** in embedding space

**Approach**:
```python
# Original CLIP space (modality gap exists)
text_embeds: centered around text_mean
image_embeds: centered around image_mean
distance(text_mean, image_mean) = large

# After double normalization (no modality gap)
text_embeds: centered around origin
image_embeds: centered around origin
distance(text_mean', image_mean') = 0
```

**Why**: Caption generation requires **cross-modal transfer**
- Input: image embedding
- Output: caption text
- Model must treat image embedding as if it were text embedding
- Diffusion bridge refines image→text in this centered space

**Analogy**:
- Original: Two cities (Text-ville and Image-town) 100 miles apart
- After centering: Both cities moved to same location
- Diffusion: Smooth road between them

---

### EZ-HOI: Standard CLIP Space

**Goal**: Maximize **discriminability** between HOI classes

**Approach**:
```python
# Use CLIP space as-is (no recentering)
text_embeds: represent HOI class semantics
image_embeds: represent visual features

# Classification via cosine similarity
logits = image_features @ text_features.T
```

**Why**: HOI detection is **within-modality matching**
- Visual features (union regions) → HOI classes (text)
- Both are **already aligned** by CLIP pretraining
- Need maximum **separation** between classes, not modality bridging

**Analogy**:
- CLIP provides a map with landmarks
- EZ-HOI learns to navigate the map better (via prompts)
- No need to redraw the map

---

## Part 6: Scale Factor Analysis

### Diffusion-Bridge: 5× Amplification

**File**: `train_norm.py:24`
```python
training_seq = load_dataset(scale=5)

def load_dataset(scale=5.0):
    caption_embeddings = [
        F.normalize(F.normalize(captions[cap_id]["embed"]) - text_mean)
        for cap_id in captions
    ]
    return scale * torch.tensor(np.array(caption_embeddings))
```

**Effect**:
```python
normalized_embed.norm() = 1.0  # Unit vector
scaled_embed.norm() = 5.0      # Amplified
```

**Why Scale by 5?**

1. **UNet Signal Strength**:
   - Diffusion UNet expects inputs with meaningful magnitude
   - Unit vectors (norm=1) may be too weak
   - Scaling to norm=5 provides stronger signal

2. **Noise Schedule Compatibility**:
   - Gaussian diffusion adds noise: `x_t = sqrt(α_t)*x_0 + sqrt(1-α_t)*ε`
   - Larger initial magnitude → better SNR at early steps
   - Empirically chosen (could be 3, 5, 10, etc.)

3. **Gradient Flow**:
   - Larger activations → larger gradients
   - Helps training convergence
   - Prevents vanishing gradients in deep UNet

**Reverse Process**: Final output is **renormalized** to unit length
```python
# make_dataset_with_diffusion_vision.py:60
generated_data = F.normalize(generated_data)
```

So the 5× scaling is **temporary** for diffusion only.

---

### EZ-HOI: No Scaling

**File**: All embeddings remain unit norm
```python
origin_text_embedding = origin_text_embedding / origin_text_embedding.norm(
    dim=-1, keepdim=True
)
# No scaling factor
```

**Why No Scaling?**

1. **Cosine Similarity**:
   - HOI classification uses: `logits = image_emb @ text_emb.T`
   - Cosine similarity is **scale-invariant**: `cos(θ) = (x·y) / (||x|| ||y||)`
   - Scaling both by 5× wouldn't change cosine values

2. **Temperature Scaling Instead**:
   - EZ-HOI uses learnable temperature in CLIP: `logit_scale = exp(temperature)`
   - Adjusts similarity magnitude during training
   - More principled than arbitrary scaling

3. **No Intermediate Processing**:
   - Text embeddings go directly to classification
   - No diffusion model that needs specific magnitude
   - Keep CLIP's original calibration

---

## Part 7: Use Case Alignment

### Diffusion-Bridge: Cross-Modal Generation

**Task**: Image → Caption (vision to language)

**Challenge**:
- Vision and text are **different modalities**
- Need to "translate" from one to another
- GPT-2 decoder expects **text-like** inputs

**Solution**:
1. Train diffusion model on **text embeddings only**
   - Learns distribution of caption embeddings
2. **Bridge** vision embeddings to text distribution
   - `vision_embed → add noise → denoise → text-like embed`
3. Feed refined embedding to GPT-2 decoder
   - GPT-2 generates caption tokens

**Why It Works**:
- Diffusion model learns: "What do caption embeddings look like?"
- Vision embeddings are transformed to match that distribution
- GPT-2 can then decode them as if they were captions

---

### EZ-HOI: Within-Modality Classification

**Task**: Union features → HOI class (both in CLIP space)

**Challenge**:
- Zero-shot: No training examples for unseen classes
- Need good embeddings for unseen HOI descriptions

**Solution**:
1. Get CLIP embeddings for **all HOI descriptions** (seen + unseen)
2. Learn **prompts** that improve alignment
   - Prompts guide CLIP to focus on action semantics
3. **Transfer knowledge** from seen to unseen
   - Unseen prompts attend to similar seen prompts

**Why It Works**:
- Both text and vision are in **same CLIP space**
- Prompts refine representations, don't bridge modalities
- Semantic similarity enables zero-shot transfer

---

## Summary: Key Philosophical Differences

| Aspect | Diffusion-Bridge | EZ-HOI |
|--------|-----------------|--------|
| **Problem** | Modality gap (vision ≠ text) | Zero-shot transfer (seen → unseen) |
| **Strategy** | Geometric recentering | Semantic prompt learning |
| **Normalization** | Double norm + mean subtract | Single L2 norm |
| **Space** | Modality-centered (origin) | Standard CLIP space |
| **Scaling** | 5× for diffusion signal | 1× for cosine similarity |
| **Learning** | Diffusion model (UNet) | Prompt adapters |
| **Goal** | Make vision "look like" text | Make text more discriminative |
| **Complexity** | High (normalization chain) | Low (standard practice) |

---

## Practical Implications

### When to Use Diffusion-Bridge Approach:

✅ **Cross-modal tasks** (image→text, text→image)
✅ **Generative models** (need specific embedding distributions)
✅ **Large modality gap** (e.g., vision vs language for generation)
✅ **No learnable alignment** (static embeddings only)

### When to Use EZ-HOI Approach:

✅ **Classification tasks** (matching in same space)
✅ **Learnable components** (prompts, adapters available)
✅ **Within-modality** (vision→vision, text→text comparisons)
✅ **Zero-shot transfer** (semantic similarity matters)

---

## Could EZ-HOI Benefit from Diffusion-Bridge Normalization?

**Potential Benefits**:
1. ✅ **Better vision-text alignment** for union features
2. ✅ **Reduced modality gap** if it exists in HOI setting

**Potential Drawbacks**:
1. ❌ **Lose CLIP's learned calibration** (embeddings are tuned together)
2. ❌ **Extra hyperparameters** (mean computation, scale factor)
3. ❌ **May hurt discriminability** (centering reduces class separation)
4. ❌ **Redundant with prompt learning** (learnable adapters already handle alignment)

**Verdict**: Probably **not beneficial** for EZ-HOI
- Prompt learning is more flexible than geometric recentering
- HOI task doesn't have significant modality gap
- CLIP space already well-suited for structured text

---

## Could Diffusion-Bridge Benefit from EZ-HOI Prompt Learning?

**Potential Benefits**:
1. ✅ **Learnable alignment** instead of fixed normalization
2. ✅ **Task-specific adaptation** for captioning

**Potential Drawbacks**:
1. ❌ **Increased training cost** (more parameters)
2. ❌ **May interfere with diffusion** (prompts change embedding distribution)
3. ❌ **GPT-2 expects specific format** (standard CLIP embeddings)

**Verdict**: **Could be beneficial** as extension
- Use prompts to improve initial CLIP embeddings
- Then apply diffusion bridge for final refinement
- But adds complexity for marginal gain

---

## Conclusion

**Diffusion-Bridge** and **EZ-HOI** solve fundamentally different problems:

- **Diffusion-Bridge**: "How do I make a vision embedding usable for text generation?"
  - Answer: Transform it geometrically to lie in text distribution

- **EZ-HOI**: "How do I classify HOI interactions in zero-shot setting?"
  - Answer: Learn prompts that improve semantic discriminability

The **double normalization + mean subtraction** in Diffusion-Bridge is brilliant for reducing modality gap in cross-modal generation, but **unnecessary** for EZ-HOI's within-space classification task.

Both approaches are **optimal for their specific goals** and demonstrate different philosophies in working with CLIP embeddings.
