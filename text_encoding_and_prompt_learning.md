# Text Encoding and Unseen Class Prompt Learning in EZ-HOI

This document explains how EZ-HOI encodes text descriptions of HOI classes and refines prompts for unseen classes through guided prompt learning.

## Overview

EZ-HOI uses **guided prompt learning** to improve zero-shot HOI detection by:
1. **Encoding text descriptions** of HOI interactions using CLIP's text encoder
2. **Learning multi-modal prompts** (MaPLe-style) with adapters
3. **Transferring knowledge from seen to unseen classes** via prompt injection

## Part 1: Text Embedding Generation

### Location
**File**: `upt_tip_cache_model_free_finetune_distillself.py`
**Function**: `get_origin_text_emb()` (lines 1861-1884)

### Input: Text Descriptions

**Source**: `hico_text_label.py`

HOI class descriptions follow the pattern:
```python
hico_text_label = {
    (4, 4): 'a photo of a person boarding an airplane',
    (17, 4): 'a photo of a person directing an airplane',
    (76, 1): 'a photo of a person riding a bicycle',
    ...
}
```

**Format**: `(verb_idx, object_idx): "text description"`

For **117 verb classes** (zero-shot verb setting):
```python
# hico_verbs_sentence from hico_list.py
classnames = [
    'a photo of a person boarding something',
    'a photo of a person carrying something',
    ...
]
```

For **600 HOI classes** (full setting):
```python
classnames = list(hico_text_label.hico_text_label.values())
# Total: 600 interaction descriptions
```

### Processing Pipeline

#### Step 1: Tokenization
```python
# Single template (default)
if use_templates == False:
    text_inputs = torch.cat([
        clip.tokenize(classname, context_length=77, truncate=True)
        for classname in tgt_class_names
    ])
    # Shape: [num_classes, 77]
```

**Example**:
- Input: `"a photo of a person riding a bicycle"`
- Output: `[49406, 320, 1125, 539, 320, 2533, 6765, 320, 10165, 49407, 0, 0, ...]`
  - `49406`: [SOS] token
  - `49407`: [EOS] token
  - `0`: [PAD] tokens

#### Step 2: Multi-Template Encoding (Optional)

When `--use_templates` is enabled:
```python
# get_multi_prompts() generates multiple templates
templates = [
    'a photo of the person {}.',
    'a picture of the person {}.',
    'the person is {}.',
    'the person {}.',
    'a person is {}.',
]

# For each HOI class, generate embeddings for all templates
all_texts_input = []
for temp in templates:
    texts_input = torch.cat([
        clip.tokenize(temp.format(hoi_action))
        for hoi_action in hico_texts
    ])
    all_texts_input.append(texts_input)
all_texts_input = torch.stack(all_texts_input, dim=0)
# Shape: [num_templates, num_classes, 77]
```

#### Step 3: CLIP Text Encoding

```python
with torch.no_grad():
    text_inputs = text_inputs.to("cuda")
    origin_text_embedding = clip_model.encode_text(text_inputs)
    # Shape: [num_classes, 512] for ViT-B or [num_classes, 768] for ViT-L
```

**CLIP's `encode_text()` process** (from `CLIP_models_adapter_prior2.py:1473-1489`):
```python
def encode_text(self, text):
    x = self.token_embedding(text)  # [batch, 77, 512/768]
    x = x + self.positional_embedding
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = self.transformer(x)  # Text transformer
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = self.ln_final(x)

    # Extract embedding at [EOS] token position
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
    return x
```

**Key operation**: Extract feature at **[EOS] token** position (not averaging)
- `text.argmax(dim=-1)` finds [EOS] position (highest token ID)
- This gives the contextual embedding for the entire sentence

#### Step 4: Template Averaging (if multi-template)

```python
if use_templates:
    origin_text_embedding = origin_text_embedding.view(
        num_templates, num_classes, -1
    ).mean(0)  # Average across templates
    # Shape: [num_classes, 512/768]
```

#### Step 5: Normalization

```python
origin_text_embedding = origin_text_embedding / origin_text_embedding.norm(
    dim=-1, keepdim=True
)
# L2 normalization for cosine similarity
```

### Output

```python
return origin_text_embedding, object_embedding

# origin_text_embedding: [117, 512/768] for verbs or [600, 512/768] for full HOI
# object_embedding: [80, 512/768] for object class names
```

**Storage**: These embeddings are stored in the UPT model as:
- `self.text_embedding`: Used for HOI classification
- `self.obj_embedding`: Used for object class guidance

---

## Part 2: Learnable Prompt Engineering

### Location
**Class**: `MultiModalPromptLearner` (lines 442-700)

### Architecture: MaPLe-Inspired Design

**MaPLe** = **M**ulti-modal **P**rompt **Le**arning
EZ-HOI extends MaPLe with HOI-specific adaptations.

### Core Components

#### 1. Shared Context Vectors (`self.ctx`)

```python
# Initialized from text (e.g., "A photo of") or randomly
ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
nn.init.normal_(ctx_vectors, std=0.02)

self.ctx = nn.Parameter(ctx_vectors)
# Shape: [n_ctx, 512/768]
# n_ctx = 2 (default, controlled by --N_CTX argument)
```

**Purpose**: Learnable prefix tokens prepended to class names

#### 2. Compound Prompts for Deep Layers

```python
self.compound_prompts_text = nn.ParameterList([
    nn.Parameter(torch.empty(n_ctx, ctx_dim))
    for _ in range(self.compound_prompts_depth - 1)  # tune_LY=9 layers
])
# Creates 8 additional prompt sets (one per deep layer)
```

**Architecture**:
- **Layer 0 (shallow)**: `self.ctx` → projected to vision dim
- **Layers 1-8 (deep)**: `self.compound_prompts_text[i]` → injected into transformer layers

#### 3. Class-Specific Text Prompts (`--txtcls_pt`)

When enabled with `--txtcls_pt`:
```python
self.txtcls_pt_adapter = Adapter(ctx_dim, down_size=args.emb_dim)
self.txtcls_ctx_pt = []
for index in range(self.compound_prompts_depth):
    self.txtcls_ctx_pt.append(
        nn.Parameter(torch.randn(n_ctx, ctx_dim))
    )
```

**Purpose**: Separate learnable prompts for each layer, adapted with **text class features**

**Adaptation mechanism**:
```python
# In forward():
temp_pt = self.txtcls_pt_adapter(
    self.txtcls_ctx_pt[index].unsqueeze(1).repeat(1, len(select_HOI_index), 1),
    (txtcls_feat[select_HOI_index].unsqueeze(1).repeat(...), None)
)
# Adapts prompts based on the text embeddings of selected HOI classes
```

#### 4. Image-Guided Prompts (`--img_clip_pt`)

When enabled with `--img_clip_pt`:
```python
self.img_clip_pt_adapter = Adapter(vis_dim, prior_size=ctx_dim, down_size=args.emb_dim)

# Load pre-extracted CLIP image features
clip_img_list = []
for fn in filenames:
    clip_img = pickle.load(open(
        os.path.join(self.clip_img_file, fn.split(".")[0]+"_clip.pkl"), 'rb'
    ))
    clip_img /= clip_img.norm(dim=-1, keepdim=True)
    clip_img_list.append(clip_img)
img_clip_prior = torch.stack(clip_img_list)

# Adapt visual prompts with CLIP image features
for index, vis_pt_i in enumerate(visual_deep_prompts):
    visual_deep_prompts[index] = self.img_clip_pt_adapter(
        vis_pt_i.unsqueeze(1).repeat(1, len(img_clip_prior), 1),
        (img_clip_prior, None)
    )
```

**Purpose**: Inject pre-extracted CLIP image features to guide visual prompts

---

## Part 3: Unseen Class Prompt Refinement (`--unseen_pt_inj`)

### The Zero-Shot Problem

In zero-shot HOI detection:
- **Seen classes**: Have training examples, can learn good prompts
- **Unseen classes**: No training examples, prompts are randomly initialized

**Solution**: Transfer knowledge from seen to unseen via **guided prompt injection**

### Mechanism: Prior Knowledge Transfer

**Location**: `MultiModalPromptLearner.forward()` (lines 661-673)

```python
if unseen_text_priors is not None:
    temp_pt = temp_pt.permute(1, 0, 2)  # [num_classes, n_ctx, dim]

    # Identify unseen and their most similar seen classes
    temp_ind_unseen = torch.tensor(unseen_text_priors[2])  # Unseen class indices
    temp_ind_simi_seen = torch.tensor(unseen_text_priors[3])  # Similar seen indices

    # Construct prior: [unseen_text_desc, similar_seen_prompts, current_unseen_prompts]
    temp_prior = torch.cat((
        unseen_text_priors[0],  # Original text embeddings of unseen classes
        temp_pt[temp_ind_simi_seen].clone().detach(),  # Prompts from similar seen
        temp_pt[temp_ind_unseen]  # Current unseen prompts
    ), dim=1)

    # Attention mask (0 = attend, 1 = ignore)
    temp_mask = torch.cat((
        unseen_text_priors[1],  # Mask for text embeddings
        torch.zeros((len(temp_prior), self.n_ctx * 2))  # Attend to prompts
    ), dim=1)

    # Refine unseen prompts using cross-attention with prior
    temp_pt[temp_ind_unseen] = self.unseen_pt_adapter(
        temp_pt[temp_ind_unseen].permute(1, 0, 2),  # Query: unseen prompts
        (temp_prior, temp_mask)  # Key/Value: prior knowledge
    ).permute(1, 0, 2)

    temp_pt = temp_pt.permute(1, 0, 2)
```

### Detailed Breakdown

#### Input: `unseen_text_priors`

Structure:
```python
unseen_text_priors = [
    text_embeddings,  # [num_unseen, embed_dim] - CLIP text features
    attention_mask,   # [num_unseen, text_len] - Mask for text
    unseen_indices,   # List of unseen class IDs
    similar_seen_indices  # List of most similar seen class IDs
]
```

**How similar seen classes are found**:
```python
# Compute similarity between unseen and seen text embeddings
unseen_embeds = text_embedding[unseen_idx]  # [num_unseen, 512/768]
seen_embeds = text_embedding[seen_idx]  # [num_seen, 512/768]

similarity = unseen_embeds @ seen_embeds.T  # Cosine similarity
most_similar_seen_idx = similarity.argmax(dim=1)  # [num_unseen]
```

**Example** (unseen verb setting):
- Unseen verb: "practicing"
- Most similar seen verb: "training" or "playing"
- Transfer learned prompts from "training" to "practicing"

#### Adapter Architecture: `self.unseen_pt_adapter`

```python
class Adapter(nn.Module):
    def __init__(self, input_size, down_size=64, ...):
        self.down_proj_mem = nn.Linear(input_size, down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj_mem = nn.Linear(down_size, input_size)

        # Cross-attention for prior injection
        self.cross_attn = nn.MultiheadAttention(...)
```

**Forward pass**:
```python
def forward(self, x, prior):
    # x: [n_ctx, num_unseen, dim] - Current unseen prompts (query)
    # prior: ([num_unseen, prior_len, dim], mask) - Prior knowledge (key/value)

    # Cross-attention: unseen prompts attend to prior knowledge
    attn_output = self.cross_attn(
        query=x,
        key=prior[0].permute(1, 0, 2),
        value=prior[0].permute(1, 0, 2),
        key_padding_mask=prior[1]
    )

    # Residual bottleneck
    down = self.non_linear_func(self.down_proj_mem(attn_output))
    up = self.up_proj_mem(down)
    return (up * self.scale) + x  # Residual connection
```

### Why This Works

1. **Text Embeddings**: Provide semantic context for what the unseen class means
2. **Similar Seen Prompts**: Offer learned visual-linguistic alignment patterns
3. **Cross-Attention**: Allows unseen prompts to selectively incorporate relevant knowledge

**Intuition**:
- Seen class "training dog" has learned how to align "training" action with dog images
- Unseen class "practicing skateboard" can borrow patterns from "training"
- Text embedding ensures "practicing" retains its unique semantic meaning

---

## Part 4: Integration in Forward Pass

### Text Encoding Flow

**In UPT model** (`upt_tip_cache_model_free_finetune_distillself.py`):

```python
class UPT(nn.Module):
    def __init__(self, ...):
        # Get initial text embeddings
        self.text_embedding, self.obj_embedding = get_origin_text_emb(
            args, clip_model, classnames, obj_class_names
        )

        # Initialize prompt learner
        self.prompt_learner = MultiModalPromptLearner(
            args, classnames, clip_model, object_class_to_target_class
        )

        # Text encoder with prompt injection
        self.text_encoder = TextEncoder(clip_model)
```

**In forward pass**:

```python
def forward(self, images, targets):
    # 1. Generate learnable prompts
    prompts, visual_prompts, text_prompts, ... = self.prompt_learner(
        txtcls_feat=self.text_embedding[select_HOI_index],
        select_HOI_index=select_HOI_index,
        unseen_text_priors=self.unseen_text_priors,  # Trigger refinement
        filenames=filenames
    )

    # 2. Encode with prompts
    text_features = self.text_encoder(
        prompts,
        tokenized_prompts,
        compound_prompts_deeper_text=text_prompts,
        txtcls_pt_list=txtcls_pt_list
    )

    # 3. Compute HOI predictions
    logits = image_features @ text_features.T  # Cosine similarity
```

---

## Summary Table

| Component | Input | Output | Purpose |
|-----------|-------|--------|---------|
| **`get_origin_text_emb()`** | Text descriptions | `[num_classes, 512/768]` | Initial CLIP text embeddings |
| **`self.ctx`** | N/A (learnable) | `[n_ctx, 512/768]` | Shared prompt prefix |
| **`self.compound_prompts_text`** | N/A (learnable) | List of `[n_ctx, 512/768]` | Deep layer prompts |
| **`self.txtcls_pt`** | Text embeddings | Adapted prompts | Class-specific prompt adaptation |
| **`self.img_clip_pt`** | Pre-extracted CLIP features | Adapted visual prompts | Image-guided prompt learning |
| **`self.unseen_pt_adapter`** | Seen prompts + text priors | Refined unseen prompts | Zero-shot knowledge transfer |

---

## Key Arguments

| Argument | Default | Effect |
|----------|---------|--------|
| `--use_templates` | False | Use multiple prompt templates and average |
| `--txtcls_pt` | False | Enable class-specific text prompt adaptation |
| `--img_clip_pt` | False | Enable image-guided visual prompt adaptation |
| `--unseen_pt_inj` | False | Enable unseen class prompt refinement |
| `--N_CTX` | 2 | Number of learnable context tokens |
| `--tune_LY` | 9 | Number of transformer layers with deep prompts |
| `--CTX_INIT` | "A photo of" | Initial prompt text (if n_ctx ≤ 4) |

---

## Example: Zero-Shot Verb Detection

**Scenario**: Train on 97 verbs, test on 20 unseen verbs

**Setup** (from `hico_text_label.py`):
```python
hico_unseen_index = {
    'unseen_verb': [1, 4, 7, 10, ...]  # 20 unseen verb indices
}
```

**Training**:
1. Text embeddings computed for all 117 verbs
2. Prompts learned only on 97 seen verbs
3. Unseen prompts initialized randomly

**Inference**:
1. Identify unseen verb indices
2. Find most similar seen verbs (cosine similarity in text embedding space)
3. For each unseen verb prompt:
   - **Query**: Current unseen prompt
   - **Prior**: [text embedding, similar seen prompt, unseen prompt]
   - **Refine**: Cross-attention + residual bottleneck
4. Use refined prompts for prediction

**Result**: Unseen verbs benefit from semantic similarity to seen verbs, improving zero-shot performance.

---

## Conclusion

EZ-HOI's text encoding and prompt learning combines:
1. **Static CLIP text embeddings** for semantic grounding
2. **Learnable multi-modal prompts** for vision-language alignment
3. **Guided prompt injection** for zero-shot transfer

This multi-stage approach enables effective zero-shot HOI detection by leveraging both pre-trained VLM knowledge and learned prompt patterns.
