# CLIP Feature Extraction for HICO-DET and V-COCO

This document explains the CLIP feature extraction scripts (`CLIP_hicodet_extract.py` and `CLIP_vcoco_extract.py`) used to pre-extract image features for the EZ-HOI framework.

## Purpose

These scripts pre-extract CLIP visual features from full images in the HICO-DET and V-COCO datasets. Pre-extraction serves two purposes:
1. **Performance**: Avoids redundant CLIP encoding during training (images are encoded once, loaded many times)
2. **Consistency**: Ensures all experiments use identical CLIP features

## Script Overview: `CLIP_hicodet_extract.py`

### Input Requirements

**1. Dataset Images**
- **Location**: `hicodet/hico_20160224_det/images/{mode}2015/`
  - `train2015/`: Training images
  - `test2015/`: Test images
- **Format**: JPEG images (e.g., `HICO_train2015_00000001.jpg`)
- **Total count**:
  - Train: 37,633 images
  - Test: 9,658 images

**2. Annotation Files**
- **Training**: `hicodet/trainval_hico.json`
- **Testing**: `hicodet/test_hico.json`

**JSON Structure**:
```json
[
  {
    "file_name": "HICO_train2015_00000001.jpg",
    "img_id": 1,
    "annotations": [
      {"bbox": [207, 32, 426, 299], "category_id": 1},
      {"bbox": [58, 97, 571, 404], "category_id": 4}
    ],
    "hoi_annotation": [
      {"subject_id": 0, "object_id": 1, "category_id": 73, "hoi_category_id": 153}
    ]
  },
  ...
]
```

**Note**: The script **only uses `file_name`** field to get the list of images. Annotations (bboxes, HOI labels) are **NOT used** during feature extraction.

**3. CLIP Model Checkpoints**
- **ViT-B/16**: `checkpoints/pretrained_CLIP/ViT-B-16.pt`
- **ViT-L/14@336px**: `checkpoints/pretrained_CLIP/ViT-L-14-336px.pt`

These are official OpenAI CLIP model weights.

### Processing Pipeline

#### Step 1: Load CLIP Model
```python
model, preprocess = clip.load(clip_mode, device)
```

**Two model variants**:
- `'ViT-B/16'`: ViT-Base with 16×16 patches, 224×224 input
- `'ViT-L/14@336px'`: ViT-Large with 14×14 patches, 336×336 input

`preprocess` is a transform pipeline that:
1. Resizes image to CLIP's input resolution
2. Center crops
3. Converts to tensor
4. Normalizes with CLIP's mean/std

#### Step 2: Process Each Image
```python
for idx, info_hoii in enumerate(hico_problems):
    filename = info_hoii['file_name']

    # Load and preprocess image
    image = preprocess(Image.open(os.path.join(img_path, filename)))
    image = image.unsqueeze(0).to(device)  # Add batch dimension

    # Extract features
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features = image_features.squeeze(0)[1:]
```

**Key operation: `[1:]` slicing**

CLIP Vision Transformer output structure:
- Input image → Conv2D patch embedding → Add [CLS] token → Transformer
- Output shape: `[batch_size, num_patches + 1, embed_dim]`
  - Index 0: **[CLS] token** (global image representation, typically used for classification)
  - Index 1+: **Patch tokens** (spatial features from image patches)

**For ViT-B/16 (224×224 input)**:
- Patch size: 16×16
- Grid size: 224÷16 = 14×14 = 196 patches
- Output: `[1, 197, 512]` (1 CLS + 196 patches, 512-dim)
- After `squeeze(0)[1:]`: `[196, 512]` → **Only patch tokens, excluding [CLS]**

**For ViT-L/14@336px (336×336 input)**:
- Patch size: 14×14
- Grid size: 336÷14 = 24×24 = 576 patches
- Output: `[1, 577, 768]` (1 CLS + 576 patches, 768-dim)
- After `squeeze(0)[1:]`: `[576, 768]` → **Only patch tokens**

**Why remove [CLS] token?**

The EZ-HOI framework processes **human-object union regions** extracted from images, not full images. The patch tokens preserve spatial information needed for:
1. Localizing features to specific image regions
2. Extracting union crop features via spatial indexing
3. Aligning with detection boxes

The [CLS] token is a global summary unsuitable for spatial operations.

#### Step 3: Save Features
```python
file = open(os.path.join(folder, filename.split(".")[0]+"_clip.pkl"), 'wb')
pickle.dump(image_features, file)
file.close()
```

Each image's features are saved as a **separate pickle file**.

### Output Structure

**Directory structure**:
```
hicodet_pkl_files/
├── clipbase_img_hicodet_train/      # ViT-B/16, train set
│   ├── HICO_train2015_00000001_clip.pkl
│   ├── HICO_train2015_00000002_clip.pkl
│   └── ... (37,633 files)
├── clipbase_img_hicodet_test/       # ViT-B/16, test set
│   ├── HICO_test2015_00000001_clip.pkl
│   └── ... (9,658 files)
├── clip336_img_hicodet_train/       # ViT-L/14@336px, train set
│   └── ... (37,633 files)
└── clip336_img_hicodet_test/        # ViT-L/14@336px, test set
    └── ... (9,658 files)
```

**Per-file content**:
```python
# Loading a single pickle file
import pickle
features = pickle.load(open('hicodet_pkl_files/clipbase_img_hicodet_train/HICO_train2015_00000001_clip.pkl', 'rb'))

# features is a torch.Tensor
print(features.shape)  # [196, 512] for ViT-B/16 or [576, 768] for ViT-L
print(features.dtype)  # torch.float32 (or float16 depending on CLIP model precision)
```

**File naming convention**:
- Original image: `HICO_train2015_00000001.jpg`
- Feature file: `HICO_train2015_00000001_clip.pkl`
- Pattern: `{image_basename}_clip.pkl`

### Storage Requirements

**ViT-B/16** (196 patches × 512 dim):
- Per image: 196 × 512 × 4 bytes (float32) = ~392 KB
- Train set: 37,633 × 392 KB ≈ **14.4 GB**
- Test set: 9,658 × 392 KB ≈ **3.7 GB**

**ViT-L/14@336px** (576 patches × 768 dim):
- Per image: 576 × 768 × 4 bytes = ~1.72 MB
- Train set: 37,633 × 1.72 MB ≈ **63 GB**
- Test set: 9,658 × 1.72 MB ≈ **16.2 GB**

**Total storage**: ~97 GB (for both models, both splits)

## Script Overview: `CLIP_vcoco_extract.py`

### Differences from HICO-DET Script

**1. Dataset Structure**
- **Image location**: `vcoco/mscoco2014/{mode}2014/`
  - `train2014/`: Training images (MSCOCO train split)
  - `val2014/`: Validation/test images (MSCOCO val split, but used as V-COCO test)
- **Annotation files**:
  - Train: `vcoco/instances_vcoco_trainval.json`
  - Test: `vcoco/instances_vcoco_test.json` (note: script uses val split for test)

**2. JSON Structure**
V-COCO uses MSCOCO format with nested structure:
```python
# Access pattern differs
for idx, info_hoii in enumerate(hico_problems['annotations']):
    filename = info_hoii['file_name']
```

Note the `['annotations']` key access - V-COCO wraps image metadata in COCO-style structure.

**3. Output Directories**
```
vcoco_pkl_files/
├── clipbase_img_vcoco_train/
├── clipbase_img_vcoco_val/      # Note: 'val' not 'test'
├── clip336_img_vcoco_train/
└── clip336_img_vcoco_val/
```

**4. Mode List**
```python
mode_list = ['train', 'val']  # Not ['train', 'test']
```

V-COCO uses MSCOCO val2014 images for its test set.

### Otherwise Identical Processing
- Same CLIP models (ViT-B/16, ViT-L/14@336px)
- Same preprocessing pipeline
- Same feature extraction (`[1:]` to remove [CLS])
- Same per-image pickle file storage

## Usage in EZ-HOI Training

### Loading During Training

The `DataFactory` class in `utils_tip_cache_and_union_finetune.py` loads these features:

```python
# In DataFactory.__getitem__
clip_img_file = 'hicodet_pkl_files/clipbase_img_hicodet_train'  # from args
filename = target['filename']  # e.g., 'HICO_train2015_00000001.jpg'

# Load pre-extracted CLIP features
feature_path = os.path.join(clip_img_file, filename.split(".")[0] + "_clip.pkl")
with open(feature_path, 'rb') as f:
    clip_features = pickle.load(f)  # [196, 512] or [576, 768]
```

### Integration with Model

These patch features are used in multiple ways:

**1. Global Image Context** (in `upt_tip_cache_model_free_finetune_distillself.py`):
```python
# Aggregate patch features for full image representation
image_features = clip_features.mean(dim=0)  # [embed_dim]
```

**2. Union Region Features**:
- Spatial crop/pooling from patch grid to get union box features
- Used for human-object interaction prediction

**3. Prompt Learning**:
- Passed to CLIP adapter modules for vision-language alignment
- Used in `--img_clip_pt` mode for learnable image prompts

## Common Issues and Solutions

### Issue 1: Path Hardcoding
**Problem**: Scripts have hardcoded paths
```python
img_path = 'hicodet/hico_20160224_det/images/' + mode+"2015"
```

**Solution**: Modify paths if your directory structure differs:
```python
# Change to your actual path
img_path = '/your/custom/path/hico_20160224_det/images/' + mode+"2015"
```

### Issue 2: CUDA Out of Memory
**Problem**: Loading full CLIP model on GPU with large images

**Solution**: Process in batches or use CPU:
```python
device = "cpu"  # Force CPU processing (slower but safer)
```

### Issue 3: Missing CLIP Package
**Problem**: `ModuleNotFoundError: No module named 'clip'`

**Solution**: Ensure using local CLIP:
```bash
export PYTHONPATH=$PYTHONPATH:"/path/to/EZ-HOI/CLIP"
cd /path/to/EZ-HOI
python CLIP_hicodet_extract.py
```

### Issue 4: Inconsistent Features
**Problem**: Features don't match between runs

**Cause**: Different CLIP model versions or preprocessing

**Solution**:
1. Always use the same CLIP checkpoint
2. Don't modify `clip.load()` preprocessing
3. Ensure `torch.backends.cudnn.deterministic = True` if reproducibility needed

## Performance Optimization

### Parallel Processing
Extract features faster with multiprocessing:

```python
from multiprocessing import Pool

def process_image(args):
    filename, img_path, model, preprocess, folder, device = args
    image = preprocess(Image.open(os.path.join(img_path, filename)))
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image).squeeze(0)[1:]

    save_path = os.path.join(folder, filename.split(".")[0]+"_clip.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(features.cpu(), f)  # Move to CPU before saving

# Use with Pool.map()
```

**Note**: CUDA models don't pickle well - use CPU for multiprocessing or process sequentially on GPU.

### Batch Processing
Process multiple images per forward pass:

```python
batch_size = 32
for i in range(0, len(hico_problems), batch_size):
    batch = hico_problems[i:i+batch_size]
    images = torch.stack([
        preprocess(Image.open(os.path.join(img_path, item['file_name'])))
        for item in batch
    ]).to(device)

    with torch.no_grad():
        features = model.encode_image(images)[:, 1:, :]  # [batch, patches, dim]

    for j, item in enumerate(batch):
        save_features(features[j], item['file_name'])
```

## Summary

| Aspect | HICO-DET | V-COCO |
|--------|----------|--------|
| **Script** | `CLIP_hicodet_extract.py` | `CLIP_vcoco_extract.py` |
| **Images** | 47,291 (37,633 train + 9,658 test) | Varies (MSCOCO subset) |
| **Input annotation** | `trainval_hico.json`, `test_hico.json` | `instances_vcoco_trainval.json`, `instances_vcoco_test.json` |
| **Image dir** | `hicodet/hico_20160224_det/images/` | `vcoco/mscoco2014/` |
| **CLIP models** | ViT-B/16, ViT-L/14@336px | ViT-B/16, ViT-L/14@336px |
| **Output shape (ViT-B)** | `[196, 512]` per image | `[196, 512]` per image |
| **Output shape (ViT-L)** | `[576, 768]` per image | `[576, 768]` per image |
| **Output dir** | `hicodet_pkl_files/{model}_img_hicodet_{split}/` | `vcoco_pkl_files/{model}_img_vcoco_{split}/` |
| **File naming** | `{image_name}_clip.pkl` | `{image_name}_clip.pkl` |
| **Storage (ViT-B)** | ~18 GB | Depends on subset size |
| **Storage (ViT-L)** | ~79 GB | Depends on subset size |

**Key Takeaway**: These scripts extract **spatial patch features** (not [CLS] tokens) from CLIP Vision Transformer for later use in HOI detection. The patch features preserve spatial information crucial for localizing human-object interactions in specific image regions.
