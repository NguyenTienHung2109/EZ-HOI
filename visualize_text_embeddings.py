"""
Visualize text and visual embeddings from trained EZ-HOI checkpoint.

This script:
1. Loads trained checkpoint and extracts text embeddings (600 HOI classes)
2. Loads cached CLIP visual features and creates visual prototypes per HOI:
   - Reads HICO annotations to map images -> HOI classes
   - Loads pre-extracted CLIP features from .pkl cache files
   - Pools features from all images containing each HOI interaction
   - Creates [600, D] visual embedding matrix
3. Applies dimensionality reduction (PCA, t-SNE, UMAP)
4. Creates 2 separate PNG visualizations:

   SEEN CLASSES (embeddings_*_seen.png):
   - Text embeddings: Blue circles
   - Visual embeddings: Green triangles
   - Gray dashed lines connecting text-visual pairs

   UNSEEN CLASSES (embeddings_*_unseen.png):
   - Text embeddings: Red circles
   - Visual embeddings: Orange triangles
   - Gray dashed lines connecting text-visual pairs

5. Supports sampling for cleaner visualizations

Usage:
    # Basic: visualize all 600 HOI classes with cached visual features
    python visualize_text_embeddings.py \
        --checkpoint checkpoints/hico_HO_pt_default_vitbase/ckpt_426660_12.pt \
        --zs_type unseen_verb \
        --visual_cache_dir hicodet_pkl_files/clipbase_img_hicodet_train \
        --annotation_file hicodet/trainval_hico.json

    # Sample 20 seen and 20 unseen HOI classes for cleaner visualization
    python visualize_text_embeddings.py \
        --zs_type unseen_verb \
        --num_seen_samples 20 \
        --num_unseen_samples 20

    # Use ViT-L visual features instead of ViT-B
    python visualize_text_embeddings.py \
        --visual_cache_dir hicodet_pkl_files/clip336_img_hicodet_train \
        --num_seen_samples 30 \
        --num_unseen_samples 15
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
from pathlib import Path
import pickle

# Add CLIP to path
sys.path.insert(0, os.path.join(os.getcwd(), 'CLIP'))

# Dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Optional: UMAP (needs to be installed separately)
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available. Install with: pip install umap-learn")

import clip
from hico_text_label import hico_text_label, hico_unseen_index, HICO_INTERACTIONS
from hico_list import hico_verbs, hico_objects, hico_verbs_sentence


def load_checkpoint_info(checkpoint_path):
    """Load checkpoint and extract configuration"""
    print(f"Loading checkpoint from: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract args if available
    args = checkpoint.get('args', None)

    print(f"\n{'='*60}")
    print("Checkpoint Information")
    print('='*60)

    if args:
        print(f"Dataset: {args.dataset}")
        print(f"Number of classes: {args.num_classes}")
        print(f"Zero-shot: {args.zs}")
        if args.zs:
            print(f"Zero-shot type: {args.zs_type}")
        print(f"Text alignment: {args.txt_align}")
        print(f"Text class prompt: {args.txtcls_pt}")
        print(f"CLIP model: {args.clip_dir_vit}")

    return checkpoint, args


def extract_text_embeddings_from_checkpoint(checkpoint, args=None, cli_args=None):
    """
    Extract text embeddings from checkpoint.

    Three approaches:
    1. If hoicls_txt is saved in checkpoint, use it directly
    2. If we have args (from checkpoint or CLI), recreate embeddings from CLIP
    3. Look for pre-computed embeddings in pkl files
    """
    print(f"\n{'='*60}")
    print("Extracting Text Embeddings")
    print('='*60)

    # Approach 1: Check if embeddings are saved in checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']

        # Look for text embeddings in state dict
        text_emb_keys = [k for k in state_dict.keys() if 'text' in k.lower() or 'hoicls' in k.lower()]
        print(f"\nFound {len(text_emb_keys)} text-related keys in checkpoint:")
        for key in text_emb_keys[:15]:  # Show first 15
            print(f"  - {key}: {state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'}")

        # Try to extract hoicls_txt if it exists as a buffer or parameter
        if 'hoicls_txt' in state_dict:
            print("\n✓ Found hoicls_txt in checkpoint")
            embeddings = state_dict['hoicls_txt']
            return embeddings.cpu().numpy()

    # Approach 2: Load from pre-computed embeddings or recreate
    print("\n⚠ hoicls_txt not found in checkpoint, will recreate from CLIP")

    # Use CLI args if checkpoint args not available
    if args is None and cli_args is None:
        raise ValueError(
            "Args not found in checkpoint and not provided via CLI.\n"
            "Please provide: --clip_model_path and --num_classes\n"
            "Example: python visualize_text_embeddings.py --clip_model_path checkpoints/pretrained_CLIP/ViT-B-16.pt --num_classes 600"
        )

    # Merge args: CLI args override checkpoint args
    if cli_args:
        num_classes = cli_args.num_classes
        clip_model_path = cli_args.clip_model_path
    elif args:
        num_classes = args.num_classes if hasattr(args, 'num_classes') else 600
        clip_model_path = args.clip_dir_vit if hasattr(args, 'clip_dir_vit') else None
    else:
        num_classes = 600
        clip_model_path = None

    print(f"\nConfiguration:")
    print(f"  Number of classes: {num_classes}")
    print(f"  CLIP model path: {clip_model_path}")

    # Get text descriptions
    if num_classes == 117:
        classnames = hico_verbs_sentence
        print(f"\nUsing 117 verb classes")
    else:
        classnames = list(hico_text_label.values())
        print(f"\nUsing {len(classnames)} HOI classes")

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if clip_model_path is None:
        print(f"\n⚠ CLIP model path not provided, using default ViT-B/16")
        clip_model, _ = clip.load("ViT-B/16", device=device)
    else:
        print(f"\nLoading CLIP model from: {clip_model_path}")
        if not os.path.exists(clip_model_path):
            print(f"⚠ CLIP model file not found: {clip_model_path}")
            print(f"  Falling back to default ViT-B/16")
            clip_model, _ = clip.load("ViT-B/16", device=device)
        else:
            try:
                # Try loading as JIT model
                clip_model = torch.jit.load(clip_model_path, map_location=device).eval()
            except:
                print(f"⚠ Failed to load as JIT model, trying as state dict...")
                try:
                    # Try loading as state dict
                    import CLIP_models_adapter_prior2
                    state_dict = torch.load(clip_model_path, map_location=device, weights_only=False)
                    if hasattr(state_dict, 'state_dict'):
                        state_dict = state_dict.state_dict()
                    clip_model = CLIP_models_adapter_prior2.build_model(state_dict)
                    clip_model = clip_model.to(device).eval()
                except Exception as e:
                    print(f"⚠ Failed to load CLIP model: {e}")
                    print(f"  Falling back to default ViT-B/16")
                    clip_model, _ = clip.load("ViT-B/16", device=device)

    print(f"✓ Loaded CLIP model")

    # Encode text
    with torch.no_grad():
        text_inputs = clip.tokenize(classnames).to(device)
        text_embeddings = clip_model.encode_text(text_inputs)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

    print(f"✓ Encoded {len(classnames)} text descriptions")
    print(f"  Embedding shape: {text_embeddings.shape}")
    print(f"  Embedding norm (mean ± std): {text_embeddings.norm(dim=-1).mean().item():.4f} ± {text_embeddings.norm(dim=-1).std().item():.4f}")

    return text_embeddings.cpu().numpy()


def get_class_labels(num_classes, zs_type=None):
    """Get text labels for HOI classes (600 only)"""
    # Full HOI mode (600 classes)
    labels = []
    for interaction in HICO_INTERACTIONS:
        action = interaction['action']
        obj = interaction['object']
        labels.append(f"{action}_{obj}")
    label_type = 'hoi'

    # Get unseen indices
    unseen_indices = []
    if zs_type and zs_type in hico_unseen_index:
        unseen_indices = hico_unseen_index[zs_type]

    return labels[:num_classes], label_type, unseen_indices


def reduce_dimensions(embeddings, method='pca', n_components=2, random_state=42):
    """Apply dimensionality reduction"""
    print(f"\n{'='*60}")
    print(f"Applying {method.upper()} Dimensionality Reduction")
    print('='*60)

    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=random_state)
        reduced = reducer.fit_transform(embeddings)
        print(f"✓ PCA completed")
        print(f"  Explained variance ratio: {reducer.explained_variance_ratio_}")
        print(f"  Total variance explained: {reducer.explained_variance_ratio_.sum():.4f}")

    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=random_state,
                      perplexity=30, n_iter=1000, verbose=1)
        reduced = reducer.fit_transform(embeddings)
        print(f"✓ t-SNE completed")

    elif method == 'umap':
        if not UMAP_AVAILABLE:
            raise ValueError("UMAP not available. Install with: pip install umap-learn")
        reducer = UMAP(n_components=n_components, random_state=random_state,
                      n_neighbors=15, min_dist=0.1)
        reduced = reducer.fit_transform(embeddings)
        print(f"✓ UMAP completed")

    else:
        raise ValueError(f"Unknown reduction method: {method}")

    return reduced


def extract_visual_embeddings_from_cache(cache_dir='hicodet_pkl_files/clipbase_img_hicodet_train',
                                         annotation_file='hicodet/trainval_hico.json',
                                         num_classes=600):
    """
    Extract visual embeddings from EZ-HOI cached CLIP features.

    Process:
    1. Load HICO annotations to map images -> HOI classes
    2. Load cached CLIP visual features for each image
    3. Pool features per HOI class to create visual prototypes
    """
    print(f"\n{'='*60}")
    print("Extracting Visual Embeddings from Cache Files")
    print('='*60)

    import json
    from collections import defaultdict

    # Check if cache directory exists
    if not os.path.exists(cache_dir):
        print(f"⚠ Cache directory not found: {cache_dir}")
        print("  Please run CLIP_hicodet_extract.py first to generate cache files")
        return None

    # Check if annotation file exists
    if not os.path.exists(annotation_file):
        print(f"⚠ Annotation file not found: {annotation_file}")
        return None

    print(f"\n✓ Cache directory: {cache_dir}")
    print(f"✓ Annotation file: {annotation_file}")

    # Step 1: Load annotations and create HOI -> images mapping
    print(f"\nLoading annotations...")
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    hoi_to_images = defaultdict(list)  # hoi_class_id -> [image filenames]

    for anno in annotations:
        filename = anno['file_name']
        for hoi in anno.get('hoi_annotation', []):
            hoi_class_id = hoi['hoi_category_id']
            if hoi_class_id < num_classes:
                hoi_to_images[hoi_class_id].append(filename)

    print(f"✓ Loaded {len(annotations)} images")
    print(f"✓ Found annotations for {len(hoi_to_images)} HOI classes")

    # Step 2: Load cached features and pool per HOI class
    print(f"\nLoading cached visual features and pooling per HOI class...")

    visual_embeddings = []
    feature_dim = None

    for hoi_id in range(num_classes):
        image_files = hoi_to_images.get(hoi_id, [])

        if len(image_files) == 0:
            # No training images for this HOI class - use zero vector
            if feature_dim is None:
                feature_dim = 512  # Default CLIP dimension
            visual_embeddings.append(np.zeros(feature_dim))
            continue

        # Load features from all images containing this HOI
        hoi_features = []
        for img_file in image_files[:50]:  # Limit to 50 images per HOI for efficiency
            cache_file = os.path.join(cache_dir, img_file.split('.')[0] + '_clip.pkl')

            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        img_feat = pickle.load(f)

                    # Features are patch tokens [N_patches, D] - take mean
                    if isinstance(img_feat, torch.Tensor):
                        img_feat = img_feat.cpu().numpy()

                    if len(img_feat.shape) == 2:
                        img_feat = img_feat.mean(axis=0)  # Pool patches

                    hoi_features.append(img_feat)

                    if feature_dim is None:
                        feature_dim = img_feat.shape[-1]

                except Exception as e:
                    continue

        # Pool features for this HOI class
        if len(hoi_features) > 0:
            hoi_proto = np.mean(hoi_features, axis=0)
            visual_embeddings.append(hoi_proto)
        else:
            if feature_dim is None:
                feature_dim = 512
            visual_embeddings.append(np.zeros(feature_dim))

        if (hoi_id + 1) % 100 == 0:
            print(f"  Processed {hoi_id + 1}/{num_classes} HOI classes...")

    visual_embeddings = np.array(visual_embeddings)

    print(f"\n✓ Created visual embeddings: {visual_embeddings.shape}")
    print(f"  Feature dimension: {feature_dim}")
    print(f"  Number of HOI classes: {num_classes}")

    # Normalize embeddings
    norms = np.linalg.norm(visual_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    visual_embeddings = visual_embeddings / norms

    print(f"✓ Normalized visual embeddings")

    return visual_embeddings


def plot_embeddings_2d_split(text_embeddings_2d, visual_embeddings_2d, labels,
                            unseen_indices, method_name, output_dir,
                            show_labels=True, zs_type=None):
    """
    Create separate visualizations for seen and unseen classes.

    Creates 2 PNG files:
    1. seen_classes.png - Text (blue) and Visual (green) embeddings for seen classes
    2. unseen_classes.png - Text (red) and Visual (orange) embeddings for unseen classes
    """

    print(f"\n>>> plot_embeddings_2d_split called")
    print(f"    Text embeddings 2D: {text_embeddings_2d.shape}")
    print(f"    Visual embeddings 2D: {visual_embeddings_2d.shape if visual_embeddings_2d is not None else 'None'}")
    print(f"    Will plot visual: {visual_embeddings_2d is not None}")

    # Separate seen and unseen indices
    seen_mask = np.ones(len(labels), dtype=bool)
    if unseen_indices:
        seen_mask[unseen_indices] = False

    seen_indices = np.where(seen_mask)[0]
    unseen_indices_arr = np.array(unseen_indices) if unseen_indices else np.array([])

    # Define colors
    text_seen_color = 'blue'
    visual_seen_color = 'green'
    text_unseen_color = 'red'
    visual_unseen_color = 'orange'

    # === 1. Plot SEEN classes ===
    if len(seen_indices) > 0:
        fig_seen, ax_seen = plt.subplots(figsize=(16, 12))

        # Plot text embeddings (seen)
        ax_seen.scatter(text_embeddings_2d[seen_indices, 0],
                       text_embeddings_2d[seen_indices, 1],
                       c=text_seen_color, alpha=0.7, s=100, marker='o',
                       label='Text Embeddings', edgecolors='black', linewidths=0.5)

        # Plot visual embeddings (seen) if available
        if visual_embeddings_2d is not None:
            ax_seen.scatter(visual_embeddings_2d[seen_indices, 0],
                           visual_embeddings_2d[seen_indices, 1],
                           c=visual_seen_color, alpha=0.7, s=100, marker='^',
                           label='Visual Embeddings', edgecolors='black', linewidths=0.5)

            # Draw connecting lines
            for i in seen_indices:
                ax_seen.plot([text_embeddings_2d[i, 0], visual_embeddings_2d[i, 0]],
                           [text_embeddings_2d[i, 1], visual_embeddings_2d[i, 1]],
                           color='gray', alpha=0.3, linewidth=1, linestyle='--')

        # Add labels if requested
        if show_labels:
            for idx in seen_indices:
                label = labels[idx]
                if len(label) > 20:
                    label = label[:17] + '...'
                ax_seen.annotate(label, (text_embeddings_2d[idx, 0], text_embeddings_2d[idx, 1]),
                               fontsize=8, alpha=0.8, color='darkblue')

        title = f'{method_name.upper()}: SEEN Classes'
        if zs_type:
            title += f' ({zs_type})'

        ax_seen.set_xlabel('Dimension 1', fontsize=14)
        ax_seen.set_ylabel('Dimension 2', fontsize=14)
        ax_seen.set_title(title, fontsize=16, fontweight='bold')
        ax_seen.legend(fontsize=12, loc='best')
        ax_seen.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save seen classes plot
        output_path_seen = output_dir / f'embeddings_{method_name}_seen.png'
        plt.savefig(output_path_seen, dpi=300, bbox_inches='tight')
        print(f"✓ Saved SEEN classes to: {output_path_seen}")
        plt.close(fig_seen)

    # === 2. Plot UNSEEN classes ===
    if len(unseen_indices_arr) > 0:
        fig_unseen, ax_unseen = plt.subplots(figsize=(16, 12))

        # Plot text embeddings (unseen)
        ax_unseen.scatter(text_embeddings_2d[unseen_indices_arr, 0],
                         text_embeddings_2d[unseen_indices_arr, 1],
                         c=text_unseen_color, alpha=0.7, s=100, marker='o',
                         label='Text Embeddings', edgecolors='black', linewidths=0.5)

        # Plot visual embeddings (unseen) if available
        if visual_embeddings_2d is not None:
            ax_unseen.scatter(visual_embeddings_2d[unseen_indices_arr, 0],
                             visual_embeddings_2d[unseen_indices_arr, 1],
                             c=visual_unseen_color, alpha=0.7, s=100, marker='^',
                             label='Visual Embeddings', edgecolors='black', linewidths=0.5)

            # Draw connecting lines
            for i in unseen_indices_arr:
                ax_unseen.plot([text_embeddings_2d[i, 0], visual_embeddings_2d[i, 0]],
                              [text_embeddings_2d[i, 1], visual_embeddings_2d[i, 1]],
                              color='gray', alpha=0.3, linewidth=1, linestyle='--')

        # Add labels if requested
        if show_labels:
            for idx in unseen_indices_arr:
                label = labels[idx]
                if len(label) > 20:
                    label = label[:17] + '...'
                ax_unseen.annotate(label, (text_embeddings_2d[idx, 0], text_embeddings_2d[idx, 1]),
                                  fontsize=8, alpha=0.8, color='darkred')

        title = f'{method_name.upper()}: UNSEEN Classes'
        if zs_type:
            title += f' ({zs_type})'

        ax_unseen.set_xlabel('Dimension 1', fontsize=14)
        ax_unseen.set_ylabel('Dimension 2', fontsize=14)
        ax_unseen.set_title(title, fontsize=16, fontweight='bold')
        ax_unseen.legend(fontsize=12, loc='best')
        ax_unseen.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save unseen classes plot
        output_path_unseen = output_dir / f'embeddings_{method_name}_unseen.png'
        plt.savefig(output_path_unseen, dpi=300, bbox_inches='tight')
        print(f"✓ Saved UNSEEN classes to: {output_path_unseen}")
        plt.close(fig_unseen)

    print(f"✓ Created {len(seen_indices)} seen + {len(unseen_indices_arr)} unseen visualizations")


def print_embedding_analysis(embeddings, labels, unseen_indices=None, top_k=10):
    """Print detailed analysis of embeddings"""
    print(f"\n{'='*60}")
    print("Embedding Analysis")
    print('='*60)

    # Basic statistics
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\nEmbedding Statistics:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Norms - Mean: {norms.mean():.4f}, Std: {norms.std():.4f}")
    print(f"  Norms - Min: {norms.min():.4f}, Max: {norms.max():.4f}")

    # Compute pairwise similarities
    embeddings_normalized = embeddings / norms[:, np.newaxis]
    similarity_matrix = embeddings_normalized @ embeddings_normalized.T

    # Get upper triangle (excluding diagonal)
    triu_indices = np.triu_indices_from(similarity_matrix, k=1)
    similarities = similarity_matrix[triu_indices]

    print(f"\nPairwise Cosine Similarities:")
    print(f"  Mean: {similarities.mean():.4f}, Std: {similarities.std():.4f}")
    print(f"  Min: {similarities.min():.4f}, Max: {similarities.max():.4f}")

    # Find most similar pairs
    print(f"\nTop {top_k} Most Similar Class Pairs:")
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    for rank, idx in enumerate(top_indices, 1):
        i, j = triu_indices[0][idx], triu_indices[1][idx]
        sim = similarities[idx]
        print(f"  {rank}. {labels[i]} <-> {labels[j]}: {sim:.4f}")

    # Find most dissimilar pairs
    print(f"\nTop {top_k} Most Dissimilar Class Pairs:")
    bottom_indices = np.argsort(similarities)[:top_k]
    for rank, idx in enumerate(bottom_indices, 1):
        i, j = triu_indices[0][idx], triu_indices[1][idx]
        sim = similarities[idx]
        print(f"  {rank}. {labels[i]} <-> {labels[j]}: {sim:.4f}")

    # If unseen classes exist, analyze seen vs unseen
    if unseen_indices and len(unseen_indices) > 0:
        print(f"\nSeen vs Unseen Analysis:")
        seen_mask = np.ones(len(labels), dtype=bool)
        seen_mask[unseen_indices] = False

        print(f"  Number of seen classes: {seen_mask.sum()}")
        print(f"  Number of unseen classes: {len(unseen_indices)}")

        # Average similarity within seen, within unseen, and across
        seen_indices = np.where(seen_mask)[0]
        unseen_indices_arr = np.array(unseen_indices)

        # Within seen
        seen_pairs = []
        for i in range(len(seen_indices)):
            for j in range(i+1, len(seen_indices)):
                seen_pairs.append(similarity_matrix[seen_indices[i], seen_indices[j]])

        # Within unseen
        unseen_pairs = []
        for i in range(len(unseen_indices_arr)):
            for j in range(i+1, len(unseen_indices_arr)):
                unseen_pairs.append(similarity_matrix[unseen_indices_arr[i], unseen_indices_arr[j]])

        # Across seen-unseen
        cross_pairs = []
        for i in seen_indices:
            for j in unseen_indices_arr:
                cross_pairs.append(similarity_matrix[i, j])

        if seen_pairs:
            print(f"  Avg similarity within seen: {np.mean(seen_pairs):.4f}")
        if unseen_pairs:
            print(f"  Avg similarity within unseen: {np.mean(unseen_pairs):.4f}")
        if cross_pairs:
            print(f"  Avg similarity seen-unseen: {np.mean(cross_pairs):.4f}")


def sample_classes(embeddings, labels, unseen_indices, num_seen=None, num_unseen=None, seed=42):
    """
    Sample a subset of seen and unseen classes for visualization.

    Args:
        embeddings: Full embedding matrix [N, D]
        labels: List of class labels
        unseen_indices: List of unseen class indices
        num_seen: Number of seen classes to sample (None = all)
        num_unseen: Number of unseen classes to sample (None = all)
        seed: Random seed for reproducibility

    Returns:
        sampled_embeddings, sampled_labels, sampled_unseen_indices, selected_indices
    """
    np.random.seed(seed)

    # Create masks
    seen_mask = np.ones(len(labels), dtype=bool)
    if unseen_indices and len(unseen_indices) > 0:
        seen_mask[unseen_indices] = False

    seen_indices = np.where(seen_mask)[0]
    unseen_indices_arr = np.array(unseen_indices) if unseen_indices else np.array([])

    # Sample seen classes
    if num_seen is not None and len(seen_indices) > num_seen:
        sampled_seen = np.random.choice(seen_indices, size=num_seen, replace=False)
        print(f"  Sampled {num_seen} seen classes from {len(seen_indices)} total")
    else:
        sampled_seen = seen_indices
        print(f"  Using all {len(seen_indices)} seen classes")

    # Sample unseen classes
    if num_unseen is not None and len(unseen_indices_arr) > num_unseen:
        sampled_unseen = np.random.choice(unseen_indices_arr, size=num_unseen, replace=False)
        print(f"  Sampled {num_unseen} unseen classes from {len(unseen_indices_arr)} total")
    else:
        sampled_unseen = unseen_indices_arr
        if len(unseen_indices_arr) > 0:
            print(f"  Using all {len(unseen_indices_arr)} unseen classes")

    # Combine selected indices
    selected_indices = np.concatenate([sampled_seen, sampled_unseen])
    selected_indices = np.sort(selected_indices)

    # Extract sampled data
    sampled_embeddings = embeddings[selected_indices]
    sampled_labels = [labels[i] for i in selected_indices]

    # Create new unseen indices mapping for sampled data
    sampled_unseen_indices = []
    for new_idx, old_idx in enumerate(selected_indices):
        if old_idx in sampled_unseen:
            sampled_unseen_indices.append(new_idx)

    print(f"\n✓ Total classes selected: {len(selected_indices)}")
    print(f"  - Seen: {len(sampled_seen)}")
    print(f"  - Unseen: {len(sampled_unseen)}")

    return sampled_embeddings, sampled_labels, sampled_unseen_indices, selected_indices


def print_all_embeddings(embeddings, labels, max_display=50):
    """Print formatted text labels with their embedding vectors"""
    print(f"\n{'='*60}")
    print("Text Labels and Embeddings")
    print('='*60)

    num_to_display = min(max_display, len(labels))
    print(f"\nShowing first {num_to_display} of {len(labels)} classes:\n")

    for i in range(num_to_display):
        print(f"[{i:3d}] {labels[i]}")
        print(f"      Embedding: {embeddings[i][:10]}... (showing first 10 dims)")
        print(f"      Norm: {np.linalg.norm(embeddings[i]):.4f}\n")

    if len(labels) > max_display:
        print(f"... ({len(labels) - max_display} more classes not shown)")
        print(f"\nLast class:")
        i = len(labels) - 1
        print(f"[{i:3d}] {labels[i]}")
        print(f"      Embedding: {embeddings[i][:10]}... (showing first 10 dims)")
        print(f"      Norm: {np.linalg.norm(embeddings[i]):.4f}")


def main():
    parser = argparse.ArgumentParser(description='Visualize text embeddings from EZ-HOI checkpoint')
    parser.add_argument('--checkpoint', type=str,
                       default='checkpoints/hico_HO_pt_default_vitbase/ckpt_426660_12.pt',
                       help='Path to checkpoint file')
    parser.add_argument('--reduction', type=str, default='all',
                       choices=['pca', 'tsne', 'umap', 'all'],
                       help='Dimensionality reduction method')
    parser.add_argument('--num_classes', type=int, default=600,
                       help='Number of HOI classes (600 for full HOI combinations)')
    parser.add_argument('--clip_model_path', type=str, default=None,
                       help='Path to CLIP model (e.g., checkpoints/pretrained_CLIP/ViT-B-16.pt). If None, uses default ViT-B/16')
    parser.add_argument('--visual_cache_dir', type=str, default='hicodet_pkl_files/clipbase_img_hicodet_train',
                       help='Directory containing cached CLIP visual features')
    parser.add_argument('--annotation_file', type=str, default='hicodet/trainval_hico.json',
                       help='HICO annotation file for mapping images to HOI classes')
    parser.add_argument('--zs_type', type=str, default=None,
                       choices=['unseen_verb', 'unseen_object', 'rare_first', 'non_rare_first', 'default', None],
                       help='Zero-shot split type (if applicable)')
    parser.add_argument('--num_seen_samples', type=int, default=None,
                       help='Number of seen classes to sample for visualization (e.g., 20). If None, shows all.')
    parser.add_argument('--num_unseen_samples', type=int, default=None,
                       help='Number of unseen classes to sample for visualization (e.g., 20). If None, shows all.')
    parser.add_argument('--sample_seed', type=int, default=42,
                       help='Random seed for class sampling (for reproducibility)')
    parser.add_argument('--show_labels', action='store_true', default=True,
                       help='Show text labels on plots')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--compare_raw', action='store_true',
                       help='Also visualize raw CLIP embeddings for comparison')
    parser.add_argument('--print_embeddings', action='store_true',
                       help='Print all embedding vectors')
    parser.add_argument('--max_display', type=int, default=50,
                       help='Maximum number of embeddings to print')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load checkpoint
    checkpoint, ckpt_args = load_checkpoint_info(args.checkpoint)

    # Determine number of classes
    num_classes = args.num_classes

    # Get zero-shot type
    zs_type = args.zs_type
    if zs_type is None and ckpt_args:
        if hasattr(ckpt_args, 'zs') and ckpt_args.zs:
            zs_type = ckpt_args.zs_type if hasattr(ckpt_args, 'zs_type') else None

    # Extract text embeddings - pass both checkpoint args and CLI args
    text_embeddings = extract_text_embeddings_from_checkpoint(checkpoint, ckpt_args, args)

    # Extract visual embeddings from cache files
    visual_embeddings = extract_visual_embeddings_from_cache(
        cache_dir=args.visual_cache_dir,
        annotation_file=args.annotation_file,
        num_classes=num_classes
    )

    # Check if visual embeddings were loaded
    if visual_embeddings is not None:
        print(f"\n✓ Visual embeddings loaded successfully: {visual_embeddings.shape}")
    else:
        print(f"\n⚠ Warning: Visual embeddings not loaded - only text embeddings will be visualized")
        print(f"  Check that:")
        print(f"    1. Cache directory exists: {args.visual_cache_dir}")
        print(f"    2. Annotation file exists: {args.annotation_file}")
        print(f"    3. Cache files (.pkl) are present in the directory")

    # Get labels
    labels, label_type, unseen_indices = get_class_labels(num_classes, zs_type)

    # Use text embeddings as default
    embeddings = text_embeddings

    # Ensure embeddings match number of classes
    if len(embeddings) != num_classes:
        print(f"Warning: Embedding count ({len(embeddings)}) != num_classes ({num_classes})")
        embeddings = embeddings[:num_classes]

    # Sample classes if requested
    if args.num_seen_samples is not None or args.num_unseen_samples is not None:
        print(f"\n{'='*60}")
        print("Sampling Classes for Visualization")
        print('='*60)

        text_embeddings, labels, unseen_indices, selected_indices = sample_classes(
            text_embeddings, labels, unseen_indices,
            num_seen=args.num_seen_samples,
            num_unseen=args.num_unseen_samples,
            seed=args.sample_seed
        )

        # Also sample visual embeddings if available
        if visual_embeddings is not None:
            print(f"  Before sampling - visual embeddings shape: {visual_embeddings.shape}")
            visual_embeddings = visual_embeddings[selected_indices]
            print(f"✓ After sampling - visual embeddings shape: {visual_embeddings.shape}")

        print(f"\nSelected class examples:")
        for i in range(min(5, len(labels))):
            color = "unseen" if i in unseen_indices else "seen"
            print(f"  [{i}] {labels[i]} ({color})")
        if len(labels) > 5:
            print(f"  ... and {len(labels) - 5} more classes")

    # Update embeddings reference
    embeddings = text_embeddings

    # Print embeddings if requested
    if args.print_embeddings:
        print_all_embeddings(embeddings, labels, args.max_display)

    # Perform embedding analysis
    print_embedding_analysis(embeddings, labels, unseen_indices)

    # Determine which reduction methods to use
    if args.reduction == 'all':
        methods = ['pca', 'tsne']
        if UMAP_AVAILABLE:
            methods.append('umap')
    else:
        methods = [args.reduction]

    # Apply dimensionality reduction and visualize
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Creating {method.upper()} Visualization (Split by Seen/Unseen)")
        print('='*60)

        # Reduce text embeddings
        text_embeddings_2d = reduce_dimensions(text_embeddings, method=method)

        # Reduce visual embeddings if available
        visual_embeddings_2d = None
        if visual_embeddings is not None:
            print(f"\nReducing visual embeddings with {method.upper()}...")
            print(f"  Visual embeddings shape before reduction: {visual_embeddings.shape}")
            visual_embeddings_2d = reduce_dimensions(visual_embeddings, method=method)
            print(f"  Visual embeddings 2D shape: {visual_embeddings_2d.shape}")
        else:
            print(f"\n⚠ No visual embeddings to reduce - skipping visual plotting")

        # Create split visualizations (seen and unseen separately)
        plot_embeddings_2d_split(
            text_embeddings_2d=text_embeddings_2d,
            visual_embeddings_2d=visual_embeddings_2d,
            labels=labels,
            unseen_indices=unseen_indices,
            method_name=method,
            output_dir=output_dir,
            show_labels=args.show_labels,
            zs_type=zs_type
        )

    # Summary
    print(f"\n{'='*60}")
    print("Visualization Complete!")
    print('='*60)
    print(f"✓ All visualizations saved to: {output_dir}")
    print(f"\nGenerated files:")
    for method in methods:
        print(f"  - embeddings_{method}_seen.png")
        print(f"  - embeddings_{method}_unseen.png")


if __name__ == '__main__':
    main()
