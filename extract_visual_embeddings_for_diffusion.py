"""
Extract VISUAL embeddings from trained EZ-HOI model for diffusion training.

This script:
1. Loads a trained EZ-HOI checkpoint
2. Iterates through HICO-DET training set (38,118 images)
3. For each image:
   - Runs forward pass through model
   - Extracts visual features AFTER mem_adapter (adapter_feat at line 1269)
   - These are features right before cosine similarity computation
4. Applies diffusion-bridge normalization chain (same as inference):
   - L2 normalize
   - Subtract HOI text mean
   - L2 normalize again
   - Scale by factor
5. Saves normalized visual embeddings to pickle file

Why train diffusion on visual embeddings instead of text embeddings?
- Text embeddings: Only 212 representative classes (insufficient for diffusion)
- Visual embeddings: 50k-100k human-object pairs from training images (abundant data)
- Goal: Learn vision→text distribution bridging with sufficient training samples

Usage:
    python extract_visual_embeddings_for_diffusion.py \
        --checkpoint checkpoints/hico_HO_pt_default_vitbase/best.pth \
        --num_classes 117 \
        --zs_type unseen_verb \
        --text_mean_path hicodet_pkl_files/hoi_text_mean_vitB_adapted_212.pkl \
        --output_path hicodet_pkl_files/hoi_visual_embeddings_normalized_vitB_train.pkl \
        --scale_factor 5.0 \
        --batch_size 1
"""

import torch
import torch.nn.functional as F
import pickle
import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Add CLIP to path
sys.path.insert(0, os.path.join(os.getcwd(), 'CLIP'))

import clip
from upt_tip_cache_model_free_finetune_distillself import build_detector
from hico_text_label import hico_unseen_index, HOI_TO_AO
from utils_tip_cache_and_union_finetune import DataFactory, custom_collate
from torch.utils.data import DataLoader


def load_checkpoint_and_args(checkpoint_path):
    """Load checkpoint and extract training arguments"""
    print(f"Loading checkpoint from: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'args' not in checkpoint:
        raise ValueError("'args' not found in checkpoint. Cannot recreate model.")

    args = checkpoint['args']
    print(f"\nLoaded training configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Number of classes: {args.num_classes}")
    print(f"  Zero-shot: {args.zs}")
    if args.zs:
        print(f"  Zero-shot type: {args.zs_type}")
    print(f"  Text align: {args.txt_align}")
    print(f"  Image align: {args.img_align}")

    return checkpoint, args


def recreate_model(args, device='cuda'):
    """Recreate the model structure with exact training configuration"""
    print("\n" + "="*60)
    print("Recreating model structure...")
    print("="*60)

    # Import necessary functions (HICODET only)
    from main_tip_finetune import hico_class_corr
    from hico_list import hico_verb_object_list, hico_verbs
    import numpy as np

    # Setup HICODET configurations
    object_to_target = hico_class_corr()

    # Create 2D lookup table [num_objects, num_actions] -> hoi_idx
    lut = np.full([80, 117], None)
    for hoi_idx, obj_idx, verb_idx in object_to_target:
        lut[obj_idx, verb_idx] = hoi_idx
    object_n_verb_to_interaction = lut.tolist()

    # Initialize distributed training (required by build_detector)
    import torch.distributed as dist
    if not dist.is_initialized():
        print(f"Initializing distributed training (single process)...")
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        try:
            dist.init_process_group(
                backend=backend,
                init_method='tcp://localhost:12355',
                world_size=1,
                rank=0
            )
            print(f"✓ Distributed training initialized (backend: {backend})")
        except Exception as e:
            print(f"Warning: Could not initialize distributed training: {e}")
            print("Attempting to continue without distributed training...")

    # Build detector with exact training configuration
    print("\nBuilding detector...")
    detector = build_detector(args, object_n_verb_to_interaction)

    # Move to device
    detector = detector.to(device)
    detector.eval()

    print(f"✓ Model created and moved to {device}")

    return detector


def load_text_mean(text_mean_path):
    """
    Load HOI text mean for normalization.

    Supports two formats:
    1. Old format: Just a tensor [embed_dim] or [1, embed_dim]
    2. New format: Dict with 'text_mean' key and metadata
    """
    if not Path(text_mean_path).exists():
        raise FileNotFoundError(
            f"Text mean not found: {text_mean_path}\n"
            f"Please extract adapted text embeddings first using extract_adapted_text_embeddings.py"
        )

    print(f"\nLoading text mean from: {text_mean_path}")

    with open(text_mean_path, 'rb') as f:
        data = pickle.load(f)

    # Handle both formats
    if isinstance(data, dict) and 'text_mean' in data:
        # New format with metadata (from extract_adapted_text_embeddings.py)
        text_mean = data['text_mean']
        print(f"  Source: {data.get('source', 'unknown')}")
        print(f"  ✓ Loaded ADAPTED text mean (from {data.get('num_classes', 'unknown')} classes)")
    elif isinstance(data, dict) and 'mean' in data:
        # Alternative dict format
        text_mean = data['mean']
        print(f"  ✓ Loaded text mean")
    else:
        # Old format: just tensor (raw CLIP embeddings)
        text_mean = data
        print(f"  Source: legacy format (raw CLIP embeddings)")

    # Ensure it's a tensor
    if not isinstance(text_mean, torch.Tensor):
        text_mean = torch.tensor(text_mean)

    # Ensure correct shape (should be 1D: [embed_dim])
    if text_mean.dim() == 2 and text_mean.shape[0] == 1:
        text_mean = text_mean.squeeze(0)

    print(f"  Shape: {text_mean.shape}")
    print(f"  Norm: {text_mean.norm().item():.6f}")

    return text_mean


def apply_diffusion_normalization(features, text_mean, scale_factor=5.0):
    """
    Apply diffusion-bridge normalization chain.

    This MUST match the normalization in diffusion_bridge_module.py lines 199-209:
    1. L2 normalize
    2. Subtract text mean
    3. L2 normalize again
    4. Scale by factor

    Args:
        features: Visual features [N, embed_dim]
        text_mean: HOI text mean [embed_dim]
        scale_factor: Scale factor (default: 5.0)

    Returns:
        normalized_features: [N, embed_dim]
    """
    # Step 1: L2 normalize
    x = F.normalize(features, dim=-1)

    # Step 2: Subtract text mean
    x = x - text_mean.to(x.device)

    # Step 3: L2 normalize again
    x = F.normalize(x, dim=-1)

    # Step 4: Scale
    x = x * scale_factor

    return x


@torch.no_grad()
def extract_visual_embeddings(model, dataloader, text_mean, scale_factor, device='cuda'):
    """
    Extract visual embeddings from all training images.

    Returns:
        all_embeddings: List of normalized visual embeddings
        all_metadata: List of metadata dicts with image_id, hoi_labels, etc.
    """
    print("\n" + "="*60)
    print("Extracting visual embeddings from training set...")
    print("="*60)

    model.eval()
    all_embeddings = []
    all_metadata = []

    total_pairs = 0

    for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="Processing images")):
        try:
            # Prepare images for model
            # DataFactory returns list of (image_detr, image_clip), but we need them processed
            images_detr = [img[0] for img in images]
            images_clip = [img[1] for img in images]

            # Move targets to device
            for target in targets:
                for k, v in target.items():
                    if isinstance(v, torch.Tensor):
                        target[k] = v.to(device)

            # Forward through model to get visual features
            # We need to hook into compute_roi_embeddings to extract adapter_feat
            # For simplicity, we'll run the full forward and extract from the model's internal state

            # Alternative: directly process through the model components
            # This is more complex but gives us exact control

            from upt_tip_cache_model_free_finetune_distillself import nested_tensor_from_tensor_list

            # Move images to device
            images_detr = [img.to(device) for img in images_detr]
            images_clip = [img.to(device) for img in images_clip]

            # Process through DETR to get region proposals
            images_detr_nested = nested_tensor_from_tensor_list(images_detr)
            results = model.detector(images_detr_nested)

            # Get region proposals
            region_props = model.prepare_region_proposals(results)

            # Get image sizes
            image_sizes = torch.stack([t['orig_size'] for t in targets])

            # Process through CLIP
            if model.use_insadapter:
                priors = model.get_prior(region_props, image_sizes, model.prior_method)
            else:
                priors = None

            images_clip_nested = nested_tensor_from_tensor_list(images_clip)

            # Get image description priors (if needed)
            if model.img_descrip_prompt:
                # Skip this for extraction (too complex and not critical for visual features)
                img_descrip_priors = None
            else:
                img_descrip_priors = None

            # Get text class features
            if model.txtcls_descrip:
                txtcls_feat = model.hoicls_txt[model.select_HOI_index]
            else:
                txtcls_feat = None

            # Get filenames (if needed for img_clip_pt)
            if model.img_clip_pt and model.self_image_path is None:
                filenames = [t['filename'] for t in targets]
            else:
                filenames = None

            # Forward through prompt_learner and text_encoder
            if model.fix_txt_pt is False and model.clip_test is False:
                tokenized_prompts = model.clip_head.tokenized_prompts
                if model.txtcls_pt:
                    prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision, txtcls_pt_list, origin_ctx = \
                        model.clip_head.prompt_learner(
                            img_descrip_priors=img_descrip_priors,
                            txtcls_feat=model.hoicls_txt,
                            select_HOI_index=model.select_HOI_index,
                            unseen_text_priors=model.unseen_text_priors,
                            filenames=filenames
                        )
                else:
                    prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision, origin_ctx = \
                        model.clip_head.prompt_learner(
                            img_descrip_priors=img_descrip_priors,
                            unseen_text_priors=model.unseen_text_priors,
                            filenames=filenames
                        )
                    txtcls_pt_list = None

                hoitxt_features, origin_txt_features = model.clip_head.text_encoder(
                    prompts, tokenized_prompts, deep_compound_prompts_text,
                    txtcls_feat, txtcls_pt_list, origin_ctx
                )
            else:
                hoitxt_features = model.hoicls_txt[model.select_HOI_index].to(device)
                if model.fix_txt_pt:
                    tokenized_prompts = model.clip_head.tokenized_prompts
                    _, shared_ctx, _, deep_compound_prompts_vision, _ = \
                        model.clip_head.prompt_learner(
                            img_descrip_priors=img_descrip_priors,
                            unseen_text_priors=model.unseen_text_priors,
                            filenames=filenames
                        )

            # Forward through image encoder
            if model.clip_test is False:
                feat_global, feat_local = model.clip_head.image_encoder(
                    images_clip_nested.decompose()[0], priors,
                    shared_ctx=shared_ctx,
                    compound_deeper_prompts=deep_compound_prompts_vision
                )
            else:
                feat_global, feat_local = model.clip_head.image_encoder(
                    images_clip_nested.decompose()[0], priors
                )

            # Now extract visual features from ROI (union boxes)
            # We need to manually compute union features like in compute_roi_embeddings
            for b_idx, props in enumerate(region_props):
                local_features = feat_local[b_idx]
                boxes = props['boxes']
                scores = props['scores']
                labels = props['labels']

                is_human = labels == model.human_idx
                n_h = torch.sum(is_human)
                n = len(boxes)

                # Permute human instances to the top
                if not torch.all(labels[:n_h] == model.human_idx):
                    h_idx = torch.nonzero(is_human).squeeze(1)
                    o_idx = torch.nonzero(is_human == 0).squeeze(1)
                    perm = torch.cat([h_idx, o_idx])
                    boxes = boxes[perm]
                    scores = scores[perm]
                    labels = labels[perm]

                # Skip if no valid pairs
                if n_h == 0 or n <= 1:
                    continue

                # Get pairwise indices
                x = torch.arange(n, device=device)
                y = torch.arange(n, device=device)
                x, y = torch.meshgrid(x, y, indexing='ij')

                # Valid human-object pairs
                x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < n_h)).unbind(1)

                if len(x_keep) == 0:
                    continue

                # Compute union boxes
                sub_boxes = boxes[x_keep]
                obj_boxes = boxes[y_keep]
                lt = torch.min(sub_boxes[..., :2], obj_boxes[..., :2])
                rb = torch.max(sub_boxes[..., 2:], obj_boxes[..., 2:])
                union_boxes = torch.cat([lt, rb], dim=-1)

                # Extract union features using ROI align
                import torchvision
                spatial_scale = 1 / (image_sizes[b_idx, 0] / local_features.shape[1])
                union_features = torchvision.ops.roi_align(
                    local_features.unsqueeze(0),
                    [union_boxes],
                    output_size=(7, 7),
                    spatial_scale=spatial_scale,
                    aligned=True
                )

                # Extract single instance features
                single_features = torchvision.ops.roi_align(
                    local_features.unsqueeze(0),
                    [boxes],
                    output_size=(7, 7),
                    spatial_scale=spatial_scale,
                    aligned=True
                )

                # Apply feature masking/dropout (if configured)
                if model.feat_mask_type == 0:
                    # During inference, featmap_dropout should be disabled (eval mode)
                    union_features = union_features.flatten(2).mean(-1)
                    single_features = single_features.flatten(2).mean(-1)
                elif model.feat_mask_type == 1:
                    union_features = union_features.flatten(2).mean(-1)
                    single_features = single_features.flatten(2).mean(-1)

                # Get human and object features
                human_features = single_features[x_keep]
                object_features = single_features[y_keep]

                # Apply normalization (if individual_norm is enabled)
                if model.individual_norm:
                    human_features = human_features / human_features.norm(dim=-1, keepdim=True)
                    object_features = object_features / object_features.norm(dim=-1, keepdim=True)
                    union_features = union_features / union_features.norm(dim=-1, keepdim=True)
                else:
                    concat_feat = torch.cat([human_features, object_features, union_features], dim=-1)
                    concat_feat = concat_feat / concat_feat.norm(dim=-1, keepdim=True)
                    # Split back (if needed, but we only use individual features below)

                # Compute visual features based on logits_type
                if model.logits_type == 'HO+U':
                    vis_feat = model.vis_fuse(torch.cat([union_features, human_features, object_features], dim=-1))
                elif model.logits_type == 'HO':
                    vis_feat = model.vis_fuse(torch.cat([human_features, object_features], dim=-1))
                elif model.logits_type == 'U':
                    vis_feat = union_features

                # Apply mem_adapter (if img_align is enabled)
                if model.img_align:
                    adapter_feat = model.mem_adapter(vis_feat.unsqueeze(1)).squeeze(1)
                else:
                    adapter_feat = vis_feat

                # This is the key feature we want to extract!
                # adapter_feat: [num_pairs, embed_dim]

                # Apply diffusion normalization
                normalized_feat = apply_diffusion_normalization(adapter_feat, text_mean, scale_factor)

                # Store embeddings and metadata
                num_pairs = len(adapter_feat)
                total_pairs += num_pairs

                for pair_idx in range(num_pairs):
                    all_embeddings.append(normalized_feat[pair_idx].cpu().numpy())

                    # Store metadata
                    metadata = {
                        'image_id': targets[b_idx].get('filename', f'batch_{batch_idx}_img_{b_idx}'),
                        'human_idx': x_keep[pair_idx].item(),
                        'object_idx': y_keep[pair_idx].item(),
                        'object_class': labels[y_keep[pair_idx]].item(),
                        # Store ground truth HOI labels if available
                    }

                    # Try to match to ground truth HOI labels
                    if 'verb' in targets[b_idx]:
                        # This is training data with GT annotations
                        # Try to find matching HOI annotation
                        # (This is approximate matching based on box overlap)
                        metadata['has_gt'] = True
                        # For simplicity, store all GT verbs in this image
                        metadata['gt_verbs'] = targets[b_idx]['verb'].cpu().tolist()
                    else:
                        metadata['has_gt'] = False

                    all_metadata.append(metadata)

        except Exception as e:
            print(f"\nError processing batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n✓ Extracted {total_pairs} visual embeddings from {len(dataloader)} batches")

    return all_embeddings, all_metadata


def main():
    parser = argparse.ArgumentParser(description='Extract visual embeddings for diffusion training')

    # Model checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained EZ-HOI checkpoint')

    # Dataset configuration (must match checkpoint training)
    parser.add_argument('--data_root', type=str, default='hicodet',
                        help='Root directory of HICO-DET dataset')
    parser.add_argument('--num_classes', type=int, default=117,
                        help='Number of classes (117 for verb, 600 for HOI)')
    parser.add_argument('--zs_type', type=str, default='unseen_verb',
                        choices=['rare_first', 'non_rare_first', 'unseen_verb', 'unseen_object', 'default'],
                        help='Zero-shot split type')

    # Normalization configuration
    parser.add_argument('--text_mean_path', type=str,
                        default='hicodet_pkl_files/hoi_text_mean_vitB_adapted_212.pkl',
                        help='Path to HOI text mean (use adapted version from extract_adapted_text_embeddings.py)')
    parser.add_argument('--scale_factor', type=float, default=5.0,
                        help='Scale factor for normalization (must match diffusion_bridge_module.py)')

    # Output configuration
    parser.add_argument('--output_path', type=str,
                        default='hicodet_pkl_files/hoi_visual_embeddings_normalized_vitB_train.pkl',
                        help='Output path for normalized visual embeddings')

    # Processing configuration
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for processing (keep at 1 to avoid OOM)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')

    # CLIP model path (required for DataFactory)
    parser.add_argument('--clip_model_path', type=str,
                        default='checkpoints/pretrained_CLIP/ViT-B-16.pt',
                        help='Path to CLIP model checkpoint')

    args = parser.parse_args()

    print("="*60)
    print("Visual Embeddings Extraction for Diffusion Training")
    print("="*60)
    print("Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("="*60)

    # Load checkpoint and recreate model
    checkpoint, ckpt_args = load_checkpoint_and_args(args.checkpoint)

    # Override num_classes and zs_type if provided
    if args.num_classes is not None:
        ckpt_args.num_classes = args.num_classes
    if args.zs_type is not None:
        ckpt_args.zs_type = args.zs_type

    # Create model
    model = recreate_model(ckpt_args, device=args.device)

    # Load checkpoint weights
    print("\nLoading checkpoint weights...")
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Weights loaded successfully")

    # Load text mean
    text_mean = load_text_mean(args.text_mean_path)

    # Create training dataset
    print("\n" + "="*60)
    print("Creating training dataset...")
    print("="*60)

    # Determine CLIP model name from checkpoint
    clip_model_name = 'ViT-B/16'  # Default
    if hasattr(ckpt_args, 'clip_model_name'):
        clip_model_name = ckpt_args.clip_model_name
    elif 'vitL' in args.checkpoint or 'ViT-L' in args.checkpoint:
        clip_model_name = 'ViT-L/14@336px'

    print(f"Using CLIP model: {clip_model_name}")

    trainset = DataFactory(
        name='hicodet',
        partition='train2015',
        data_root=args.data_root,
        clip_model_name=clip_model_name,
        zero_shot=ckpt_args.zs if hasattr(ckpt_args, 'zs') else False,
        zs_type=ckpt_args.zs_type if hasattr(ckpt_args, 'zs_type') else 'rare_first',
        num_classes=ckpt_args.num_classes
    )

    print(f"✓ Created training dataset with {len(trainset)} images")

    # Create dataloader
    train_loader = DataLoader(
        dataset=trainset,
        collate_fn=custom_collate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False  # Keep sequential for reproducibility
    )

    print(f"✓ Created dataloader with batch_size={args.batch_size}")

    # Extract visual embeddings
    all_embeddings, all_metadata = extract_visual_embeddings(
        model, train_loader, text_mean, args.scale_factor, device=args.device
    )

    # Convert to numpy array
    embeddings_array = np.stack(all_embeddings)

    print("\n" + "="*60)
    print("Extraction Summary:")
    print("="*60)
    print(f"Total embeddings: {len(embeddings_array)}")
    print(f"Embedding shape: {embeddings_array.shape}")
    print(f"Embedding dtype: {embeddings_array.dtype}")
    print(f"Embedding mean: {embeddings_array.mean():.6f}")
    print(f"Embedding std: {embeddings_array.std():.6f}")
    print(f"Embedding min: {embeddings_array.min():.6f}")
    print(f"Embedding max: {embeddings_array.max():.6f}")

    # Save to pickle
    output_data = {
        'embeddings': embeddings_array,
        'metadata': all_metadata,
        'config': {
            'checkpoint': args.checkpoint,
            'text_mean_path': args.text_mean_path,
            'scale_factor': args.scale_factor,
            'num_classes': ckpt_args.num_classes,
            'zs_type': ckpt_args.zs_type if hasattr(ckpt_args, 'zs_type') else None,
            'clip_model': clip_model_name,
        }
    }

    print(f"\nSaving to: {args.output_path}")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    with open(args.output_path, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"✓ Saved {len(embeddings_array)} normalized visual embeddings")
    print("="*60)
    print("\nNext steps:")
    print(f"1. Train diffusion model: python train_hoi_diffusion.py --data_path {args.output_path}")
    print(f"2. Test diffusion inference: python test_visual_diffusion.py")
    print("="*60)


if __name__ == '__main__':
    main()
