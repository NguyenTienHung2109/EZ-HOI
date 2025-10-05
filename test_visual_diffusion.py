"""
Test visual diffusion bridge on sample images.

This script:
1. Loads a trained diffusion model
2. Loads a trained EZ-HOI checkpoint
3. Extracts visual features from test images
4. Applies diffusion bridge transformation
5. Compares HOI predictions with/without diffusion
6. Visualizes the effect of diffusion on feature distribution

Usage:
    python test_visual_diffusion.py \
        --diffusion_model hoi_diffusion_results/model-500.pt \
        --checkpoint checkpoints/hico_HO_pt_default_vitbase/best.pth \
        --text_mean_path hicodet_pkl_files/hoi_text_mean_vitB_adapted_212.pkl \
        --adapted_text_pkl hicodet_pkl_files/hoi_adapted_text_embeddings_vitB_212.pkl \
        --num_test_images 10 \
        --inference_steps 100
"""

import torch
import torch.nn.functional as F
import pickle
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add CLIP to path
sys.path.insert(0, os.path.join(os.getcwd(), 'CLIP'))

from diffusion_bridge_module import DiffusionBridgeHOI
from upt_tip_cache_model_free_finetune_distillself import build_detector
from utils_tip_cache_and_union_finetune import DataFactory, custom_collate
from torch.utils.data import DataLoader
from hico_text_label import hico_text_label, HOI_TO_AO


def load_checkpoint_and_model(checkpoint_path, device='cuda'):
    """Load EZ-HOI checkpoint and recreate model"""
    print("="*60)
    print("Loading EZ-HOI checkpoint...")
    print("="*60)

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'args' not in checkpoint:
        raise ValueError("'args' not found in checkpoint")

    args = checkpoint['args']
    print(f"  Dataset: {args.dataset}")
    print(f"  Num classes: {args.num_classes}")
    print(f"  Zero-shot: {args.zs}")
    if args.zs:
        print(f"  ZS type: {args.zs_type}")

    # Recreate model (simplified version without full initialization)
    from main_tip_finetune import hico_class_corr
    import numpy as np

    object_to_target = hico_class_corr()
    lut = np.full([80, 117], None)
    for hoi_idx, obj_idx, verb_idx in object_to_target:
        lut[obj_idx, verb_idx] = hoi_idx
    object_n_verb_to_interaction = lut.tolist()

    # Initialize distributed training if needed
    import torch.distributed as dist
    if not dist.is_initialized():
        print(f"Initializing distributed training (single process)...")
        try:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(
                backend=backend,
                init_method='tcp://localhost:12356',
                world_size=1,
                rank=0
            )
        except Exception as e:
            print(f"Warning: Could not initialize distributed: {e}")

    print("\nBuilding detector...")
    detector = build_detector(args, object_n_verb_to_interaction)
    detector = detector.to(device)
    detector.load_state_dict(checkpoint['model_state_dict'])
    detector.eval()

    print("✓ Model loaded successfully")

    return detector, args


def load_adapted_text_embeddings(adapted_text_pkl):
    """Load adapted text embeddings"""
    print(f"\nLoading adapted text embeddings from: {adapted_text_pkl}")

    with open(adapted_text_pkl, 'rb') as f:
        data = pickle.load(f)

    embeddings = data['embeddings']  # [212, 512]
    select_HOI_index = data['select_HOI_index']  # List of 212 HOI IDs

    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings)

    print(f"  Shape: {embeddings.shape}")
    print(f"  Num classes: {len(select_HOI_index)}")

    return embeddings, select_HOI_index


@torch.no_grad()
def extract_visual_features_from_image(model, image, target, device='cuda'):
    """
    Extract visual features (adapter_feat) from a single image.

    Returns:
        adapter_feat: [num_pairs, embed_dim]
        metadata: dict with pair information
    """
    from upt_tip_cache_model_free_finetune_distillself import nested_tensor_from_tensor_list
    import torchvision

    # Prepare image
    image_detr, image_clip = image
    images_detr = [image_detr.to(device)]
    images_clip = [image_clip.to(device)]

    # Move target to device
    for k, v in target.items():
        if isinstance(v, torch.Tensor):
            target[k] = v.to(device)

    # Forward through DETR
    images_detr_nested = nested_tensor_from_tensor_list(images_detr)
    results = model.detector(images_detr_nested)

    # Get region proposals
    region_props = model.prepare_region_proposals(results)

    # Get image sizes
    image_sizes = target['orig_size'].unsqueeze(0)

    # Process through CLIP
    if model.use_insadapter:
        priors = model.get_prior(region_props, image_sizes, model.prior_method)
    else:
        priors = None

    images_clip_nested = nested_tensor_from_tensor_list(images_clip)

    # Get text features
    if model.txtcls_descrip:
        txtcls_feat = model.hoicls_txt[model.select_HOI_index]
    else:
        txtcls_feat = None

    # Forward through prompt_learner and text_encoder
    if model.fix_txt_pt is False and model.clip_test is False:
        tokenized_prompts = model.clip_head.tokenized_prompts
        if model.txtcls_pt:
            prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision, txtcls_pt_list, origin_ctx = \
                model.clip_head.prompt_learner(
                    img_descrip_priors=None,
                    txtcls_feat=model.hoicls_txt,
                    select_HOI_index=model.select_HOI_index,
                    unseen_text_priors=model.unseen_text_priors,
                    filenames=None
                )
        else:
            prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision, origin_ctx = \
                model.clip_head.prompt_learner(
                    img_descrip_priors=None,
                    unseen_text_priors=model.unseen_text_priors,
                    filenames=None
                )
            txtcls_pt_list = None

        hoitxt_features, origin_txt_features = model.clip_head.text_encoder(
            prompts, tokenized_prompts, deep_compound_prompts_text,
            txtcls_feat, txtcls_pt_list, origin_ctx
        )
    else:
        hoitxt_features = model.hoicls_txt[model.select_HOI_index].to(device)

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

    # Extract union features
    props = region_props[0]
    local_features = feat_local[0]
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
        return None, None

    # Get pairwise indices
    x = torch.arange(n, device=device)
    y = torch.arange(n, device=device)
    x, y = torch.meshgrid(x, y, indexing='ij')

    # Valid human-object pairs
    x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < n_h)).unbind(1)

    if len(x_keep) == 0:
        return None, None

    # Compute union boxes
    sub_boxes = boxes[x_keep]
    obj_boxes = boxes[y_keep]
    lt = torch.min(sub_boxes[..., :2], obj_boxes[..., :2])
    rb = torch.max(sub_boxes[..., 2:], obj_boxes[..., 2:])
    union_boxes = torch.cat([lt, rb], dim=-1)

    # Extract union features using ROI align
    spatial_scale = 1 / (image_sizes[0, 0] / local_features.shape[1])
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

    # Apply feature masking
    if model.feat_mask_type in [0, 1]:
        union_features = union_features.flatten(2).mean(-1)
        single_features = single_features.flatten(2).mean(-1)

    # Get human and object features
    human_features = single_features[x_keep]
    object_features = single_features[y_keep]

    # Apply normalization
    if model.individual_norm:
        human_features = human_features / human_features.norm(dim=-1, keepdim=True)
        object_features = object_features / object_features.norm(dim=-1, keepdim=True)
        union_features = union_features / union_features.norm(dim=-1, keepdim=True)

    # Compute visual features based on logits_type
    if model.logits_type == 'HO+U':
        vis_feat = model.vis_fuse(torch.cat([union_features, human_features, object_features], dim=-1))
    elif model.logits_type == 'HO':
        vis_feat = model.vis_fuse(torch.cat([human_features, object_features], dim=-1))
    elif model.logits_type == 'U':
        vis_feat = union_features

    # Apply mem_adapter
    if model.img_align:
        adapter_feat = model.mem_adapter(vis_feat.unsqueeze(1)).squeeze(1)
    else:
        adapter_feat = vis_feat

    # Metadata
    metadata = {
        'num_pairs': len(adapter_feat),
        'object_classes': labels[y_keep].cpu().tolist(),
        'filename': target.get('filename', 'unknown')
    }

    return adapter_feat, metadata


def compute_predictions(visual_feat, text_embeddings, select_HOI_index):
    """
    Compute HOI predictions from visual features.

    Args:
        visual_feat: [num_pairs, embed_dim]
        text_embeddings: [212, embed_dim]
        select_HOI_index: List of 212 HOI IDs

    Returns:
        predictions: [num_pairs, 212] similarity scores
        top_k_hois: List of (hoi_id, score) tuples for each pair
    """
    # Normalize features
    visual_feat = F.normalize(visual_feat, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)

    # Compute cosine similarity
    similarities = visual_feat @ text_embeddings.T  # [num_pairs, 212]

    # Get top-5 predictions for each pair
    top_k = 5
    top_scores, top_indices = similarities.topk(top_k, dim=-1)

    # Convert to HOI IDs
    top_k_hois = []
    for pair_idx in range(len(visual_feat)):
        pair_preds = []
        for k in range(top_k):
            hoi_idx_in_212 = top_indices[pair_idx, k].item()
            hoi_id = select_HOI_index[hoi_idx_in_212]
            score = top_scores[pair_idx, k].item()
            pair_preds.append((hoi_id, score))
        top_k_hois.append(pair_preds)

    return similarities, top_k_hois


def visualize_comparison(visual_feat_baseline, visual_feat_diffusion,
                         text_embeddings, select_HOI_index,
                         save_path='diffusion_effect.png'):
    """
    Visualize the effect of diffusion bridge on visual features.

    Creates plots showing:
    1. Cosine similarity distributions (before/after)
    2. Feature norm distributions
    3. Top-1 prediction changes
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Cosine similarity to text embeddings
    text_norm = F.normalize(text_embeddings, dim=-1)

    # Baseline similarities
    sims_baseline = (F.normalize(visual_feat_baseline, dim=-1) @ text_norm.T).cpu().numpy()
    max_sims_baseline = sims_baseline.max(axis=1)

    # Diffusion similarities
    sims_diffusion = (F.normalize(visual_feat_diffusion, dim=-1) @ text_norm.T).cpu().numpy()
    max_sims_diffusion = sims_diffusion.max(axis=1)

    # Plot similarity distributions
    axes[0, 0].hist(max_sims_baseline, bins=30, alpha=0.5, label='Baseline', color='blue')
    axes[0, 0].hist(max_sims_diffusion, bins=30, alpha=0.5, label='After Diffusion', color='red')
    axes[0, 0].set_xlabel('Max Cosine Similarity to Text')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Similarity to Text Embeddings')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Feature norms
    norms_baseline = visual_feat_baseline.norm(dim=-1).cpu().numpy()
    norms_diffusion = visual_feat_diffusion.norm(dim=-1).cpu().numpy()

    axes[0, 1].hist(norms_baseline, bins=30, alpha=0.5, label='Baseline', color='blue')
    axes[0, 1].hist(norms_diffusion, bins=30, alpha=0.5, label='After Diffusion', color='red')
    axes[0, 1].set_xlabel('Feature Norm')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Feature Norm Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Similarity improvement scatter
    axes[1, 0].scatter(max_sims_baseline, max_sims_diffusion, alpha=0.5)
    axes[1, 0].plot([0, 1], [0, 1], 'r--', label='No change')
    axes[1, 0].set_xlabel('Baseline Similarity')
    axes[1, 0].set_ylabel('Diffusion Similarity')
    axes[1, 0].set_title('Similarity Change per Pair')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Top-1 prediction changes
    top1_baseline = sims_baseline.argmax(axis=1)
    top1_diffusion = sims_diffusion.argmax(axis=1)
    changed = (top1_baseline != top1_diffusion).sum()
    unchanged = (top1_baseline == top1_diffusion).sum()

    axes[1, 1].bar(['Unchanged', 'Changed'], [unchanged, changed], color=['blue', 'red'])
    axes[1, 1].set_ylabel('Number of Pairs')
    axes[1, 1].set_title(f'Top-1 Prediction Changes ({changed}/{len(top1_baseline)} changed)')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Test visual diffusion bridge')

    # Model paths
    parser.add_argument('--diffusion_model', type=str,
                        default='hoi_diffusion_results/model-500.pt',
                        help='Path to trained diffusion model')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/hico_HO_pt_default_vitbase/best.pth',
                        help='Path to EZ-HOI checkpoint')
    parser.add_argument('--text_mean_path', type=str,
                        default='hicodet_pkl_files/hoi_text_mean_vitB_adapted_212.pkl',
                        help='Path to adapted text mean')
    parser.add_argument('--adapted_text_pkl', type=str,
                        default='hicodet_pkl_files/hoi_adapted_text_embeddings_vitB_212.pkl',
                        help='Path to adapted text embeddings')

    # Dataset
    parser.add_argument('--data_root', type=str, default='hicodet',
                        help='Root directory of HICO-DET dataset')
    parser.add_argument('--num_test_images', type=int, default=10,
                        help='Number of test images to process')

    # Diffusion parameters
    parser.add_argument('--inference_steps', type=int, default=100,
                        help='Number of DDIM inference steps (lower=faster, higher=better quality)')
    parser.add_argument('--scale_factor', type=float, default=5.0,
                        help='Normalization scale factor')

    # Output
    parser.add_argument('--output_dir', type=str, default='test_diffusion_results',
                        help='Output directory for visualizations')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    print("="*60)
    print("Visual Diffusion Bridge Testing")
    print("="*60)
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load diffusion bridge
    print("Loading diffusion bridge module...")
    diffusion_bridge = DiffusionBridgeHOI(
        diffusion_path=args.diffusion_model,
        text_mean_path=args.text_mean_path,
        inference_steps=args.inference_steps,
        scale_factor=args.scale_factor,
        verbose=True
    )
    diffusion_bridge = diffusion_bridge.to(args.device)
    diffusion_bridge.eval()

    # Load EZ-HOI model
    model, model_args = load_checkpoint_and_model(args.checkpoint, device=args.device)

    # Load adapted text embeddings
    text_embeddings, select_HOI_index = load_adapted_text_embeddings(args.adapted_text_pkl)
    text_embeddings = text_embeddings.to(args.device)

    # Create test dataset
    print("\n" + "="*60)
    print("Loading test dataset...")
    print("="*60)

    clip_model_name = 'ViT-B/16'
    if hasattr(model_args, 'clip_model_name'):
        clip_model_name = model_args.clip_model_name

    testset = DataFactory(
        name='hicodet',
        partition='test2015',
        data_root=args.data_root,
        clip_model_name=clip_model_name,
        zero_shot=False,
        num_classes=model_args.num_classes
    )

    print(f"✓ Loaded test dataset with {len(testset)} images")

    # Test on sample images
    print("\n" + "="*60)
    print(f"Testing diffusion bridge on {args.num_test_images} images...")
    print("="*60)

    all_visual_baseline = []
    all_visual_diffusion = []

    for img_idx in range(min(args.num_test_images, len(testset))):
        print(f"\nProcessing image {img_idx + 1}/{args.num_test_images}...")

        image, target = testset[img_idx]

        # Extract visual features
        visual_feat, metadata = extract_visual_features_from_image(
            model, image, target, device=args.device
        )

        if visual_feat is None:
            print("  ⚠️  No valid human-object pairs, skipping...")
            continue

        print(f"  Found {metadata['num_pairs']} human-object pairs")

        # Baseline: no diffusion
        visual_baseline = visual_feat.clone()

        # With diffusion
        visual_diffusion = diffusion_bridge(visual_feat)

        # Compute predictions
        preds_baseline, top_k_baseline = compute_predictions(
            visual_baseline, text_embeddings, select_HOI_index
        )
        preds_diffusion, top_k_diffusion = compute_predictions(
            visual_diffusion, text_embeddings, select_HOI_index
        )

        # Show top predictions for first pair
        if metadata['num_pairs'] > 0:
            print(f"\n  Top-5 predictions for first pair:")
            print(f"    Baseline:")
            for hoi_id, score in top_k_baseline[0]:
                verb_idx, obj_idx = HOI_TO_AO[hoi_id]
                label = hico_text_label[(verb_idx, obj_idx)]
                print(f"      {label}: {score:.4f}")

            print(f"    After Diffusion:")
            for hoi_id, score in top_k_diffusion[0]:
                verb_idx, obj_idx = HOI_TO_AO[hoi_id]
                label = hico_text_label[(verb_idx, obj_idx)]
                print(f"      {label}: {score:.4f}")

        # Collect for visualization
        all_visual_baseline.append(visual_baseline)
        all_visual_diffusion.append(visual_diffusion)

    # Concatenate all features
    if len(all_visual_baseline) > 0:
        all_visual_baseline = torch.cat(all_visual_baseline, dim=0)
        all_visual_diffusion = torch.cat(all_visual_diffusion, dim=0)

        print(f"\n\n" + "="*60)
        print("Overall Statistics")
        print("="*60)
        print(f"Total pairs processed: {len(all_visual_baseline)}")

        # Visualize comparison
        viz_path = Path(args.output_dir) / 'diffusion_effect.png'
        visualize_comparison(
            all_visual_baseline,
            all_visual_diffusion,
            text_embeddings,
            select_HOI_index,
            save_path=viz_path
        )

        print("\n" + "="*60)
        print("Testing completed!")
        print("="*60)
        print(f"\nResults saved to: {args.output_dir}/")
        print(f"\nNext steps:")
        print(f"1. Review visualization: {viz_path}")
        print(f"2. If diffusion improves alignment, integrate into full HOI detection")
        print(f"3. Run full evaluation on test set with/without diffusion")
        print("="*60)
    else:
        print("\n⚠️  No valid pairs found in test images")


if __name__ == '__main__':
    main()
