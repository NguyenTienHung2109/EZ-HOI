"""
Extract Adapted Embeddings for Visualization

This script extracts visual and text embeddings from a trained EZ-HOI model
AFTER all adaptations (prompt learning, mem_adapter, etc.) but BEFORE diffusion.

It then applies the diffusion bridge to create "after diffusion" embeddings,
and saves both versions for visualization.

Usage:
    python extract_adapted_embeddings.py \
        --checkpoint checkpoints/your_model/best.pth \
        --data_root hicodet \
        --num_samples 1000 \
        --output embeddings_for_viz.pkl

Output:
    embeddings_for_viz.pkl containing:
    - visual_before: Visual embeddings before diffusion [N, embed_dim]
    - visual_after: Visual embeddings after diffusion [N, embed_dim]
    - text_adapted: Adapted text embeddings [212, embed_dim]
    - metadata: List of dicts with image info and GT labels
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

from upt_tip_cache_model_free_finetune_distillself import build_detector
from utils_tip_cache_and_union_finetune import DataFactory, custom_collate
from torch.utils.data import DataLoader
from diffusion_bridge_module import DiffusionBridgeHOI


def load_model_from_checkpoint(checkpoint_path, device='cuda'):
    """
    Load trained model from checkpoint and prepare for extraction.

    Returns:
        model: Loaded model in eval mode
        args: Training arguments from checkpoint
    """
    print("="*60)
    print("Loading Model from Checkpoint")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'args' not in checkpoint:
        raise ValueError("Checkpoint must contain 'args' to recreate model")

    args = checkpoint['args']

    print(f"\nModel Configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Num classes: {args.num_classes}")
    print(f"  Zero-shot: {args.zs if hasattr(args, 'zs') else False}")
    print(f"  Image align: {args.img_align if hasattr(args, 'img_align') else False}")
    print(f"  Text align: {args.txt_align if hasattr(args, 'txt_align') else False}")

    # Initialize distributed training if needed
    import torch.distributed as dist
    if not dist.is_initialized():
        print(f"\nInitializing distributed training (single process)...")
        try:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(
                backend=backend,
                init_method='tcp://localhost:12357',
                world_size=1,
                rank=0
            )
            print(f"✓ Distributed initialized (backend: {backend})")
        except Exception as e:
            print(f"Warning: Could not initialize distributed: {e}")

    # Build model structure
    print(f"\nBuilding model...")
    from main_tip_finetune import hico_class_corr
    import numpy as np

    object_to_target = hico_class_corr()
    lut = np.full([80, 117], None)
    for hoi_idx, obj_idx, verb_idx in object_to_target:
        lut[obj_idx, verb_idx] = hoi_idx
    object_n_verb_to_interaction = lut.tolist()

    model = build_detector(args, object_n_verb_to_interaction)

    # Load weights
    print(f"Loading checkpoint weights...")
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Move to device and set eval mode
    model = model.to(device)
    model.eval()

    # Enable extraction mode
    model.extraction_mode = True

    print(f"✓ Model loaded successfully")
    print(f"✓ Extraction mode: ENABLED")

    return model, args


def create_dataloader(args, partition='test2015', batch_size=1, num_workers=4):
    """
    Create dataloader for extraction.

    Args:
        args: Training arguments from checkpoint
        partition: 'train2015' or 'test2015'
        batch_size: Batch size (keep at 1 for simplicity)
        num_workers: Number of dataloader workers

    Returns:
        dataloader: PyTorch DataLoader
    """
    print("\n" + "="*60)
    print("Creating DataLoader")
    print("="*60)

    # Determine CLIP model name
    clip_model_name = 'ViT-B/16'  # Default
    if hasattr(args, 'clip_model_name'):
        clip_model_name = args.clip_model_name
    elif hasattr(args, 'clip_dir_vit'):
        if 'ViT-L' in args.clip_dir_vit:
            clip_model_name = 'ViT-L/14@336px'

    print(f"CLIP model: {clip_model_name}")
    print(f"Partition: {partition}")

    # Create dataset
    dataset = DataFactory(
        name='hicodet',
        partition=partition,
        data_root=args.data_root if hasattr(args, 'data_root') else 'hicodet',
        clip_model_name=clip_model_name,
        zero_shot=args.zs if hasattr(args, 'zs') else False,
        zs_type=args.zs_type if hasattr(args, 'zs_type') else 'rare_first',
        num_classes=args.num_classes
    )

    print(f"✓ Dataset created: {len(dataset)} images")

    # Create dataloader
    dataloader = DataLoader(
        dataset=dataset,
        collate_fn=custom_collate,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False
    )

    print(f"✓ DataLoader created (batch_size={batch_size})")

    return dataloader


@torch.no_grad()
def extract_embeddings(model, dataloader, num_samples=1000, device='cuda'):
    """
    Extract adapted visual and text embeddings from model.

    Args:
        model: Model with extraction_mode=True
        dataloader: DataLoader for images
        num_samples: Number of images to process (None = all)
        device: Device to use

    Returns:
        dict with:
        - visual_embeddings: [N_pairs, embed_dim]
        - text_embeddings: [N_classes, embed_dim]
        - metadata: List of dicts with image info
    """
    print("\n" + "="*60)
    print("Extracting Embeddings")
    print("="*60)

    model.eval()

    visual_embeddings = []
    text_embeddings = None
    metadata = []

    total_pairs = 0

    for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="Processing images")):
        # Limit number of samples
        if num_samples is not None and batch_idx >= num_samples:
            break

        try:
            # Prepare images
            if isinstance(images[0], (list, tuple)):
                images_detr = [img[0].to(device) for img in images]
                images_clip = [img[1].to(device) for img in images]
                images_to_pass = [(d, c) for d, c in zip(images_detr, images_clip)]
            else:
                images_to_pass = [img.to(device) for img in images]

            # Move targets to device
            targets_to_pass = []
            for target in targets:
                target_dict = {}
                for k, v in target.items():
                    if isinstance(v, torch.Tensor):
                        target_dict[k] = v.to(device)
                    else:
                        target_dict[k] = v
                targets_to_pass.append(target_dict)

            # Forward pass (triggers extraction hooks)
            _ = model(images_to_pass, targets_to_pass)

            # Extract visual embeddings (stored by forward pass)
            if hasattr(model, '_extracted_visual_feat'):
                visual_feat = model._extracted_visual_feat

                if visual_feat is not None and len(visual_feat) > 0:
                    visual_embeddings.append(visual_feat)
                    total_pairs += len(visual_feat)

                    # Store metadata for this image
                    for pair_idx in range(len(visual_feat)):
                        meta = {
                            'image_id': targets[0].get('filename', f'img_{batch_idx}'),
                            'batch_idx': batch_idx,
                            'pair_idx': pair_idx,
                        }

                        # Add GT labels if available
                        if 'hoi' in targets[0]:
                            meta['gt_hoi'] = targets[0]['hoi'].cpu().tolist()
                        if 'verb' in targets[0]:
                            meta['gt_verb'] = targets[0]['verb'].cpu().tolist()

                        metadata.append(meta)

            # Extract text embeddings (only once, same for all images)
            if text_embeddings is None and hasattr(model, '_extracted_text_feat'):
                text_embeddings = model._extracted_text_feat

        except Exception as e:
            print(f"\nError processing batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Concatenate all visual embeddings
    if len(visual_embeddings) > 0:
        visual_embeddings = torch.cat(visual_embeddings, dim=0)
    else:
        raise ValueError("No visual embeddings extracted! Check if extraction_mode is enabled.")

    if text_embeddings is None:
        raise ValueError("No text embeddings extracted! Check if extraction_mode is enabled.")

    print(f"\n✓ Extraction complete:")
    print(f"  Visual embeddings: {visual_embeddings.shape}")
    print(f"  Text embeddings: {text_embeddings.shape}")
    print(f"  Total pairs: {total_pairs}")
    print(f"  Total images: {batch_idx + 1}")

    return {
        'visual_embeddings': visual_embeddings,
        'text_embeddings': text_embeddings,
        'metadata': metadata
    }


def apply_diffusion_bridge(visual_embeddings, diffusion_path, text_mean_path,
                          inference_steps=100, scale_factor=5.0, device='cuda'):
    """
    Apply diffusion bridge to visual embeddings.

    Args:
        visual_embeddings: [N, embed_dim] tensor
        diffusion_path: Path to trained diffusion model
        text_mean_path: Path to text mean for normalization
        inference_steps: Number of DDIM steps
        scale_factor: Normalization scale factor
        device: Device to use

    Returns:
        visual_after_diffusion: [N, embed_dim] tensor
    """
    print("\n" + "="*60)
    print("Applying Diffusion Bridge")
    print("="*60)

    # Check if diffusion model exists
    if not os.path.exists(diffusion_path):
        print(f"⚠️  Diffusion model not found: {diffusion_path}")
        print(f"⚠️  Skipping diffusion transformation")
        print(f"⚠️  You can train diffusion model with: python train_hoi_diffusion.py")
        return None

    if not os.path.exists(text_mean_path):
        print(f"⚠️  Text mean not found: {text_mean_path}")
        print(f"⚠️  Skipping diffusion transformation")
        return None

    # Load diffusion bridge
    print(f"Loading diffusion bridge...")
    print(f"  Model: {diffusion_path}")
    print(f"  Text mean: {text_mean_path}")
    print(f"  Inference steps: {inference_steps}")

    bridge = DiffusionBridgeHOI(
        diffusion_path=diffusion_path,
        text_mean_path=text_mean_path,
        inference_steps=inference_steps,
        scale_factor=scale_factor,
        verbose=True
    )

    bridge = bridge.to(device)
    bridge.eval()

    # Apply diffusion
    print(f"\nApplying diffusion to {len(visual_embeddings)} embeddings...")
    visual_embeddings = visual_embeddings.to(device)

    # Process in batches to avoid OOM
    batch_size = 128
    visual_after_list = []

    for i in tqdm(range(0, len(visual_embeddings), batch_size), desc="Diffusion batches"):
        batch = visual_embeddings[i:i+batch_size]
        batch_after = bridge(batch)
        visual_after_list.append(batch_after.cpu())

    visual_after_diffusion = torch.cat(visual_after_list, dim=0)

    print(f"✓ Diffusion applied")
    print(f"  Output shape: {visual_after_diffusion.shape}")

    return visual_after_diffusion


def main():
    parser = argparse.ArgumentParser(description='Extract adapted embeddings for visualization')

    # Model checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained EZ-HOI checkpoint')

    # Dataset
    parser.add_argument('--partition', type=str, default='test2015',
                       choices=['train2015', 'test2015'],
                       help='Dataset partition to use')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of images to process (None=all)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size (keep at 1)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers')

    # Diffusion (optional)
    parser.add_argument('--apply_diffusion', action='store_true',
                       help='Apply diffusion bridge (requires trained model)')
    parser.add_argument('--diffusion_model', type=str,
                       default='hoi_diffusion_results/model-500.pt',
                       help='Path to trained diffusion model')
    parser.add_argument('--text_mean_path', type=str,
                       default='hicodet_pkl_files/hoi_text_mean_vitB_adapted_212.pkl',
                       help='Path to text mean for normalization')
    parser.add_argument('--inference_steps', type=int, default=100,
                       help='Number of DDIM inference steps')
    parser.add_argument('--scale_factor', type=float, default=5.0,
                       help='Normalization scale factor')

    # Output
    parser.add_argument('--output', type=str, default='embeddings_for_viz.pkl',
                       help='Output path for embeddings')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("Extract Adapted Embeddings for Visualization")
    print("="*60)
    print("Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("="*60)

    # Load model
    model, model_args = load_model_from_checkpoint(args.checkpoint, device=args.device)

    # Create dataloader
    dataloader = create_dataloader(
        model_args,
        partition=args.partition,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Extract embeddings
    embeddings = extract_embeddings(
        model,
        dataloader,
        num_samples=args.num_samples,
        device=args.device
    )

    # Apply diffusion (optional)
    visual_after_diffusion = None
    if args.apply_diffusion:
        visual_after_diffusion = apply_diffusion_bridge(
            embeddings['visual_embeddings'],
            args.diffusion_model,
            args.text_mean_path,
            args.inference_steps,
            args.scale_factor,
            device=args.device
        )

    # Prepare output
    output_data = {
        'visual_before': embeddings['visual_embeddings'].cpu().numpy(),
        'visual_after': visual_after_diffusion.cpu().numpy() if visual_after_diffusion is not None else None,
        'text_adapted': embeddings['text_embeddings'].cpu().numpy(),
        'metadata': embeddings['metadata'],
        'config': {
            'checkpoint': args.checkpoint,
            'partition': args.partition,
            'num_samples': args.num_samples,
            'diffusion_applied': visual_after_diffusion is not None,
            'diffusion_model': args.diffusion_model if args.apply_diffusion else None,
            'inference_steps': args.inference_steps if args.apply_diffusion else None,
        }
    }

    # Save
    print("\n" + "="*60)
    print("Saving Results")
    print("="*60)
    print(f"Output file: {args.output}")

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)

    with open(args.output, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"✓ Saved embeddings:")
    print(f"  Visual (before): {output_data['visual_before'].shape}")
    if output_data['visual_after'] is not None:
        print(f"  Visual (after):  {output_data['visual_after'].shape}")
    print(f"  Text (adapted):  {output_data['text_adapted'].shape}")
    print(f"  Metadata:        {len(output_data['metadata'])} entries")

    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print(f"1. Visualize embeddings:")
    print(f"   python visualize_embedding_distributions.py --embeddings {args.output}")
    print(f"\n2. View results in: visualization_results/")
    print("="*60)


if __name__ == '__main__':
    main()
