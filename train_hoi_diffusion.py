"""
Train diffusion model on HOI embeddings (text or visual).

Adapted from diffusion-bridge/ddpm/train_norm.py but for HOI-specific distribution.

Supported training modes:
1. Text embeddings: 212 representative classes (adapted through learnable prompts)
   - Limited data, may overfit
   - Use only if you have no alternative

2. Visual embeddings: 50k-100k human-object pairs from training images (RECOMMENDED)
   - Abundant data, stable training
   - Learns vision→text distribution bridging
   - Extract using extract_visual_embeddings_for_diffusion.py

Key differences from original diffusion-bridge:
- Dataset: HOI-specific embeddings vs 400k MSCOCO captions
- Training steps: 300k-500k vs 3M (depending on dataset size)
- Distribution: Structured HOI distribution vs free-form captions
"""

import torch
import pickle
import numpy as np
import torch.nn.functional as F
import argparse
import os
import sys
from pathlib import Path

# Add diffusion-bridge to path
DIFFUSION_BRIDGE_PATH = os.path.join(os.getcwd(), 'diffusion-bridge', 'ddpm')
sys.path.insert(0, DIFFUSION_BRIDGE_PATH)

try:
    from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d import (
        Unet1D, GaussianDiffusion1D_norm, Trainer1D, Dataset1D
    )
except ImportError:
    print("ERROR: Could not import diffusion modules.")
    print(f"Make sure diffusion-bridge is properly set up at: {DIFFUSION_BRIDGE_PATH}")
    print("Install with: cd diffusion-bridge/ddpm && pip install -e .")
    sys.exit(1)


def load_hoi_embeddings(data_path):
    """
    Load HOI embeddings (text or visual).

    Supports two formats:
    1. Text embeddings: from extract_adapted_text_embeddings.py
       - data['embeddings']: [212, embed_dim]
       - data['num_classes']: number of classes

    2. Visual embeddings: from extract_visual_embeddings_for_diffusion.py
       - data['embeddings']: [N, embed_dim] where N = 50k-100k pairs
       - data['metadata']: list of metadata dicts
       - data['config']: extraction configuration
    """
    print(f"Loading HOI embeddings from: {data_path}")

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    embeddings = data['embeddings']  # [num_samples, embed_dim]

    # Ensure embeddings is a tensor
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings)

    # Detect embedding type
    if 'metadata' in data:
        # Visual embeddings
        embedding_type = 'visual'
        num_samples = len(embeddings)
        config = data.get('config', {})
        clip_model = config.get('clip_model', 'Unknown')
        scale_factor = config.get('scale_factor', 5.0)

        print(f"  Type: Visual embeddings (from training images)")
        print(f"  CLIP model: {clip_model}")
        print(f"  Number of samples: {num_samples:,}")
        print(f"  Checkpoint: {config.get('checkpoint', 'Unknown')}")
    else:
        # Text embeddings
        embedding_type = 'text'
        num_samples = data.get('num_classes', len(embeddings))
        clip_model = data.get('clip_model', 'Unknown')
        scale_factor = data.get('scale_factor', 5.0)

        print(f"  Type: Text embeddings (adapted HOI classes)")
        print(f"  CLIP model: {clip_model}")
        print(f"  Number of classes: {num_samples}")

    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    print(f"  Scale factor: {scale_factor}")

    # Compute statistics
    if isinstance(embeddings, torch.Tensor):
        norms = embeddings.norm(dim=-1)
        print(f"  Embedding norms: {norms.mean().item():.4f} ± {norms.std().item():.4f}")
        print(f"  Embedding range: [{embeddings.min().item():.4f}, {embeddings.max().item():.4f}]")
    else:
        norms = np.linalg.norm(embeddings, axis=-1)
        print(f"  Embedding norms: {norms.mean():.4f} ± {norms.std():.4f}")
        print(f"  Embedding range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")

    # Add embedding type to data info
    data['embedding_type'] = embedding_type
    data['num_samples'] = num_samples

    return embeddings.type(torch.float32), data


def setup_diffusion_model(args, embed_dim):
    """Setup 1D UNet and Gaussian diffusion model"""
    print("\n" + "="*60)
    print("Setting up diffusion model...")
    print("="*60)

    # 1D UNet for embedding sequences
    model = Unet1D(
        dim=embed_dim,  # Match embedding dimension (512 for ViT-B, 768 for ViT-L)
        init_dim=args.init_dim,
        dim_mults=args.dim_mults,
        channels=1  # Single channel (embedding vector)
    )

    print(f"\nUNet configuration:")
    print(f"  Embedding dim: {embed_dim}")
    print(f"  Init dim: {args.init_dim}")
    print(f"  Dim multipliers: {args.dim_mults}")
    print(f"  Channels: 1")

    # Gaussian diffusion wrapper
    diffusion = GaussianDiffusion1D_norm(
        model,
        seq_length=embed_dim,
        timesteps=args.timesteps,
        objective=args.objective
    )

    print(f"\nDiffusion configuration:")
    print(f"  Timesteps: {args.timesteps}")
    print(f"  Objective: {args.objective}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {num_params:,}")

    return diffusion


def setup_trainer(diffusion, dataset, args):
    """Setup trainer for diffusion model"""
    print("\n" + "="*60)
    print("Setting up trainer...")
    print("="*60)

    trainer = Trainer1D(
        diffusion,
        dataset=dataset,
        train_batch_size=args.batch_size,
        train_lr=args.learning_rate,
        train_num_steps=args.train_steps,
        gradient_accumulate_every=args.gradient_accumulate,
        ema_decay=args.ema_decay,
        amp=args.use_amp,
        results_folder=args.results_folder
    )

    print(f"Training configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Total steps: {args.train_steps:,}")
    print(f"  Gradient accumulation: {args.gradient_accumulate}")
    print(f"  EMA decay: {args.ema_decay}")
    print(f"  Mixed precision (AMP): {args.use_amp}")
    print(f"  Results folder: {args.results_folder}")

    return trainer


def main():
    parser = argparse.ArgumentParser(description='Train diffusion model on HOI embeddings (visual recommended, text also supported)')

    # Data paths
    parser.add_argument('--data_path', type=str,
                        default='hicodet_pkl_files/hoi_visual_embeddings_normalized_vitB_train.pkl',
                        help='Path to normalized HOI embeddings (visual recommended, text also supported)')

    # Model architecture
    parser.add_argument('--init_dim', type=int, default=32,
                        help='Initial dimension for UNet')
    parser.add_argument('--dim_mults', type=int, nargs='+', default=[1, 2, 4, 8],
                        help='Dimension multipliers for UNet layers')

    # Diffusion settings
    parser.add_argument('--timesteps', type=int, default=1000,
                        help='Number of diffusion timesteps')
    parser.add_argument('--objective', type=str, default='pred_x0',
                        choices=['pred_x0', 'pred_noise', 'pred_v'],
                        help='Diffusion objective (pred_x0 recommended)')

    # Training settings
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=8e-5,
                        help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=500000,
                        help='Total training steps (500k for visual, 300k for text)')
    parser.add_argument('--gradient_accumulate', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--ema_decay', type=float, default=0.995,
                        help='Exponential moving average decay')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision')

    # Output settings
    parser.add_argument('--results_folder', type=str, default='./hoi_diffusion_results',
                        help='Folder to save model checkpoints')

    args = parser.parse_args()

    print("="*60)
    print("HOI Diffusion Model Training")
    print("="*60)
    print("Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")

    # Load data
    embeddings, data_info = load_hoi_embeddings(args.data_path)

    # Create dataset
    print("\n" + "="*60)
    print("Creating dataset...")
    print("="*60)
    dataset = Dataset1D(embeddings.unsqueeze(1))  # Add channel dimension
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample shape: {dataset[0].shape}")

    # Setup model
    embed_dim = embeddings.shape[1]
    diffusion = setup_diffusion_model(args, embed_dim)

    # Setup trainer
    trainer = setup_trainer(diffusion, dataset, args)

    # Create results folder
    Path(args.results_folder).mkdir(parents=True, exist_ok=True)

    # Save training config
    config_path = Path(args.results_folder) / 'training_config.txt'
    with open(config_path, 'w') as f:
        f.write("HOI Diffusion Training Configuration\n")
        f.write("="*60 + "\n\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
        f.write("\nData Info:\n")
        for key, value in data_info.items():
            if key != 'embeddings' and key != 'classnames':
                f.write(f"{key}: {value}\n")
    print(f"\n✓ Saved training config to: {config_path}")

    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    print("\nCheckpoints will be saved every 10k steps")
    print(f"Monitor training in: {args.results_folder}/")
    print("\nTraining will take approximately:")
    print(f"  Steps: {args.train_steps:,}")
    if data_info.get('embedding_type') == 'visual':
        print(f"  Time: ~8-15 hours (visual embeddings, depending on GPU)")
    else:
        print(f"  Time: ~6-12 hours (text embeddings, depending on GPU)")
    print("="*60 + "\n")

    trainer.train()

    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print(f"\nTrained model saved to: {args.results_folder}/")
    print(f"\nNext steps:")
    print(f"1. Test diffusion bridge: python test_visual_diffusion.py")
    print(f"2. Integrate into EZ-HOI inference using DiffusionBridgeHOI module")
    print(f"3. Update model checkpoint to enable diffusion_bridge parameter")
    print("="*60)


if __name__ == '__main__':
    main()
