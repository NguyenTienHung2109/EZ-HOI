"""
Train diffusion model on HOI text embeddings.

Adapted from diffusion-bridge/ddpm/train_norm.py but for HOI-specific text distribution.

Key differences:
- Dataset: 600 HOI text embeddings vs 400k MSCOCO captions
- Training steps: 300k vs 3M (smaller dataset converges faster)
- Distribution: Structured HOI templates vs free-form captions
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
    """Load HOI text embeddings (already normalized by extract_hoi_text_embeddings.py)"""
    print(f"Loading HOI text embeddings from: {data_path}")

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    embeddings = data['embeddings']  # [num_classes, embed_dim]
    num_classes = data['num_classes']
    clip_model = data['clip_model']
    scale_factor = data.get('scale_factor', 5.0)

    print(f"  CLIP model: {clip_model}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    print(f"  Scale factor: {scale_factor}")
    print(f"  Embedding norms (mean ± std): "
          f"{embeddings.norm(dim=-1).mean().item():.4f} ± "
          f"{embeddings.norm(dim=-1).std().item():.4f}")

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
    parser = argparse.ArgumentParser(description='Train diffusion model on HOI text embeddings')

    # Data paths
    parser.add_argument('--data_path', type=str,
                        default='hicodet_pkl_files/hoi_text_embeddings_normalized_vitB_600.pkl',
                        help='Path to normalized HOI text embeddings')

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
    parser.add_argument('--train_steps', type=int, default=300000,
                        help='Total training steps (reduced from 3M for small dataset)')
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
    print(f"  Time: ~6-12 hours (depending on GPU)")
    print("="*60 + "\n")

    trainer.train()

    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print(f"\nTrained model saved to: {args.results_folder}/")
    print(f"\nNext steps:")
    print(f"1. Create diffusion bridge module using trained model")
    print(f"2. Integrate into EZ-HOI for inference")
    print("="*60)


if __name__ == '__main__':
    main()
