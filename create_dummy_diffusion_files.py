#!/usr/bin/env python3
"""
Create dummy diffusion checkpoint and text mean files for testing.

This script creates minimal valid files that allow the code to run
without actual trained diffusion models. Use only for testing!

Usage:
    python create_dummy_diffusion_files.py
"""

import torch
import pickle
import os
from pathlib import Path

def create_dummy_diffusion_checkpoint(output_path, embed_dim=512):
    """
    Create a dummy diffusion checkpoint with valid structure.

    This creates a minimal UNet1D + GaussianDiffusion1D_norm state dict
    that will pass the loading checks.
    """
    print(f"\nCreating dummy diffusion checkpoint...")
    print(f"  Embedding dimension: {embed_dim}")
    print(f"  Output path: {output_path}")

    # Create dummy state dict matching the expected architecture
    # Based on Unet1D(dim=512, init_dim=32, dim_mults=(1,2,4,8), channels=1)
    state_dict = {}

    # Initial convolution layer - this is used to infer embed_dim
    state_dict['model.init_conv.weight'] = torch.randn(32, 1, embed_dim)
    state_dict['model.init_conv.bias'] = torch.randn(32)

    # Time MLP
    state_dict['model.time_mlp.0.weight'] = torch.randn(128, 32)
    state_dict['model.time_mlp.0.bias'] = torch.randn(128)
    state_dict['model.time_mlp.2.weight'] = torch.randn(128, 128)
    state_dict['model.time_mlp.2.bias'] = torch.randn(128)

    # Downsample blocks (simplified)
    dims = [32, 64, 128, 256]
    for i, dim in enumerate(dims):
        # Basic conv layers
        state_dict[f'model.downs.{i}.0.block1.weight'] = torch.randn(dim, dim, 7)
        state_dict[f'model.downs.{i}.0.block1.bias'] = torch.randn(dim)
        state_dict[f'model.downs.{i}.0.block2.weight'] = torch.randn(dim, dim, 7)
        state_dict[f'model.downs.{i}.0.block2.bias'] = torch.randn(dim)

    # Middle block
    state_dict['model.mid_block1.weight'] = torch.randn(256, 256, 7)
    state_dict['model.mid_block1.bias'] = torch.randn(256)
    state_dict['model.mid_block2.weight'] = torch.randn(256, 256, 7)
    state_dict['model.mid_block2.bias'] = torch.randn(256)

    # Upsample blocks (simplified)
    for i, dim in enumerate(reversed(dims)):
        state_dict[f'model.ups.{i}.0.block1.weight'] = torch.randn(dim, dim*2, 7)
        state_dict[f'model.ups.{i}.0.block1.bias'] = torch.randn(dim)
        state_dict[f'model.ups.{i}.0.block2.weight'] = torch.randn(dim, dim, 7)
        state_dict[f'model.ups.{i}.0.block2.bias'] = torch.randn(dim)

    # Final layers
    state_dict['model.final_conv.0.weight'] = torch.randn(32, 32)
    state_dict['model.final_conv.0.bias'] = torch.randn(32)
    state_dict['model.final_conv.2.weight'] = torch.randn(1, 32, embed_dim)
    state_dict['model.final_conv.2.bias'] = torch.randn(1)

    # Diffusion-specific parameters
    state_dict['betas'] = torch.linspace(0.0001, 0.02, 1000)
    state_dict['alphas_cumprod'] = torch.cumprod(1 - state_dict['betas'], dim=0)
    state_dict['sqrt_alphas_cumprod'] = torch.sqrt(state_dict['alphas_cumprod'])
    state_dict['sqrt_one_minus_alphas_cumprod'] = torch.sqrt(1 - state_dict['alphas_cumprod'])

    # Save as checkpoint dict
    checkpoint = {
        'model': state_dict,
        'epoch': 300,
        'note': 'DUMMY CHECKPOINT FOR TESTING ONLY - NOT A REAL MODEL'
    }

    # Create directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save
    torch.save(checkpoint, output_path)

    print(f"  ✓ Created dummy diffusion checkpoint")
    print(f"  Size: {os.path.getsize(output_path) / 1024:.2f} KB")
    print(f"  State dict keys: {len(state_dict)}")
    print(f"  WARNING: This is a DUMMY file for testing only!")

    return output_path


def create_dummy_text_mean(output_path, embed_dim=512):
    """
    Create a dummy text mean tensor.

    This creates a normalized random tensor that mimics a real text mean.
    """
    print(f"\nCreating dummy text mean...")
    print(f"  Embedding dimension: {embed_dim}")
    print(f"  Output path: {output_path}")

    # Create random normalized vector (mimics average of text embeddings)
    text_mean = torch.randn(embed_dim)
    text_mean = text_mean / text_mean.norm()  # Normalize
    text_mean = text_mean * 0.05  # Scale to realistic magnitude (~0.05 norm)

    # Save in the expected format
    data = {
        'text_mean': text_mean,
        'num_classes': 'unknown',
        'clip_model': f'dummy_{embed_dim}',
        'source': 'DUMMY FILE FOR TESTING ONLY - NOT REAL DATA',
        'note': 'This is a randomly generated tensor, not actual COCO text mean!'
    }

    # Create directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"  ✓ Created dummy text mean")
    print(f"  Size: {os.path.getsize(output_path) / 1024:.2f} KB")
    print(f"  Shape: {text_mean.shape}")
    print(f"  Norm: {text_mean.norm().item():.6f}")
    print(f"  WARNING: This is a DUMMY file for testing only!")

    return output_path


def main():
    print("=" * 70)
    print("Creating Dummy Diffusion Files for Testing")
    print("=" * 70)
    print("\nWARNING: These are DUMMY files with random weights!")
    print("They allow the code to run but will NOT produce meaningful results.")
    print("Replace with real trained diffusion checkpoint before actual training!")
    print("=" * 70)

    # Determine embedding dimension (512 for ViT-B/16, 768 for ViT-L/14)
    embed_dim = 512  # Change to 768 for ViT-L/14

    # Paths for dummy files
    dummy_dir = "dummy_diffusion_files"
    diffusion_path = f"{dummy_dir}/dummy_diffusion_vitB.pt"
    text_mean_path = f"{dummy_dir}/dummy_text_mean_vitB.pkl"

    # Create files
    create_dummy_diffusion_checkpoint(diffusion_path, embed_dim=embed_dim)
    create_dummy_text_mean(text_mean_path, embed_dim=embed_dim)

    # Also create ViT-L variants
    print("\n" + "=" * 70)
    print("Creating ViT-L/14 variants...")
    print("=" * 70)

    diffusion_path_L = f"{dummy_dir}/dummy_diffusion_vitL.pt"
    text_mean_path_L = f"{dummy_dir}/dummy_text_mean_vitL.pkl"

    create_dummy_diffusion_checkpoint(diffusion_path_L, embed_dim=768)
    create_dummy_text_mean(text_mean_path_L, embed_dim=768)

    # Update config file
    print("\n" + "=" * 70)
    print("Updating diffusion_bridge_config.yaml...")
    print("=" * 70)

    config_content = f"""# Diffusion Bridge Configuration (USING DUMMY FILES FOR TESTING)
#
# WARNING: This config uses DUMMY files with random weights!
# The code will run but results will be meaningless.
# Replace with real COCO diffusion checkpoint before actual training!

# ============================================================
# DUMMY FILES (for testing code only)
# ============================================================

# Path to DUMMY diffusion checkpoint (random weights!)
model_path: '{diffusion_path}'

# Path to DUMMY text mean (random vector!)
text_mean_path: '{text_mean_path}'

# ============================================================
# Diffusion Sampling Steps
# ============================================================

training_steps: 100
inference_steps: 600

# ============================================================
# Diffusion Parameters
# ============================================================

scale_factor: 5.0

# ============================================================
# TO USE REAL DIFFUSION:
# ============================================================
# 1. Train diffusion model on MS-COCO (or obtain pretrained checkpoint)
# 2. Update model_path to point to real checkpoint
# 3. Update text_mean_path to point to real COCO text mean
# 4. Remove this warning comment
#
"""

    config_path = "configs/diffusion_bridge_config.yaml"
    with open(config_path, 'w') as f:
        f.write(config_content)

    print(f"  ✓ Updated {config_path}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary - Dummy Files Created")
    print("=" * 70)
    print(f"\n✓ ViT-B/16 files:")
    print(f"  - {diffusion_path}")
    print(f"  - {text_mean_path}")
    print(f"\n✓ ViT-L/14 files:")
    print(f"  - {diffusion_path_L}")
    print(f"  - {text_mean_path_L}")
    print(f"\n✓ Config updated: {config_path}")

    print("\n" + "=" * 70)
    print("Next Steps")
    print("=" * 70)
    print("\n1. Test the code runs without errors:")
    print("   bash scripts/hico_train_vitB_zs_diff.sh")
    print("\n2. Check for any runtime issues")
    print("\n3. Replace dummy files with real COCO diffusion checkpoint:")
    print("   - Update model_path in configs/diffusion_bridge_config.yaml")
    print("   - Update text_mean_path in configs/diffusion_bridge_config.yaml")
    print("\n4. Run actual training with real checkpoint")

    print("\n" + "=" * 70)
    print("⚠️  IMPORTANT WARNINGS")
    print("=" * 70)
    print("\n- These dummy files use RANDOM WEIGHTS")
    print("- Training will run but results will be MEANINGLESS")
    print("- Only use for testing code integration")
    print("- Do NOT evaluate or publish results with dummy files")
    print("- Replace with real trained diffusion before real experiments")

    print("\n" + "=" * 70)
    print("✅ Done! Dummy files created for testing.")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
