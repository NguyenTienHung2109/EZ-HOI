"""
Pre-compute Diffusion-Bridged Vision Embeddings for EZ-HOI

This script bridges existing CLIP vision embeddings using a trained diffusion model,
eliminating the need for runtime diffusion inference during training.

Usage:
    python precompute_bridged_vision_embeddings.py \
        --input_dir hicodet_pkl_files/clipbase_img_hicodet_train \
        --output_dir hicodet_pkl_files/clipbase_img_hicodet_train_bridged \
        --diffusion_model hoi_diffusion_results/model-300.pt \
        --text_mean hicodet_pkl_files/hoi_text_mean_vitB_600.pkl \
        --embed_dim 512 \
        --inference_steps 600 \
        --scale_factor 5.0

For ViT-L:
    python precompute_bridged_vision_embeddings.py \
        --input_dir hicodet_pkl_files/clip336_img_hicodet_train \
        --output_dir hicodet_pkl_files/clip336_img_hicodet_train_bridged \
        --diffusion_model hoi_diffusion_results/model-vitL-300.pt \
        --text_mean hicodet_pkl_files/hoi_text_mean_vitL_600.pkl \
        --embed_dim 768 \
        --inference_steps 600 \
        --scale_factor 5.0
"""

import os
import sys
import pickle
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

# Ensure local modules are imported first
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffusion_bridge_module import DiffusionBridgeHOI


def bridge_single_pkl(pkl_path, diffusion_bridge, device):
    """
    Load a single pkl file, apply diffusion bridge, return bridged embedding.

    Args:
        pkl_path: Path to input pkl file
        diffusion_bridge: DiffusionBridgeHOI module
        device: torch.device

    Returns:
        bridged_embedding: Tensor after diffusion bridging
    """
    # Load original embedding
    with open(pkl_path, 'rb') as f:
        embedding = pickle.load(f)

    # Ensure tensor format
    if not isinstance(embedding, torch.Tensor):
        embedding = torch.tensor(embedding)

    # Move to device and ensure correct shape [1, embed_dim]
    if embedding.dim() == 1:
        embedding = embedding.unsqueeze(0)
    embedding = embedding.to(device)

    # Apply diffusion bridge
    with torch.no_grad():
        bridged_embedding = diffusion_bridge(embedding)

    # Move back to CPU and squeeze
    bridged_embedding = bridged_embedding.cpu().squeeze(0)

    return bridged_embedding


def process_directory(input_dir, output_dir, diffusion_bridge, device):
    """
    Process all pkl files in input directory and save bridged versions to output directory.

    Args:
        input_dir: Source directory containing original pkl files
        output_dir: Target directory for bridged pkl files
        diffusion_bridge: DiffusionBridgeHOI module
        device: torch.device
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all pkl files
    pkl_files = list(Path(input_dir).glob("*_clip.pkl"))

    if len(pkl_files) == 0:
        print(f"Warning: No pkl files found in {input_dir}")
        return

    print(f"Found {len(pkl_files)} pkl files to process")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Process each file
    success_count = 0
    error_count = 0

    for pkl_path in tqdm(pkl_files, desc="Bridging embeddings", dynamic_ncols=True):
        try:
            # Bridge the embedding
            bridged_embedding = bridge_single_pkl(pkl_path, diffusion_bridge, device)

            # Save to output directory with same filename
            output_path = os.path.join(output_dir, pkl_path.name)
            with open(output_path, 'wb') as f:
                pickle.dump(bridged_embedding, f)

            success_count += 1

        except Exception as e:
            print(f"\nError processing {pkl_path.name}: {e}")
            error_count += 1
            continue

    print()
    print("="*60)
    print(f"Processing complete!")
    print(f"  Successfully bridged: {success_count} files")
    if error_count > 0:
        print(f"  Errors: {error_count} files")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Pre-compute diffusion-bridged vision embeddings")

    # Input/Output paths
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing original CLIP vision embeddings')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save bridged embeddings')

    # Diffusion bridge configuration
    parser.add_argument('--diffusion_model', type=str, required=True,
                        help='Path to trained diffusion model checkpoint (.pt)')
    parser.add_argument('--text_mean', type=str, required=True,
                        help='Path to HOI text mean file (.pkl)')
    parser.add_argument('--embed_dim', type=int, default=512,
                        help='Embedding dimension (512 for ViT-B, 768 for ViT-L)')
    parser.add_argument('--inference_steps', type=int, default=600,
                        help='Number of DDIM sampling steps (default: 600)')
    parser.add_argument('--scale_factor', type=float, default=5.0,
                        help='Normalization scale factor (default: 5.0)')

    # Device configuration
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()

    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    # Check if diffusion model and text mean exist
    if not os.path.exists(args.diffusion_model):
        raise FileNotFoundError(f"Diffusion model not found: {args.diffusion_model}")
    if not os.path.exists(args.text_mean):
        raise FileNotFoundError(f"Text mean file not found: {args.text_mean}")

    # Initialize diffusion bridge
    print("Initializing Diffusion Bridge...")
    print("="*60)
    diffusion_bridge = DiffusionBridgeHOI(
        diffusion_path=args.diffusion_model,
        text_mean_path=args.text_mean,
        inference_steps=args.inference_steps,
        scale_factor=args.scale_factor,
        embed_dim=args.embed_dim,
        verbose=True
    )
    diffusion_bridge = diffusion_bridge.to(device)
    diffusion_bridge.eval()
    print()

    # Process directory
    process_directory(args.input_dir, args.output_dir, diffusion_bridge, device)

    print()
    print("All done! You can now use the bridged embeddings for training.")
    print(f"Bridged embeddings saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
