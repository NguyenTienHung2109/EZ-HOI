"""
Extract Vision Embedding Mean for HICO-DET

This script computes the normalized mean of vision embeddings from pre-extracted
CLIP image features, following the same procedure as diffusion-bridge's
compute_embed_means.py.

Formula: vision_mean = (1/N) * Σ(normalize(vision_embed_i))

Usage:
    python extract_vision_mean.py --clip_model ViT-B/16
    python extract_vision_mean.py --clip_model ViT-L/14@336px
"""

import os
import pickle
import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse


def main(clip_model_type='ViT-B/16', split='train'):
    """
    Compute normalized vision embedding mean from pre-extracted CLIP features.

    Args:
        clip_model_type: CLIP model variant ('ViT-B/16' or 'ViT-L/14@336px')
        split: Dataset split ('train' or 'test')
    """
    # Determine paths based on CLIP model type
    if clip_model_type == 'ViT-L/14@336px':
        folder_name = 'clip336_img_hicodet_' + split
        output_file = f'hicodet_pkl_files/hoi_vision_mean_vitL.pkl'
        embed_dim = 768
    else:  # ViT-B/16
        folder_name = 'clipbase_img_hicodet_' + split
        output_file = f'hicodet_pkl_files/hoi_vision_mean_vitB.pkl'
        embed_dim = 512

    clip_features_dir = os.path.join('hicodet_pkl_files', folder_name)

    # Check if directory exists
    if not os.path.exists(clip_features_dir):
        raise FileNotFoundError(
            f"CLIP features directory not found: {clip_features_dir}\n"
            f"Please run CLIP_hicodet_extract.py first to extract features."
        )

    print("="*70)
    print("Extracting Vision Embedding Mean for HICO-DET")
    print("="*70)
    print(f"CLIP model: {clip_model_type}")
    print(f"Split: {split}")
    print(f"Input directory: {clip_features_dir}")
    print(f"Output file: {output_file}")
    print(f"Embedding dimension: {embed_dim}")
    print("="*70 + "\n")

    # Get all pickle files
    pkl_files = [f for f in os.listdir(clip_features_dir) if f.endswith('_clip.pkl')]

    if len(pkl_files) == 0:
        raise ValueError(
            f"No CLIP feature files found in {clip_features_dir}\n"
            f"Expected files with pattern: *_clip.pkl"
        )

    print(f"Found {len(pkl_files)} CLIP feature files\n")

    # Initialize mean accumulator
    vision_mean = torch.zeros(1, embed_dim)

    print("Computing normalized vision mean...")
    print("Formula: vision_mean = (1/N) * Σ(normalize(vision_embed_i))\n")

    # Process each file
    for pkl_file in tqdm(pkl_files, desc="Processing images"):
        pkl_path = os.path.join(clip_features_dir, pkl_file)

        try:
            with open(pkl_path, 'rb') as f:
                # Load CLIP image features
                # Note: CLIP_hicodet_extract.py saves image_features.squeeze(0)[1:]
                # which removes batch dim and CLS token, giving patch features
                # We need to use the full image embedding instead
                image_features = pickle.load(f)

                # Check if this is patch features or global feature
                if image_features.dim() == 2:
                    # This is patch features [num_patches, embed_dim]
                    # For computing mean, we should use global pooling or CLS token
                    # But since the extraction script already removed CLS token,
                    # let's use mean pooling across patches
                    image_embed = image_features.mean(dim=0, keepdim=True)  # [1, embed_dim]
                else:
                    # Single embedding vector
                    image_embed = image_features.unsqueeze(0) if image_features.dim() == 1 else image_features

                # Normalize and accumulate
                # Following diffusion-bridge: img_mean += img_embed / img_embed.norm()
                vision_mean += image_embed / image_embed.norm()

        except Exception as e:
            print(f"\nWarning: Failed to process {pkl_file}: {e}")
            continue

    # Compute final mean
    vision_mean = vision_mean / len(pkl_files)

    # Save to pickle file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(vision_mean, f)

    print("\n" + "="*70)
    print("✓ Vision mean computed successfully!")
    print("="*70)
    print(f"Total images processed: {len(pkl_files)}")
    print(f"Vision mean shape: {vision_mean.shape}")
    print(f"Vision mean norm: {vision_mean.norm().item():.6f}")
    print(f"Saved to: {output_file}")
    print("="*70 + "\n")

    # Verify the saved file
    print("Verifying saved file...")
    with open(output_file, 'rb') as f:
        loaded_mean = pickle.load(f)
    print(f"✓ Loaded mean shape: {loaded_mean.shape}")
    print(f"✓ Loaded mean norm: {loaded_mean.norm().item():.6f}")
    print("\nReady to use in diffusion_bridge_module.py!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract vision embedding mean from HICO-DET CLIP features')
    parser.add_argument('--clip_model', type=str, default='ViT-B/16',
                        choices=['ViT-B/16', 'ViT-L/14@336px'],
                        help='CLIP model variant')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'test'],
                        help='Dataset split to use for computing mean')

    args = parser.parse_args()

    main(args.clip_model, args.split)
