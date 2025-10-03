"""
Extract HOI text embeddings and compute statistics for diffusion bridge training.

This script:
1. Loads CLIP model (ViT-B/16 or ViT-L/14@336px)
2. Extracts text embeddings for ALL HOI classes (600 for full, 117 for verbs)
3. Computes modality mean using diffusion-bridge normalization strategy
4. Saves raw embeddings, normalized embeddings, and statistics
"""

import torch
import torch.nn.functional as F
import pickle
import argparse
import os
from pathlib import Path

# Add CLIP to path
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'CLIP'))

import clip
import hico_text_label
from hico_list import hico_verbs_sentence


def extract_text_embeddings(args):
    """Extract CLIP text embeddings for all HOI classes"""

    print(f"Loading CLIP model: {args.clip_model}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model, preprocess = clip.load(args.clip_model, device=device)
    clip_model.eval()

    # Get class names based on setting
    if args.num_classes == 600:
        classnames = list(hico_text_label.hico_text_label.values())
        print(f"Extracting embeddings for 600 full HOI classes")
    elif args.num_classes == 117:
        classnames = hico_verbs_sentence
        print(f"Extracting embeddings for 117 verb classes")
    else:
        raise ValueError(f"Unsupported num_classes: {args.num_classes}")

    print(f"Number of classes: {len(classnames)}")
    print(f"Example class: {classnames[0]}")

    # Tokenize and encode
    print("Tokenizing text...")
    text_tokens = torch.cat([
        clip.tokenize(classname, context_length=77, truncate=True)
        for classname in classnames
    ]).to(device)

    print("Encoding text with CLIP...")
    with torch.no_grad():
        text_embeddings = clip_model.encode_text(text_tokens)  # [num_classes, 512/768]

    text_embeddings = text_embeddings.cpu()
    print(f"Text embeddings shape: {text_embeddings.shape}")
    print(f"Embedding dimension: {text_embeddings.shape[1]}")

    return text_embeddings, classnames


def compute_statistics_and_normalize(text_embeddings, args):
    """
    Compute modality mean and normalize embeddings using diffusion-bridge strategy.

    Strategy:
    1. Compute mean from NORMALIZED embeddings (not raw)
    2. Apply double normalization: normalize(normalize(x) - mean) * scale
    """

    print("\n" + "="*60)
    print("Computing statistics (diffusion-bridge style)...")
    print("="*60)

    # Step 1: Compute mean from normalized embeddings
    print("\nStep 1: Computing mean from normalized embeddings...")
    text_mean = torch.zeros(1, text_embeddings.shape[1])

    for emb in text_embeddings:
        normalized_emb = F.normalize(emb.unsqueeze(0), dim=-1)
        text_mean += normalized_emb

    text_mean = text_mean / len(text_embeddings)

    print(f"Text mean shape: {text_mean.shape}")
    print(f"Text mean norm: {text_mean.norm().item():.4f}")

    # Step 2: Apply double normalization chain
    print("\nStep 2: Applying double normalization chain...")
    normalized_embeddings = []

    for i, emb in enumerate(text_embeddings):
        # First normalization
        x1 = F.normalize(emb.unsqueeze(0), dim=-1)

        # Subtract mean
        x2 = x1 - text_mean

        # Second normalization
        x3 = F.normalize(x2, dim=-1)

        # Scale
        x4 = x3 * args.scale_factor

        normalized_embeddings.append(x4.squeeze(0))

        if i == 0:
            print(f"\nExample normalization (first class):")
            print(f"  Original norm:          {emb.norm().item():.4f}")
            print(f"  After 1st normalize:    {x1.norm().item():.4f}")
            print(f"  After subtract mean:    {x2.norm().item():.4f}")
            print(f"  After 2nd normalize:    {x3.norm().item():.4f}")
            print(f"  After scale by {args.scale_factor}:      {x4.norm().item():.4f}")

    normalized_embeddings = torch.stack(normalized_embeddings)
    print(f"\nNormalized embeddings shape: {normalized_embeddings.shape}")
    print(f"Final embedding norms (mean): {normalized_embeddings.norm(dim=-1).mean().item():.4f}")
    print(f"Final embedding norms (std):  {normalized_embeddings.norm(dim=-1).std().item():.4f}")

    return text_mean, normalized_embeddings


def save_embeddings(text_embeddings, text_mean, normalized_embeddings, classnames, args):
    """Save all embeddings and statistics"""

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Determine file suffix based on CLIP model
    if 'ViT-B/16' in args.clip_model or 'ViT-B-16' in args.clip_model:
        suffix = 'vitB'
    elif 'ViT-L/14@336px' in args.clip_model or 'ViT-L-14-336px' in args.clip_model:
        suffix = 'vitL'
    else:
        suffix = 'custom'

    # Add num_classes to suffix
    suffix = f"{suffix}_{args.num_classes}"

    print("\n" + "="*60)
    print("Saving embeddings and statistics...")
    print("="*60)

    # Save raw embeddings
    raw_path = output_dir / f'hoi_text_embeddings_raw_{suffix}.pkl'
    with open(raw_path, 'wb') as f:
        pickle.dump({
            'embeddings': text_embeddings,
            'classnames': classnames,
            'clip_model': args.clip_model,
            'num_classes': args.num_classes
        }, f)
    print(f"\n✓ Saved raw embeddings to: {raw_path}")

    # Save normalized embeddings (for diffusion training)
    norm_path = output_dir / f'hoi_text_embeddings_normalized_{suffix}.pkl'
    with open(norm_path, 'wb') as f:
        pickle.dump({
            'embeddings': normalized_embeddings,
            'classnames': classnames,
            'clip_model': args.clip_model,
            'num_classes': args.num_classes,
            'scale_factor': args.scale_factor
        }, f)
    print(f"✓ Saved normalized embeddings to: {norm_path}")

    # Save text mean
    mean_path = output_dir / f'hoi_text_mean_{suffix}.pkl'
    with open(mean_path, 'wb') as f:
        pickle.dump(text_mean, f)
    print(f"✓ Saved text mean to: {mean_path}")

    # Save summary info
    summary_path = output_dir / f'hoi_text_embeddings_summary_{suffix}.txt'
    with open(summary_path, 'w') as f:
        f.write("HOI Text Embeddings Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"CLIP Model: {args.clip_model}\n")
        f.write(f"Number of classes: {args.num_classes}\n")
        f.write(f"Embedding dimension: {text_embeddings.shape[1]}\n")
        f.write(f"Scale factor: {args.scale_factor}\n\n")
        f.write(f"Text mean norm: {text_mean.norm().item():.6f}\n")
        f.write(f"Normalized embedding norms (mean ± std): "
                f"{normalized_embeddings.norm(dim=-1).mean().item():.4f} ± "
                f"{normalized_embeddings.norm(dim=-1).std().item():.4f}\n\n")
        f.write("Example classes:\n")
        for i in range(min(5, len(classnames))):
            f.write(f"  {i}: {classnames[i]}\n")
    print(f"✓ Saved summary to: {summary_path}")

    print("\n" + "="*60)
    print("All files saved successfully!")
    print("="*60)

    return {
        'raw_path': raw_path,
        'norm_path': norm_path,
        'mean_path': mean_path,
        'summary_path': summary_path
    }


def main():
    parser = argparse.ArgumentParser(description='Extract HOI text embeddings for diffusion bridge')

    # Model settings
    parser.add_argument('--clip_model', type=str, default='ViT-B/16',
                        choices=['ViT-B/16', 'ViT-L/14@336px'],
                        help='CLIP model variant (must match EZ-HOI training)')
    parser.add_argument('--num_classes', type=int, default=600,
                        choices=[600, 117],
                        help='Number of HOI classes (600=full, 117=verbs)')

    # Normalization settings
    parser.add_argument('--scale_factor', type=float, default=5.0,
                        help='Scale factor for normalized embeddings (diffusion-bridge uses 5.0)')

    # Output settings
    parser.add_argument('--output_dir', type=str, default='hicodet_pkl_files',
                        help='Directory to save embeddings')

    args = parser.parse_args()

    print("="*60)
    print("HOI Text Embedding Extraction")
    print("="*60)
    print(f"Configuration:")
    print(f"  CLIP model: {args.clip_model}")
    print(f"  Number of classes: {args.num_classes}")
    print(f"  Scale factor: {args.scale_factor}")
    print(f"  Output directory: {args.output_dir}")
    print("="*60 + "\n")

    # Extract embeddings
    text_embeddings, classnames = extract_text_embeddings(args)

    # Compute statistics and normalize
    text_mean, normalized_embeddings = compute_statistics_and_normalize(text_embeddings, args)

    # Save everything
    paths = save_embeddings(text_embeddings, text_mean, normalized_embeddings, classnames, args)

    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print(f"1. Use normalized embeddings for diffusion training:")
    print(f"   {paths['norm_path']}")
    print(f"\n2. Use text mean for diffusion bridge:")
    print(f"   {paths['mean_path']}")
    print(f"\n3. Train diffusion model:")
    print(f"   python train_hoi_diffusion.py \\")
    print(f"     --data_path {paths['norm_path']} \\")
    print(f"     --text_mean_path {paths['mean_path']}")
    print("="*60)


if __name__ == '__main__':
    main()
