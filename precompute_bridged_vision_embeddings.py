"""
Pre-compute Diffusion-Bridged Vision Embeddings for EZ-HOI

This script bridges existing CLIP vision embeddings using a trained diffusion model,
eliminating the need for runtime diffusion inference during training.

Usage:
    python precompute_bridged_vision_embeddings.py \
        --input_dir hicodet_pkl_files/clipbase_img_hicodet_train \
        --output_dir hicodet_pkl_files/clipbase_img_hicodet_train_bridged \
        --diffusion_model hoi_diffusion_results/model-300.pt \
        --vision_mean hicodet_pkl_files/hoi_vision_mean_vitB_600.pkl \
        --embed_dim 512 \
        --inference_steps 600 \
        --scale_factor 5.0

For ViT-L:
    python precompute_bridged_vision_embeddings.py \
        --input_dir hicodet_pkl_files/clip336_img_hicodet_train \
        --output_dir hicodet_pkl_files/clip336_img_hicodet_train_bridged \
        --diffusion_model hoi_diffusion_results/model-vitL-300.pt \
        --vision_mean hicodet_pkl_files/hoi_vision_mean_vitL_600.pkl \
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
from joblib import Parallel, delayed

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


def process_files_on_gpu_joblib_single_model(files, output_dir, diffusion_model_path, vision_mean_path, device, embed_dim, inference_steps, scale_factor):
    """
    Process a list of files on a specific GPU using a single model instance.

    Args:
        files: List of pkl files to process
        output_dir: Directory to save the bridged embeddings
        diffusion_model_path: Path to the diffusion model checkpoint
        vision_mean_path: Path to the vision mean file
        device: torch.device
        embed_dim: Embedding dimension
        inference_steps: Number of inference steps
        scale_factor: Scale factor for normalization
    """
    # Initialize a single model for this GPU
    diffusion_bridge = DiffusionBridgeHOI(
        diffusion_path=diffusion_model_path,
        vision_mean_path=vision_mean_path,
        inference_steps=inference_steps,
        scale_factor=scale_factor,
        embed_dim=embed_dim,
        verbose=False
    ).to(device)  # Ensure model is moved to the correct device
    diffusion_bridge.eval()

    success_count = 0
    error_count = 0
    
    for pkl_path in tqdm(files, desc=f"Bridging embeddings on {device}", dynamic_ncols=True):
        try:
            # Bridge the embedding
            bridged_embedding = bridge_single_pkl(pkl_path, diffusion_bridge, device)

            # Save to output directory with same filename
            output_path = os.path.join(output_dir, pkl_path.name)
            with open(output_path, 'wb') as f:
                pickle.dump(bridged_embedding, f)
            success_count += 1

        except Exception as e:
            print(f"\nError processing {pkl_path.name} on {device}: {e}")
            error_count += 1
    
    print(f"\nFinished processing on {device}: {success_count} succeeded, {error_count} failed.")


def process_directory_joblib_single_model(input_dir, output_dir, diffusion_model_path, vision_mean_path, num_gpus, embed_dim, inference_steps, scale_factor):
    """
    Process all pkl files in input directory in parallel across multiple GPUs using joblib.
    Each GPU creates a single model instance to process its assigned files.

    Args:
        input_dir: Source directory containing original pkl files
        output_dir: Target directory for bridged pkl files
        diffusion_model_path: Path to the diffusion model checkpoint
        vision_mean_path: Path to the vision mean file
        num_gpus: Number of GPUs to use
        embed_dim: Embedding dimension
        inference_steps: Number of inference steps
        scale_factor: Scale factor for normalization
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

    # Divide files among GPUs
    files_per_gpu = len(pkl_files) // num_gpus
    file_splits = [pkl_files[i * files_per_gpu: (i + 1) * files_per_gpu] for i in range(num_gpus)]


    # Assign remaining files to the last GPU
    if len(pkl_files) % num_gpus != 0:
        file_splits[-1].extend(pkl_files[num_gpus * files_per_gpu:])    

    print(f"Using {num_gpus} GPUs for processing")
    print("File distribution among GPUs:")
    for gpu_id, files in enumerate(file_splits):
        print(f"  GPU {gpu_id}: {len(files)} files")

    # Process files in parallel using joblib
    Parallel(n_jobs=num_gpus)(
        delayed(process_files_on_gpu_joblib_single_model)(
            files, output_dir, diffusion_model_path, vision_mean_path, f"cuda:{gpu_id}", embed_dim, inference_steps, scale_factor
        ) for gpu_id, files in enumerate(file_splits)
    )


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
    parser.add_argument('--vision_mean', type=str, required=True,
                        help='Path to HOI vision mean file (.pkl)')
    parser.add_argument('--embed_dim', type=int, default=512,
                        help='Embedding dimension (512 for ViT-B, 768 for ViT-L)')
    parser.add_argument('--inference_steps', type=int, default=600,
                        help='Number of DDIM sampling steps (default: 600)')
    parser.add_argument('--scale_factor', type=float, default=5.0,
                        help='Normalization scale factor (default: 5.0)')

    # Device configuration
    parser.add_argument('--num_gpus', type=int, default=torch.cuda.device_count(),
                        help='Number of GPUs to use (default: all available)')

    args = parser.parse_args()

    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    # Check if diffusion model and vision mean exist
    if not os.path.exists(args.diffusion_model):
        raise FileNotFoundError(f"Diffusion model not found: {args.diffusion_model}")
    if not os.path.exists(args.vision_mean):
        raise FileNotFoundError(f"Vision mean file not found: {args.vision_mean}")

    # Process directory in parallel across GPUs using joblib
    process_directory_joblib_single_model(
        args.input_dir, args.output_dir, args.diffusion_model, args.vision_mean,
        args.num_gpus, args.embed_dim, args.inference_steps, args.scale_factor
    )

    print()
    print("All done! You can now use the bridged embeddings for training.")
    print(f"Bridged embeddings saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
