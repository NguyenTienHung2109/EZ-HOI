"""
Diffusion Bridge Module for HOI Detection

This module applies a trained diffusion model to bridge the modality gap between
vision features (union crops) and text features (HOI descriptions).

Process:
1. Normalize vision features
2. Apply diffusion-bridge normalization chain (subtract mean, renormalize, scale)
3. Run DDIM sampling to refine features toward text distribution
4. Reverse normalization to standard CLIP space

Usage:
    bridge = DiffusionBridgeHOI(
        diffusion_path='hoi_diffusion_results/model-300.pt',
        text_mean_path='hicodet_pkl_files/hoi_text_mean_vitB_600.pkl',
        inference_steps=600
    )

    # At inference time
    vision_features = extract_union_features(image)  # [batch, 512]
    bridged_features = bridge(vision_features)  # [batch, 512]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import sys
import os
from pathlib import Path


class DiffusionBridgeHOI(nn.Module):
    """
    Frozen diffusion model for bridging vision→text modality gap in HOI detection.

    This module:
    - Loads a pretrained diffusion model (trained on HOI text embeddings)
    - Applies diffusion-bridge normalization to vision features
    - Uses DDIM sampling to refine features toward text distribution
    - Returns features in standard CLIP space for classification
    """

    def __init__(self, diffusion_path, text_mean_path, inference_steps=600,
                 scale_factor=5.0, verbose=False):
        """
        Args:
            diffusion_path: Path to trained diffusion model (.pt file)
            text_mean_path: Path to HOI text mean (.pkl file)
            inference_steps: Number of DDIM steps (100-1000, trade-off speed/quality)
            scale_factor: Scale factor for normalization chain (default: 5.0)
            verbose: Print detailed information during initialization
        """
        super().__init__()

        self.inference_steps = inference_steps
        self.scale_factor = scale_factor
        self.verbose = verbose

        if verbose:
            print("="*60)
            print("Initializing Diffusion Bridge Module")
            print("="*60)
            print(f"Diffusion model: {diffusion_path}")
            print(f"Text mean: {text_mean_path}")
            print(f"Inference steps: {inference_steps}")
            print(f"Scale factor: {scale_factor}")

        # Load trained diffusion model
        self.diffusion = self._load_diffusion_model(diffusion_path)

        # Load HOI text mean
        self.hoi_text_mean = self._load_text_mean(text_mean_path)

        # Freeze all parameters (inference only, no gradients)
        self.diffusion.eval()
        for param in self.diffusion.parameters():
            param.requires_grad = False

        if verbose:
            print(f"✓ Diffusion bridge initialized")
            print(f"  Text mean shape: {self.hoi_text_mean.shape}")
            print(f"  Text mean norm: {self.hoi_text_mean.norm().item():.6f}")
            print("="*60)

    def _load_diffusion_model(self, diffusion_path):
        """Load pretrained diffusion model from checkpoint"""
        # Add diffusion-bridge to path
        DIFFUSION_BRIDGE_PATH = os.path.join(os.getcwd(), 'diffusion-bridge', 'ddpm')
        if DIFFUSION_BRIDGE_PATH not in sys.path:
            sys.path.insert(0, DIFFUSION_BRIDGE_PATH)

        try:
            from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d import (
                Unet1D, GaussianDiffusion1D_norm
            )
        except ImportError as e:
            raise ImportError(
                f"Could not import diffusion modules. "
                f"Make sure diffusion-bridge is set up: {e}"
            )

        if not Path(diffusion_path).exists():
            raise FileNotFoundError(
                f"Diffusion model not found: {diffusion_path}\n"
                f"Please train the diffusion model first using train_hoi_diffusion.py"
            )

        if self.verbose:
            print(f"\nLoading diffusion model from: {diffusion_path}")

        # Load checkpoint
        checkpoint = torch.load(diffusion_path, map_location='cpu')

        # Extract model state dict
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Infer embedding dimension from state dict
        # The model's first layer should give us the embedding dimension
        for key in state_dict.keys():
            if 'init_conv' in key and 'weight' in key:
                embed_dim = state_dict[key].shape[2]  # seq_length dimension
                break
        else:
            # Fallback: assume 512 (ViT-B/16)
            embed_dim = 512
            if self.verbose:
                print(f"  Warning: Could not infer embedding dim, assuming {embed_dim}")

        if self.verbose:
            print(f"  Embedding dimension: {embed_dim}")

        # Reconstruct model architecture (must match training)
        model = Unet1D(
            dim=embed_dim,
            init_dim=32,
            dim_mults=(1, 2, 4, 8),
            channels=1
        )

        diffusion = GaussianDiffusion1D_norm(
            model,
            seq_length=embed_dim,
            timesteps=1000,
            objective='pred_x0',
            sampling_timesteps=self.inference_steps  # Use DDIM with fewer steps
        )

        # Load weights
        diffusion.load_state_dict(state_dict, strict=True)

        if self.verbose:
            num_params = sum(p.numel() for p in diffusion.parameters())
            print(f"  ✓ Loaded diffusion model ({num_params:,} parameters)")

        return diffusion

    def _load_text_mean(self, text_mean_path):
        """Load HOI text mean for normalization"""
        if not Path(text_mean_path).exists():
            raise FileNotFoundError(
                f"Text mean not found: {text_mean_path}\n"
                f"Please run extract_hoi_text_embeddings.py first"
            )

        if self.verbose:
            print(f"\nLoading text mean from: {text_mean_path}")

        with open(text_mean_path, 'rb') as f:
            text_mean = pickle.load(f)

        # Ensure it's a tensor
        if not isinstance(text_mean, torch.Tensor):
            text_mean = torch.tensor(text_mean)

        # Register as buffer (moves with model to GPU, but not trained)
        self.register_buffer('_text_mean_buffer', text_mean)

        if self.verbose:
            print(f"  ✓ Loaded text mean (shape: {text_mean.shape})")

        return text_mean

    @torch.no_grad()
    def forward(self, vision_features):
        """
        Bridge vision features to text distribution using diffusion.

        Args:
            vision_features: Vision embeddings from union crops [batch, embed_dim]

        Returns:
            bridged_features: Text-like vision embeddings [batch, embed_dim]
        """
        # Step 1: First L2 normalization
        x = F.normalize(vision_features, dim=-1)

        # Step 2: Subtract HOI text mean (center in modality space)
        x = x - self._text_mean_buffer.to(x.device)

        # Step 3: Second L2 normalization (project to unit sphere)
        x = F.normalize(x, dim=-1)

        # Step 4: Scale by factor (amplify signal for diffusion)
        x = x * self.scale_factor

        # Step 5: Add channel dimension for 1D convolution
        x = x.unsqueeze(1)  # [batch, 1, embed_dim]

        # Step 6: Apply DDIM sampling (refine toward text distribution)
        # This is the key step: diffusion model learned what text embeddings look like,
        # now we use it to transform vision embeddings to be more text-like
        x_bridged = self.diffusion.ddim_sample_with_img(x, inference_step=self.inference_steps)

        # Step 7: Remove channel dimension
        x_bridged = x_bridged.squeeze(1)  # [batch, embed_dim]

        # Step 8: Normalize back to unit sphere (standard CLIP space)
        x_bridged = F.normalize(x_bridged, dim=-1)

        return x_bridged

    def extra_repr(self):
        """String representation for printing model"""
        return (f"inference_steps={self.inference_steps}, "
                f"scale_factor={self.scale_factor}, "
                f"text_mean_shape={self._text_mean_buffer.shape}")


def test_diffusion_bridge():
    """
    Test function to verify diffusion bridge works correctly.

    Run with: python diffusion_bridge_module.py
    """
    print("\n" + "="*60)
    print("Testing Diffusion Bridge Module")
    print("="*60 + "\n")

    # Test parameters
    diffusion_path = 'hoi_diffusion_results/model-300.pt'
    text_mean_path = 'hicodet_pkl_files/hoi_text_mean_vitB_600.pkl'

    # Check if files exist
    if not Path(diffusion_path).exists():
        print(f"⚠️  Diffusion model not found: {diffusion_path}")
        print("Please train the diffusion model first:")
        print("  python train_hoi_diffusion.py")
        return

    if not Path(text_mean_path).exists():
        print(f"⚠️  Text mean not found: {text_mean_path}")
        print("Please extract text embeddings first:")
        print("  python extract_hoi_text_embeddings.py")
        return

    # Create module
    print("Creating diffusion bridge module...")
    bridge = DiffusionBridgeHOI(
        diffusion_path=diffusion_path,
        text_mean_path=text_mean_path,
        inference_steps=100,  # Use fewer steps for testing
        verbose=True
    )

    # Create dummy vision features
    batch_size = 4
    embed_dim = 512
    vision_features = torch.randn(batch_size, embed_dim)
    vision_features = F.normalize(vision_features, dim=-1)  # Normalize like real features

    print(f"\nInput vision features:")
    print(f"  Shape: {vision_features.shape}")
    print(f"  Norms: {vision_features.norm(dim=-1)}")

    # Apply diffusion bridge
    print(f"\nApplying diffusion bridge (inference_steps=100)...")
    bridged_features = bridge(vision_features)

    print(f"\nOutput bridged features:")
    print(f"  Shape: {bridged_features.shape}")
    print(f"  Norms: {bridged_features.norm(dim=-1)}")

    # Measure change
    cosine_sim = F.cosine_similarity(vision_features, bridged_features, dim=-1)
    print(f"\nCosine similarity (input vs output):")
    print(f"  Mean: {cosine_sim.mean().item():.4f}")
    print(f"  Std:  {cosine_sim.std().item():.4f}")

    print("\n✓ Diffusion bridge test completed successfully!")
    print("="*60 + "\n")


if __name__ == '__main__':
    test_diffusion_bridge()
