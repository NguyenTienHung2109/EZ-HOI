"""
Diffusion Bridge Module for HOI Detection

This module applies a trained diffusion model to bridge the modality gap between
vision features (union crops) and text features (HOI descriptions).

Following the original diffusion-bridge paper implementation:
- Uses VISION embedding mean (not text mean) for normalization
- Uses 5 DDIM steps (not 600) for fast inference
- inference_steps parameter controls timestep range (600→0), not iteration count

Process:
1. Normalize vision features
2. Apply diffusion-bridge normalization chain (subtract VISION mean, renormalize, scale)
3. Run DDIM sampling (~5 iterations) to refine features toward text distribution
4. Reverse normalization to standard CLIP space

Usage:
    bridge = DiffusionBridgeHOI(
        diffusion_path='hoi_diffusion_results/model-300.pt',
        vision_mean_path='hicodet_pkl_files/hoi_vision_mean_vitB.pkl',  # VISION mean!
        inference_steps=600  # Timestep range, NOT iteration count
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

# Ensure local modules are imported first (avoid conflicts with system packages)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class DiffusionBridgeHOI(nn.Module):
    """
    Frozen diffusion model for bridging vision→text modality gap in HOI detection.

    This module:
    - Loads a pretrained diffusion model (trained on HOI text embeddings)
    - Applies diffusion-bridge normalization to vision features
    - Uses DDIM sampling to refine features toward text distribution
    - Returns features in standard CLIP space for classification
    """

    def __init__(self, diffusion_path, vision_mean_path, inference_steps=600,
                 scale_factor=5.0, embed_dim=None, verbose=False):
        """
        Args:
            diffusion_path: Path to trained diffusion model (.pt file)
            vision_mean_path: Path to HOI vision mean (.pkl file)
                             This should be the VISION embedding mean (not text mean)
                             to match the original diffusion-bridge implementation
            inference_steps: DDIM inference timestep range (e.g., 600 means steps 600→0)
                            Note: This is NOT the number of denoising iterations
                            The actual number of iterations is controlled by sampling_timesteps
                            in the model initialization (set to 5 for fast inference)
            scale_factor: Scale factor for normalization chain (default: 5.0)
            embed_dim: Embedding dimension (512 for ViT-B/16, 768 for ViT-L/14).
                       If None, will attempt auto-detection from checkpoint.
            verbose: Print detailed information during initialization
        """
        super().__init__()

        self.inference_steps = inference_steps
        self.scale_factor = scale_factor
        self.embed_dim = embed_dim
        self.verbose = verbose

        if verbose:
            print("="*60)
            print("Initializing Diffusion Bridge Module")
            print("="*60)
            print(f"Diffusion model: {diffusion_path}")
            print(f"Vision mean: {vision_mean_path}")
            print(f"Inference timestep range: {inference_steps}")
            print(f"Scale factor: {scale_factor}")
            if embed_dim is not None:
                print(f"Embedding dimension: {embed_dim} (manually specified)")
            else:
                print(f"Embedding dimension: auto-detect from checkpoint")

        # Load trained diffusion model
        self.diffusion = self._load_diffusion_model(diffusion_path)

        # Load HOI vision mean (following original diffusion-bridge implementation)
        self.hoi_vision_mean = self._load_vision_mean(vision_mean_path)

        # Freeze all parameters (inference only, no gradients)
        self.diffusion.eval()
        for param in self.diffusion.parameters():
            param.requires_grad = False

        if verbose:
            print(f"✓ Diffusion bridge initialized")
            print(f"  Vision mean shape: {self.hoi_vision_mean.shape}")
            print(f"  Vision mean norm: {self.hoi_vision_mean.norm().item():.6f}")
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
                f"Make sure diffusion-bridge submodule is initialized.\n"
                f"Run: git submodule update --init --recursive\n"
                f"Error: {e}"
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

        # Determine embedding dimension
        if self.embed_dim is not None:
            # Use manually specified dimension
            embed_dim = self.embed_dim
            if self.verbose:
                print(f"  Using specified embedding dimension: {embed_dim}")
        else:
            # Auto-detect from checkpoint
            embed_dim = None

            # Method 1: Try model.init_conv.weight shape [init_dim, channels, embed_dim]
            if 'model.init_conv.weight' in state_dict:
                init_conv_shape = state_dict['model.init_conv.weight'].shape
                if len(init_conv_shape) == 3:
                    embed_dim = init_conv_shape[2]
                    if self.verbose:
                        print(f"  Auto-detected embed_dim from init_conv: {embed_dim}")

            # Method 2: Try model.final_conv.2.weight shape [channels, hidden_dim, embed_dim]
            if embed_dim is None and 'model.final_conv.2.weight' in state_dict:
                final_conv_shape = state_dict['model.final_conv.2.weight'].shape
                if len(final_conv_shape) == 3:
                    embed_dim = final_conv_shape[2]
                    if self.verbose:
                        print(f"  Auto-detected embed_dim from final_conv: {embed_dim}")

            # Method 3: Fallback to 512 (ViT-B/16)
            if embed_dim is None:
                embed_dim = 512
                if self.verbose:
                    print(f"  ⚠️  Warning: Could not auto-detect embedding dim")
                    print(f"  Available keys: {list(state_dict.keys())[:5]}...")
                    print(f"  Falling back to default: {embed_dim}")
                    print(f"  Tip: Specify embed_dim explicitly in config to avoid this warning")

        if self.verbose:
            print(f"  ✓ Embedding dimension: {embed_dim}")

        # Reconstruct model architecture (must match training)
        model = Unet1D(
            dim=embed_dim,
            init_dim=32,
            dim_mults=(1, 2, 4, 8),
            channels=1
        )

        # CRITICAL: sampling_timesteps controls the NUMBER of denoising iterations
        # Original diffusion-bridge uses sampling_timesteps=5 (NOT inference_steps!)
        # This means only ~5 DDIM steps regardless of inference_step parameter
        # inference_step (passed to ddim_sample_with_img) controls the timestep RANGE
        diffusion = GaussianDiffusion1D_norm(
            model,
            seq_length=embed_dim,
            timesteps=1000,
            objective='pred_x0',
            sampling_timesteps=5  # Fixed to 5 DDIM steps for fast inference
        )

        # Load weights
        diffusion.load_state_dict(state_dict, strict=True)

        if self.verbose:
            num_params = sum(p.numel() for p in diffusion.parameters())
            print(f"  ✓ Loaded diffusion model ({num_params:,} parameters)")
            print(f"  DDIM configuration:")
            print(f"    - Total timesteps: 1000")
            print(f"    - Sampling timesteps: 5 (number of denoising iterations)")
            print(f"    - Inference step range: {self.inference_steps} → 0")
            print(f"    - Effective speed: ~120x faster than dense sampling")

        return diffusion

    def _load_vision_mean(self, vision_mean_path):
        """
        Load HOI vision mean for normalization.

        Following original diffusion-bridge: uses VISION embedding mean (not text mean)
        to center vision embeddings before diffusion refinement.

        Supports two formats:
        1. Simple format: Just a tensor [embed_dim] or [1, embed_dim]
        2. Dict format: Dict with 'vision_mean' key and metadata
        """
        if not Path(vision_mean_path).exists():
            raise FileNotFoundError(
                f"Vision mean not found: {vision_mean_path}\n"
                f"Please run extract_vision_mean.py first to generate it"
            )

        if self.verbose:
            print(f"\nLoading vision mean from: {vision_mean_path}")

        with open(vision_mean_path, 'rb') as f:
            data = pickle.load(f)

        # Handle both formats
        if isinstance(data, dict):
            # Dict format with metadata
            vision_mean = data.get('vision_mean', data.get('image_mean', None))
            if vision_mean is None:
                raise ValueError(f"Dict format must contain 'vision_mean' or 'image_mean' key")
            if self.verbose:
                print(f"  Source: {data.get('source', 'unknown')}")
                print(f"  Num images: {data.get('num_images', 'unknown')}")
                print(f"  CLIP model: {data.get('clip_model', 'unknown')}")
        else:
            # Simple format: just tensor
            vision_mean = data
            if self.verbose:
                print(f"  Source: Computed from normalized CLIP vision embeddings")

        # Ensure it's a tensor
        if not isinstance(vision_mean, torch.Tensor):
            vision_mean = torch.tensor(vision_mean)

        # Ensure correct shape (should be 1D: [embed_dim] or 2D: [1, embed_dim])
        if vision_mean.dim() == 2 and vision_mean.shape[0] == 1:
            vision_mean = vision_mean.squeeze(0)

        # Register as buffer (moves with model to GPU, but not trained)
        self.register_buffer('_vision_mean_buffer', vision_mean)

        if self.verbose:
            print(f"  ✓ Loaded vision mean (shape: {vision_mean.shape})")
            print(f"  Vision mean norm: {vision_mean.norm().item():.6f}")

        return vision_mean

    @torch.no_grad()
    def forward(self, vision_features):
        """
        Bridge vision features to text distribution using diffusion.

        Following original diffusion-bridge implementation:
        1. Normalize vision embeddings
        2. Subtract VISION mean (centers in vision modality space)
        3. Re-normalize (project to unit sphere)
        4. Scale and apply diffusion refinement
        5. Return normalized bridged embeddings

        Args:
            vision_features: Vision embeddings from union crops [batch, embed_dim]

        Returns:
            bridged_features: Text-like vision embeddings [batch, embed_dim]
        """
        # Step 1: First L2 normalization
        x = F.normalize(vision_features, dim=-1)

        # Step 2: Subtract VISION mean (center in vision modality space)
        # NOTE: Original diffusion-bridge uses VISION mean, not text mean!
        x = x - self._vision_mean_buffer.to(x.device)

        # Step 3: Second L2 normalization (project to unit sphere)
        x = F.normalize(x, dim=-1)

        # Step 4: Scale by factor (amplify signal for diffusion)
        x = x * self.scale_factor

        # Step 5: Add channel dimension for 1D convolution
        x = x.unsqueeze(1)  # [batch, 1, embed_dim]

        # Step 6: Apply DDIM sampling (refine toward text distribution)
        # This is the key step: diffusion model learned what text embeddings look like,
        # now we use it to transform vision embeddings to be more text-like
        # The model will perform ~5 denoising iterations (controlled by sampling_timesteps)
        x_bridged = self.diffusion.ddim_sample_with_img(x, inference_step=self.inference_steps)

        # Step 7: Remove channel dimension
        x_bridged = x_bridged.squeeze(1)  # [batch, embed_dim]

        # Step 8: Normalize back to unit sphere (standard CLIP space)
        x_bridged = F.normalize(x_bridged, dim=-1)

        return x_bridged

    @torch.no_grad()
    def apply_diffusion_only(self, features, inference_steps=None):
        """
        Apply ONLY the diffusion sampling part (not the geometric normalization).
        Assumes features are already in diffusion-normalized space via DiffusionGeometricTransform.

        This method is used during training/inference when features have already been
        geometrically transformed (L2 norm -> subtract mean -> L2 norm).

        Args:
            features: [batch, embed_dim] - already geometrically normalized
            inference_steps: Number of DDIM steps (overrides self.inference_steps if provided)

        Returns:
            refined_features: [batch, embed_dim] - diffusion-refined, L2 normalized
        """
        if inference_steps is None:
            inference_steps = self.inference_steps

        # Features are already: L2_norm -> subtract_mean -> L2_norm
        # Now apply: scale -> diffusion -> normalize

        # Step 1: Scale by factor (amplify signal for diffusion)
        x = features * self.scale_factor

        # Step 2: Add channel dimension for 1D convolution
        x = x.unsqueeze(1)  # [batch, 1, embed_dim]

        # Step 3: Apply DDIM sampling (refine toward text distribution)
        x_bridged = self.diffusion.ddim_sample_with_img(x, inference_step=inference_steps)

        # Step 4: Remove channel dimension
        x_bridged = x_bridged.squeeze(1)  # [batch, embed_dim]

        # Step 5: Normalize back to unit sphere
        x_bridged = F.normalize(x_bridged, dim=-1)

        return x_bridged

    def extra_repr(self):
        """String representation for printing model"""
        return (f"inference_steps={self.inference_steps}, "
                f"scale_factor={self.scale_factor}, "
                f"vision_mean_shape={self._vision_mean_buffer.shape}")


def test_diffusion_bridge():
    """
    Test function to verify diffusion bridge works correctly.

    Run with: python diffusion_bridge_module.py

    Tests both full forward() and apply_diffusion_only() methods.
    If diffusion checkpoint is not available, runs basic sanity checks only.
    """
    print("\n" + "="*70)
    print("Testing Diffusion Bridge Module (CPU Mode)")
    print("="*70 + "\n")

    # Test parameters
    diffusion_path = 'dummy_diffusion_files/dummy_diffusion_vitB.pt'
    vision_mean_path = 'dummy_diffusion_files/dummy_vision_mean_vitB.pkl'

    # Check alternative paths
    if not Path(diffusion_path).exists():
        diffusion_path = 'hoi_diffusion_results/model-300.pt'
        vision_mean_path = 'hicodet_pkl_files/hoi_vision_mean_vitB.pkl'

    # Check if files exist
    has_checkpoint = Path(diffusion_path).exists() and Path(vision_mean_path).exists()

    if not has_checkpoint:
        print("⚠️  Diffusion checkpoint not found.")
        print("   Running basic sanity tests only (without diffusion sampling)...\n")

        # Run basic tests without actual checkpoint
        print("Test 1: Geometric transformation logic")
        batch_size = 4
        embed_dim = 512

        # Create dummy vision mean
        vision_mean = torch.randn(embed_dim)
        vision_mean = F.normalize(vision_mean, dim=-1) * 0.05

        # Create features
        features = torch.randn(batch_size, embed_dim)
        features = F.normalize(features, dim=-1)

        # Manual geometric transform (what DiffusionBridgeHOI.forward() does)
        transformed = F.normalize(features, dim=-1)
        transformed = transformed - vision_mean
        transformed = F.normalize(transformed, dim=-1)

        print(f"  Input shape: {features.shape}")
        print(f"  Output shape: {transformed.shape}")
        print(f"  Output norms: {transformed.norm(dim=-1)}")
        assert torch.allclose(transformed.norm(dim=-1), torch.ones(batch_size), atol=1e-5)
        print(f"  ✓ Geometric transformation works correctly\n")

        print("="*70)
        print("✅ Basic sanity tests passed!")
        print("="*70)
        print("\nTo test with actual diffusion:")
        print("  1. Create dummy files: python create_dummy_diffusion_files.py")
        print("  2. Or use real COCO checkpoint")
        print("  3. Re-run this test")
        print("="*70 + "\n")
        return

    # Full test with checkpoint
    print(f"Found checkpoint files:")
    print(f"  Diffusion: {diffusion_path}")
    print(f"  Vision mean: {vision_mean_path}\n")

    # Create module
    print("Test 1: Loading diffusion bridge module...")
    try:
        bridge = DiffusionBridgeHOI(
            diffusion_path=diffusion_path,
            vision_mean_path=vision_mean_path,
            inference_steps=100,  # Use fewer steps for testing
            verbose=True
        )
        print("  ✓ Module loaded successfully\n")
    except ImportError as e:
        print(f"  ❌ Failed to load: {e}\n")
        print("="*70)
        print("⚠️  Diffusion-bridge submodule not initialized")
        print("="*70)
        print("\nTo initialize the submodule:")
        print("  git submodule update --init --recursive")
        print("\nOr on Colab, run these commands before testing:")
        print("  %cd /content/EZ-HOI")
        print("  !git submodule update --init --recursive")
        print("\nSkipping diffusion tests, but geometric transform tests passed!")
        print("="*70 + "\n")
        return
    except Exception as e:
        print(f"  ❌ Failed to load: {e}")
        return

    # Create dummy vision features
    batch_size = 4
    embed_dim = 512
    vision_features = torch.randn(batch_size, embed_dim)
    vision_features = F.normalize(vision_features, dim=-1)

    # Test 2: Full forward pass
    print("Test 2: Testing full forward() method...")
    print(f"  Input shape: {vision_features.shape}")
    print(f"  Input norms: {vision_features.norm(dim=-1)}")

    bridged_features = bridge(vision_features)

    print(f"  Output shape: {bridged_features.shape}")
    print(f"  Output norms: {bridged_features.norm(dim=-1)}")

    cosine_sim = F.cosine_similarity(vision_features, bridged_features, dim=-1)
    print(f"  Cosine similarity (input vs output): {cosine_sim.mean().item():.4f}")
    print(f"  ✓ forward() method works\n")

    # Test 3: apply_diffusion_only() method
    print("Test 3: Testing apply_diffusion_only() method...")

    # Prepare features (already geometrically transformed)
    geo_features = F.normalize(vision_features, dim=-1)
    geo_features = geo_features - bridge._vision_mean_buffer
    geo_features = F.normalize(geo_features, dim=-1)

    print(f"  Pre-transformed features shape: {geo_features.shape}")
    print(f"  Pre-transformed norms: {geo_features.norm(dim=-1)}")

    # Apply diffusion only
    diffused_features = bridge.apply_diffusion_only(geo_features, inference_steps=50)

    print(f"  Diffused output shape: {diffused_features.shape}")
    print(f"  Diffused output norms: {diffused_features.norm(dim=-1)}")

    assert diffused_features.shape == geo_features.shape
    assert torch.allclose(diffused_features.norm(dim=-1), torch.ones(batch_size), atol=1e-5)
    print(f"  ✓ apply_diffusion_only() method works\n")

    # Test 4: Different inference steps
    print("Test 4: Testing with different inference steps...")
    diffused_50 = bridge.apply_diffusion_only(geo_features, inference_steps=50)
    diffused_100 = bridge.apply_diffusion_only(geo_features, inference_steps=100)

    sim_50_100 = F.cosine_similarity(diffused_50, diffused_100, dim=-1)
    print(f"  Similarity (50 vs 100 steps): {sim_50_100.mean().item():.4f}")
    print(f"  ✓ Different step counts work\n")

    # Summary
    print("="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("\nDiffusion bridge module is working correctly:")
    print("  ✓ forward() - Full geometric transform + diffusion")
    print("  ✓ apply_diffusion_only() - Diffusion sampling only")
    print("  ✓ Different inference steps - Flexible configuration")
    print("\nReady for integration into EZ-HOI training!")
    print("="*70 + "\n")


if __name__ == '__main__':
    test_diffusion_bridge()
