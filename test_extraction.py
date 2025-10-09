"""
Test Extraction: Quick verification script

This script performs a quick test to verify that:
1. Model modifications are correctly applied
2. Extraction mode works
3. Visual and text embeddings can be captured
4. Embeddings have correct shapes

Usage:
    python test_extraction.py --checkpoint your_checkpoint.pth

Expected output:
    ✓ Model loaded successfully
    ✓ Extraction mode enabled
    ✓ Visual embeddings extracted: torch.Size([N, 512])
    ✓ Text embeddings extracted: torch.Size([212, 512])
    ✓ All checks passed!
"""

import torch
import argparse
import os
import sys

# Add CLIP to path
sys.path.insert(0, os.path.join(os.getcwd(), 'CLIP'))

from upt_tip_cache_model_free_finetune_distillself import build_detector
from utils_tip_cache_and_union_finetune import DataFactory, custom_collate
from torch.utils.data import DataLoader


def test_extraction(checkpoint_path, device='cuda'):
    """
    Run quick extraction test.

    Args:
        checkpoint_path: Path to checkpoint
        device: Device to use

    Returns:
        True if all tests pass, False otherwise
    """
    print("\n" + "="*60)
    print("TESTING EXTRACTION FUNCTIONALITY")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print("="*60 + "\n")

    try:
        # 1. Load checkpoint
        print("[1/6] Loading checkpoint...")
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return False

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if 'args' not in checkpoint:
            print(f"❌ Checkpoint missing 'args'")
            return False

        args = checkpoint['args']
        print(f"✓ Checkpoint loaded")

        # 2. Initialize distributed training
        print("\n[2/6] Initializing distributed training...")
        import torch.distributed as dist
        if not dist.is_initialized():
            try:
                backend = "nccl" if torch.cuda.is_available() else "gloo"
                dist.init_process_group(
                    backend=backend,
                    init_method='tcp://localhost:12358',
                    world_size=1,
                    rank=0
                )
                print(f"✓ Distributed initialized (backend: {backend})")
            except Exception as e:
                print(f"⚠️  Warning: Could not initialize distributed: {e}")

        # 3. Build model
        print("\n[3/6] Building model...")
        from main_tip_finetune import hico_class_corr
        import numpy as np

        object_to_target = hico_class_corr()
        lut = np.full([80, 117], None)
        for hoi_idx, obj_idx, verb_idx in object_to_target:
            lut[obj_idx, verb_idx] = hoi_idx
        object_n_verb_to_interaction = lut.tolist()

        model = build_detector(args, object_n_verb_to_interaction)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model = model.to(device)
        model.eval()
        print(f"✓ Model built and loaded")

        # 4. Enable extraction mode
        print("\n[4/6] Enabling extraction mode...")
        model.extraction_mode = True

        # Verify the flag is set
        if not hasattr(model, 'extraction_mode') or not model.extraction_mode:
            print(f"❌ Failed to set extraction_mode")
            return False

        print(f"✓ Extraction mode enabled")

        # 5. Create test dataloader
        print("\n[5/6] Creating test dataloader...")

        clip_model_name = 'ViT-B/16'
        if hasattr(args, 'clip_model_name'):
            clip_model_name = args.clip_model_name
        elif hasattr(args, 'clip_dir_vit') and 'ViT-L' in args.clip_dir_vit:
            clip_model_name = 'ViT-L/14@336px'

        dataset = DataFactory(
            name='hicodet',
            partition='test2015',
            data_root=args.data_root if hasattr(args, 'data_root') else 'hicodet',
            clip_model_name=clip_model_name,
            zero_shot=args.zs if hasattr(args, 'zs') else False,
            zs_type=args.zs_type if hasattr(args, 'zs_type') else 'rare_first',
            num_classes=args.num_classes
        )

        dataloader = DataLoader(
            dataset=dataset,
            collate_fn=custom_collate,
            batch_size=1,
            num_workers=2,
            pin_memory=True,
            shuffle=False
        )

        print(f"✓ Dataloader created ({len(dataset)} images)")

        # 6. Test extraction on one batch
        print("\n[6/6] Testing extraction on sample batch...")

        images, targets = next(iter(dataloader))

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

        # Forward pass
        with torch.no_grad():
            _ = model(images_to_pass, targets_to_pass)

        # Check if embeddings were captured
        visual_extracted = False
        text_extracted = False
        visual_shape = None
        text_shape = None

        if hasattr(model, '_extracted_visual_feat'):
            visual_feat = model._extracted_visual_feat
            if visual_feat is not None and len(visual_feat) > 0:
                visual_extracted = True
                visual_shape = visual_feat.shape
                print(f"✓ Visual embeddings extracted: {visual_shape}")
            else:
                print(f"❌ Visual embeddings empty")
        else:
            print(f"❌ Model missing '_extracted_visual_feat' attribute")
            print(f"   Did you add the extraction code to compute_roi_embeddings()?")

        if hasattr(model, '_extracted_text_feat'):
            text_feat = model._extracted_text_feat
            if text_feat is not None and len(text_feat) > 0:
                text_extracted = True
                text_shape = text_feat.shape
                print(f"✓ Text embeddings extracted: {text_shape}")
            else:
                print(f"❌ Text embeddings empty")
        else:
            print(f"❌ Model missing '_extracted_text_feat' attribute")
            print(f"   Did you add the extraction code to forward()?")

        # Final verification
        print("\n" + "="*60)
        print("TEST RESULTS")
        print("="*60)

        if visual_extracted and text_extracted:
            print("✅ ALL CHECKS PASSED!")
            print("\nExtracted shapes:")
            print(f"  Visual: {visual_shape}")
            print(f"  Text:   {text_shape}")

            # Check if shapes are reasonable
            expected_embed_dim = 512 if 'ViT-B' in clip_model_name else 768
            if visual_shape[-1] != expected_embed_dim:
                print(f"\n⚠️  Warning: Expected embedding dim {expected_embed_dim}, got {visual_shape[-1]}")
            if text_shape[-1] != expected_embed_dim:
                print(f"\n⚠️  Warning: Expected embedding dim {expected_embed_dim}, got {text_shape[-1]}")

            print("\n" + "="*60)
            print("You can now run full extraction:")
            print("  python extract_adapted_embeddings.py \\")
            print(f"    --checkpoint {checkpoint_path} \\")
            print("    --num_samples 1000")
            print("="*60)
            return True
        else:
            print("❌ EXTRACTION FAILED")
            print("\nPossible issues:")
            if not visual_extracted:
                print("  - Visual embeddings not captured")
                print("  - Check MODIFICATION_GUIDE.md for correct code placement")
                print("  - Ensure modification is in compute_roi_embeddings() method")
            if not text_extracted:
                print("  - Text embeddings not captured")
                print("  - Check MODIFICATION_GUIDE.md for correct code placement")
                print("  - Ensure modification is after text_encoder call")
            print("\nPlease review MODIFICATION_GUIDE.md and add the required 4 lines")
            print("="*60)
            return False

    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Test extraction functionality')

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    success = test_extraction(args.checkpoint, device=args.device)

    if success:
        print("\n✅ Test completed successfully!")
        return 0
    else:
        print("\n❌ Test failed! Please review errors above.")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
