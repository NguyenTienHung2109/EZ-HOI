#!/usr/bin/env python3
"""
Diagnostic script to check CLIP feature dimensions in pickle files.

This script checks if the pre-extracted CLIP features have the correct dimensions.
Expected: 512 for ViT-B/16, 768 for ViT-L/14

Usage:
    python check_clip_dimensions.py
"""

import os
import pickle
import sys

def check_pickle_dimensions(pkl_dir, expected_dim=512):
    """
    Check dimensions of all pickle files in a directory.

    Args:
        pkl_dir: Directory containing .pkl files
        expected_dim: Expected embedding dimension (512 for ViT-B, 768 for ViT-L)

    Returns:
        True if all files have correct dimensions, False otherwise
    """
    if not os.path.exists(pkl_dir):
        print(f"❌ Directory not found: {pkl_dir}")
        return False

    pkl_files = [f for f in os.listdir(pkl_dir) if f.endswith('.pkl')]

    if not pkl_files:
        print(f"⚠️  No .pkl files found in: {pkl_dir}")
        return True

    print(f"\n{'='*70}")
    print(f"Checking: {pkl_dir}")
    print(f"Expected dimension: {expected_dim}")
    print(f"Total files: {len(pkl_files)}")
    print(f"{'='*70}\n")

    all_correct = True

    for i, pkl_file in enumerate(pkl_files, 1):
        pkl_path = os.path.join(pkl_dir, pkl_file)

        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)

            # Check if it's a tensor (has shape attribute)
            if hasattr(data, 'shape'):
                shape = data.shape

                # Check last dimension (embedding dimension)
                if len(shape) > 0:
                    actual_dim = shape[-1]
                else:
                    actual_dim = shape[0] if len(shape) == 1 else "unknown"

                # Print progress every 100 files
                if i % 100 == 0:
                    print(f"  Checked {i}/{len(pkl_files)} files...")

                # Check for dimension mismatch
                if actual_dim != expected_dim:
                    print(f"\n{'='*70}")
                    print(f"❌ DIMENSION MISMATCH FOUND!")
                    print(f"{'='*70}")
                    print(f"File: {pkl_file}")
                    print(f"Full path: {pkl_path}")
                    print(f"Expected dimension: {expected_dim}")
                    print(f"Actual dimension: {actual_dim}")
                    print(f"Full shape: {shape}")
                    print(f"Data type: {type(data)}")
                    print(f"{'='*70}\n")

                    all_correct = False

                    # STOP immediately on first mismatch
                    print("⛔ Stopping check - found problematic file!")
                    return False
            else:
                print(f"⚠️  Warning: {pkl_file} doesn't have shape attribute (type: {type(data)})")

        except Exception as e:
            print(f"❌ Error loading {pkl_file}: {e}")
            all_correct = False

    if all_correct:
        print(f"\n✅ All {len(pkl_files)} files have correct dimensions ({expected_dim})\n")

    return all_correct


def main():
    print("\n" + "="*70)
    print("CLIP Feature Dimension Diagnostic Tool")
    print("="*70)

    # Directories to check
    dirs_to_check = [
        ("hicodet_pkl_files/clipbase_img_hicodet_train", 512),   # ViT-B/16 train
        ("hicodet_pkl_files/clipbase_img_hicodet_test", 512),    # ViT-B/16 test
        ("hicodet_pkl_files/clip336_img_hicodet_train", 768),    # ViT-L/14 train
        ("hicodet_pkl_files/clip336_img_hicodet_test", 768),     # ViT-L/14 test
    ]

    all_passed = True

    for pkl_dir, expected_dim in dirs_to_check:
        passed = check_pickle_dimensions(pkl_dir, expected_dim)
        if not passed:
            all_passed = False
            break  # Stop on first error

    # Final summary
    print("\n" + "="*70)
    if all_passed:
        print("✅ RESULT: All CLIP features have correct dimensions!")
        print("="*70 + "\n")
        sys.exit(0)
    else:
        print("❌ RESULT: Found dimension mismatch in pickle files!")
        print("="*70)
        print("\nPossible causes:")
        print("1. CLIP extraction used wrong CLIP model version")
        print("2. Features were extracted with modified CLIP code")
        print("3. PYTHONPATH pointed to wrong CLIP implementation")
        print("\nSolution:")
        print("1. Re-run CLIP_hicodet_extract.py with correct PYTHONPATH:")
        print("   export PYTHONPATH=$PYTHONPATH:\"$PWD/CLIP\"")
        print("   python CLIP_hicodet_extract.py")
        print("2. Or regenerate features from scratch")
        print("="*70 + "\n")
        sys.exit(1)


if __name__ == '__main__':
    main()
