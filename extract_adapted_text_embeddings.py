"""
Extract ADAPTED text embeddings from trained EZ-HOI model.

This script:
1. Loads a trained EZ-HOI checkpoint
2. Recreates the model structure with all configurations
3. Forward passes text through learned adapters (txtmem_adapter, act_descriptor_attn)
4. Extracts ADAPTED text embeddings (after all modifications)
5. Applies diffusion-bridge normalization strategy
6. Saves embeddings for diffusion training

Key insight: Text embeddings in EZ-HOI are modified by learnable modules during training.
Diffusion should be trained on these ADAPTED embeddings, not raw CLIP embeddings, to ensure
vision-text alignment consistency at inference time.

Usage:
    python extract_adapted_text_embeddings.py \
        --checkpoint checkpoints/hico_HO_pt_default_vitbase/best.pth \
        --num_classes 600 \
        --zs_type unseen_verb \
        --output_dir hicodet_pkl_files
"""

import torch
import torch.nn.functional as F
import pickle
import argparse
import os
import sys
from pathlib import Path

# Add CLIP to path
sys.path.insert(0, os.path.join(os.getcwd(), 'CLIP'))

import clip
from upt_tip_cache_model_free_finetune_distillself import build_detector
from hico_text_label import hico_unseen_index
import hico_text_label


def load_checkpoint_and_args(checkpoint_path):
    """Load checkpoint and extract training arguments"""
    print(f"Loading checkpoint from: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'args' not in checkpoint:
        print("Warning: 'args' not found in checkpoint. You need to provide all arguments manually.")
        return checkpoint, None

    args = checkpoint['args']
    print(f"\nLoaded training configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Number of classes: {args.num_classes}")
    print(f"  Zero-shot: {args.zs}")
    if args.zs:
        print(f"  Zero-shot type: {args.zs_type}")
    print(f"  VLM text: {args.vlmtxt if hasattr(args, 'vlmtxt') else 'None'}")
    print(f"  Text align: {args.txt_align}")
    print(f"  Text class prompt: {args.txtcls_pt}")
    print(f"  Unseen prompt injection: {args.unseen_pt_inj}")
    print(f"  Action descriptor: {args.act_descriptor if hasattr(args, 'act_descriptor') else False}")

    return checkpoint, args


def recreate_model(args, device='cuda'):
    """Recreate the model structure with exact training configuration"""
    print("\n" + "="*60)
    print("Recreating model structure...")
    print("="*60)

    # Import necessary functions
    from main_tip_finetune import hico_class_corr, vcoco_class_corr
    from hico_list import hico_verb_object_list, hico_verbs
    from vcoco_list import vcoco_verb_object_list, vcoco_verbs

    # Setup dataset-specific configurations
    if args.dataset == 'hicodet':
        object_to_target = hico_class_corr()
        object_n_verb_to_interaction = hico_verb_object_list()
    else:
        object_to_target = vcoco_class_corr()
        object_n_verb_to_interaction = vcoco_verb_object_list()

    # Build detector (this will initialize all modules including text adapters)
    print(f"Building detector with configuration from checkpoint...")
    model = build_detector(
        args=args,
        class_corr=object_to_target,
        object_n_verb_to_interaction=object_n_verb_to_interaction,
        clip_model_path=args.clip_dir_vit,
        num_anno=None,
        verb2interaction=None
    )

    model = model.to(device)
    model.eval()

    print(f"✓ Model recreated successfully")
    return model


def extract_adapted_embeddings(model, device='cuda'):
    """
    Extract adapted text embeddings from the model.

    Flow:
    1. Get initial HOI text embeddings (hoicls_txt)
    2. If txt_align: Apply txtmem_adapter
    3. If act_descriptor: Apply act_descriptor_attn
    4. Return final adapted embeddings
    """
    print("\n" + "="*60)
    print("Extracting adapted text embeddings...")
    print("="*60)

    with torch.no_grad():
        # Get initial text embeddings (already loaded in model during build_detector)
        hoicls_txt = model.hoicls_txt.to(device)  # [num_classes, embed_dim]
        select_HOI_index = model.select_HOI_index

        # Select relevant classes
        hoitxt_features = hoicls_txt[select_HOI_index]  # [selected_classes, embed_dim]

        print(f"\nInitial embeddings shape: {hoitxt_features.shape}")
        print(f"Number of selected classes: {len(select_HOI_index)}")
        print(f"Embedding dimension: {hoitxt_features.shape[1]}")

        # Apply text adapter if enabled
        if model.txt_align:
            print(f"\n✓ Applying text adapter (txtmem_adapter)...")
            adapted_features = model.txtmem_adapter(hoitxt_features.unsqueeze(0)).squeeze(0)
            print(f"  After adapter shape: {adapted_features.shape}")
        else:
            print(f"\n✗ Text adapter disabled, using original embeddings")
            adapted_features = hoitxt_features

        # Apply action descriptor attention if enabled
        if len(model.act_descriptor_feat_select) == 2 and len(model.act_descriptor_feat_select[0]) > 0:
            print(f"\n✓ Applying action descriptor attention...")
            # act_descriptor_feat_select: (descriptor_features, sequence_indices)
            adapted_features = model.act_descriptor_attn(
                adapted_features.unsqueeze(0),
                (model.act_descriptor_feat_select[0][model.act_descriptor_feat_select[1]], None)
            ).squeeze(0)
            print(f"  After descriptor attention shape: {adapted_features.shape}")
        else:
            print(f"\n✗ Action descriptor attention disabled")

        # Map back to full 600 classes if needed
        if len(select_HOI_index) < len(hoicls_txt):
            print(f"\n✓ Mapping selected classes back to full {len(hoicls_txt)} classes...")
            full_embeddings = torch.zeros_like(hoicls_txt)
            full_embeddings[select_HOI_index] = adapted_features
            final_embeddings = full_embeddings
        else:
            final_embeddings = adapted_features

        print(f"\nFinal adapted embeddings shape: {final_embeddings.shape}")
        print(f"Embedding norms (mean ± std): {final_embeddings.norm(dim=-1).mean().item():.4f} ± {final_embeddings.norm(dim=-1).std().item():.4f}")

    return final_embeddings.cpu(), select_HOI_index


def apply_diffusion_normalization(embeddings, scale_factor=5.0):
    """
    Apply diffusion-bridge normalization strategy.

    Strategy (from diffusion-bridge paper):
    1. Compute modality mean from NORMALIZED embeddings
    2. Apply double normalization chain: normalize(normalize(x) - mean) * scale
    """
    print("\n" + "="*60)
    print("Applying diffusion-bridge normalization...")
    print("="*60)

    # Step 1: Compute mean from normalized embeddings
    print("\nStep 1: Computing text modality mean...")
    text_mean = torch.zeros(1, embeddings.shape[1])

    for emb in embeddings:
        normalized_emb = F.normalize(emb.unsqueeze(0), dim=-1)
        text_mean += normalized_emb

    text_mean = text_mean / len(embeddings)
    print(f"  Text mean shape: {text_mean.shape}")
    print(f"  Text mean norm: {text_mean.norm().item():.4f}")

    # Step 2: Apply double normalization chain
    print("\nStep 2: Applying double normalization chain...")
    normalized_embeddings = []

    for i, emb in enumerate(embeddings):
        # First normalization
        x1 = F.normalize(emb.unsqueeze(0), dim=-1)

        # Subtract mean
        x2 = x1 - text_mean

        # Second normalization
        x3 = F.normalize(x2, dim=-1)

        # Scale
        x4 = x3 * scale_factor

        normalized_embeddings.append(x4.squeeze(0))

        if i == 0:
            print(f"\nExample normalization (first class):")
            print(f"  Original norm:          {emb.norm().item():.4f}")
            print(f"  After 1st normalize:    {x1.norm().item():.4f}")
            print(f"  After subtract mean:    {x2.norm().item():.4f}")
            print(f"  After 2nd normalize:    {x3.norm().item():.4f}")
            print(f"  After scale by {scale_factor}:      {x4.norm().item():.4f}")

    normalized_embeddings = torch.stack(normalized_embeddings)
    print(f"\nNormalized embeddings shape: {normalized_embeddings.shape}")
    print(f"Final embedding norms (mean ± std): {normalized_embeddings.norm(dim=-1).mean().item():.4f} ± {normalized_embeddings.norm(dim=-1).std().item():.4f}")

    return text_mean, normalized_embeddings


def save_embeddings(raw_embeddings, text_mean, normalized_embeddings,
                   select_HOI_index, args, output_dir):
    """Save all embeddings and statistics"""

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Determine file suffix
    clip_model = args.clip_dir_vit
    if 'ViT-B-16' in clip_model or 'ViT-B/16' in clip_model:
        suffix = 'vitB'
    elif 'ViT-L-14-336px' in clip_model or 'ViT-L/14@336px' in clip_model:
        suffix = 'vitL'
    else:
        suffix = 'custom'

    suffix = f"{suffix}_{args.num_classes}"

    # Add additional identifier for adapted embeddings
    if args.zs:
        suffix = f"adapted_{args.zs_type}_{suffix}"
    else:
        suffix = f"adapted_{suffix}"

    print("\n" + "="*60)
    print("Saving embeddings and statistics...")
    print("="*60)

    # Save raw adapted embeddings
    raw_path = output_dir / f'hoi_text_embeddings_{suffix}_raw.pkl'
    with open(raw_path, 'wb') as f:
        pickle.dump({
            'embeddings': raw_embeddings,
            'select_HOI_index': select_HOI_index,
            'num_classes': args.num_classes,
            'checkpoint': args.resume if hasattr(args, 'resume') else 'unknown',
            'txt_align': args.txt_align,
            'txtcls_pt': args.txtcls_pt,
            'act_descriptor': args.act_descriptor if hasattr(args, 'act_descriptor') else False,
            'vlmtxt': args.vlmtxt if hasattr(args, 'vlmtxt') else None
        }, f)
    print(f"\n✓ Saved raw adapted embeddings to: {raw_path}")

    # Save normalized embeddings (for diffusion training)
    norm_path = output_dir / f'hoi_text_embeddings_{suffix}_normalized.pkl'
    with open(norm_path, 'wb') as f:
        pickle.dump({
            'embeddings': normalized_embeddings,
            'select_HOI_index': select_HOI_index,
            'num_classes': args.num_classes,
            'scale_factor': 5.0,
            'checkpoint': args.resume if hasattr(args, 'resume') else 'unknown',
            'txt_align': args.txt_align,
            'txtcls_pt': args.txtcls_pt
        }, f)
    print(f"✓ Saved normalized embeddings to: {norm_path}")

    # Save text mean
    mean_path = output_dir / f'hoi_text_mean_{suffix}.pkl'
    with open(mean_path, 'wb') as f:
        pickle.dump(text_mean, f)
    print(f"✓ Saved text mean to: {mean_path}")

    # Save summary
    summary_path = output_dir / f'hoi_text_embeddings_{suffix}_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("Adapted HOI Text Embeddings Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"Checkpoint: {args.resume if hasattr(args, 'resume') else 'unknown'}\n")
        f.write(f"Number of classes: {args.num_classes}\n")
        f.write(f"Selected classes: {len(select_HOI_index)}\n")
        f.write(f"Embedding dimension: {raw_embeddings.shape[1]}\n")
        f.write(f"Scale factor: 5.0\n\n")
        f.write("Adaptation modules applied:\n")
        f.write(f"  - Text align (txtmem_adapter): {args.txt_align}\n")
        f.write(f"  - Text class prompts: {args.txtcls_pt}\n")
        f.write(f"  - Action descriptor: {args.act_descriptor if hasattr(args, 'act_descriptor') else False}\n")
        f.write(f"  - VLM text: {args.vlmtxt if hasattr(args, 'vlmtxt') else 'None'}\n\n")
        f.write(f"Text mean norm: {text_mean.norm().item():.6f}\n")
        f.write(f"Normalized embedding norms (mean ± std): "
                f"{normalized_embeddings.norm(dim=-1).mean().item():.4f} ± "
                f"{normalized_embeddings.norm(dim=-1).std().item():.4f}\n")
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
    parser = argparse.ArgumentParser(
        description='Extract adapted text embeddings from trained EZ-HOI model'
    )

    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained EZ-HOI checkpoint (.pth file)')

    # Optional arguments (can be inferred from checkpoint if available)
    parser.add_argument('--num_classes', type=int, default=None,
                        help='Number of HOI classes (600 or 117). If None, read from checkpoint.')
    parser.add_argument('--zs_type', type=str, default=None,
                        help='Zero-shot type. If None, read from checkpoint.')
    parser.add_argument('--output_dir', type=str, default='hicodet_pkl_files',
                        help='Directory to save embeddings')
    parser.add_argument('--scale_factor', type=float, default=5.0,
                        help='Scale factor for normalization (diffusion-bridge uses 5.0)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run on')

    args = parser.parse_args()

    print("="*60)
    print("Extract Adapted Text Embeddings from Trained EZ-HOI")
    print("="*60)
    print(f"Configuration:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Scale factor: {args.scale_factor}")
    print(f"  Device: {args.device}")
    print("="*60 + "\n")

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = 'cpu'

    # Load checkpoint and training args
    checkpoint, ckpt_args = load_checkpoint_and_args(args.checkpoint)

    if ckpt_args is None:
        raise ValueError(
            "Cannot extract args from checkpoint. "
            "Please ensure the checkpoint was saved with training arguments."
        )

    # Override checkpoint args with command-line args if provided
    if args.num_classes is not None:
        ckpt_args.num_classes = args.num_classes
    if args.zs_type is not None:
        ckpt_args.zs_type = args.zs_type

    # Recreate model structure
    model = recreate_model(ckpt_args, device=args.device)

    # Load weights
    print("\nLoading model weights...")
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Loaded model_state_dict")
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        print("✓ Loaded state_dict")
    else:
        # Try loading checkpoint directly as state dict
        model.load_state_dict(checkpoint)
        print("✓ Loaded checkpoint as state_dict")

    # Extract adapted embeddings
    adapted_embeddings, select_HOI_index = extract_adapted_embeddings(
        model, device=args.device
    )

    # Apply diffusion normalization
    text_mean, normalized_embeddings = apply_diffusion_normalization(
        adapted_embeddings, scale_factor=args.scale_factor
    )

    # Save everything
    paths = save_embeddings(
        raw_embeddings=adapted_embeddings,
        text_mean=text_mean,
        normalized_embeddings=normalized_embeddings,
        select_HOI_index=select_HOI_index,
        args=ckpt_args,
        output_dir=args.output_dir
    )

    # Print next steps
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print(f"1. Train diffusion model on adapted embeddings:")
    print(f"   python train_hoi_diffusion.py \\")
    print(f"     --data_path {paths['norm_path']} \\")
    print(f"     --train_steps 300000 \\")
    print(f"     --results_folder hoi_diffusion_results_adapted")
    print(f"\n2. Use trained diffusion for inference:")
    print(f"   Edit scripts/hico_test_vitB_zs_diffusion.sh to use:")
    print(f"     --diffusion_model_path hoi_diffusion_results_adapted/model-300.pt")
    print(f"     --diffusion_text_mean_path {paths['mean_path']}")
    print("="*60)


if __name__ == '__main__':
    main()
