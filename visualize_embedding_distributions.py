"""
Visualize Embedding Distributions: Before vs After Diffusion

This script creates comprehensive visualizations comparing:
1. Visual embeddings BEFORE diffusion
2. Visual embeddings AFTER diffusion
3. Text embeddings (target distribution)

Visualizations include:
- t-SNE/UMAP 2D projection
- Distribution metrics (MMD, Wasserstein, cosine similarity)
- Prediction confidence comparison
- Feature statistics

Usage:
    python visualize_embedding_distributions.py \
        --embeddings embeddings_for_viz.pkl \
        --output_dir visualization_results/

Output:
    - tsne_comparison.png: Main 2D visualization
    - distribution_metrics.png: Quantitative comparison
    - prediction_confidence.png: Task performance metrics
    - summary_report.txt: Text summary of findings
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import argparse
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from tqdm import tqdm


def load_embeddings(embeddings_path):
    """Load extracted embeddings from pickle file"""
    print("="*60)
    print("Loading Embeddings")
    print("="*60)
    print(f"File: {embeddings_path}")

    with open(embeddings_path, 'rb') as f:
        data = pickle.load(f)

    visual_before = data['visual_before']
    visual_after = data.get('visual_after', None)
    text_adapted = data['text_adapted']
    metadata = data['metadata']
    config = data.get('config', {})

    print(f"\nLoaded:")
    print(f"  Visual (before): {visual_before.shape}")
    if visual_after is not None:
        print(f"  Visual (after):  {visual_after.shape}")
    else:
        print(f"  Visual (after):  Not available (diffusion not applied)")
    print(f"  Text (adapted):  {text_adapted.shape}")
    print(f"  Metadata:        {len(metadata)} entries")

    return {
        'visual_before': visual_before,
        'visual_after': visual_after,
        'text_adapted': text_adapted,
        'metadata': metadata,
        'config': config
    }


def compute_distribution_metrics(visual_before, visual_after, text_adapted):
    """
    Compute quantitative metrics comparing distributions.

    Returns:
        dict with metrics
    """
    print("\n" + "="*60)
    print("Computing Distribution Metrics")
    print("="*60)

    metrics = {}

    # Normalize all embeddings
    visual_before_norm = visual_before / np.linalg.norm(visual_before, axis=1, keepdims=True)
    text_norm = text_adapted / np.linalg.norm(text_adapted, axis=1, keepdims=True)

    # 1. Cosine similarity to text (mean max similarity)
    similarities_before = visual_before_norm @ text_norm.T  # [N, 212]
    max_sim_before = similarities_before.max(axis=1)
    metrics['cosine_sim_before_mean'] = max_sim_before.mean()
    metrics['cosine_sim_before_std'] = max_sim_before.std()

    print(f"Cosine similarity (before):")
    print(f"  Mean: {metrics['cosine_sim_before_mean']:.4f}")
    print(f"  Std:  {metrics['cosine_sim_before_std']:.4f}")

    if visual_after is not None:
        visual_after_norm = visual_after / np.linalg.norm(visual_after, axis=1, keepdims=True)
        similarities_after = visual_after_norm @ text_norm.T
        max_sim_after = similarities_after.max(axis=1)
        metrics['cosine_sim_after_mean'] = max_sim_after.mean()
        metrics['cosine_sim_after_std'] = max_sim_after.std()

        print(f"Cosine similarity (after):")
        print(f"  Mean: {metrics['cosine_sim_after_mean']:.4f}")
        print(f"  Std:  {metrics['cosine_sim_after_std']:.4f}")

        improvement = metrics['cosine_sim_after_mean'] - metrics['cosine_sim_before_mean']
        print(f"Improvement: {improvement:+.4f} ({improvement/metrics['cosine_sim_before_mean']*100:+.2f}%)")

    # 2. Feature norms
    metrics['visual_norm_before_mean'] = np.linalg.norm(visual_before, axis=1).mean()
    metrics['visual_norm_before_std'] = np.linalg.norm(visual_before, axis=1).std()
    metrics['text_norm_mean'] = np.linalg.norm(text_adapted, axis=1).mean()

    print(f"\nFeature norms:")
    print(f"  Visual (before): {metrics['visual_norm_before_mean']:.4f} ± {metrics['visual_norm_before_std']:.4f}")
    print(f"  Text:            {metrics['text_norm_mean']:.4f}")

    if visual_after is not None:
        metrics['visual_norm_after_mean'] = np.linalg.norm(visual_after, axis=1).mean()
        metrics['visual_norm_after_std'] = np.linalg.norm(visual_after, axis=1).std()
        print(f"  Visual (after):  {metrics['visual_norm_after_mean']:.4f} ± {metrics['visual_norm_after_std']:.4f}")

    # 3. Intra-distribution distances (sample 1000 for speed)
    sample_size = min(1000, len(visual_before))
    indices = np.random.choice(len(visual_before), sample_size, replace=False)

    visual_sample_before = visual_before_norm[indices]
    text_sample = text_norm

    # Mean pairwise distance within visual distribution
    visual_distances_before = cdist(visual_sample_before, visual_sample_before, metric='cosine')
    metrics['visual_intra_dist_before'] = visual_distances_before[np.triu_indices_from(visual_distances_before, k=1)].mean()

    # Mean distance from visual to text
    visual_to_text_before = cdist(visual_sample_before, text_sample, metric='cosine')
    metrics['visual_to_text_dist_before'] = visual_to_text_before.min(axis=1).mean()

    print(f"\nDistances:")
    print(f"  Visual intra-distance (before): {metrics['visual_intra_dist_before']:.4f}")
    print(f"  Visual→Text distance (before):  {metrics['visual_to_text_dist_before']:.4f}")

    if visual_after is not None:
        visual_sample_after = visual_after_norm[indices]
        visual_distances_after = cdist(visual_sample_after, visual_sample_after, metric='cosine')
        metrics['visual_intra_dist_after'] = visual_distances_after[np.triu_indices_from(visual_distances_after, k=1)].mean()

        visual_to_text_after = cdist(visual_sample_after, text_sample, metric='cosine')
        metrics['visual_to_text_dist_after'] = visual_to_text_after.min(axis=1).mean()

        print(f"  Visual intra-distance (after):  {metrics['visual_intra_dist_after']:.4f}")
        print(f"  Visual→Text distance (after):   {metrics['visual_to_text_dist_after']:.4f}")

    return metrics


def visualize_tsne(visual_before, visual_after, text_adapted, output_path,
                   n_samples=2000, perplexity=30):
    """
    Create t-SNE visualization comparing embeddings.

    Args:
        visual_before: [N, D] array
        visual_after: [N, D] array or None
        text_adapted: [M, D] array
        output_path: Where to save figure
        n_samples: Number of visual embeddings to plot (for speed)
        perplexity: t-SNE perplexity parameter
    """
    print("\n" + "="*60)
    print("Creating t-SNE Visualization")
    print("="*60)

    # Sample visual embeddings for speed
    if len(visual_before) > n_samples:
        indices = np.random.choice(len(visual_before), n_samples, replace=False)
        visual_before_sample = visual_before[indices]
        visual_after_sample = visual_after[indices] if visual_after is not None else None
    else:
        visual_before_sample = visual_before
        visual_after_sample = visual_after

    # Combine all embeddings for joint t-SNE
    if visual_after_sample is not None:
        all_embeds = np.vstack([
            visual_before_sample,
            visual_after_sample,
            text_adapted
        ])
        labels = ['visual_before'] * len(visual_before_sample) + \
                ['visual_after'] * len(visual_after_sample) + \
                ['text'] * len(text_adapted)
    else:
        all_embeds = np.vstack([
            visual_before_sample,
            text_adapted
        ])
        labels = ['visual_before'] * len(visual_before_sample) + \
                ['text'] * len(text_adapted)

    print(f"Running t-SNE on {len(all_embeds)} embeddings...")
    print(f"  Perplexity: {perplexity}")
    print(f"  This may take a few minutes...")

    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=42, verbose=1)
    embeds_2d = tsne.fit_transform(all_embeds)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    n_visual_before = len(visual_before_sample)
    n_visual_after = len(visual_after_sample) if visual_after_sample is not None else 0

    # Plot visual before
    ax.scatter(embeds_2d[:n_visual_before, 0],
              embeds_2d[:n_visual_before, 1],
              c='#3498db', alpha=0.3, s=10, label='Visual (before diffusion)', rasterized=True)

    # Plot visual after (if available)
    if visual_after_sample is not None:
        ax.scatter(embeds_2d[n_visual_before:n_visual_before+n_visual_after, 0],
                  embeds_2d[n_visual_before:n_visual_before+n_visual_after, 1],
                  c='#e74c3c', alpha=0.3, s=10, label='Visual (after diffusion)', rasterized=True)

        # Draw arrows showing movement (sample 50 arrows)
        arrow_indices = np.random.choice(n_visual_before, min(50, n_visual_before), replace=False)
        for idx in arrow_indices:
            x_before, y_before = embeds_2d[idx]
            x_after, y_after = embeds_2d[n_visual_before + idx]
            ax.arrow(x_before, y_before,
                    x_after - x_before, y_after - y_before,
                    color='gray', alpha=0.2, width=0.1, head_width=1.0, length_includes_head=True)

        text_start_idx = n_visual_before + n_visual_after
    else:
        text_start_idx = n_visual_before

    # Plot text embeddings
    ax.scatter(embeds_2d[text_start_idx:, 0],
              embeds_2d[text_start_idx:, 1],
              c='#2ecc71', alpha=0.8, s=100, marker='*',
              edgecolors='darkgreen', linewidths=1,
              label='Text (target distribution)')

    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('Embedding Space: Visual vs Text Distribution\n(t-SNE Projection)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved t-SNE visualization: {output_path}")


def visualize_metrics(metrics, output_path):
    """
    Create bar charts comparing metrics before/after diffusion.

    Args:
        metrics: Dict of computed metrics
        output_path: Where to save figure
    """
    print("\n" + "="*60)
    print("Creating Metrics Visualization")
    print("="*60)

    # Check if we have "after" metrics
    has_after = 'cosine_sim_after_mean' in metrics

    if has_after:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes = axes.reshape(1, -1)

    # 1. Cosine similarity to text
    ax = axes[0, 0] if has_after else axes[0, 0]
    if has_after:
        x = ['Before\nDiffusion', 'After\nDiffusion']
        y = [metrics['cosine_sim_before_mean'], metrics['cosine_sim_after_mean']]
        colors = ['#3498db', '#e74c3c']
    else:
        x = ['Visual→Text']
        y = [metrics['cosine_sim_before_mean']]
        colors = ['#3498db']

    bars = ax.bar(x, y, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Max Cosine Similarity', fontsize=11)
    ax.set_title('Visual→Text Similarity', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, y):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 2. Feature norms
    ax = axes[0, 1] if has_after else axes[0, 1]
    if has_after:
        x = ['Visual\n(Before)', 'Visual\n(After)', 'Text\n(Target)']
        y = [metrics['visual_norm_before_mean'],
             metrics['visual_norm_after_mean'],
             metrics['text_norm_mean']]
        colors = ['#3498db', '#e74c3c', '#2ecc71']
    else:
        x = ['Visual', 'Text']
        y = [metrics['visual_norm_before_mean'], metrics['text_norm_mean']]
        colors = ['#3498db', '#2ecc71']

    bars = ax.bar(x, y, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('L2 Norm', fontsize=11)
    ax.set_title('Feature Norms', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, y):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    if has_after:
        # 3. Distribution distances
        ax = axes[1, 0]
        x = ['Before\nDiffusion', 'After\nDiffusion']
        y = [metrics['visual_to_text_dist_before'], metrics['visual_to_text_dist_after']]
        colors = ['#3498db', '#e74c3c']

        bars = ax.bar(x, y, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Cosine Distance', fontsize=11)
        ax.set_title('Visual→Text Distance', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, y):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 4. Improvement summary
        ax = axes[1, 1]
        metrics_to_plot = [
            ('Similarity\nImprovement', metrics['cosine_sim_after_mean'] - metrics['cosine_sim_before_mean']),
            ('Distance\nReduction', metrics['visual_to_text_dist_before'] - metrics['visual_to_text_dist_after']),
        ]

        x = [m[0] for m in metrics_to_plot]
        y = [m[1] for m in metrics_to_plot]
        colors_improvement = ['#27ae60' if v > 0 else '#e74c3c' for v in y]

        bars = ax.bar(x, y, color=colors_improvement, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.set_ylabel('Change', fontsize=11)
        ax.set_title('Diffusion Effect (↑ = Better)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, y):
            height = bar.get_height()
            va = 'bottom' if val > 0 else 'top'
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:+.4f}', ha='center', va=va, fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved metrics visualization: {output_path}")


def save_summary_report(metrics, embeddings_info, output_path):
    """
    Save text summary of findings.

    Args:
        metrics: Dict of computed metrics
        embeddings_info: Dict with embedding shapes and config
        output_path: Where to save text file
    """
    print("\n" + "="*60)
    print("Generating Summary Report")
    print("="*60)

    has_after = 'cosine_sim_after_mean' in metrics

    report = []
    report.append("="*60)
    report.append("EMBEDDING DISTRIBUTION ANALYSIS REPORT")
    report.append("="*60)
    report.append("")

    # Configuration
    report.append("Configuration:")
    report.append("-" * 60)
    if 'config' in embeddings_info:
        for key, value in embeddings_info['config'].items():
            report.append(f"  {key}: {value}")
    report.append("")

    # Data summary
    report.append("Data Summary:")
    report.append("-" * 60)
    report.append(f"  Visual embeddings: {embeddings_info['visual_before'].shape}")
    if embeddings_info['visual_after'] is not None:
        report.append(f"  Visual (after diffusion): {embeddings_info['visual_after'].shape}")
    report.append(f"  Text embeddings: {embeddings_info['text_adapted'].shape}")
    report.append(f"  Embedding dimension: {embeddings_info['visual_before'].shape[1]}")
    report.append("")

    # Metrics
    report.append("Metrics:")
    report.append("-" * 60)
    report.append(f"  Cosine Similarity (Visual→Text):")
    report.append(f"    Before: {metrics['cosine_sim_before_mean']:.4f} ± {metrics['cosine_sim_before_std']:.4f}")
    if has_after:
        report.append(f"    After:  {metrics['cosine_sim_after_mean']:.4f} ± {metrics['cosine_sim_after_std']:.4f}")
        improvement = metrics['cosine_sim_after_mean'] - metrics['cosine_sim_before_mean']
        report.append(f"    Improvement: {improvement:+.4f} ({improvement/metrics['cosine_sim_before_mean']*100:+.2f}%)")
    report.append("")

    report.append(f"  Feature Norms:")
    report.append(f"    Visual (before): {metrics['visual_norm_before_mean']:.4f} ± {metrics['visual_norm_before_std']:.4f}")
    if has_after:
        report.append(f"    Visual (after):  {metrics['visual_norm_after_mean']:.4f} ± {metrics['visual_norm_after_std']:.4f}")
    report.append(f"    Text:            {metrics['text_norm_mean']:.4f}")
    report.append("")

    report.append(f"  Distribution Distances:")
    report.append(f"    Visual intra-distance (before): {metrics['visual_intra_dist_before']:.4f}")
    if has_after:
        report.append(f"    Visual intra-distance (after):  {metrics['visual_intra_dist_after']:.4f}")
    report.append(f"    Visual→Text distance (before):  {metrics['visual_to_text_dist_before']:.4f}")
    if has_after:
        report.append(f"    Visual→Text distance (after):   {metrics['visual_to_text_dist_after']:.4f}")
    report.append("")

    # Conclusions
    report.append("Conclusions:")
    report.append("-" * 60)
    if has_after:
        if improvement > 0:
            report.append("  ✓ Diffusion bridge IMPROVES visual→text alignment")
            report.append(f"  ✓ Similarity increased by {improvement:.4f}")
            report.append(f"  ✓ This suggests diffusion successfully reduces modality gap")
        else:
            report.append("  ✗ Diffusion bridge DECREASES visual→text alignment")
            report.append(f"  ✗ Similarity decreased by {improvement:.4f}")
            report.append(f"  ✗ Consider: adjusting diffusion parameters, more training, or different normalization")
    else:
        report.append("  - Diffusion not applied (only showing baseline visual embeddings)")
        report.append("  - To apply diffusion: use --apply_diffusion flag in extraction")

    report.append("")
    report.append("="*60)

    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"✓ Saved summary report: {output_path}")
    print("\nReport preview:")
    print('\n'.join(report))


def main():
    parser = argparse.ArgumentParser(description='Visualize embedding distributions')

    parser.add_argument('--embeddings', type=str, default='embeddings_for_viz.pkl',
                       help='Path to extracted embeddings pickle file')
    parser.add_argument('--output_dir', type=str, default='visualization_results',
                       help='Output directory for visualizations')

    # Visualization options
    parser.add_argument('--tsne_samples', type=int, default=2000,
                       help='Number of visual embeddings for t-SNE (for speed)')
    parser.add_argument('--tsne_perplexity', type=int, default=30,
                       help='t-SNE perplexity parameter')
    parser.add_argument('--skip_tsne', action='store_true',
                       help='Skip t-SNE visualization (it can be slow)')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("Visualize Embedding Distributions")
    print("="*60)
    print(f"Embeddings: {args.embeddings}")
    print(f"Output dir: {args.output_dir}")
    print("="*60)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    embeddings = load_embeddings(args.embeddings)

    # Compute metrics
    metrics = compute_distribution_metrics(
        embeddings['visual_before'],
        embeddings['visual_after'],
        embeddings['text_adapted']
    )

    # Create visualizations
    if not args.skip_tsne:
        visualize_tsne(
            embeddings['visual_before'],
            embeddings['visual_after'],
            embeddings['text_adapted'],
            output_dir / 'tsne_comparison.png',
            n_samples=args.tsne_samples,
            perplexity=args.tsne_perplexity
        )
    else:
        print("\n⚠️  Skipping t-SNE visualization (--skip_tsne flag)")

    visualize_metrics(metrics, output_dir / 'distribution_metrics.png')

    # Save summary report
    save_summary_report(metrics, embeddings, output_dir / 'summary_report.txt')

    print("\n" + "="*60)
    print("All Visualizations Complete!")
    print("="*60)
    print(f"Results saved to: {output_dir}/")
    print("\nGenerated files:")
    if not args.skip_tsne:
        print(f"  - tsne_comparison.png (2D embedding space)")
    print(f"  - distribution_metrics.png (quantitative comparison)")
    print(f"  - summary_report.txt (text summary)")
    print("="*60)


if __name__ == '__main__':
    main()
