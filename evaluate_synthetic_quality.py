import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import rbf_kernel
import argparse
from tqdm import tqdm

# Attempt to import UMAP
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

def compute_mmd(x, y, gamma=1.0):
    """Compute Maximum Mean Discrepancy (MMD) between two distributions."""
    xx = rbf_kernel(x, x, gamma)
    yy = rbf_kernel(y, y, gamma)
    xy = rbf_kernel(x, y, gamma)
    return xx.mean() + yy.mean() - 2 * xy.mean()

def plot_comparison(real_emb, syn_emb, labels, title, filename):
    """Plot Real vs Synthetic in 2D space."""
    plt.figure(figsize=(10, 8))
    plt.scatter(real_emb[:, 0], real_emb[:, 1], c='blue', alpha=0.3, label='Real', s=10)
    plt.scatter(syn_emb[:, 0], syn_emb[:, 1], c='red', alpha=0.5, label='Synthetic', s=15, marker='x')
    plt.title(title, fontsize=15, fontweight='bold')
    plt.xlabel('Dim 1'); plt.ylabel('Dim 2')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=200, bbox_inches='tight'); plt.close()
    print(f"  Saved: {filename}")

def plot_by_activity(emb, labels, title, filename):
    """Plot dots colored by Activity ID."""
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(emb[:, 0], emb[:, 1], c=labels, cmap='tab10', s=15, alpha=0.7)
    plt.colorbar(scatter, label='Activity ID')
    plt.title(title, fontsize=15, fontweight='bold')
    plt.xlabel('Dim 1'); plt.ylabel('Dim 2'); plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=200, bbox_inches='tight'); plt.close()
    print(f"  Saved: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Synthetic Token Quality")
    parser.add_argument("--real_tokens", type=str, required=True, help="Path to real token_store.hdf5")
    parser.add_argument("--syn_tokens", type=str, required=True, help="Path to synthetic_tokens.hdf5")
    parser.add_argument("--out_dir", type=str, default="quality_analysis")
    parser.add_argument("--sample_size", type=int, default=1000, help="Max windows to plot per type")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("\n" + "="*60)
    print("Synthetic Quality Analysis: Real vs Synthetic Distributions")
    print("="*60)

    # 1. Load Data
    with h5py.File(args.real_tokens, 'r') as f_real, h5py.File(args.syn_tokens, 'r') as f_syn:
        # We use mean-pooled tokens (N, 64) for distribution comparison
        x_real = f_real['mean_tokens'][:]
        y_real = f_real['labels'][:]
        
        x_syn = f_syn['mean_tokens'][:]
        y_syn = f_syn['labels'][:]

    # 2. Filter / Sample for visualization
    # To keep the plots clean, we sample up to args.sample_size windows
    idx_real = np.random.choice(len(x_real), min(len(x_real), args.sample_size), replace=False)
    idx_syn = np.random.choice(len(x_syn), min(len(x_syn), args.sample_size), replace=False)
    
    x_real_sub = x_real[idx_real]
    x_syn_sub = x_syn[idx_syn]
    
    combined = np.vstack([x_real_sub, x_syn_sub])
    
    # 3. TSNE
    print(f"\nRunning t-SNE on {len(combined)} samples...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca')
    emb_tsne = tsne.fit_transform(combined)
    
    plot_comparison(
        emb_tsne[:len(idx_real)], 
        emb_tsne[len(idx_real):], 
        None, 
        "CAR-IMU: Real vs Synthetic (t-SNE)",
        os.path.join(args.out_dir, "quality_tsne.png")
    )
    
    # New plot by activity
    combined_labels = np.concatenate([y_real[idx_real], y_syn[idx_syn]])
    plot_by_activity(
        emb_tsne, combined_labels,
        "CAR-IMU: Clustering by Activity (t-SNE)",
        os.path.join(args.out_dir, "activity_tsne.png")
    )

    # 4. UMAP
    if HAS_UMAP:
        print(f"Running UMAP on {len(combined)} samples...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        emb_umap = reducer.fit_transform(combined)
        
        plot_comparison(
            emb_umap[:len(idx_real)], 
            emb_umap[len(idx_real):], 
            None, 
            "CAR-IMU: Real vs Synthetic (UMAP)",
            os.path.join(args.out_dir, "quality_umap.png")
        )

        plot_by_activity(
            emb_umap, combined_labels,
            "CAR-IMU: Clustering by Activity (UMAP)",
            os.path.join(args.out_dir, "activity_umap.png")
        )
    else:
        print("\n[Skip] UMAP (umap-learn not installed)")

    # 5. Statistical Metrics
    print("\n" + "-"*40)
    print(f"{'Activity':<15} {'MMD (↓)':>10} {'Cos Sim (↑)':>12}")
    print("-"*40)

    unique_classes = np.unique(y_real).astype(int)
    for cls_id in unique_classes:
        mask_real = (y_real == cls_id)
        mask_syn = (y_syn == cls_id)
        
        if mask_real.sum() < 10 or mask_syn.sum() < 10:
            continue
            
        real_cls = x_real[mask_real]
        syn_cls = x_syn[mask_syn]
        
        # MMD (using a subset for speed if needed, but 64-d is fast)
        mmd_val = compute_mmd(real_cls[:500], syn_cls[:500])
        
        # Centroid Cosine Similarity
        c_real = real_cls.mean(axis=0)
        c_syn = syn_cls.mean(axis=0)
        cos_sim = np.dot(c_real, c_syn) / (np.linalg.norm(c_real) * np.linalg.norm(c_syn) + 1e-8)
        
        print(f"{cls_id:<15} {mmd_val:>10.4f} {cos_sim:>12.4f}")

    # 6. TSTR Evaluation (Train on Synthetic, Test on Real)
    print("\n" + "="*60)
    print("TSTR: Train on Synthetic, Test on Real (Cross-Subject)")
    print("="*60)
    
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score
    
    # Train on ALL synthetic
    scaler = StandardScaler()
    x_syn_scaled = scaler.fit_transform(x_syn)
    
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42)
    print(f"  Training MLP on {len(x_syn)} synthetic windows...")
    clf.fit(x_syn_scaled, y_syn)
    
    # Test on ALL real
    x_real_scaled = scaler.transform(x_real)
    y_pred = clf.predict(x_real_scaled)
    tstr_f1 = f1_score(y_real, y_pred, average='macro')
    
    print(f"\n  ⭐ TSTR Macro-F1: {tstr_f1:.4f}")
    print("  (If > 0.60, your synthetic data is highly realistic!)")

    print("-"*40)
    print(f"\n✅ Quality analysis complete. Plots saved to: {args.out_dir}/")

if __name__ == "__main__":
    main()
