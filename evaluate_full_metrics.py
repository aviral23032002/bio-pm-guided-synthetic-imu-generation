import os
import h5py
import numpy as np
import argparse
import torch
import sys
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.manifold import TSNE

# Attempt to import UMAP
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

def compute_zcr(x):
    return ((x[:-1] * x[1:]) < 0).sum() / len(x)

def compute_sma(x):
    return np.sum(np.abs(x)) / len(x)

def plot_overlay(emb, idx_r, title, path):
    plt.figure(figsize=(10, 8))
    plt.scatter(emb[:idx_r, 0], emb[:idx_r, 1], c='blue', alpha=0.4, label='Real', s=10)
    plt.scatter(emb[idx_r:, 0], emb[idx_r:, 1], c='red', alpha=0.6, label='Synthetic', s=15, marker='x')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(); plt.grid(True, alpha=0.2)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {path}")

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--real_tokens", default="results_week1_test/token_store.hdf5")
    # parser.add_argument("--syn_tokens",  default="synthetic_tokens/synthetic_tokens.hdf5")
    # parser.add_argument("--real_waves",  default="preprocessed_biopm_test") 
    # parser.add_argument("--syn_waves",   default="synthetic_waveforms/synthetic_waveforms.hdf5")
    # parser.add_argument("--out_dir",     default="full_pipeline_report")
    parser.add_argument("--real_tokens", default="results_week1_test/token_store.hdf5")
    parser.add_argument("--syn_tokens",  default="synthetic_tokens/synthetic_tokens.hdf5")
    parser.add_argument("--real_waves",  default="preprocessed_biopm_test") 
    parser.add_argument("--syn_waves",   default="synthetic_waveforms/synthetic_waveforms.hdf5")
    parser.add_argument("--out_dir",     default="full_pipeline_report")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print("\n" + "="*80)
    print("CAR-IMU ADVANCED PIPELINE REPORT (Metrics + TSNE + UMAP)")
    print("="*80)

    # 1. Load Data
    with h5py.File(args.real_tokens, 'r') as f_real, h5py.File(args.syn_tokens, 'r') as f_syn:
        y_real = f_real['labels'][:]
        x_real_m = f_real['mean_tokens'][:]
        x_syn_m = f_syn['mean_tokens'][:]
        y_syn_l = f_syn['labels'][:]

    with h5py.File(args.syn_waves, 'r') as f_sw:
        sw = f_sw['waveforms'][:]
        sl = f_sw['labels'][:]
    
    import glob
    rw_list, rl_list = [], []
    for path in sorted(glob.glob(os.path.join(args.real_waves, "*.h5")))[:10]:
        with h5py.File(path, 'r') as f:
            rw_list.append(f['window_acc_raw'][:])
            rl_list.append(f['window_label'][:])
    rw = np.concatenate(rw_list)
    rl = np.concatenate(rl_list)

    # 2. Numerical Metrics
    print(f"  {'Activity':<12} {'Cos Sim':>10} {'Spec Corr':>10} {'ZCR Err':>10} {'SMA Err':>10}")
    print("  " + "-"*56)

    for cid in range(6):
        m_r = rl == cid
        m_s = sl == cid
        m_rt = y_real == cid
        m_st = y_syn_l == cid
        
        if m_r.sum() > 5 and m_s.sum() > 5:
            c_r = x_real_m[m_rt].mean(axis=0)
            c_s = x_syn_m[m_st].mean(axis=0)
            t_sim = np.dot(c_r, c_s) / (np.linalg.norm(c_r) * np.linalg.norm(c_s) + 1e-8)
            psd_r = np.abs(np.fft.rfft(rw[m_r], axis=1)).mean(axis=(0,2))
            psd_s = np.abs(np.fft.rfft(sw[m_s], axis=1)).mean(axis=(0,2))
            s_corr, _ = pearsonr(psd_r, psd_s)
            zcr_r = np.mean([compute_zcr(w[:, 0]) for w in rw[m_r]])
            zcr_s = np.mean([compute_zcr(w[:, 0]) for w in sw[m_s]])
            zcr_err = np.abs(zcr_r - zcr_s)
            sma_r = np.mean([compute_sma(w) for w in rw[m_r]])
            sma_s = np.mean([compute_sma(w) for w in sw[m_s]])
            sma_err = np.abs(sma_r - sma_s)
            print(f"  Class {cid:<7} {t_sim:10.4f} {s_corr:10.4f} {zcr_err:10.4f} {sma_err:10.4f}")

    # 3. Preparation for Plotting
    n_plot = 1000
    idx_r = np.random.choice(len(x_real_m), min(len(x_real_m), n_plot), replace=False)
    idx_s = np.random.choice(len(x_syn_m), min(len(x_syn_m), n_plot), replace=False)
    combined = np.vstack([x_real_m[idx_r], x_syn_m[idx_s]])

    # 4. TSNE
    print("\nRunning t-SNE visualization...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_tsne = tsne.fit_transform(combined)
    plot_overlay(emb_tsne, len(idx_r), "CAR-IMU: Real vs Synthetic Overlay (t-SNE)", os.path.join(args.out_dir, "pipeline_tsne_overlay.png"))

    # 5. UMAP
    if HAS_UMAP:
        print("\nRunning UMAP visualization...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        emb_umap = reducer.fit_transform(combined)
        plot_overlay(emb_umap, len(idx_r), "CAR-IMU: Real vs Synthetic Overlay (UMAP)", os.path.join(args.out_dir, "pipeline_umap_overlay.png"))
    else:
        print("\n[Skip] UMAP (umap-learn not installed)")

    print("\n" + "="*80)
    print("✅ Full pipeline report complete.")

if __name__ == "__main__":
    main()
