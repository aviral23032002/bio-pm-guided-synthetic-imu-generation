#!/usr/bin/env python3
"""
evaluate_feature_ablation.py — Compare HAR performance across different feature sets.

MODES:
  - 320d: Axis-wise Mean (x,y,z) + Global Std + Gravity (64 * 5)
  - 192d: Global Mean + Global Std + Gravity (64 * 3)
  - 64d:  Global Mean only (64 * 1)

USAGE:
    python evaluate_feature_ablation.py --feature_mode 320d
    python evaluate_feature_ablation.py --feature_mode 192d
    python evaluate_feature_ablation.py --feature_mode 64d
"""

import os
import argparse
import warnings
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from joblib import Parallel, delayed

from car_imu_decoder import ACTIVITY_NAMES, NUM_CLASSES

# ── Device Detection ──────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()

# Suppress warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_features(tokens: np.ndarray, gravity: np.ndarray, mode: str) -> np.ndarray:
    """
    Compute features from (N, 192, 64) tokens based on the selected mode.
    """
    N = tokens.shape[0]
    
    if mode == "64d":
        # Global mean only
        return tokens.mean(axis=1)

    elif mode == "192d":
        # Global mean + Global std + Gravity
        mean_all = tokens.mean(axis=1)
        std_all  = tokens.std(axis=1)
        return np.concatenate([mean_all, std_all, gravity], axis=1)

    elif mode == "320d":
        # Axis-wise mean + Global std + Gravity
        # tokens are interleaved: x1, y1, z1, x2, y2, z2 ...
        mean_x = tokens[:, 0::3, :].mean(axis=1)
        mean_y = tokens[:, 1::3, :].mean(axis=1)
        mean_z = tokens[:, 2::3, :].mean(axis=1)
        std_all = tokens.std(axis=1)
        return np.concatenate([mean_x, mean_y, mean_z, std_all, gravity], axis=1)
    
    else:
        raise ValueError(f"Unknown feature mode: {mode}")

def sanitize_features(feats: np.ndarray) -> np.ndarray:
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(feats, -30.0, 30.0).astype(np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(real_path: str, syn_path: str, mode: str):
    print(f"Loading data with mode: {mode}")
    
    # Load Real
    with h5py.File(real_path, "r") as f:
        r_tokens  = f["tokens"][:]
        r_labels  = f["labels"][:].astype(int)
        r_pids    = f["subject_ids"][:].astype(int)
        r_gravity = f["gravity_vecs"][:] if "gravity_vecs" in f else np.zeros((len(r_tokens), 64))

    real_feats = compute_features(r_tokens, r_gravity, mode)
    real_feats = sanitize_features(real_feats)

    # Load Synthetic
    syn_feats, syn_labels = None, None
    if os.path.exists(syn_path):
        with h5py.File(syn_path, "r") as f:
            s_tokens = f["tokens"][:]
            s_labels = f["labels"][:].astype(int)
            if "gravity_vecs" in f:
                s_gravity = f["gravity_vecs"][:]
            else:
                # Fallback to class-mean gravity from real data
                s_gravity = np.stack([
                    r_gravity[r_labels == lbl].mean(axis=0)
                    for lbl in s_labels
                ])
        syn_feats = compute_features(s_tokens, s_gravity, mode)
        syn_feats = sanitize_features(syn_feats)
        syn_labels = s_labels

    print(f"  Real features: {real_feats.shape}")
    if syn_feats is not None:
        print(f"  Synth features: {syn_feats.shape}")
    
    return real_feats, r_labels, r_pids, syn_feats, syn_labels

# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFIER (MLP)
# ══════════════════════════════════════════════════════════════════════════════

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def train_mlp(X_train, y_train, X_test, device, epochs=50):
    n_val = max(1, int(len(X_train) * 0.1))
    X_tr, y_tr = X_train[:-n_val], y_train[:-n_val]
    X_va, y_va = X_train[-n_val:], y_train[-n_val:]

    def to_t(x, dtype): return torch.tensor(x, dtype=dtype).to(device)
    Xtr_t, ytr_t = to_t(X_tr, torch.float32), to_t(y_tr, torch.long)
    Xva_t, yva_t = to_t(X_va, torch.float32), to_t(y_va, torch.long)
    Xte_t = to_t(X_test, torch.float32)

    model = SimpleMLP(X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_loss, best_state = float("inf"), None
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        criterion(model(Xtr_t), ytr_t).backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            v_loss = criterion(model(Xva_t), yva_t).item()
            if v_loss < best_loss:
                best_loss = v_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        return model(Xte_t).argmax(dim=1).cpu().numpy()

# ══════════════════════════════════════════════════════════════════════════════
# LOSO LOOP
# ══════════════════════════════════════════════════════════════════════════════

import threading

# Lock for clean printing from multiple threads
print_lock = threading.Lock()

def run_fold(fold, test_subj, real_feats, real_labels, real_pids, syn_feats, syn_labels):
    test_mask = real_pids == test_subj
    train_mask = ~test_mask

    X_tr_r, y_tr_r = real_feats[train_mask], real_labels[train_mask]
    X_te, y_te = real_feats[test_mask], real_labels[test_mask]

    # Scaler
    sc = StandardScaler().fit(X_tr_r)
    X_tr_r_sc = sc.transform(X_tr_r)
    X_te_sc   = sc.transform(X_te)

    # ── [A] REAL ONLY ──────────────────────────────────────────
    # MLP
    pred_mlp_r = train_mlp(X_tr_r_sc, y_tr_r, X_te_sc, DEVICE)
    f1_mlp_r   = f1_score(y_te, pred_mlp_r, average="macro", zero_division=0)
    
    # LR
    lr_r = LogisticRegression(max_iter=1000, C=1.0, solver="saga", random_state=42)
    lr_r.fit(X_tr_r_sc, y_tr_r)
    pred_lr_r = lr_r.predict(X_te_sc)
    f1_lr_r   = f1_score(y_te, pred_lr_r, average="macro", zero_division=0)

    # ── [B] AUGMENTED ───────────────────────────────────────────
    f1_mlp_a, f1_lr_a = 0.0, 0.0
    if syn_feats is not None:
        X_tr_a = np.concatenate([X_tr_r, syn_feats], axis=0)
        y_tr_a = np.concatenate([y_tr_r, syn_labels], axis=0)
        X_tr_a_sc = sc.transform(X_tr_a)
        
        # MLP
        pred_mlp_a = train_mlp(X_tr_a_sc, y_tr_a, X_te_sc, DEVICE)
        f1_mlp_a   = f1_score(y_te, pred_mlp_a, average="macro", zero_division=0)

        # LR
        lr_a = LogisticRegression(max_iter=1000, C=1.0, solver="saga", random_state=42)
        lr_a.fit(X_tr_a_sc, y_tr_a)
        pred_lr_a = lr_a.predict(X_te_sc)
        f1_lr_a   = f1_score(y_te, pred_lr_a, average="macro", zero_division=0)

    with print_lock:
        print(f"  Fold {fold+1:<3} Subj {test_subj:<4} | "
              f"MLP (R:{f1_mlp_r:.3f} A:{f1_mlp_a:.3f}) | "
              f"LR (R:{f1_lr_r:.3f} A:{f1_lr_a:.3f})")

    return {
        "mlp_r": f1_mlp_r, "mlp_a": f1_mlp_a,
        "lr_r": f1_lr_r, "lr_a": f1_lr_a
    }

def run_evaluation(args, mode, syn_path):
    print(f"\n" + "-"*75)
    print(f"RUNNING: Mode={mode}, Tokens={os.path.basename(os.path.dirname(syn_path))}")
    print("-" * 75)
    
    r_feats, r_labels, r_pids, s_feats, s_labels = load_data(args.real_tokens, syn_path, mode)
    subjects = np.unique(r_pids)

    print(f"\n  Starting LOSO ({len(subjects)} subjects)...")
    print(f"  {'Fold':<7} {'Subj':<8} | {'MLP (Real/Aug)':<15} | {'LR (Real/Aug)':<15}")
    print("  " + "-" * 65)

    results = Parallel(n_jobs=args.n_jobs, backend="threading")(
        delayed(run_fold)(i, s, r_feats, r_labels, r_pids, s_feats, s_labels)
        for i, s in enumerate(subjects)
    )

    summary = {
        "mlp_r": np.mean([r["mlp_r"] for r in results]),
        "mlp_a": np.mean([r["mlp_a"] for r in results]),
        "lr_r":  np.mean([r["lr_r"]  for r in results]),
        "lr_a":  np.mean([r["lr_a"]  for r in results]),
    }
    
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_mode", type=str, default="320d", choices=["320d", "192d", "64d"])
    parser.add_argument("--real_tokens", default="results_week1/token_store.hdf5")
    parser.add_argument("--syn_tokens", default="synthetic_tokens/synthetic_tokens.hdf5",
                        help="Path to standard synthetic tokens")
    parser.add_argument("--syn_tokens_ca", default="synthetic_tokens_confusion_aware/synthetic_tokens.hdf5",
                        help="Path to confusion-aware synthetic tokens")
    parser.add_argument("--run_all", action="store_true", 
                        help="Run the full ablation: 64d(Std), 192d(Std), 320d(CA)")
    parser.add_argument("--n_jobs", type=int, default=8)
    args = parser.parse_args()

    configs = []
    if args.run_all:
        configs = [
            ("64d",  args.syn_tokens,    "Standard (64d)"),
            ("192d", args.syn_tokens,    "Standard (192d)"),
            ("320d", args.syn_tokens_ca, "Confusion-Aware (320d)")
        ]
    else:
        configs = [(args.feature_mode, args.syn_tokens, f"Custom ({args.feature_mode})")]

    all_summaries = []
    for mode, path, label in configs:
        if not os.path.exists(path):
            print(f"  ⚠ Skipping {label}: {path} not found")
            continue
            
        res = run_evaluation(args, mode, path)
        all_summaries.append({
            "label": label,
            "mlp_r": res["mlp_r"], "mlp_a": res["mlp_a"],
            "lr_r": res["lr_r"], "lr_a": res["lr_a"]
        })

    # Final Comparison Table
    print("\n" + "="*85)
    print(f"{'Configuration':<25} | {'MLP Real':>10} {'MLP Aug':>10} {'Δ MLP':>8} | {'LR Real':>10} {'LR Aug':>10} {'Δ LR':>8}")
    print("="*85)
    for s in all_summaries:
        d_mlp = s['mlp_a'] - s['mlp_r']
        d_lr  = s['lr_a']  - s['lr_r']
        print(f"{s['label']:<25} | {s['mlp_r']:10.4f} {s['mlp_a']:10.4f} {d_mlp:>8.3f} | "
              f"{s['lr_r']:10.4f} {s['lr_a']:10.4f} {d_lr:>8.3f}")
    print("="*85)

if __name__ == "__main__":
    main()
