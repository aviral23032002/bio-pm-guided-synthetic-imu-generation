#!/usr/bin/env python3
"""
evaluate_cuda.py — High-performance CUDA evaluation with axis-wise pooling and Parallelization.
Optimized for A100 / high-end GPUs.
"""

import os
import argparse
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

# ── Hardware Setup ────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 6
ACTIVITY_NAMES = ["Walking", "Jogging", "Stairs", "Sitting", "Standing", "Lying"]

# ── Model Definition ─────────────────────────────────────────────────────────
class FastMLP(nn.Module):
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
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# ── Training Function ─────────────────────────────────────────────────────────
def train_torch_mlp(X_train, y_train, X_test, device,
                    num_classes=6, epochs=150, patience=10,
                    batch_size=2048, lr=1e-3):
    # Split for validation (10%)
    n_val = max(1, int(len(X_train) * 0.1))
    X_tr, y_tr = X_train[:-n_val], y_train[:-n_val]
    X_va, y_va = X_train[-n_val:], y_train[-n_val:]

    def to_t(x, dtype):
        return torch.tensor(x, dtype=dtype).to(device)

    Xtr_t, ytr_t = to_t(X_tr, torch.float32), to_t(y_tr, torch.long)
    Xva_t, yva_t = to_t(X_va, torch.float32), to_t(y_va, torch.long)
    Xte_t        = to_t(X_test, torch.float32)

    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=batch_size, shuffle=True)
    model = FastMLP(X_train.shape[1], num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_loss, best_state, no_improve = float("inf"), None, 0

    for epoch in range(epochs):
        model.train()
        for Xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            criterion(model(Xb), yb).backward()
            optimizer.step()
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            v_loss = criterion(model(Xva_t), yva_t).item()
        
        if v_loss < best_loss:
            best_loss = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience: break

    if best_state is None:
        return np.zeros(len(X_test))

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        return model(Xte_t).argmax(dim=1).cpu().numpy()

# ── Data Loading & Pooling ───────────────────────────────────────────────────
def load_data(real_path, syn_path):
    print(f"\nLoading real tokens: {real_path}")
    with h5py.File(real_path, "r") as f:
        real_tokens = f["tokens"][:].astype(np.float32)
        real_labels = f["labels"][:].astype(int)
        real_pids   = f["subject_ids"][:].astype(int)
        real_grav   = f["gravity_vecs"][:].astype(np.float32)

    # Axis-wise pooling for real data
    mean_x = real_tokens[:, 0::3, :].mean(axis=1)
    mean_y = real_tokens[:, 1::3, :].mean(axis=1)
    mean_z = real_tokens[:, 2::3, :].mean(axis=1)
    std    = real_tokens.std(axis=1)
    real_feats = np.concatenate([mean_x, mean_y, mean_z, std, real_grav], axis=1) # 320-d

    syn_feats, syn_labels = None, None
    if os.path.exists(syn_path):
        print(f"Loading synthetic tokens: {syn_path}")
        with h5py.File(syn_path, "r") as f:
            syn_tokens = f["tokens"][:].astype(np.float32)
            syn_labels = f["labels"][:].astype(int)
        
        s_mean_x = syn_tokens[:, 0::3, :].mean(axis=1)
        s_mean_y = syn_tokens[:, 1::3, :].mean(axis=1)
        s_mean_z = syn_tokens[:, 2::3, :].mean(axis=1)
        s_std    = syn_tokens.std(axis=1)
        
        # Class-mean gravity proxy
        class_grav = {}
        for c in range(NUM_CLASSES):
            mask = real_labels == c
            class_grav[c] = real_grav[mask].mean(axis=0) if mask.any() else np.zeros(64)
        
        syn_grav = np.stack([class_grav[l] for l in syn_labels])
        syn_feats = np.concatenate([s_mean_x, s_mean_y, s_mean_z, s_std, syn_grav], axis=1)

    return real_feats, real_labels, real_pids, syn_feats, syn_labels

def run_single_fold(fold, test_subj, real_feats, real_labels, real_pids,
                    syn_feats, syn_labels, device, n_classes, batch_size, epochs):
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    test_mask = (real_pids == test_subj)
    train_mask = ~test_mask
    
    X_tr_r, y_tr_r = real_feats[train_mask], real_labels[train_mask]
    X_te,   y_te   = real_feats[test_mask],  real_labels[test_mask]

    # Scaler
    sc = StandardScaler().fit(X_tr_r)
    X_tr_r_sc = sc.transform(X_tr_r)
    X_te_sc    = sc.transform(X_te)

    # Condition A: Real Only
    pred_a = train_torch_mlp(X_tr_r_sc, y_tr_r, X_te_sc, device, batch_size=batch_size, epochs=epochs)
    f1_a = f1_score(y_te, pred_a, average="macro", zero_division=0)

    # Condition B: Augmented
    if syn_feats is not None:
        X_tr_b = np.concatenate([X_tr_r, syn_feats], axis=0)
        y_tr_b = np.concatenate([y_tr_r, syn_labels], axis=0)
        
        X_tr_b_sc = sc.transform(X_tr_b) # Use same scaler (fit on real)
        X_te_b_sc = sc.transform(X_te)
        
        pred_b = train_torch_mlp(X_tr_b_sc, y_tr_b, X_te_b_sc, device, batch_size=batch_size, epochs=epochs)
        f1_b = f1_score(y_te, pred_b, average="macro", zero_division=0)
    else:
        f1_b = 0.0

    print(f"  {fold+1:<5} {test_subj:<6} {f1_a:>10.3f} {f1_b:>10.3f}")
    return f1_a, f1_b

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", required=True)
    parser.add_argument("--syn", required=True)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()

    print(f"=================================================================")
    print(f"Device: {DEVICE} | Parallelizing across folds | Batch Size: {args.batch_size}")
    print(f"=================================================================")

    r_feats, r_labels, r_pids, s_feats, s_labels = load_data(args.real, args.syn)
    subjs = np.unique(r_pids)
    
    print(f"\n🚀 Running LOSO ({len(subjs)} folds)...")
    print(f"  {'Fold':<5} {'Subj':<6} {'Real_F1':>10} {'Aug_F1':>10}")
    print("  " + "-" * 40)

    # Parallelize
    results = Parallel(n_jobs=-1)(
        delayed(run_single_fold)(
            i, subj, r_feats, r_labels, r_pids, s_feats, s_labels, DEVICE, NUM_CLASSES, args.batch_size, args.epochs
        )
        for i, subj in enumerate(subjs)
    )

    res_a = [r[0] for r in results]
    res_b = [r[1] for r in results]

    print("\n" + "="*40)
    print(f"FINAL MACRO-F1 SUMMARY (320-d)")
    print(f"Real Only:  {np.mean(res_a):.4f}")
    print(f"Augmented:  {np.mean(res_b):.4f}")
    print(f"Delta:      {np.mean(res_b) - np.mean(res_a):+.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
