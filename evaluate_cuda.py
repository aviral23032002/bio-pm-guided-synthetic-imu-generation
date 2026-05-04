#!/usr/bin/env python3
import os
import argparse
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

# ── Hardware Setup ────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=================================================================")
print(f"CAR-IMU — High Performance CUDA Evaluation")
print(f"Device: {DEVICE}")
print(f"=================================================================")

# ── Model Definition ─────────────────────────────────────────────────────────
class FastMLP(nn.Module):
    def __init__(self, input_dim, num_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),      nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# ── Pure Torch Training Loop ──────────────────────────────────────────────────
def train_model(X_train_t, y_train_t, X_test_t, input_dim, device, epochs=200):
    # Split for validation (10%)
    n_val = max(1, int(len(X_train_t) * 0.1))
    X_tr, y_tr = X_train_t[:-n_val], y_train_t[:-n_val]
    X_va, y_va = X_train_t[-n_val:], y_train_t[-n_val:]

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=2048, shuffle=True, pin_memory=True)
    model = FastMLP(input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    best_loss = float("inf")
    best_state = None
    patience, no_improve = 20, 0

    for epoch in range(epochs):
        model.train()
        for Xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            criterion(model(Xb), yb).backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            v_loss = criterion(model(X_va), y_va).item()
        
        if v_loss < best_loss:
            best_loss = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience: break

    if best_state is None:
        return np.zeros(len(X_test_t))

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        return model(X_test_t).argmax(dim=1).cpu().numpy()

# ── Data Loading ─────────────────────────────────────────────────────────────
def load_data(real_path, syn_path):
    print("\nLoading HDF5 data...")
    with h5py.File(real_path, "r") as f:
        real_tokens = f["tokens"][:].astype(np.float32)
        real_labels = f["labels"][:].astype(int)
        real_pids   = f["subject_ids"][:].astype(int)
        real_grav   = f["gravity_vecs"][:].astype(np.float32)

    # Pre-compute features (N, 192)
    real_mean = real_tokens.mean(axis=1)
    real_std  = real_tokens.std(axis=1)
    real_feats = np.concatenate([real_mean, real_std, real_grav], axis=1)

    syn_feats, syn_labels = None, None
    if os.path.exists(syn_path):
        with h5py.File(syn_path, "r") as f:
            syn_tokens = f["tokens"][:].astype(np.float32)
            syn_labels = f["labels"][:].astype(int)
        
        syn_mean = syn_tokens.mean(axis=1)
        syn_std  = syn_tokens.std(axis=1)
        
        # Class-mean gravity proxy
        class_grav = {}
        for c in range(6):
            mask = real_labels == c
            class_grav[c] = real_grav[mask].mean(axis=0) if mask.any() else np.zeros(64)
        
        syn_grav = np.stack([class_grav[l] for l in syn_labels])
        syn_feats = np.concatenate([syn_mean, syn_std, syn_grav], axis=1)

    return real_feats, real_labels, real_pids, syn_feats, syn_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", required=True)
    parser.add_argument("--syn", required=True)
    args = parser.parse_args()

    r_feats, r_labels, r_pids, s_feats, s_labels = load_data(args.real, args.syn)
    
    # Move to Torch/GPU early
    r_f_t = torch.tensor(r_feats, device=DEVICE)
    r_l_t = torch.tensor(r_labels, device=DEVICE)
    s_f_t = torch.tensor(s_feats, device=DEVICE) if s_feats is not None else None
    s_l_t = torch.tensor(s_labels, device=DEVICE) if s_labels is not None else None
    
    subjs = np.unique(r_pids)
    results_a, results_b = [], []

    print(f"\n🚀 Running LOSO ({len(subjs)} folds)...")
    print(f"{'Fold':<5} {'Subj':<6} {'A_MLP':>8} {'B_MLP':>8}")
    print("-" * 35)

    for i, subj in enumerate(subjs):
        test_mask = (r_pids == subj)
        train_mask = ~test_mask
        
        # Condition A: Real Only
        X_tr_r, y_tr_r = r_f_t[train_mask], r_l_t[train_mask]
        X_te,   y_te   = r_f_t[test_mask],  r_labels[test_mask]
        
        # Pure Torch Scaling (GPU)
        mu, std = X_tr_r.mean(dim=0), X_tr_r.std(dim=0) + 1e-8
        X_tr_r_sc = (X_tr_r - mu) / std
        X_te_sc = (r_f_t[test_mask] - mu) / std

        pred_a = train_model(X_tr_r_sc, y_tr_r, X_te_sc, 192, DEVICE)
        f1_a = f1_score(y_te, pred_a, average="macro", zero_division=0)
        results_a.append(f1_a)

        # Condition B: Augmented
        if s_f_t is not None:
            X_tr_b = torch.cat([X_tr_r, s_f_t], dim=0)
            y_tr_b = torch.cat([y_tr_r, s_l_t], dim=0)
            
            # Re-scale for augmented set
            mu_b, std_b = X_tr_b.mean(dim=0), X_tr_b.std(dim=0) + 1e-8
            X_tr_b_sc = (X_tr_b - mu_b) / std_b
            X_te_b_sc = (r_f_t[test_mask] - mu_b) / std_b
            
            pred_b = train_model(X_tr_b_sc, y_tr_b, X_te_b_sc, 192, DEVICE)
            f1_b = f1_score(y_te, pred_b, average="macro", zero_division=0)
            results_b.append(f1_b)
        else:
            f1_b = 0.0

        print(f"{i:<5} {subj:<6} {f1_a:>8.3f} {f1_b:>8.3f}")

    print("\n" + "="*35)
    print(f"FINAL MACRO-F1 (MLP)")
    print(f"Real Only:  {np.mean(results_a):.3f}")
    print(f"Augmented:  {np.mean(results_b):.3f}")
    print(f"Delta:      {np.mean(results_b) - np.mean(results_a):+.3f}")
    print("="*35)

if __name__ == "__main__":
    main()
