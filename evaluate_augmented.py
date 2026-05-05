#!/usr/bin/env python3
"""
evaluate_augmented.py — Compare real-only vs real+synthetic HAR classifier with Attention Pooling.

WHAT THIS DOES:
    Runs LOSO cross-validation with TWO conditions using PyTorch:
      [A] Real-only:  train on real 192x64 token sequences
      [B] Augmented:  train on real + synthetic 192x64 token sequences (CAR-IMU)

    It uses an Attention Pooling layer to learn which of the 192 tokens are most
    important for classification. It also pairs synthetic tokens with random real
    gravity vectors from the same class so the classifier has orientation context.

USAGE:
    python evaluate_augmented.py
"""

import os
import argparse
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from car_imu_decoder import ACTIVITY_NAMES, NUM_CLASSES

def get_best_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")

# ══════════════════════════════════════════════════════════════════════════════
# PYTORCH MODEL DEFINITION
# ══════════════════════════════════════════════════════════════════════════════
class AttentionPooling(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (B, 192, 64)
        weights = torch.softmax(self.scorer(x).squeeze(-1), dim=-1) # (B, 192)
        return (x * weights.unsqueeze(-1)).sum(dim=1)  # (B, 64)


class AttentionMLP(nn.Module):
    def __init__(self, token_dim=64, gravity_dim=64, num_classes=6):
        super().__init__()
        self.attn_pool = AttentionPooling(token_dim)
        self.classifier = nn.Sequential(
            nn.Linear(token_dim + gravity_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, tokens, gravity):
        # tokens:  (B, 192, 64)
        # gravity: (B, 64)
        pooled = self.attn_pool(tokens)               # (B, 64)
        x      = torch.cat([pooled, gravity], dim=-1) # (B, 128)
        return self.classifier(x)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def sanitize_features(feats: np.ndarray, name: str = "") -> np.ndarray:
    n_bad = (~np.isfinite(feats)).sum()
    if n_bad > 0:
        print(f"  ⚠  {name}: {n_bad} non-finite values replaced with 0")
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    feats = np.clip(feats, -30.0, 30.0)
    return feats.astype(np.float32)

def load_real_data(path: str):
    with h5py.File(path, "r") as f:
        # Load raw 192-d tokens instead of mean-pooled
        tokens = f["tokens"][:].astype(np.float32)       # (N, 192, 64)
        labels = f["labels"][:].astype(int)
        pids   = f["subject_ids"][:].astype(int)
        
        # Load gravity vectors if they exist
        if "gravity_vecs" in f:
            gravity = f["gravity_vecs"][:].astype(np.float32) # (N, 64)
        else:
            gravity = np.zeros((len(tokens), 64), dtype=np.float32)
            print("  ⚠ gravity_vecs not found in token_store, using zeros.")
            
    tokens = sanitize_features(tokens, "real tokens")
    gravity = sanitize_features(gravity, "real gravity")
    print(f"  Real tokens:  {tokens.shape}  subjects: {np.unique(pids).tolist()}")
    return tokens, gravity, labels, pids

def load_synthetic_data(path: str, real_gravity: np.ndarray, real_labels: np.ndarray):
    with h5py.File(path, "r") as f:
        tokens = f["tokens"][:].astype(np.float32)       # (N_syn, 192, 64)
        labels = f["labels"][:].astype(int)
        
    tokens = sanitize_features(tokens, "synthetic tokens")
    
    # ── Automatically sample Real Gravity for Synthetic Tokens ──
    # Since CAR-IMU only generates movement tokens, we pair them with real
    # gravity vectors of the same class to give the classifier orientation context.
    gravity = np.zeros((len(tokens), real_gravity.shape[1]), dtype=np.float32)
    for c in range(NUM_CLASSES):
        syn_mask = (labels == c)
        n_syn = syn_mask.sum()
        if n_syn > 0:
            real_mask = (real_labels == c)
            real_grav_c = real_gravity[real_mask]
            if len(real_grav_c) > 0:
                indices = np.random.choice(len(real_grav_c), n_syn, replace=True)
                gravity[syn_mask] = real_grav_c[indices]
                
    print(f"  Synthetic:    {tokens.shape}")
    for cls_id in range(NUM_CLASSES):
        n = (labels == cls_id).sum()
        if n > 0:
            print(f"    {ACTIVITY_NAMES[cls_id]:<12} +{n}")
            
    return tokens, gravity, labels

# ══════════════════════════════════════════════════════════════════════════════
# PYTORCH TRAINER
# ══════════════════════════════════════════════════════════════════════════════
def train_torch_mlp(X_tr_tok, X_tr_grav, y_tr, X_te_tok, X_te_grav, device, epochs=30):
    model = AttentionMLP(token_dim=64, gravity_dim=64, num_classes=NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    dataset = TensorDataset(
        torch.tensor(X_tr_tok, dtype=torch.float32), 
        torch.tensor(X_tr_grav, dtype=torch.float32), 
        torch.tensor(y_tr, dtype=torch.long)
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)
    
    X_te_tok_t = torch.tensor(X_te_tok, dtype=torch.float32).to(device)
    X_te_grav_t = torch.tensor(X_te_grav, dtype=torch.float32).to(device)
    
    for ep in range(epochs):
        model.train()
        for tok, grav, lbl in loader:
            tok, grav, lbl = tok.to(device), grav.to(device), lbl.to(device)
            optimizer.zero_grad()
            out = model(tok, grav)
            loss = criterion(out, lbl)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        out = model(X_te_tok_t, X_te_grav_t)
        preds = out.argmax(dim=-1).cpu().numpy()
        
    return preds

# ══════════════════════════════════════════════════════════════════════════════
# LOSO
# ══════════════════════════════════════════════════════════════════════════════
def run_loso_both_conditions(
    real_tok, real_grav, real_labels, real_pids,
    syn_tok, syn_grav, syn_labels, device
):
    subjects = np.unique(real_pids)
    n_classes = NUM_CLASSES

    macro_real_mlp, macro_aug_mlp = [], []
    per_class_real = {i: [] for i in range(n_classes)}
    per_class_aug  = {i: [] for i in range(n_classes)}

    has_syn = syn_tok is not None and len(syn_tok) > 0

    print(f"\n  LOSO ({len(subjects)} folds) with PyTorch AttentionMLP...")
    print(f"  {'Fold':<5} {'Subj':<6} {'Real F1':>8} {'Aug F1':>8}")
    print("  " + "-" * 35)

    for fold, test_subj in enumerate(subjects):
        test_mask  = real_pids == test_subj
        train_mask = ~test_mask

        # Condition A: Real Only
        X_tr_tok_r = real_tok[train_mask]
        X_tr_grav_r = real_grav[train_mask]
        y_tr_r = real_labels[train_mask]
        
        X_te_tok = real_tok[test_mask]
        X_te_grav = real_grav[test_mask]
        y_te = real_labels[test_mask]

        # Condition B: Real + Synthetic
        if has_syn:
            X_tr_tok_a = np.concatenate([X_tr_tok_r, syn_tok], axis=0)
            X_tr_grav_a = np.concatenate([X_tr_grav_r, syn_grav], axis=0)
            y_tr_a = np.concatenate([y_tr_r, syn_labels], axis=0)
        else:
            X_tr_tok_a, X_tr_grav_a, y_tr_a = X_tr_tok_r, X_tr_grav_r, y_tr_r

        # Train and Evaluate Condition A (Real)
        pred_mlp_r = train_torch_mlp(X_tr_tok_r, X_tr_grav_r, y_tr_r, X_te_tok, X_te_grav, device)
        f1_mlp_r   = f1_score(y_te, pred_mlp_r, average="macro", zero_division=0)

        # Train and Evaluate Condition B (Augmented)
        pred_mlp_a = train_torch_mlp(X_tr_tok_a, X_tr_grav_a, y_tr_a, X_te_tok, X_te_grav, device)
        f1_mlp_a   = f1_score(y_te, pred_mlp_a, average="macro", zero_division=0)

        # Collect Macros
        macro_real_mlp.append(f1_mlp_r)
        macro_aug_mlp.append(f1_mlp_a)

        # Per-class F1
        for cls_id in range(n_classes):
            if (y_te == cls_id).sum() > 0:
                per_class_real[cls_id].append(
                    f1_score(y_te == cls_id, pred_mlp_r == cls_id, average="binary", zero_division=0))
                per_class_aug[cls_id].append(
                    f1_score(y_te == cls_id, pred_mlp_a == cls_id, average="binary", zero_division=0))

        print(f"  {fold+1:<5} {int(test_subj):<6} {f1_mlp_r:>8.3f} {f1_mlp_a:>8.3f}")

    return {
        "macro_real_mlp":  np.array(macro_real_mlp),
        "macro_aug_mlp":   np.array(macro_aug_mlp),
        "per_class_real":  per_class_real,
        "per_class_aug":   per_class_aug,
    }

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--real_tokens", default="results_week1/token_store.hdf5")
    p.add_argument("--syn_tokens",  default="synthetic_tokens/synthetic_tokens.hdf5")
    p.add_argument("--out_dir",     default="results_week2")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_best_device()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 65)
    print("CAR-IMU — Augmented HAR Evaluation (PyTorch + Attention)")
    print("=" * 65)

    print("\nLoading data ...")
    real_tok, real_grav, real_labels, real_pids = load_real_data(args.real_tokens)

    if os.path.exists(args.syn_tokens):
        syn_tok, syn_grav, syn_labels = load_synthetic_data(args.syn_tokens, real_grav, real_labels)
    else:
        print(f"  ⚠ Not found: {args.syn_tokens} — running real-only.")
        syn_tok, syn_grav, syn_labels = None, None, None

    res = run_loso_both_conditions(
        real_tok, real_grav, real_labels, real_pids,
        syn_tok, syn_grav, syn_labels, device
    )

    print("\n" + "=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)
    print(f"\n  {'Condition':<40} {'MLP F1':>8}")
    print("  " + "-" * 50)
    
    r_mlp = res["macro_real_mlp"].mean()
    a_mlp = res["macro_aug_mlp"].mean()
    d_mlp = a_mlp - r_mlp

    print(f"  {'[A] Real only  (Attention, LOSO)':<40} {r_mlp:>8.3f}")
    print(f"  {'[B] Real+Synth (Attention, LOSO)':<40} {a_mlp:>8.3f}")
    print(f"  {'Δ = [B] - [A]':<40} {d_mlp:>+8.3f}")

    print(f"\n  Per-class F1  (AttentionMLP, LOSO mean):")
    print(f"  {'Class':<14} {'Real only':>12} {'Augmented':>12} {'Δ':>8}  ")
    print("  " + "-" * 52)

    for cls_id in range(NUM_CLASSES):
        vals_r = res["per_class_real"][cls_id]
        vals_a = res["per_class_aug"][cls_id]
        r = np.mean(vals_r) if vals_r else float("nan")
        a = np.mean(vals_a) if vals_a else float("nan")
        d = a - r
        minority = " ← minority" if cls_id in [2, 3, 4, 5] else ""
        print(f"  {ACTIVITY_NAMES[cls_id]:<14} {r:>12.3f} {a:>12.3f} {d:>+8.3f}{minority}")

    print("\n" + "=" * 65)
    if d_mlp > 0.01:
        print(f"  ✅ CAR-IMU augmentation HELPS!  Macro-F1 {d_mlp:+.3f}")
    elif d_mlp > -0.01:
        print(f"  ➡️  CAR-IMU augmentation NEUTRAL  (Δ={d_mlp:+.3f})")
    else:
        print(f"  ⚠️  CAR-IMU augmentation HURTS  (Δ={d_mlp:+.3f})")
    print("=" * 65)

    save_path = os.path.join(args.out_dir, "augmented_results_attention.npy")
    np.save(save_path, res, allow_pickle=True)
    print(f"\n  Saved: {save_path}")

if __name__ == "__main__":
    main()
