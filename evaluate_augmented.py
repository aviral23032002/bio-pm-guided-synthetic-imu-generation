#!/usr/bin/env python3
"""
evaluate_augmented.py — Compare real-only vs real+synthetic HAR classifier.

WHAT THIS DOES:
    Runs LOSO cross-validation with TWO conditions:
      [A] Real-only:  train on real 192-d token features (mean + std + gravity)
      [B] Augmented:  train on real + synthetic 192-d token features (CAR-IMU)

    Collects per-class F1 inside the LOSO loop (single pass, no redundancy).

USAGE:
    python evaluate_augmented.py
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

# ── Device Detection ──────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
DEVICE = get_device()

# Suppress sklearn numerical warnings — the results are still valid
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from car_imu_decoder import ACTIVITY_NAMES, NUM_CLASSES

WEEK1_BASELINE_F1 = 0.689


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def sanitize_features(feats: np.ndarray, name: str = "") -> np.ndarray:
    """
    Replace NaN/Inf with 0 and clip extreme values.
    Synthetic tokens can occasionally have exploding values during AR generation.
    """
    n_bad = (~np.isfinite(feats)).sum()
    if n_bad > 0:
        print(f"  ⚠  {name}: {n_bad} non-finite values replaced with 0")
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    # Clip to ±30 (real Bio-PM tokens have norm ~6, so ±30 is very conservative)
    feats = np.clip(feats, -30.0, 30.0)
    return feats.astype(np.float32)


def load_real_tokens(path: str):
    with h5py.File(path, "r") as f:
        # 1. Load tokens (N, 192, 64) — NOT mean_tokens
        tokens      = f["tokens"][:].astype(np.float32)
        labels      = f["labels"][:].astype(int)
        subject_ids = f["subject_ids"][:].astype(int)
        
        # 2. Load gravity_vecs (N, 64)
        if "gravity_vecs" in f:
            gravity_vecs = f["gravity_vecs"][:].astype(np.float32)
            has_gravity = True
        else:
            print("  ⚠ ERROR: gravity_vecs not found in token_store.hdf5.")
            print("     Run week1_analysis.py with gravity extraction enabled first.")
            print("     Falling back to 128-d features (mean + std only).")
            gravity_vecs = None
            has_gravity = False

    # Verify interleaving before computing
    n_x = tokens[:, 0::3, :].shape[1]
    n_y = tokens[:, 1::3, :].shape[1]
    n_z = tokens[:, 2::3, :].shape[1]
    print(f"  Axis token counts — x:{n_x} y:{n_y} z:{n_z} (total={n_x+n_y+n_z})")
    
    if not (n_x == n_y == n_z):
        print("  ⚠ Unequal axis counts — falling back to global mean (192-d)")
        mean_feats = tokens.mean(axis=1)
        std_feats  = tokens.std(axis=1)
        if has_gravity:
            features = np.concatenate([mean_feats, std_feats, gravity_vecs], axis=1)
        else:
            features = np.concatenate([mean_feats, std_feats], axis=1)
    else:
        mean_x    = tokens[:, 0::3, :].mean(axis=1)   # (N, 64)
        mean_y    = tokens[:, 1::3, :].mean(axis=1)   # (N, 64)
        mean_z    = tokens[:, 2::3, :].mean(axis=1)   # (N, 64)
        std_feats = tokens.std(axis=1)                 # (N, 64)
        if has_gravity:
            features  = np.concatenate([
                mean_x, mean_y, mean_z, std_feats, gravity_vecs
            ], axis=1)   # (N, 320)
        else:
            features  = np.concatenate([
                mean_x, mean_y, mean_z, std_feats
            ], axis=1)   # (N, 256)

    # 7. Sanitize
    features = sanitize_features(features, "real tokens")
    
    # 8. Print stats
    dim = features.shape[1]
    print(f"  Real tokens:  {features.shape} ({dim}-d) subjects: {len(np.unique(subject_ids))} found")
    
    # 9 & 10. Return
    return features, labels, subject_ids, gravity_vecs


def load_synthetic_tokens(path: str, real_gravity: np.ndarray, real_labels: np.ndarray):
    with h5py.File(path, "r") as f:
        # 1. Load tokens (N, 192, 64)
        tokens = f["tokens"][:].astype(np.float32)
        labels = f["labels"][:].astype(int)

    mean_x    = tokens[:, 0::3, :].mean(axis=1)
    mean_y    = tokens[:, 1::3, :].mean(axis=1)
    mean_z    = tokens[:, 2::3, :].mean(axis=1)
    std_feats = tokens.std(axis=1)

    # 5. Compute class-mean gravity from real data:
    if real_gravity is not None:
        class_gravity = {}
        for cls_id in range(NUM_CLASSES):
            mask = real_labels == cls_id
            if mask.sum() > 0:
                class_gravity[cls_id] = real_gravity[mask].mean(axis=0)  # (64,)
            else:
                class_gravity[cls_id] = np.zeros(64, dtype=np.float32)

        # 6. Assign class-mean gravity to each synthetic sample:
        syn_gravity = np.stack([class_gravity[cls_id] for cls_id in labels])
        # 7. Concatenate (320-d)
        features  = np.concatenate([
            mean_x, mean_y, mean_z, std_feats, syn_gravity
        ], axis=1)   # (N, 320)
    else:
        # Fallback (256-d)
        features  = np.concatenate([
            mean_x, mean_y, mean_z, std_feats
        ], axis=1)   # (N, 256)

    # 8. Sanitize
    features = sanitize_features(features, "synthetic tokens")
    
    # 9. Print stats
    print(f"  Synthetic:    {features.shape} ({features.shape[1]}-d)")
    for cls_id in range(NUM_CLASSES):
        n = (labels == cls_id).sum()
        if n > 0:
            print(f"    {ACTIVITY_NAMES[cls_id]:<12} +{n}")
            
    # 10. Return
    return features, labels



# ── Torch MLP Implementation ──────────────────────────────────────────────────
class TorchMLP(nn.Module):
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

def train_torch_mlp(X_train, y_train, X_test, device,
                    num_classes=6, epochs=150, patience=10,
                    batch_size=256, lr=1e-3):
    # Train/val split
    n_val    = max(1, int(len(X_train) * 0.1))
    X_tr, y_tr = X_train[:-n_val], y_train[:-n_val]
    X_va, y_va = X_train[-n_val:], y_train[-n_val:]

    # Tensors
    def to_t(x, dtype):
        return torch.tensor(x, dtype=dtype).to(device)

    Xtr_t = to_t(X_tr,   torch.float32)
    ytr_t = to_t(y_tr,   torch.long)
    Xva_t = to_t(X_va,   torch.float32)
    yva_t = to_t(y_va,   torch.long)
    Xte_t = to_t(X_test, torch.float32)

    # DataLoader
    loader = DataLoader(
        TensorDataset(Xtr_t, ytr_t),
        batch_size=batch_size,
        shuffle=True
    )

    # Model, optimizer, scheduler
    model     = TorchMLP(X_train.shape[1], num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # Training loop with early stopping
    best_loss, best_state, no_improve = float("inf"), None, 0

    for epoch in range(epochs):
        model.train()
        for Xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(Xb), yb).backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xva_t), yva_t).item()

        if val_loss < best_loss:
            best_loss  = val_loss
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # Restore best and predict
    model.load_state_dict(
        {k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        return model(Xte_t).argmax(dim=1).cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# LOSO — collects both macro-F1 and per-class F1 in a single pass
# ══════════════════════════════════════════════════════════════════════════════
def run_single_fold(fold, test_subj, real_feats, real_labels, real_pids,
                    syn_feats, syn_labels, device, n_classes):
    """
    Logic for a single LOSO fold.
    """
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    test_mask  = real_pids == test_subj
    train_mask = ~test_mask

    X_tr_r = real_feats[train_mask]
    y_tr_r = real_labels[train_mask]
    X_te   = real_feats[test_mask]
    y_te   = real_labels[test_mask]

    has_syn = syn_feats is not None and len(syn_feats) > 0

    # Augmented training set
    if has_syn:
        X_tr_a = np.concatenate([X_tr_r, syn_feats], axis=0)
        y_tr_a = np.concatenate([y_tr_r, syn_labels], axis=0)
    else:
        X_tr_a, y_tr_a = X_tr_r, y_tr_r

    # ── Normalise ────────
    sc = StandardScaler()
    sc.fit(X_tr_r)
    X_tr_r_sc = sc.transform(X_tr_r)
    X_te_r    = sc.transform(X_te)
    X_tr_a_sc = sc.transform(X_tr_a)
    X_te_a    = sc.transform(X_te)

    X_tr_a_sc = np.clip(X_tr_a_sc, -10, 10)

    # ── Linear probe (Logistic Regression) ────────
    lr_r = LogisticRegression(max_iter=1000, C=1.0, solver="saga", random_state=42)
    lr_r.fit(X_tr_r_sc, y_tr_r)
    pred_lr_r = lr_r.predict(X_te_r)
    f1_lr_r   = f1_score(y_te, pred_lr_r, average="macro", zero_division=0)

    lr_a = LogisticRegression(max_iter=1000, C=1.0, solver="saga", random_state=42)
    lr_a.fit(X_tr_a_sc, y_tr_a)
    pred_lr_a = lr_a.predict(X_te_a)
    f1_lr_a   = f1_score(y_te, pred_lr_a, average="macro", zero_division=0)

    # ── MLP (PyTorch) ────────
    # Re-detect device inside worker for safety with joblib/loky
    if device.type == "mps" or device.type == "cuda":
        worker_device = device
    else:
        worker_device = torch.device("cpu")

    pred_mlp_r = train_torch_mlp(X_tr_r_sc, y_tr_r, X_te_r, worker_device, num_classes=n_classes)
    f1_mlp_r   = f1_score(y_te, pred_mlp_r, average="macro", zero_division=0)

    pred_mlp_a = train_torch_mlp(X_tr_a_sc, y_tr_a, X_te_a, worker_device, num_classes=n_classes)
    f1_mlp_a   = f1_score(y_te, pred_mlp_a, average="macro", zero_division=0)

    # Per-class F1 (MLP only)
    fold_per_class_real = {}
    fold_per_class_aug  = {}
    for cls_id in range(n_classes):
        if (y_te == cls_id).sum() > 0:
            fold_per_class_real[cls_id] = f1_score(y_te == cls_id, pred_mlp_r == cls_id,
                                                   average="binary", zero_division=0)
            fold_per_class_aug[cls_id]  = f1_score(y_te == cls_id, pred_mlp_a == cls_id,
                                                   average="binary", zero_division=0)
        else:
            fold_per_class_real[cls_id] = None
            fold_per_class_aug[cls_id]  = None

    print(f"  {fold+1:<5} {int(test_subj):<6} {f1_lr_r:>8.3f} {f1_mlp_r:>8.3f} {f1_lr_a:>8.3f} {f1_mlp_a:>8.3f}")

    return {
        "f1_lr_r": f1_lr_r, "f1_lr_a": f1_lr_a,
        "f1_mlp_r": f1_mlp_r, "f1_mlp_a": f1_mlp_a,
        "per_class_real": fold_per_class_real,
        "per_class_aug": fold_per_class_aug
    }

def run_loso_both_conditions(
    real_feats, real_labels, real_pids,
    syn_feats, syn_labels,
    device=None,
):
    if device is None:
        device = get_device()

    subjects = np.unique(real_pids)
    n_classes = NUM_CLASSES

    print(f"\n  LOSO ({len(subjects)} folds) Parallelized...")
    print(f"  {'Fold':<5} {'Subj':<6} {'A_LR':>8} {'A_MLP':>8} {'B_LR':>8} {'B_MLP':>8}")
    print("  " + "-" * 45)

    # Parallel execution across folds
    fold_results = Parallel(n_jobs=-1)(
        delayed(run_single_fold)(
            fold, test_subj, real_feats, real_labels, real_pids,
            syn_feats, syn_labels, device, n_classes
        )
        for fold, test_subj in enumerate(subjects)
    )

    # Aggregate results
    res = {
        "macro_real_lr":   np.array([r["f1_lr_r"] for r in fold_results]),
        "macro_aug_lr":    np.array([r["f1_lr_a"] for r in fold_results]),
        "macro_real_mlp":  np.array([r["f1_mlp_r"] for r in fold_results]),
        "macro_aug_mlp":   np.array([r["f1_mlp_a"] for r in fold_results]),
        "per_class_real":  {i: [] for i in range(n_classes)},
        "per_class_aug":   {i: [] for i in range(n_classes)},
    }

    for r in fold_results:
        for cls_id in range(n_classes):
            if r["per_class_real"][cls_id] is not None:
                res["per_class_real"][cls_id].append(r["per_class_real"][cls_id])
            if r["per_class_aug"][cls_id] is not None:
                res["per_class_aug"][cls_id].append(r["per_class_aug"][cls_id])

    return res


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
    os.makedirs(args.out_dir, exist_ok=True)

    print("\n" + "=" * 65)
    print("CAR-IMU — Augmented HAR Evaluation")
    print(f"Device: {DEVICE}")
    print("=" * 65)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\nLoading data ...")
    real_feats, real_labels, real_pids, real_gravity = load_real_tokens(args.real_tokens)

    if os.path.exists(args.syn_tokens):
        syn_feats, syn_labels = load_synthetic_tokens(
            args.syn_tokens, real_gravity, real_labels)
    else:
        print(f"  ⚠ Not found: {args.syn_tokens} — running real-only.")
        syn_feats, syn_labels = None, None

    # ── Run LOSO (single pass, both conditions) ───────────────────────────────
    res = run_loso_both_conditions(
        real_feats, real_labels, real_pids,
        syn_feats, syn_labels,
        device=DEVICE,
    )

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)
    print("  Feature: mean_x(64)+mean_y(64)+mean_z(64)+std(64)+gravity(64) = 320-d")
    print("  Pooling: axis-wise (0::3, 1::3, 2::3) + global std")
    print(f"  Device:  {DEVICE}")

    print(f"\n  {'Condition':<40} {'LR F1':>8} {'MLP F1':>8}")
    print("  " + "-" * 58)
    print(f"  {'[REF] Week1 real-only 1028-d baseline':<40} {'0.691':>8} {'0.689':>8}")

    r_lr  = res["macro_real_lr"].mean()
    r_mlp = res["macro_real_mlp"].mean()
    a_lr  = res["macro_aug_lr"].mean()
    a_mlp = res["macro_aug_mlp"].mean()

    # Determine dimensionality for label
    dim = real_feats.shape[1]
    print(f"  {'[A] Real only  ('+str(dim)+'-d tokens, LOSO)':<40} {r_lr:>8.3f} {r_mlp:>8.3f}")
    print(f"  {'[B] Real+Synth ('+str(dim)+'-d tokens, LOSO)':<40} {a_lr:>8.3f} {a_mlp:>8.3f}")

    d_lr  = a_lr  - r_lr
    d_mlp = a_mlp - r_mlp
    print(f"  {'Δ = [B] - [A]':<40} {d_lr:>+8.3f} {d_mlp:>+8.3f}")

    # ── Per-class F1 (MLP) ────────────────────────────────────────────────────
    print(f"\n  Per-class F1  (MLP, LOSO mean):")
    print(f"  {'Class':<14} {'Real only':>12} {'Augmented':>12} {'Δ':>8}  ")
    print("  " + "-" * 52)

    for cls_id in range(NUM_CLASSES):
        vals_r = res["per_class_real"][cls_id]
        vals_a = res["per_class_aug"][cls_id]
        r = np.mean(vals_r) if vals_r else float("nan")
        a = np.mean(vals_a) if vals_a else float("nan")
        minority = " ← minority" if cls_id in [2, 3, 4, 5] else ""
        print(f"  {ACTIVITY_NAMES[cls_id]:<14} {r:>12.3f} {a:>12.3f} {a-r:>+8.3f}{minority}")

    # ── Save ──────────────────────────────────────────────────────────────────
    save_path = os.path.join(args.out_dir, "augmented_results.npy")
    np.save(save_path, res, allow_pickle=True)
    print(f"\n  Saved: {save_path}")


if __name__ == "__main__":
    main()
