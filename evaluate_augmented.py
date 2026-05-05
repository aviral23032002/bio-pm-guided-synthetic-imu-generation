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
    """
    Load synthetic tokens and compute features.
    If 'gravity_vecs' is available in the HDF5 (confusion_aware), load it directly.
    Otherwise fall back to class-mean gravity from real data.
    """
    with h5py.File(path, "r") as f:
        tokens = f["tokens"][:]              # (N, 192, 64)
        labels = f["labels"][:].astype(int)

        # Load gravity directly if saved (confusion_aware strategy)
        if "gravity_vecs" in f:
            gravity = f["gravity_vecs"][:]   # (N, 64)
            print("  Synthetic gravity: loaded from file (per-token)")
        else:
            # Fall back to class-mean gravity from real data
            gravity = np.stack([
                real_gravity[real_labels == lbl].mean(axis=0)
                for lbl in labels
            ])   # (N, 64)
            print("  Synthetic gravity: class-mean from real data (fallback)")

    # Compute 320-d axis-wise features
    mean_x    = tokens[:, 0::3, :].mean(axis=1)   # (N, 64)
    mean_y    = tokens[:, 1::3, :].mean(axis=1)   # (N, 64)
    mean_z    = tokens[:, 2::3, :].mean(axis=1)   # (N, 64)
    std_feats = tokens.std(axis=1)                 # (N, 64)

    features = np.concatenate([
        mean_x, mean_y, mean_z, std_feats, gravity
    ], axis=1)   # (N, 320)

    features = sanitize_features(features, "synthetic tokens")

    print(f"  Synthetic:    {features.shape}")
    for cls_id in range(NUM_CLASSES):
        n = (labels == cls_id).sum()
        if n > 0:
            print(f"    {ACTIVITY_NAMES[cls_id]:<12} +{n}")

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
                    num_classes=6, epochs=100, patience=7,
                    batch_size=2048, lr=1e-3):
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
def run_single_fold(
    fold, test_subj, real_feats, real_labels, real_pids,
    syn_feats, syn_labels, device, n_classes,
    run_lr=True, run_mlp=True, batch_size=2048, epochs=100
):
    test_mask  = real_pids == test_subj
    train_mask = ~test_mask

    # Fold split
    X_tr_r, y_tr_r = real_feats[train_mask], real_labels[train_mask]
    X_te,   y_te   = real_feats[test_mask],  real_labels[test_mask]

    # Combined training set (Augmented)
    X_tr_a = np.concatenate([X_tr_r, syn_feats], axis=0)
    y_tr_a = np.concatenate([y_tr_r, syn_labels], axis=0)

    # Scaler (fit on real data only to avoid leakage)
    sc = StandardScaler().fit(X_tr_r)
    X_tr_r_sc = sc.transform(X_tr_r)
    X_tr_a_sc = sc.transform(X_tr_a)
    X_te_sc   = sc.transform(X_te)

    # ── Linear probe ─────────────────────────────────────────────
    if run_lr:
        lr_r = LogisticRegression(max_iter=1000, C=1.0,
                                  solver="saga", random_state=42)
        lr_r.fit(X_tr_r_sc, y_tr_r)
        pred_lr_r = lr_r.predict(X_te_sc)
        f1_lr_r   = f1_score(y_te, pred_lr_r,
                             average="macro", zero_division=0)

        lr_a = LogisticRegression(max_iter=1000, C=1.0,
                                  solver="saga", random_state=42)
        lr_a.fit(X_tr_a_sc, y_tr_a)
        pred_lr_a = lr_a.predict(X_te_sc)
        f1_lr_a   = f1_score(y_te, pred_lr_a,
                             average="macro", zero_division=0)
    else:
        f1_lr_r = f1_lr_a = float("nan")
        pred_lr_r = pred_lr_a = np.zeros_like(y_te)

    # ── MLP ──────────────────────────────────────────────────────
    if run_mlp:
        pred_mlp_r = train_torch_mlp(
            X_tr_r_sc, y_tr_r, X_te_sc,
            device, num_classes=NUM_CLASSES,
            batch_size=batch_size, epochs=epochs)
        f1_mlp_r = f1_score(y_te, pred_mlp_r,
                            average="macro", zero_division=0)

        pred_mlp_a = train_torch_mlp(
            X_tr_a_sc, y_tr_a, X_te_sc,
            device, num_classes=NUM_CLASSES,
            batch_size=batch_size, epochs=epochs)
        f1_mlp_a = f1_score(y_te, pred_mlp_a,
                            average="macro", zero_division=0)
    else:
        f1_mlp_r = f1_mlp_a = float("nan")
        pred_mlp_r = pred_mlp_a = np.zeros_like(y_te)

    # Per-class F1
    active_pred_r = pred_mlp_r if run_mlp else pred_lr_r
    active_pred_a = pred_mlp_a if run_mlp else pred_lr_a

    per_class_real = {}
    per_class_aug  = {}
    for cls_id in range(n_classes):
        if (y_te == cls_id).sum() > 0:
            per_class_real[cls_id] = f1_score(y_te == cls_id,
                                             active_pred_r == cls_id,
                                             average="binary", zero_division=0)
            per_class_aug[cls_id]  = f1_score(y_te == cls_id,
                                             active_pred_a == cls_id,
                                             average="binary", zero_division=0)

    return {
        "fold": fold,
        "test_subj": test_subj,
        "f1_lr_r": f1_lr_r, "f1_lr_a": f1_lr_a,
        "f1_mlp_r": f1_mlp_r, "f1_mlp_a": f1_mlp_a,
        "per_class_real": per_class_real,
        "per_class_aug":  per_class_aug,
    }


def run_loso_both_conditions(
    real_feats, real_labels, real_pids,
    syn_feats, syn_labels,
    device=None,
    run_lr=True,
    run_mlp=True,
    n_jobs=16,
    batch_size=4096,
    epochs=50
):
    """
    Perform Leave-One-Subject-Out cross validation across 2 conditions.
    """
    subjects = np.unique(real_pids)
    n_classes = NUM_CLASSES

    header = f"  {'Fold':<5} {'Subj':<6}"
    if run_lr:
        header += f" {'A_LR':>8} {'B_LR':>8}"
    if run_mlp:
        header += f" {'A_MLP':>8} {'B_MLP':>8}"
    print(header)
    print("  " + "-" * len(header.rstrip()))

    # Parallel execution across folds (threading is faster for MPS than loky)
    fold_results = Parallel(n_jobs=n_jobs, backend="threading", verbose=10)(
        delayed(run_single_fold)(
            fold, test_subj, real_feats, real_labels, real_pids,
            syn_feats, syn_labels, device, n_classes,
            run_lr=run_lr, run_mlp=run_mlp, batch_size=batch_size, epochs=epochs
        )
        for fold, test_subj in enumerate(subjects)
    )

    # Reconstruct results arrays
    macro_real_lr = np.array([r["f1_lr_r"] for r in fold_results])
    macro_aug_lr  = np.array([r["f1_lr_a"] for r in fold_results])
    macro_real_mlp = np.array([r["f1_mlp_r"] for r in fold_results])
    macro_aug_mlp  = np.array([r["f1_mlp_a"] for r in fold_results])

    per_class_real = {c: [] for c in range(n_classes)}
    per_class_aug  = {c: [] for c in range(n_classes)}
    
    # Sort results by fold ID for clean printing
    fold_results = sorted(fold_results, key=lambda x: x["fold"])
    
    for r in fold_results:
        # Print fold summary
        fold_str = f"  {r['fold']+1:<5} {int(r['test_subj']):<6}"
        if run_lr:
            fold_str += f" {r['f1_lr_r']:>8.3f} {r['f1_lr_a']:>8.3f}"
        if run_mlp:
            fold_str += f" {r['f1_mlp_r']:>8.3f} {r['f1_mlp_a']:>8.3f}"
        print(fold_str)

        # Collect per-class
        for cls_id in range(n_classes):
            if cls_id in r["per_class_real"]:
                per_class_real[cls_id].append(r["per_class_real"][cls_id])
                per_class_aug[cls_id].append(r["per_class_aug"][cls_id])

    return {
        "macro_real_lr":  macro_real_lr,
        "macro_aug_lr":   macro_aug_lr,
        "macro_real_mlp": macro_real_mlp,
        "macro_aug_mlp":  macro_aug_mlp,
        "per_class_real": per_class_real,
        "per_class_aug":  per_class_aug,
        "run_lr":         run_lr,
        "run_mlp":        run_mlp,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--real_tokens", default="results_week1/token_store.hdf5")
    p.add_argument("--syn_tokens",  default="synthetic_tokens/synthetic_tokens.hdf5")
    p.add_argument("--out_dir",     default="results_aug")
    p.add_argument("--classifier", type=str, default="both",
                   choices=["lr", "mlp", "both"],
                   help="Which classifier(s) to run: 'lr' = LogisticRegression only, 'mlp' = PyTorch MLP only, 'both' = run both (default)")
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=100)
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

    # ── Run Evaluation ────────────────────────────────────────────────────────
    run_lr  = args.classifier in ["lr",  "both"]
    run_mlp = args.classifier in ["mlp", "both"]

    print(f"  Classifiers: {args.classifier.upper()}")

    res = run_loso_both_conditions(
        real_feats, real_labels, real_pids,
        syn_feats, syn_labels,
        device=DEVICE,
        run_lr=run_lr,
        run_mlp=run_mlp,
        batch_size=args.batch_size,
        epochs=args.epochs
    )

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)
    print("  Feature: mean_x(64)+mean_y(64)+mean_z(64)+std(64)+gravity(64) = 320-d")
    print("  Pooling: axis-wise (0::3, 1::3, 2::3) + global std")
    print(f"  Device:  {DEVICE}")

    r_lr  = res["macro_real_lr"].mean()
    r_mlp = res["macro_real_mlp"].mean()
    a_lr  = res["macro_aug_lr"].mean()
    a_mlp = res["macro_aug_mlp"].mean()

    # Format values — show "  —  " for inactive classifiers
    def fmt(val, active):
        return f"{val:>8.3f}" if active else f"{'—':>8}"

    run_lr  = res["run_lr"]
    run_mlp = res["run_mlp"]

    print(f"  {'Condition':<40} {'LR F1':>8} {'MLP F1':>8}")
    print("  " + "-" * 58)
    print(f"  {'[REF] Week1 real-only 1028-d baseline':<40} {'0.691':>8} {'0.689':>8}")
    print(f"  {'[A] Real only  (320-d tokens, LOSO)':<40} {fmt(r_lr, run_lr)} {fmt(r_mlp, run_mlp)}")
    print(f"  {'[B] Real+Synth (320-d tokens, LOSO)':<40} {fmt(a_lr, run_lr)} {fmt(a_mlp, run_mlp)}")

    d_lr  = a_lr  - r_lr
    d_mlp = a_mlp - r_mlp
    print(f"  {'Δ = [B] - [A]':<40} {fmt(d_lr, run_lr)} {fmt(d_mlp, run_mlp)}")

    # ── Per-class F1 (MLP or LR) ──────────────────────────────────────────────
    primary_name = "MLP" if run_mlp else "LR"
    print(f"\n  Per-class F1  ({primary_name}, LOSO mean):")
    print(f"  {'Class':<14} {'Real only':>12} {'Augmented':>12} {'Δ':>8}  ")
    print("  " + "-" * 52)

    for cls_id in range(NUM_CLASSES):
        vals_r = res["per_class_real"][cls_id]
        vals_a = res["per_class_aug"][cls_id]
        r = np.mean(vals_r) if vals_r else float("nan")
        a = np.mean(vals_a) if vals_a else float("nan")
        minority = " ← minority" if cls_id in [2, 3, 4, 5] else ""
        print(f"  {ACTIVITY_NAMES[cls_id]:<14} {r:>12.3f} {a:>12.3f} {a-r:>+8.3f}{minority}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    primary_delta = d_mlp if run_mlp else d_lr
    print("\n  Verdict:")
    if primary_delta > 0.01:
        print(f"  ✅ Augmentation HELPS  {primary_name} Macro-F1 {primary_delta:+.3f}")
    elif primary_delta > -0.01:
        print(f"  ➡️  Augmentation NEUTRAL (Δ={primary_delta:+.3f})")
    else:
        print(f"  ⚠️  Augmentation HURTS  (Δ={primary_delta:+.3f})")

    # ── Save ──────────────────────────────────────────────────────────────────
    save_path = os.path.join(args.out_dir, "augmented_results.npy")
    np.save(save_path, res, allow_pickle=True)
    print(f"\n  Results saved to {save_path}")
    print(f"\n  Saved: {save_path}")


if __name__ == "__main__":
    main()
