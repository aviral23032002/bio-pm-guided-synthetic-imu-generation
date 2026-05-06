#!/usr/bin/env python3
"""
evaluate_augmented.py — Compare real-only vs real+synthetic HAR classifier.

WHAT THIS DOES:
    Runs LOSO cross-validation with TWO conditions:
      [A] Real-only:  train on real 320-d token features (mean_x + mean_y + mean_z + std + gravity)
      [B] Augmented:  train on real + synthetic 320-d token features (CAR-IMU / boundary_aware)

    Collects per-class F1 inside the LOSO loop (single pass, no redundancy).

CHANGES FROM PREVIOUS VERSION:
    - "minority" marker replaced with "augmented" marker based on actual syn_labels
    - Crash guard added when real_gravity is None and synthetic gravity unavailable
    - n_jobs changed from hardcoded 16 to -1 (all available cores)

USAGE:
    python evaluate_augmented.py \
        --real_tokens results_wisdm_v2_6class/token_store.hdf5 \
        --syn_tokens  synthetic_tokens_boundary_aware/synthetic_tokens.hdf5 \
        --out_dir     results_boundary_aware \
        --classifier  both
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

# Suppress sklearn numerical warnings
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
    """Replace NaN/Inf with 0 and clip extreme values."""
    n_bad = (~np.isfinite(feats)).sum()
    if n_bad > 0:
        print(f"  ⚠  {name}: {n_bad} non-finite values replaced with 0")
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    feats = np.clip(feats, -30.0, 30.0)
    return feats.astype(np.float32)


def load_real_tokens(path: str):
    """
    Load real token sequences and compute 320-d axis-wise features.
    Returns: features (N, 320), labels (N,), subject_ids (N,), gravity_vecs (N, 64)
    """
    with h5py.File(path, "r") as f:
        tokens      = f["tokens"][:].astype(np.float32)        # (N, 192, 64)
        labels      = f["labels"][:].astype(int)
        subject_ids = f["subject_ids"][:].astype(int)

        if "gravity_vecs" in f:
            gravity_vecs = f["gravity_vecs"][:].astype(np.float32)  # (N, 64)
            has_gravity  = True
        else:
            print("  ⚠ ERROR: gravity_vecs not found in token_store.hdf5.")
            print("     Run week1_analysis.py with gravity extraction enabled.")
            print("     Falling back to 256-d features (mean_xyz + std only).")
            gravity_vecs = None
            has_gravity  = False

    # Verify axis interleaving
    n_x = tokens[:, 0::3, :].shape[1]
    n_y = tokens[:, 1::3, :].shape[1]
    n_z = tokens[:, 2::3, :].shape[1]
    print(f"  Axis token counts — x:{n_x} y:{n_y} z:{n_z} (total={n_x+n_y+n_z})")

    if not (n_x == n_y == n_z):
        print("  ⚠ Unequal axis counts — falling back to global mean")
        mean_feats = tokens.mean(axis=1)
        std_feats  = tokens.std(axis=1)
        features   = (np.concatenate([mean_feats, std_feats, gravity_vecs], axis=1)
                      if has_gravity
                      else np.concatenate([mean_feats, std_feats], axis=1))
    else:
        mean_x    = tokens[:, 0::3, :].mean(axis=1)   # (N, 64)
        mean_y    = tokens[:, 1::3, :].mean(axis=1)   # (N, 64)
        mean_z    = tokens[:, 2::3, :].mean(axis=1)   # (N, 64)
        std_feats = tokens.std(axis=1)                 # (N, 64)
        features  = (np.concatenate([mean_x, mean_y, mean_z, std_feats, gravity_vecs], axis=1)
                     if has_gravity
                     else np.concatenate([mean_x, mean_y, mean_z, std_feats], axis=1))

    features = sanitize_features(features, "real tokens")
    dim      = features.shape[1]
    print(f"  Real tokens:  {features.shape} ({dim}-d) "
          f"subjects: {len(np.unique(subject_ids))} found")

    return features, labels, subject_ids, gravity_vecs


def load_synthetic_tokens(
    path:         str,
    real_gravity: np.ndarray,   # (N_real, 64) — used as fallback if no gravity in file
    real_labels:  np.ndarray,   # (N_real,)    — used to compute class-mean gravity
):
    """
    Load synthetic tokens and compute 320-d axis-wise features.

    Gravity priority:
      1. Load from HDF5 gravity_vecs if saved (boundary_aware / confusion_aware)
      2. Compute class-mean from real_gravity (fallback for older files)
      3. Use zeros if real_gravity is also unavailable
    """
    with h5py.File(path, "r") as f:
        tokens = f["tokens"][:].astype(np.float32)   # (N, 192, 64)
        labels = f["labels"][:].astype(int)

        if "gravity_vecs" in f:
            gravity = f["gravity_vecs"][:].astype(np.float32)   # (N, 64)
            print("  Synthetic gravity: loaded from file (per-token) ✅")
        elif real_gravity is not None:
            # Fallback: class-mean gravity from real data
            gravity = np.stack([
                real_gravity[real_labels == lbl].mean(axis=0)
                for lbl in labels
            ])   # (N, 64)
            print("  Synthetic gravity: class-mean from real data (fallback)")
        else:
            # No gravity available anywhere — use zeros
            gravity = np.zeros((len(labels), 64), dtype=np.float32)
            print("  ⚠ Synthetic gravity: zeros (no gravity source available)")

    # Compute 320-d axis-wise features (matches generate_synthetic.py exactly)
    mean_x    = tokens[:, 0::3, :].mean(axis=1)   # (N, 64)
    mean_y    = tokens[:, 1::3, :].mean(axis=1)   # (N, 64)
    mean_z    = tokens[:, 2::3, :].mean(axis=1)   # (N, 64)
    std_feats = tokens.std(axis=1)                 # (N, 64)

    features = np.concatenate(
        [mean_x, mean_y, mean_z, std_feats, gravity], axis=1)   # (N, 320)

    features = sanitize_features(features, "synthetic tokens")

    print(f"  Synthetic:    {features.shape}")
    for cls_id in range(NUM_CLASSES):
        n = (labels == cls_id).sum()
        if n > 0:
            print(f"    {ACTIVITY_NAMES[cls_id]:<12} +{n}")

    return features, labels


# ══════════════════════════════════════════════════════════════════════════════
# TORCH MLP
# ══════════════════════════════════════════════════════════════════════════════
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


def train_torch_mlp(
    X_train, y_train, X_test, device,
    num_classes=6, epochs=60, patience=5,
    lr=2e-3,
):
    """
    Optimized for M4 MPS: uses full-batch training to eliminate DataLoader overhead.
    """
    n_val      = max(1, int(len(X_train) * 0.1))
    X_tr, y_tr = X_train[:-n_val], y_train[:-n_val]
    X_va, y_va = X_train[-n_val:], y_train[-n_val:]

    def to_t(x, dtype):
        return torch.from_numpy(x).to(device, dtype=dtype, non_blocking=True)

    # Move entire datasets to GPU at once
    Xtr_t = to_t(X_tr,   torch.float32)
    ytr_t = to_t(y_tr,   torch.long)
    Xva_t = to_t(X_va,   torch.float32)
    yva_t = to_t(y_va,   torch.long)
    Xte_t = to_t(X_test, torch.float32)

    model     = TorchMLP(X_train.shape[1], num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_loss, best_state, no_improve = float("inf"), None, 0

    # Fast Full-Batch Training Loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        preds = model(Xtr_t)
        loss  = criterion(preds, ytr_t)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(Xva_t), yva_t).item()
            if val_loss < best_loss:
                best_loss  = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience: break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    
    model.eval()
    with torch.no_grad():
        return model(Xte_t).argmax(dim=1).cpu().numpy()


import threading

# Lock for clean printing from multiple threads
print_lock = threading.Lock()

# ══════════════════════════════════════════════════════════════════════════════
# SINGLE FOLD
# ══════════════════════════════════════════════════════════════════════════════
def run_single_fold(
    fold, test_subj,
    real_feats, real_labels, real_pids,
    syn_feats, syn_labels,
    device, n_classes,
    run_lr=True, run_mlp=True,
    epochs=60,
):
    test_mask  = real_pids == test_subj
    train_mask = ~test_mask

    X_tr_r, y_tr_r = real_feats[train_mask], real_labels[train_mask]
    X_te,   y_te   = real_feats[test_mask],  real_labels[test_mask]

    has_syn = syn_feats is not None and len(syn_feats) > 0
    if has_syn:
        X_tr_a = np.concatenate([X_tr_r, syn_feats], axis=0)
        y_tr_a = np.concatenate([y_tr_r, syn_labels], axis=0)
    else:
        X_tr_a, y_tr_a = X_tr_r, y_tr_r

    # Scaler fitted on real training data only
    sc        = StandardScaler().fit(X_tr_r)
    X_tr_r_sc = sc.transform(X_tr_r)
    X_tr_a_sc = np.clip(sc.transform(X_tr_a), -10, 10)
    X_te_sc   = sc.transform(X_te)

    # ── Logistic Regression (CPU) ─────────────────────────────────────────────
    if run_lr:
        lr_r = LogisticRegression(max_iter=1000, C=1.0, solver="saga", random_state=42)
        lr_r.fit(X_tr_r_sc, y_tr_r)
        f1_lr_r = f1_score(y_te, lr_r.predict(X_te_sc), average="macro", zero_division=0)

        lr_a = LogisticRegression(max_iter=1000, C=1.0, solver="saga", random_state=42)
        lr_a.fit(X_tr_a_sc, y_tr_a)
        f1_lr_a = f1_score(y_te, lr_a.predict(X_te_sc), average="macro", zero_division=0)
    else:
        f1_lr_r = f1_lr_a = float("nan")

    # ── PyTorch MLP (MPS) ─────────────────────────────────────────────────────
    if run_mlp:
        pred_mlp_r = train_torch_mlp(X_tr_r_sc, y_tr_r, X_te_sc, device, num_classes=n_classes, epochs=epochs)
        f1_mlp_r   = f1_score(y_te, pred_mlp_r, average="macro", zero_division=0)

        pred_mlp_a = train_torch_mlp(X_tr_a_sc, y_tr_a, X_te_sc, device, num_classes=n_classes, epochs=epochs)
        f1_mlp_a   = f1_score(y_te, pred_mlp_a, average="macro", zero_division=0)
    else:
        f1_mlp_r = f1_mlp_a = float("nan")
        pred_mlp_r = pred_mlp_a = np.zeros_like(y_te)

    # Per-class metrics summary
    active_pred_r = pred_mlp_r if run_mlp else np.zeros_like(y_te)
    active_pred_a = pred_mlp_a if run_mlp else np.zeros_like(y_te)

    per_class_real = {}
    per_class_aug  = {}
    for cls_id in range(n_classes):
        if (y_te == cls_id).sum() > 0:
            per_class_real[cls_id] = f1_score(y_te == cls_id, active_pred_r == cls_id, average="binary", zero_division=0)
            per_class_aug[cls_id]  = f1_score(y_te == cls_id, active_pred_a == cls_id, average="binary", zero_division=0)

    # Live Fold Logging
    with print_lock:
        log_str = f"  Fold {fold+1:<3} Subj {int(test_subj):<4} |"
        if run_lr:  log_str += f" LR(R:{f1_lr_r:.3f} A:{f1_lr_a:.3f}) |"
        if run_mlp: log_str += f" MLP(R:{f1_mlp_r:.3f} A:{f1_mlp_a:.3f}) |"
        print(log_str)

    return {
        "fold": fold, "test_subj": test_subj,
        "f1_lr_r": f1_lr_r, "f1_lr_a": f1_lr_a,
        "f1_mlp_r": f1_mlp_r, "f1_mlp_a": f1_mlp_a,
        "per_class_real": per_class_real, "per_class_aug": per_class_aug
    }


# ══════════════════════════════════════════════════════════════════════════════
# LOSO RUNNER
# ══════════════════════════════════════════════════════════════════════════════
def run_loso_both_conditions(
    real_feats, real_labels, real_pids,
    syn_feats, syn_labels,
    device=None,
    run_lr=True,
    run_mlp=True,
    epochs=60,
):
    subjects  = np.unique(real_pids)
    n_classes = NUM_CLASSES

    print(f"\n  LOSO ({len(subjects)} folds) ...")
    header = f"  {'Fold':<5} {'Subj':<6}"
    if run_lr:  header += f" {'A_LR':>8} {'B_LR':>8}"
    if run_mlp: header += f" {'A_MLP':>8} {'B_MLP':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    # Parallel execution optimized for M4 MPS
    # Using threading backend to share GPU context and n_jobs=8 to avoid overloading MPS
    fold_results = Parallel(n_jobs=8, backend="threading", verbose=0)(
        delayed(run_single_fold)(
            fold, test_subj,
            real_feats, real_labels, real_pids,
            syn_feats, syn_labels,
            device, n_classes,
            run_lr=run_lr, run_mlp=run_mlp,
            epochs=epochs,
        )
        for fold, test_subj in enumerate(subjects)
    )

    # Sort by fold for clean printing
    fold_results = sorted(fold_results, key=lambda x: x["fold"])

    macro_real_lr  = np.array([r["f1_lr_r"]  for r in fold_results])
    macro_aug_lr   = np.array([r["f1_lr_a"]  for r in fold_results])
    macro_real_mlp = np.array([r["f1_mlp_r"] for r in fold_results])
    macro_aug_mlp  = np.array([r["f1_mlp_a"] for r in fold_results])

    per_class_real = {c: [] for c in range(n_classes)}
    per_class_aug  = {c: [] for c in range(n_classes)}

    for r in fold_results:
        fold_str = f"  {r['fold']+1:<5} {int(r['test_subj']):<6}"
        if run_lr:
            fold_str += f" {r['f1_lr_r']:>8.3f} {r['f1_lr_a']:>8.3f}"
        if run_mlp:
            fold_str += f" {r['f1_mlp_r']:>8.3f} {r['f1_mlp_a']:>8.3f}"
        print(fold_str)

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
# ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--real_tokens", default="results_week1/token_store.hdf5")
    p.add_argument("--syn_tokens",
                   default="synthetic_tokens/synthetic_tokens.hdf5")
    p.add_argument("--out_dir",     default="results_aug")
    p.add_argument("--classifier",  type=str, default="both",
                   choices=["lr", "mlp", "both"],
                   help="Which classifier(s) to run")
    p.add_argument("--batch_size",  type=int, default=2048)
    p.add_argument("--epochs",      type=int, default=100)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("\n" + "=" * 65)
    print("CAR-IMU — Augmented HAR Evaluation")
    print(f"Device: {DEVICE}")
    print("=" * 65)

    # ── Load real tokens ──────────────────────────────────────────────────────
    print("\nLoading data ...")
    real_feats, real_labels, real_pids, real_gravity = load_real_tokens(
        args.real_tokens)

    # ── Load synthetic tokens ─────────────────────────────────────────────────
    if os.path.exists(args.syn_tokens):
        syn_feats, syn_labels = load_synthetic_tokens(
            args.syn_tokens, real_gravity, real_labels)
    else:
        print(f"  ⚠ Not found: {args.syn_tokens} — running real-only.")
        syn_feats  = np.empty((0, real_feats.shape[1]), dtype=np.float32)
        syn_labels = np.empty(0, dtype=int)

    # ── Determine which classes were augmented (for table marker) ─────────────
    augmented_classes = set(np.unique(syn_labels)) if len(syn_labels) > 0 else set()

    # ── Run LOSO ──────────────────────────────────────────────────────────────
    run_lr  = args.classifier in ["lr",  "both"]
    run_mlp = args.classifier in ["mlp", "both"]
    print(f"  Classifiers: {args.classifier.upper()}")

    res = run_loso_both_conditions(
        real_feats, real_labels, real_pids,
        syn_feats, syn_labels,
        device=DEVICE,
        run_lr=run_lr,
        run_mlp=run_mlp,
        epochs=args.epochs,
    )

    # ── Results summary ───────────────────────────────────────────────────────
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
    d_lr  = a_lr  - r_lr
    d_mlp = a_mlp - r_mlp

    def fmt(val, active):
        return f"{val:>8.3f}" if active else f"{'—':>8}"

    run_lr  = res["run_lr"]
    run_mlp = res["run_mlp"]

    print(f"  {'Condition':<40} {'LR F1':>8} {'MLP F1':>8}")
    print("  " + "-" * 58)
    print(f"  {'[REF] Week1 real-only 1028-d baseline':<40} "
          f"{'0.691':>8} {'0.689':>8}")
    print(f"  {'[A] Real only  (320-d tokens, LOSO)':<40} "
          f"{fmt(r_lr, run_lr)} {fmt(r_mlp, run_mlp)}")
    print(f"  {'[B] Real+Synth (320-d tokens, LOSO)':<40} "
          f"{fmt(a_lr, run_lr)} {fmt(a_mlp, run_mlp)}")
    print(f"  {'Δ = [B] - [A]':<40} "
          f"{fmt(d_lr, run_lr)} {fmt(d_mlp, run_mlp)}")

    # ── Per-class F1 ──────────────────────────────────────────────────────────
    primary_name = "MLP" if run_mlp else "LR"
    print(f"\n  Per-class F1  ({primary_name}, LOSO mean):")
    print(f"  {'Class':<14} {'Real only':>12} {'Augmented':>12} {'Δ':>8}  ")
    print("  " + "-" * 55)

    for cls_id in range(NUM_CLASSES):
        vals_r = res["per_class_real"][cls_id]
        vals_a = res["per_class_aug"][cls_id]
        r = np.mean(vals_r) if vals_r else float("nan")
        a = np.mean(vals_a) if vals_a else float("nan")
        d = a - r

        # Mark classes that actually received synthetic tokens
        marker = " ← augmented" if cls_id in augmented_classes else ""
        print(f"  {ACTIVITY_NAMES[cls_id]:<14} {r:>12.3f} {a:>12.3f} {d:>+8.3f}{marker}")

    # ── Augmented classes summary ─────────────────────────────────────────────
    if augmented_classes:
        aug_names = [ACTIVITY_NAMES[c] for c in sorted(augmented_classes)]
        print(f"\n  Augmented classes: {aug_names}")
        print(f"  Synthetic tokens:  {len(syn_labels)} "
              f"({len(syn_labels)/len(real_labels)*100:.1f}% of real)")

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
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    main()