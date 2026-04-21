#!/usr/bin/env python3
"""
evaluate_augmented.py — Compare real-only vs real+synthetic HAR classifier.

WHAT THIS DOES:
    Runs LOSO cross-validation with TWO conditions:
      [A] Real-only:  train on real 64-d token features
      [B] Augmented:  train on real + synthetic 64-d token features (CAR-IMU)

    Collects per-class F1 inside the LOSO loop (single pass, no redundancy).

USAGE:
    python evaluate_augmented.py
"""

import os
import argparse
import warnings
import numpy as np
import h5py
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

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
        features    = f["mean_tokens"][:].astype(np.float32)   # (N, 64)
        labels      = f["labels"][:].astype(int)
        subject_ids = f["subject_ids"][:].astype(int)
    features = sanitize_features(features, "real tokens")
    print(f"  Real tokens:  {features.shape}  subjects: {np.unique(subject_ids).tolist()}")
    return features, labels, subject_ids


def load_synthetic_tokens(path: str):
    with h5py.File(path, "r") as f:
        features = f["mean_tokens"][:].astype(np.float32)   # (N_syn, 64)
        labels   = f["labels"][:].astype(int)
    features = sanitize_features(features, "synthetic tokens")
    print(f"  Synthetic:    {features.shape}")
    for cls_id in range(NUM_CLASSES):
        n = (labels == cls_id).sum()
        if n > 0:
            print(f"    {ACTIVITY_NAMES[cls_id]:<12} +{n}")
    return features, labels



# ══════════════════════════════════════════════════════════════════════════════
# LOSO — collects both macro-F1 and per-class F1 in a single pass
# ══════════════════════════════════════════════════════════════════════════════
def run_loso_both_conditions(
    real_feats:  np.ndarray,
    real_labels: np.ndarray,
    real_pids:   np.ndarray,
    syn_feats:   np.ndarray,
    syn_labels:  np.ndarray,
):
    """
    One LOSO pass that simultaneously trains:
      - Condition A: real only (MLP)
      - Condition B: real + synthetic (MLP)
      - Condition A: real only (Linear probe)
      - Condition B: real + synthetic (Linear probe)

    Returns:
        results dict with macro-F1 and per-class F1 for both conditions.
    """
    subjects = np.unique(real_pids)
    n_classes = NUM_CLASSES

    macro_real_lr,  macro_aug_lr  = [], []
    macro_real_mlp, macro_aug_mlp = [], []
    per_class_real  = {i: [] for i in range(n_classes)}
    per_class_aug   = {i: [] for i in range(n_classes)}

    has_syn = syn_feats is not None and len(syn_feats) > 0

    print(f"\n  LOSO ({len(subjects)} folds) ...")
    print(f"  {'Fold':<5} {'Subj':<6} {'A_LR':>8} {'A_MLP':>8} {'B_LR':>8} {'B_MLP':>8}")
    print("  " + "-" * 45)

    for fold, test_subj in enumerate(subjects):
        test_mask  = real_pids == test_subj
        train_mask = ~test_mask

        X_tr_r = real_feats[train_mask]
        y_tr_r = real_labels[train_mask]
        X_te   = real_feats[test_mask]
        y_te   = real_labels[test_mask]

        # Augmented training set
        if has_syn:
            X_tr_a = np.concatenate([X_tr_r, syn_feats], axis=0)
            y_tr_a = np.concatenate([y_tr_r, syn_labels], axis=0)
        else:
            X_tr_a, y_tr_a = X_tr_r, y_tr_r

        # ── Normalise — fit ONLY on real train data, apply everywhere ────────
        # Fitting on synthetic too would corrupt the scaler with outliers.
        sc = StandardScaler()
        sc.fit(X_tr_r)                          # fit on real only
        X_tr_r_sc = sc.transform(X_tr_r)
        X_te_r    = sc.transform(X_te)
        X_tr_a_sc = sc.transform(X_tr_a)        # same scaler for augmented
        X_te_a    = sc.transform(X_te)

        # Clip after scaling to remove any residual outliers (>10σ = noise)
        X_tr_a_sc = np.clip(X_tr_a_sc, -10, 10)

        # ── Linear probe ─────────────────────────────────────────────────────
        lr_r = LogisticRegression(max_iter=1000, C=1.0, solver="saga",
                                  random_state=42)
        lr_r.fit(X_tr_r_sc, y_tr_r)
        pred_lr_r = lr_r.predict(X_te_r)
        f1_lr_r   = f1_score(y_te, pred_lr_r, average="macro", zero_division=0)

        lr_a = LogisticRegression(max_iter=1000, C=1.0, solver="saga",
                                  random_state=42)
        lr_a.fit(X_tr_a_sc, y_tr_a)
        pred_lr_a = lr_a.predict(X_te_a)
        f1_lr_a   = f1_score(y_te, pred_lr_a, average="macro", zero_division=0)


        # ── MLP ──────────────────────────────────────────────────────────────
        mlp_r = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                               random_state=42, early_stopping=True,
                               validation_fraction=0.1, n_iter_no_change=15)
        mlp_r.fit(X_tr_r_sc, y_tr_r)
        pred_mlp_r = mlp_r.predict(X_te_r)
        f1_mlp_r   = f1_score(y_te, pred_mlp_r, average="macro", zero_division=0)

        mlp_a = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                               random_state=42, early_stopping=True,
                               validation_fraction=0.1, n_iter_no_change=15)
        mlp_a.fit(X_tr_a_sc, y_tr_a)
        pred_mlp_a = mlp_a.predict(X_te_a)
        f1_mlp_a   = f1_score(y_te, pred_mlp_a, average="macro", zero_division=0)

        # ── Collect ───────────────────────────────────────────────────────────
        macro_real_lr.append(f1_lr_r);  macro_aug_lr.append(f1_lr_a)
        macro_real_mlp.append(f1_mlp_r); macro_aug_mlp.append(f1_mlp_a)

        # Per-class F1 (MLP only)
        for cls_id in range(n_classes):
            if (y_te == cls_id).sum() > 0:
                per_class_real[cls_id].append(
                    f1_score(y_te == cls_id, pred_mlp_r == cls_id,
                             average="binary", zero_division=0))
                per_class_aug[cls_id].append(
                    f1_score(y_te == cls_id, pred_mlp_a == cls_id,
                             average="binary", zero_division=0))

        print(f"  {fold+1:<5} {int(test_subj):<6} "
              f"{f1_lr_r:>8.3f} {f1_mlp_r:>8.3f} "
              f"{f1_lr_a:>8.3f} {f1_mlp_a:>8.3f}")

    return {
        "macro_real_lr":   np.array(macro_real_lr),
        "macro_aug_lr":    np.array(macro_aug_lr),
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
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 65)
    print("CAR-IMU — Augmented HAR Evaluation")
    print("=" * 65)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\nLoading data ...")
    real_feats, real_labels, real_pids = load_real_tokens(args.real_tokens)

    if os.path.exists(args.syn_tokens):
        syn_feats, syn_labels = load_synthetic_tokens(args.syn_tokens)
    else:
        print(f"  ⚠ Not found: {args.syn_tokens} — running real-only.")
        syn_feats, syn_labels = None, None

    # ── Run LOSO (single pass, both conditions) ───────────────────────────────
    res = run_loso_both_conditions(
        real_feats, real_labels, real_pids,
        syn_feats, syn_labels
    )

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)
    print(f"\n  {'Condition':<40} {'LR F1':>8} {'MLP F1':>8}")
    print("  " + "-" * 58)
    print(f"  {'[REF] Week1 real-only 1028-d baseline':<40} {'0.691':>8} {'0.689':>8}")

    r_lr  = res["macro_real_lr"].mean()
    r_mlp = res["macro_real_mlp"].mean()
    a_lr  = res["macro_aug_lr"].mean()
    a_mlp = res["macro_aug_mlp"].mean()

    print(f"  {'[A] Real only  (64-d tokens, LOSO)':<40} {r_lr:>8.3f} {r_mlp:>8.3f}")
    print(f"  {'[B] Real+Synth (64-d tokens, LOSO)':<40} {a_lr:>8.3f} {a_mlp:>8.3f}")

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
        d = a - r
        minority = " ← minority" if cls_id in [2, 3, 4, 5] else ""
        print(f"  {ACTIVITY_NAMES[cls_id]:<14} {r:>12.3f} {a:>12.3f} {d:>+8.3f}{minority}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    if d_mlp > 0.01:
        print(f"  ✅ CAR-IMU augmentation HELPS!  MLP Macro-F1 {d_mlp:+.3f}")
    elif d_mlp > -0.01:
        print(f"  ➡️  CAR-IMU augmentation NEUTRAL  (Δ={d_mlp:+.3f})")
    else:
        print(f"  ⚠️  CAR-IMU augmentation HURTS  (Δ={d_mlp:+.3f})")
        print(f"       → Try: python generate_synthetic.py --temperature 0.3")
    print("=" * 65)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_path = os.path.join(args.out_dir, "augmented_results.npy")
    np.save(save_path, res, allow_pickle=True)
    print(f"\n  Saved: {save_path}")


if __name__ == "__main__":
    main()
