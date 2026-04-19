#!/usr/bin/env python3
"""
Minimal downstream task example — classification with BioPM features.

Shows how to load extracted features and train a simple classifier.

Run from the CS690TR directory:
    python examples/example_downstream.py --features features/biopm_features.npz

Or with synthetic data for demo:
    python examples/example_downstream.py --demo
"""

import os
import sys
import argparse
import numpy as np


def demo_with_synthetic_data():
    """Run a demo with synthetic features to show the workflow."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
    from sklearn.metrics import f1_score, classification_report

    print("=" * 60)
    print("Downstream Classification Demo (synthetic features)")
    print("=" * 60)
    print()

    # Simulate 500 samples, 1028-d features, 5 classes, 5 subjects
    np.random.seed(42)
    N, D, K, S = 500, 1028, 5, 5
    X = np.random.randn(N, D).astype(np.float32)
    y = np.random.randint(0, K, N)
    pids = np.random.choice(S, N)

    # Add class-dependent signal so the classifier can learn something
    for c in range(K):
        X[y == c, c * 10:(c + 1) * 10] += 2.0

    print(f"Features: {X.shape}")
    print(f"Labels:   {y.shape} — {K} classes")
    print(f"Subjects: {S} unique")
    print()

    # --- Leave-One-Subject-Out cross-validation ---
    logo = LeaveOneGroupOut()
    scaler = StandardScaler()

    f1_scores = []
    for train_idx, test_idx in logo.split(X, y, groups=pids):
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        clf = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
        clf.fit(X_tr, y[train_idx])
        y_pred = clf.predict(X_te)
        f1_scores.append(f1_score(y[test_idx], y_pred, average='macro'))

    print(f"LOSO Macro-F1 per fold: {[f'{s:.3f}' for s in f1_scores]}")
    print(f"Mean Macro-F1: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
    print()
    print("Replace synthetic data with real BioPM features for your project!")


def classify_real_features(features_path):
    """Load real features and run classification."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import f1_score, classification_report

    data = np.load(features_path)
    X = data['features']
    y = data['labels'].astype(int)
    pids = data['pids'].astype(int)

    print("=" * 60)
    print("Downstream Classification with BioPM Features")
    print("=" * 60)
    print(f"Features: {X.shape}")
    print(f"Labels:   {sorted(np.unique(y).tolist())}")
    print(f"Subjects: {sorted(np.unique(pids).tolist())}")
    print()

    logo = LeaveOneGroupOut()
    scaler = StandardScaler()

    f1_scores = []
    all_y_true, all_y_pred = [], []

    for train_idx, test_idx in logo.split(X, y, groups=pids):
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        clf = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
        clf.fit(X_tr, y[train_idx])
        y_pred = clf.predict(X_te)
        f1_scores.append(f1_score(y[test_idx], y_pred, average='macro'))
        all_y_true.extend(y[test_idx].tolist())
        all_y_pred.extend(y_pred.tolist())

    print(f"LOSO Macro-F1 per fold: {[f'{s:.3f}' for s in f1_scores]}")
    print(f"Mean: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
    print()
    print("Classification Report (pooled):")
    print(classification_report(all_y_true, all_y_pred))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=str, default=None,
                   help="Path to .npz with features, labels, pids")
    p.add_argument("--demo", action="store_true",
                   help="Run with synthetic data")
    args = p.parse_args()

    if args.demo or args.features is None:
        demo_with_synthetic_data()
    else:
        classify_real_features(args.features)


if __name__ == "__main__":
    main()
