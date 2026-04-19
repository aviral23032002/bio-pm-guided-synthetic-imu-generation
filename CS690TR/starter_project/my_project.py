#!/usr/bin/env python3
"""
Starter template for your CS690T final project using BioPM features.

=== HOW TO USE THIS FILE ===

  1. Copy this file and rename it for your project
  2. Edit the CONFIGURATION section below
  3. Run preprocessing first (scripts/preprocess_data.py)
  4. Run feature extraction (scripts/extract_features.py)
  5. Edit the downstream_analysis() function for your task

This file provides the skeleton.  Fill in your analysis code where
indicated by "# YOUR CODE HERE" comments.

=== USAGE ===

  python starter_project/my_project.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ===================================================================
# CONFIGURATION — edit these paths for your project
# ===================================================================
FEATURES_PATH = "../features/biopm_features.npz"
OUTPUT_DIR = "../results/"
# ===================================================================


def load_features(path):
    """Load BioPM features from .npz file."""
    data = np.load(path)
    X = data['features']    # (N, 1028) BioPM embeddings
    y = data['labels']      # (N,) integer activity labels
    pids = data['pids']     # (N,) subject IDs
    print(f"Loaded {X.shape[0]} samples, {X.shape[1]}-d features")
    print(f"  Classes: {sorted(np.unique(y).astype(int).tolist())}")
    print(f"  Subjects: {sorted(np.unique(pids).astype(int).tolist())}")
    return X, y, pids


def downstream_analysis(X, y, pids):
    """
    YOUR DOWNSTREAM TASK GOES HERE.

    Ideas:
      - Activity classification (LogisticRegression, SVM, Random Forest)
      - Clustering (k-means, DBSCAN)
      - Visualization (t-SNE, UMAP)
      - Biomarker discovery (feature importance, correlation analysis)
      - Regression (predict some continuous outcome)
      - Anomaly detection
      - Subject identification
      - Transfer learning comparison

    The feature vector X has shape (N, 1028):
      [0:64]     = mean of transformer tokens
      [64:128]   = std of transformer tokens
      [128:1028] = gravity signal (300 timesteps × 3 axes)
    """

    # --- Example: Simple classification ---
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import f1_score

    logo = LeaveOneGroupOut()
    scaler = StandardScaler()
    f1_scores = []

    for train_idx, test_idx in logo.split(X, y, groups=pids):
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        clf = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
        clf.fit(X_tr, y[train_idx].astype(int))
        y_pred = clf.predict(X_te)
        f1_scores.append(f1_score(y[test_idx].astype(int), y_pred,
                                  average='macro'))

    print(f"\nResults:")
    print(f"  LOSO Macro-F1: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
    print(f"  Per-fold: {[f'{s:.3f}' for s in f1_scores]}")

    # YOUR CODE HERE: add your own analysis
    # ...

    return f1_scores


def main():
    print("=" * 60)
    print("CS690T Final Project — BioPM Feature Analysis")
    print("=" * 60)
    print()

    if not os.path.exists(FEATURES_PATH):
        print(f"Features file not found: {FEATURES_PATH}")
        print()
        print("First run:")
        print("  1. python scripts/preprocess_data.py --raw_data_dir ... --output_dir ...")
        print("  2. python scripts/extract_features.py --data_dir ... --checkpoint ...")
        print()
        print("Or update FEATURES_PATH at the top of this file.")
        sys.exit(1)

    X, y, pids = load_features(FEATURES_PATH)
    results = downstream_analysis(X, y, pids)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "f1_scores.npy"), results)
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
