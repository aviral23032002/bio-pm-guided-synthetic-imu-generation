#!/usr/bin/env python3
"""
Minimal feature extraction example — shows the shapes at each stage.

This demonstrates what happens when you run extract_features.py,
without requiring actual data or a checkpoint.  It creates dummy
tensors to illustrate the model's input/output shapes.

Run from the CS690TR directory:
    python examples/example_feature_extraction.py
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models.biopm import BioPMModel, masked_mean_std
import torch.nn.functional as F


def main():
    print("=" * 60)
    print("Feature Extraction Shape Demo (random weights)")
    print("=" * 60)
    print()
    print("NOTE: This uses random weights for shape demonstration.")
    print("For real features, use scripts/extract_features.py with a checkpoint.")
    print()

    # --- Model ---
    model = BioPMModel(n_classes=11)
    model.eval()
    print(f"Model created: BioPMModel(n_classes=11)")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print()

    # --- Fake input (matches preprocessed data format) ---
    B = 4            # batch size
    L = 192          # max movement elements per window
    NORM = 32        # normalized ME length
    T_GRAV = 300     # gravity window length (after interpolation)

    # Movement element patches: (B, L, 32)
    patches = torch.randn(B, L, NORM)
    # Position info: (B, L) — fractional positions [0, 1]
    pos_info = torch.rand(B, L)
    # Additional embedding: (B, L, 5) — axis(0), duration(1), min, max, dir
    # Axis codes must be 0..3 (x=0, y=1, z=2); other fields are floats
    add_emb = torch.zeros(B, L, 5)
    add_emb[:, :, 0] = torch.randint(0, 3, (B, L)).float()
    add_emb[:, :, 1] = torch.rand(B, L) * 10
    # Mask: (B, L) — 0 = visible (no masking at inference time)
    mask = torch.zeros(B, L)

    print(f"Input shapes:")
    print(f"  patches:   {tuple(patches.shape)}  (movement element values)")
    print(f"  pos_info:  {tuple(pos_info.shape)}  (fractional positions)")
    print(f"  add_emb:   {tuple(add_emb.shape)}  (axis, duration, metadata)")
    print(f"  mask:      {tuple(mask.shape)}  (all zeros = no masking)")
    print()

    # --- Encoder forward pass ---
    with torch.no_grad():
        # Accelerometer stream → (B, L, 64)
        acc_tokens = model.encoder_acc(patches, pos_info, mask, add_emb)
        print(f"encoder_acc output:  {tuple(acc_tokens.shape)}")

        # Pool: mean + std → (B, 128)
        acc_pooled = masked_mean_std(acc_tokens)
        print(f"after mean+std pool: {tuple(acc_pooled.shape)}")

        # Gravity stream
        gravity = torch.randn(B, T_GRAV, 3)
        g = gravity.transpose(1, 2)  # (B, 3, T_GRAV)
        g_flat = g.reshape(B, -1)    # (B, 900)
        print(f"gravity flattened:   {tuple(g_flat.shape)}")

        # Fused feature
        fused = torch.cat([acc_pooled, g_flat], dim=-1)
        print(f"fused feature:       {tuple(fused.shape)}")

    print()
    print("Feature vector breakdown (1028-d):")
    print("  [0:64]     = mean of transformer tokens (64-d)")
    print("  [64:128]   = std of transformer tokens (64-d)")
    print("  [128:1028] = gravity signal flattened (300×3 = 900-d)")
    print()
    print("Use these 1028-d vectors for downstream tasks:")
    print("  - Classification (sklearn, XGBoost, etc.)")
    print("  - Clustering (k-means, DBSCAN, etc.)")
    print("  - Visualization (t-SNE, UMAP)")
    print("  - Regression")
    print("  - Retrieval")


if __name__ == "__main__":
    main()
