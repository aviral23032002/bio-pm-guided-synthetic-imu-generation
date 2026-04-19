#!/usr/bin/env python3
"""
extract_features.py — Extract BioPM embeddings from preprocessed data.

This is the primary student-facing entry point for feature extraction.

=== WHAT THIS DOES ===

  1. Loads preprocessed Data_MeLabel_*.h5 files
  2. Loads the pretrained BioPM / 50MR encoder
  3. Runs inference (no masking) on each window
  4. Extracts a 1028-d feature vector per window:
       [mean(64-d) | std(64-d) | gravity(900-d)]
  5. Saves features, labels, and subject IDs to .npz

=== USAGE ===

  python scripts/extract_features.py \
      --data_dir     /path/to/preprocessed_h5_files \
      --checkpoint   checkpoints/checkpoint.pt \
      --output       features/my_features.npz \
      --device       cpu \
      --batch_size   32

=== OUTPUT FORMAT ===

  The output .npz file contains:
    features:  (N, 1028) float32  — BioPM embeddings
    labels:    (N,)      float32  — activity labels
    pids:      (N,)      int      — subject identifiers

Load with:
    data = np.load('features/my_features.npz')
    X = data['features']   # (N, 1028)
    y = data['labels']     # (N,)
    pids = data['pids']    # (N,)
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.inference.feature_extractor import extract_features


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract BioPM features from preprocessed data")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Directory with Data_MeLabel_*.h5 files")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to 50MR checkpoint.pt")
    p.add_argument("--output", type=str, default="features/biopm_features.npz",
                   help="Output .npz file path (default: features/biopm_features.npz)")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Batch size for inference (default: 32)")
    p.add_argument("--device", type=str, default="cpu",
                   choices=["cpu", "cuda"],
                   help="Device for inference (default: cpu)")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("BioPM Feature Extraction")
    print("=" * 60)
    print(f"  Data dir:    {args.data_dir}")
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Output:      {args.output}")
    print(f"  Device:      {args.device}")
    print(f"  Batch size:  {args.batch_size}")
    print()

    if not os.path.isfile(args.checkpoint):
        print(f"ERROR: Checkpoint not found at {args.checkpoint}")
        print("Place the 50MR checkpoint.pt in the checkpoints/ directory.")
        sys.exit(1)

    features, labels, pids = extract_features(
        data_root=args.data_dir,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        device=args.device,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez(args.output, features=features, labels=labels, pids=pids)

    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"  Features shape: {features.shape}")
    print(f"  Labels shape:   {labels.shape}")
    print(f"  Unique labels:  {sorted(np.unique(labels).astype(int).tolist())}")
    print(f"  Unique subjects: {sorted(np.unique(pids).astype(int).tolist())}")
    print(f"  Saved to: {args.output}")
    print()
    print("Next steps:")
    print("  data = np.load('" + args.output + "')")
    print("  X = data['features']   # shape (N, 1028)")
    print("  y = data['labels']     # shape (N,)")
    print("  pids = data['pids']    # shape (N,)")


if __name__ == "__main__":
    main()
