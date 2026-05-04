#!/usr/bin/env python3
"""
extract_gravity_vectors.py — Run Bio-PM encoder_gravity on all real windows
and append the resulting (N, 64) vectors to token_store.hdf5.

These gravity vectors encode body orientation / posture per window.
They are needed by the gravity-conditioned TokenDecoder.

Usage:
    python extract_gravity_vectors.py
    python extract_gravity_vectors.py --checkpoint CS690TR/checkpoints/checkpoint.pt
"""

import os
import sys
import argparse
import glob
import numpy as np
import h5py
import torch

# ── Bio-PM imports ─────────────────────────────────────────────────────────────
BIOPM_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CS690TR")
sys.path.insert(0, BIOPM_DIR)
from src.models.biopm import load_pretrained_encoder

ACTIVITY_NAMES = {
    0: "Walking", 1: "Jogging", 2: "Upstairs",
    3: "Downstairs", 4: "Sitting", 5: "Standing"
}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# encoder_gravity uses AdaptiveAvgPool1d which has an MPS bug:
# "input sizes must be divisible by output sizes" on non-divisible lengths.
# Force gravity encoder to CPU regardless of available hardware.
GRAVITY_DEVICE = torch.device("cpu")


def extract(args):
    device = get_device()
    print(f"Device: {device}")

    # ── Load Bio-PM (frozen) ──────────────────────────────────────────────────
    print(f"\nLoading Bio-PM: {args.checkpoint}")
    # Force CPU: encoder_gravity uses AdaptiveAvgPool1d which has an MPS bug
    # ("input sizes must be divisible by output sizes" on non-divisible lengths)
    model = load_pretrained_encoder(args.checkpoint, device="cpu")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print("  encoder_gravity ready (CPU — MPS AdaptivePool unsupported)")

    # ── Load x_gravity from Data_MeLabel files IN SORTED ORDER ───────────────
    # IMPORTANT: must match the exact ordering used in week1_analysis.py
    # Both use sorted(glob.glob("Data_MeLabel_*.h5")) → same alphabetical order
    print(f"\nLoading gravity windows from: {args.biopm_dir}")
    all_gravity = []   # list of (N_s, 300, 3)
    all_labels  = []
    subj_order  = []

    for path in sorted(glob.glob(os.path.join(args.biopm_dir, "Data_MeLabel_*.h5"))):
        subj = int(os.path.basename(path).split("_")[-1].split(".")[0])
        with h5py.File(path, "r") as f:
            grav   = f["x_gravity"][:].astype(np.float32)    # (N_s, 300, 3)
            labels = f["window_label"][:].astype(int)
        all_gravity.append(grav)
        all_labels.append(labels)
        subj_order.append(subj)
        print(f"  Subject {subj:>2}: x_gravity {grav.shape}")

    gravity_windows = np.concatenate(all_gravity, axis=0)  # (N, 300, 3)
    labels_all      = np.concatenate(all_labels,  axis=0)  # (N,)
    N = len(gravity_windows)
    print(f"\nTotal windows: {N}")

    # ── Run encoder_gravity in batches ────────────────────────────────────────
    print(f"\nRunning encoder_gravity (batch_size={args.batch_size}) ...")
    gravity_vecs = []

    with torch.no_grad():
        for start in range(0, N, args.batch_size):
            batch = torch.from_numpy(
                gravity_windows[start:start + args.batch_size]
            ).cpu()                                             # keep on CPU

            gvec = model.encoder_gravity(batch)                # (B, 64) on CPU
            gravity_vecs.append(gvec.numpy())

            if (start // args.batch_size) % 20 == 0:
                pct = 100 * min(start + args.batch_size, N) / N
                print(f"  {pct:.0f}% ({min(start + args.batch_size, N)}/{N})")

    gravity_vecs = np.concatenate(gravity_vecs, axis=0).astype(np.float32)  # (N, 64)
    print(f"\nGravity vectors extracted: {gravity_vecs.shape}")

    # ── Verify count matches token_store ──────────────────────────────────────
    with h5py.File(args.token_store, "r") as f:
        N_tokens = f["tokens"].shape[0]

    if N != N_tokens:
        print(f"\n⚠️  WARNING: gravity count ({N}) ≠ token count ({N_tokens})")
        print("   The files may have been processed differently.")
        print("   Proceeding anyway — check alignment manually.")
    else:
        print(f"✅ Count matches token_store: {N} == {N_tokens}")

    # ── Append to token_store.hdf5 ────────────────────────────────────────────
    print(f"\nAppending gravity_vecs to {args.token_store} ...")
    with h5py.File(args.token_store, "a") as f:
        if "gravity_vecs" in f:
            del f["gravity_vecs"]
            print("  (deleted existing gravity_vecs)")
        f.create_dataset("gravity_vecs", data=gravity_vecs, compression="gzip")
        print(f"  gravity_vecs: {gravity_vecs.shape}")

    # ── Per-class gravity stats ───────────────────────────────────────────────
    print("\nPer-class gravity vector norms:")
    print(f"  {'Class':<12} {'N':>6} {'Mean L2':>10} {'Std L2':>10}")
    print("  " + "-" * 42)
    for cls_id in range(6):
        mask = labels_all == cls_id
        if mask.sum() == 0:
            continue
        vecs = gravity_vecs[mask]
        norms = np.linalg.norm(vecs, axis=1)
        print(f"  {ACTIVITY_NAMES[cls_id]:<12} {mask.sum():>6} "
              f"{norms.mean():>10.4f} {norms.std():>10.4f}")

    # ── Save class-mean gravity for synthetic generation ──────────────────────
    class_gravity_means = {}
    for cls_id in range(6):
        mask = labels_all == cls_id
        if mask.sum() > 0:
            class_gravity_means[cls_id] = gravity_vecs[mask].mean(axis=0)  # (64,)

    out_path = os.path.join(os.path.dirname(args.token_store), "class_gravity_means.npy")
    np.save(out_path, class_gravity_means)
    print(f"\n✅ Class-mean gravity saved: {out_path}")
    print("   (used for synthetic window gravity at decode time)")
    print(f"\n✅ Done! gravity_vecs appended to {args.token_store}")
    print("\nNext step:")
    print("   python train_token_decoder.py   # will now use gravity + raw target")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default="CS690TR/checkpoints/checkpoint.pt")
    p.add_argument("--biopm_dir",   default="preprocessed_wisdm_v2_6class")
    p.add_argument("--token_store", default="results_wisdm_v2_6class/token_store.hdf5")
    p.add_argument("--batch_size",  type=int, default=128)
    return p.parse_args()


if __name__ == "__main__":
    extract(parse_args())
