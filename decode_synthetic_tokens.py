#!/usr/bin/env python3
"""
decode_synthetic_tokens.py — Convert synthetic token sequences → raw IMU waveforms
using the trained GravityTokenDecoder CNN.

For real data: gravity_vec comes from encoder_gravity (Bio-PM).
For synthetic data: gravity_vec = class-mean from real data (class_gravity_means.npy).

Usage:
    python decode_synthetic_tokens.py
    python decode_synthetic_tokens.py --syn_tokens synthetic_tokens/synthetic_tokens.hdf5
"""

import os
import argparse
import numpy as np
import h5py
import torch

from train_token_decoder import GravityTokenDecoder, get_device

ACTIVITY_NAMES = {
    0: "Walking", 1: "Jogging", 2: "Upstairs",
    3: "Downstairs", 4: "Sitting", 5: "Standing"
}


def load_decoder(checkpoint_path: str, device: torch.device):
    print(f"Loading decoder: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = GravityTokenDecoder(token_dim=64, grav_dim=64, out_len=300,
                                out_channels=3, hidden=128)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()

    print(f"  Best val MSE: {ckpt['val_mse']:.4f}  (epoch {ckpt['epoch']})")
    return model, ckpt


def decode(args):
    device = get_device(args.device)

    # ── Load decoder ──────────────────────────────────────────────────────────
    model, ckpt = load_decoder(args.checkpoint, device)

    tok_mean  = ckpt["tok_mean"]    # (64,)
    tok_std   = ckpt["tok_std"]     # (64,)
    grav_mean = ckpt["grav_mean"]   # (64,)
    grav_std  = ckpt["grav_std"]    # (64,)
    raw_mean  = ckpt["raw_mean"]    # (1,1,3)
    raw_std   = ckpt["raw_std"]     # (1,1,3)

    # ── Load class-mean gravity vectors ───────────────────────────────────────
    print(f"\nLoading class-mean gravity: {args.class_gravity}")
    class_gravity_means = np.load(args.class_gravity, allow_pickle=True).item()
    print("  Class gravity norms:")
    for cls_id, vec in class_gravity_means.items():
        print(f"    {ACTIVITY_NAMES[cls_id]:<12} L2={np.linalg.norm(vec):.4f}")

    # ── Load synthetic tokens ─────────────────────────────────────────────────
    print(f"\nLoading synthetic tokens: {args.syn_tokens}")
    with h5py.File(args.syn_tokens, "r") as f:
        syn_tokens = f["tokens"][:].astype(np.float32)   # (N, 192, 64)
        syn_labels = f["labels"][:].astype(int)

    print(f"  Synthetic tokens: {syn_tokens.shape}")
    for cls_id in range(6):
        n = (syn_labels == cls_id).sum()
        if n > 0:
            print(f"    {ACTIVITY_NAMES[cls_id]:<12} {n}")

    # ── Normalise tokens ──────────────────────────────────────────────────────
    syn_tokens_norm = (syn_tokens - tok_mean[None, None, :]) / tok_std[None, None, :]

    # ── Build gravity vectors: class-mean for each synthetic window ───────────
    # Normalise using same stats as training
    syn_gravity = np.zeros((len(syn_labels), 64), dtype=np.float32)
    for cls_id, mean_vec in class_gravity_means.items():
        mask = syn_labels == cls_id
        syn_gravity[mask] = mean_vec
    syn_gravity_norm = (syn_gravity - grav_mean[None, :]) / grav_std[None, :]

    # ── Batch decode ──────────────────────────────────────────────────────────
    print(f"\nDecoding {len(syn_tokens)} sequences ...")
    all_waveforms = []

    with torch.no_grad():
        for start in range(0, len(syn_tokens_norm), args.batch_size):
            batch_tok  = torch.from_numpy(
                syn_tokens_norm[start:start + args.batch_size]).to(device)
            batch_grav = torch.from_numpy(
                syn_gravity_norm[start:start + args.batch_size]).to(device)

            waveforms = model(batch_tok, batch_grav)   # (B, 300, 3)
            all_waveforms.append(waveforms.cpu().numpy())

            pct = 100 * min(start + args.batch_size, len(syn_tokens)) / len(syn_tokens)
            if (start // args.batch_size) % 10 == 0:
                print(f"  {pct:.0f}%")

    syn_waveforms = np.concatenate(all_waveforms, axis=0)  # (N, 300, 3)

    # ── Denormalise ───────────────────────────────────────────────────────────
    syn_waveforms = syn_waveforms * raw_std[0] + raw_mean[0]

    print(f"\nDecoded waveforms: {syn_waveforms.shape}")
    print(f"  Range: [{syn_waveforms.min():.3f}, {syn_waveforms.max():.3f}]")

    print("\n  Per-class stats:")
    print(f"  {'Class':<12} {'N':>6} {'Mean':>8} {'Std':>8}")
    print("  " + "-" * 38)
    for cls_id in range(6):
        mask = syn_labels == cls_id
        if mask.sum() == 0:
            continue
        w = syn_waveforms[mask]
        print(f"  {ACTIVITY_NAMES[cls_id]:<12} {mask.sum():>6} {w.mean():>8.4f} {w.std():>8.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "synthetic_waveforms.hdf5")
    with h5py.File(out_path, "w") as f:
        f.create_dataset("waveforms",   data=syn_waveforms.astype(np.float32),
                         compression="gzip")
        f.create_dataset("labels",      data=syn_labels.astype(np.int32))
        f.create_dataset("subject_ids", data=np.full(len(syn_labels), -1, np.int32))
        f.attrs["source"] = "CAR-IMU + GravityTokenDecoder"

    print(f"\n✅ Saved: {out_path}")
    print(f"   waveforms: {syn_waveforms.shape}  (N, 300, 3)")

    # ── Sanity plot ───────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        unique_cls = sorted(np.unique(syn_labels))
        fig, axes = plt.subplots(len(unique_cls), 3,
                                 figsize=(12, 2.5 * len(unique_cls)))
        if len(unique_cls) == 1:
            axes = [axes]
        for row, cls_id in enumerate(unique_cls):
            sample = syn_waveforms[syn_labels == cls_id][0]
            for col, ax_name in enumerate(["X", "Y", "Z"]):
                axes[row][col].plot(sample[:, col])
                axes[row][col].set_title(f"{ACTIVITY_NAMES[cls_id]} — {ax_name}")
                axes[row][col].grid(alpha=0.3)
        plt.suptitle("Synthetic Raw Waveforms (CAR-IMU + GravityDecoder)", fontsize=11)
        plt.tight_layout()
        plot_path = os.path.join(args.out_dir, "synthetic_waveforms_preview.png")
        plt.savefig(plot_path, dpi=120)
        print(f"   Preview: {plot_path}")
    except Exception as e:
        print(f"  (Plot skipped: {e})")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     default="decoder_checkpoints/decoder_best.pt")
    p.add_argument("--syn_tokens",     default="synthetic_tokens/synthetic_tokens.hdf5")
    p.add_argument("--class_gravity",  default="results_week1/class_gravity_means.npy")
    p.add_argument("--out_dir",        default="synthetic_waveforms")
    p.add_argument("--batch_size",     type=int, default=128)
    p.add_argument("--device",         default="auto",
                   choices=["auto", "mps", "cuda", "cpu"])
    return p.parse_args()


if __name__ == "__main__":
    decode(parse_args())
