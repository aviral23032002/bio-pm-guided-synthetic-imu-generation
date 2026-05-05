#!/usr/bin/env python3
"""
decode_synthetic_patches.py — Convert synthetic tokens → raw IMU waveforms
using the trained PatchDecoder (per-token MLP).

For each synthetic token sequence (192, 64):
  1. Predict patch waveform (32,), axis (0/1/2), position (0..1), duration per token
  2. Resample each 32-sample patch to its predicted original duration
  3. Place each patch at its position on its axis in a (300, 3) window
  4. Add per-class gravity DC offset

Usage:
    python decode_synthetic_patches.py
"""

import os
import argparse
import numpy as np
import h5py
import torch
from scipy.interpolate import interp1d

from train_patch_decoder import PatchDecoder, get_device

ACTIVITY_NAMES = {0:"Walking",1:"Jogging",2:"Upstairs",
                  3:"Downstairs",4:"Sitting",5:"Standing"}

WINDOW_LEN = 300   # samples per window (10s @ 30Hz)


def resample_patch(patch_32: np.ndarray, target_len: int) -> np.ndarray:
    """Resample a 32-sample normalized patch to target_len samples."""
    if target_len <= 1:
        return np.full(target_len, patch_32.mean())
    x_old = np.linspace(0, 1, 32)
    x_new = np.linspace(0, 1, target_len)
    return interp1d(x_old, patch_32, kind="linear")(x_new)


def stitch_patches(patch_wavs, axes, positions, durations, window_len=300) -> np.ndarray:
    """
    Reconstruct a (window_len, 3) waveform from per-patch predictions.

    patch_wavs:  (L, 32) predicted patch waveforms
    axes:        (L,)    predicted axis index (0=X,1=Y,2=Z)
    positions:   (L,)    predicted fractional position in window (0..1)
    durations:   (L,)    predicted original patch length (samples)
    """
    waveform = np.zeros((window_len, 3), dtype=np.float32)
    counts   = np.zeros((window_len, 3), dtype=np.float32)

    for j in range(len(patch_wavs)):
        ax  = int(np.clip(axes[j], 0, 2))
        dur = max(1, int(round(float(durations[j]))))
        dur = min(dur, window_len)
        pos = float(np.clip(positions[j], 0, 1))

        start = int(pos * window_len)
        end   = min(start + dur, window_len)
        actual_len = end - start
        if actual_len <= 0:
            continue

        resampled = resample_patch(patch_wavs[j], actual_len)
        waveform[start:end, ax] += resampled
        counts[start:end, ax]   += 1

    # Average where patches overlap
    mask = counts > 0
    waveform[mask] /= counts[mask]
    return waveform


def decode(args):
    device = get_device(args.device)

    # ── Load patch decoder ────────────────────────────────────────────────────
    print(f"Loading patch decoder: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model = PatchDecoder(hidden=256).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])

    tok_mean = ckpt["tok_mean"];  tok_std  = ckpt["tok_std"]
    wav_mean = ckpt["wav_mean"];  wav_std  = ckpt["wav_std"]
    dur_mean = ckpt["dur_mean"];  dur_std  = ckpt["dur_std"]
    print(f"  Best val patch MSE: {ckpt['val_wav_mse']:.4f}")

    # ── Load class gravity means ──────────────────────────────────────────────
    print(f"Loading class gravity: {args.class_gravity}")
    class_gravity = np.load(args.class_gravity, allow_pickle=True).item()

    # ── Load synthetic tokens ─────────────────────────────────────────────────
    print(f"Loading synthetic tokens: {args.syn_tokens}")
    with h5py.File(args.syn_tokens, "r") as f:
        syn_tokens = f["tokens"][:].astype(np.float32)   # (N, 192, 64)
        syn_labels = f["labels"][:].astype(int)
    N = len(syn_tokens)
    print(f"  {N} synthetic sequences")

    # Normalise tokens
    syn_norm = (syn_tokens - tok_mean[None, None, :]) / tok_std[None, None, :]

    # ── Batch-predict patches for all tokens ──────────────────────────────────
    print("Predicting patches ...")
    B, L = syn_norm.shape[:2]
    flat_tokens = syn_norm.reshape(B * L, 64)                  # (N*192, 64)

    all_wavs, all_axes, all_pos, all_dur = [], [], [], []

    with torch.no_grad():
        for start in range(0, len(flat_tokens), args.batch_size):
            batch = torch.from_numpy(flat_tokens[start:start+args.batch_size]).to(device)
            p_wav, p_ax, p_pos, p_dur = model(batch)
            all_wavs.append(p_wav.cpu().numpy())
            all_axes.append(p_ax.argmax(1).cpu().numpy())
            all_pos.append(p_pos.cpu().numpy())
            all_dur.append(p_dur.cpu().numpy())

    pred_wavs = np.concatenate(all_wavs).reshape(B, L, 32)    # (N, 192, 32)
    pred_axes = np.concatenate(all_axes).reshape(B, L)         # (N, 192)
    pred_pos  = np.concatenate(all_pos).reshape(B, L)          # (N, 192)
    pred_dur  = np.concatenate(all_dur).reshape(B, L)          # (N, 192)

    # Denormalise patch waveforms and durations
    pred_wavs = pred_wavs * wav_std[None, None, :] + wav_mean[None, None, :]
    pred_dur  = pred_dur * dur_std + dur_mean

    # ── Stitch patches → waveforms ────────────────────────────────────────────
    print("Stitching patches into waveforms ...")
    waveforms = np.zeros((N, WINDOW_LEN, 3), dtype=np.float32)

    for i in range(N):
        waveforms[i] = stitch_patches(
            pred_wavs[i], pred_axes[i], pred_pos[i], pred_dur[i],
            window_len=WINDOW_LEN
        )

    # ── Add per-class gravity DC ──────────────────────────────────────────────
    # Convert gravity 64-d vector → approximate DC per axis
    # We use the raw gravity signal stats loaded from real data
    # Gravity is a (300,3) signal; its mean per axis IS the DC offset
    # We approximate by loading raw gravity windows and computing class means
    if args.gravity_raw_dir and os.path.exists(args.gravity_raw_dir):
        import glob as _glob
        import h5py as _h5py
        cls_grav_dc = {}
        for path in sorted(_glob.glob(os.path.join(args.gravity_raw_dir, "Data_MeLabel_*.h5"))):
            with _h5py.File(path, "r") as f:
                grav_raw = f["x_gravity"][:].astype(np.float32)   # (N_s, 300, 3)
                lbls     = f["window_label"][:].astype(int)
            for cls_id in range(6):
                mask = lbls == cls_id
                if mask.sum() > 0:
                    cls_grav_dc.setdefault(cls_id, []).append(grav_raw[mask].mean(axis=(0,1)))
        for cls_id in cls_grav_dc:
            cls_grav_dc[cls_id] = np.mean(cls_grav_dc[cls_id], axis=0)  # (3,)

        for cls_id in range(6):
            mask = syn_labels == cls_id
            if mask.sum() > 0 and cls_id in cls_grav_dc:
                waveforms[mask] += cls_grav_dc[cls_id][None, None, :]

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "synthetic_waveforms.hdf5")
    with h5py.File(out_path, "w") as f:
        f.create_dataset("waveforms",   data=waveforms, compression="gzip")
        f.create_dataset("labels",      data=syn_labels.astype(np.int32))
        f.create_dataset("subject_ids", data=np.full(N, -1, np.int32))
        f.attrs["source"] = "CAR-IMU + PatchDecoder"

    print(f"\n✅ Saved: {out_path}")
    print(f"   waveforms: {waveforms.shape}  (N, 300, 3)")

    # ── Mega-Grid Plot: Real vs 20 Synthetic Samples ──────────────────────────
    try:
        import matplotlib.pyplot as plt
        plots_dir = os.path.join(args.out_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        print("Generating Real vs Synthetic comparison plot...")
        # Load 1 real token per class
        with h5py.File(args.real_tokens, "r") as f:
            r_tok = f["tokens"][:]
            r_lbl = f["labels"][:]
            
        real_recons = {}
        for cls_id in range(6):
            if (r_lbl == cls_id).sum() == 0: continue
            sample_tok = r_tok[r_lbl == cls_id][0:1] # (1, 192, 64)
            # normalize
            sample_norm = (sample_tok - tok_mean[None, None, :]) / tok_std[None, None, :]
            with torch.no_grad():
                batch = torch.from_numpy(sample_norm.reshape(-1, 64)).to(device)
                p_wav, p_ax, p_pos, p_dur = model(batch)
            
            p_wav = p_wav.cpu().numpy().reshape(1, 192, 32) * wav_std[None, None, :] + wav_mean[None, None, :]
            p_ax = p_ax.argmax(1).cpu().numpy().reshape(1, 192)
            p_pos = p_pos.cpu().numpy().reshape(1, 192)
            p_dur = p_dur.cpu().numpy().reshape(1, 192) * dur_std + dur_mean
            
            w = stitch_patches(p_wav[0], p_ax[0], p_pos[0], p_dur[0], window_len=WINDOW_LEN)
            if 'cls_grav_dc' in locals() and cls_id in cls_grav_dc:
                w += cls_grav_dc[cls_id][None, :]
            real_recons[cls_id] = w
            
        # Plotting
        unique_cls = sorted(np.unique(syn_labels))
        fig, axes = plt.subplots(6, 21, figsize=(60, 15))
        
        for row, cls_id in enumerate(range(6)):
            cls_name = ACTIVITY_NAMES.get(cls_id, f"Class_{cls_id}")
            
            # Plot Real Reconstructed in Column 0
            ax = axes[row, 0]
            if cls_id in real_recons:
                w = real_recons[cls_id]
                ax.plot(w[:, 0], color='r', alpha=0.8)
                ax.plot(w[:, 1], color='g', alpha=0.8)
                ax.plot(w[:, 2], color='b', alpha=0.8)
            ax.set_title(f"REAL: {cls_name}")
            ax.grid(alpha=0.3)
            
            # Plot 20 Synthetics in Columns 1-20
            syn_samples = waveforms[syn_labels == cls_id][:20] if cls_id in unique_cls else []
            for col in range(1, 21):
                ax = axes[row, col]
                if col - 1 < len(syn_samples):
                    w = syn_samples[col - 1]
                    ax.plot(w[:, 0], color='r', alpha=0.8)
                    ax.plot(w[:, 1], color='g', alpha=0.8)
                    ax.plot(w[:, 2], color='b', alpha=0.8)
                    ax.set_title(f"SYN {col}: {cls_name}")
                ax.grid(alpha=0.3)
                
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, "real_vs_synthetic_grid.png")
        plt.savefig(plot_path, dpi=100)
        plt.close(fig)
        print(f"   Saved mega-grid plot to: {plot_path}")
    except Exception as e:
        print(f"  (Plot skipped: {e})")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    default="decoder_checkpoints/patch_decoder_best.pt")
    p.add_argument("--syn_tokens",    default="synthetic_tokens/synthetic_tokens.hdf5")
    p.add_argument("--real_tokens",   default="results_week1/token_store.hdf5")
    p.add_argument("--class_gravity", default="results_week1/class_gravity_means.npy")
    p.add_argument("--gravity_raw_dir", default="preprocessed_biopm")
    p.add_argument("--out_dir",       default="synthetic_waveforms")
    p.add_argument("--batch_size",    type=int, default=2048)
    p.add_argument("--device",        default="auto")
    return p.parse_args()


if __name__ == "__main__":
    decode(parse_args())
