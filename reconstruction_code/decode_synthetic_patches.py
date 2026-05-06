#!/usr/bin/env python3
"""
decode_synthetic_patches.py — Convert synthetic tokens → raw IMU waveforms
using the trained PatchDecoder. Uses the 'Continuous Chain' strategy for 
lowest RMSE and zero dropouts.
"""
import os
import argparse
import numpy as np
import h5py
import torch
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from train_patch_decoder import PatchDecoder, get_device

ACTIVITY_NAMES = {0: "Walking", 1: "Jogging", 2: "Upstairs", 3: "Downstairs", 4: "Sitting", 5: "Standing"}
WINDOW_LEN = 300 

def resample_patch(patch_32: np.ndarray, target_len: int) -> np.ndarray:
    """Resample a 32-sample normalized patch to target_len samples."""
    if target_len <= 1:
        return np.full(target_len, patch_32.mean())
    x_old = np.linspace(0, 1, 32)
    x_new = np.linspace(0, 1, target_len)
    return interp1d(x_old, patch_32, kind="linear")(x_new)

def stitch_patches_chain(patch_wavs, axes, positions, durations, window_len=300) -> np.ndarray:
    """
    Reconstruct a (window_len, 3) waveform using a 'Continuous Chain' strategy.
    Ensures zero gaps by forcing patches to fill the entire timeline per axis.
    """
    waveform = np.zeros((window_len, 3), dtype=np.float32)

    for ax in range(3):
        ax_mask = (axes == ax)
        if not ax_mask.any(): continue
        
        # Extract and sort patches for THIS axis
        ax_wavs = patch_wavs[ax_mask]
        ax_pos  = positions[ax_mask]
        sort_idx = np.argsort(ax_pos)
        ax_wavs = ax_wavs[sort_idx]
        
        # Divide the timeline into equal segments based on patch count
        n_patches = len(ax_wavs)
        seg_size  = window_len / n_patches
        
        for i in range(n_patches):
            start = int(i * seg_size)
            end   = int((i + 1) * seg_size) if i < n_patches - 1 else window_len
            actual_len = end - start
            if actual_len <= 0: continue
            
            resampled = resample_patch(ax_wavs[i], actual_len)
            waveform[start:end, ax] = resampled
            
    return waveform

def decode(args):
    device = get_device(args.device)
    
    print(f"Loading patch decoder: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = PatchDecoder(hidden=256).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    
    tok_mean = ckpt["tok_mean"];  tok_std  = ckpt["tok_std"]
    wav_mean = ckpt["wav_mean"];  wav_std  = ckpt["wav_std"]
    dur_mean = ckpt["dur_mean"];  dur_std  = ckpt["dur_std"]
    print(f"  Best val patch MSE: {ckpt['val_wav_mse']:.4f}")

    print(f"Loading synthetic tokens: {args.syn_tokens}")
    with h5py.File(args.syn_tokens, "r") as f:
        tokens = f["tokens"][:].astype(np.float32)
        labels = f["labels"][:].astype(int)
    
    N = len(tokens)
    print(f"  {N} synthetic sequences")
    
    # Normalise tokens
    tokens_norm = (tokens - tok_mean[None, None, :]) / tok_std[None, None, :]
    
    print("Reconstructing via Continuous Chain ...")
    all_waveforms = np.zeros((N, WINDOW_LEN, 3), dtype=np.float32)
    
    bs = 64
    for i in range(0, N, bs):
        end_idx = min(i+bs, N)
        batch_tok = torch.from_numpy(tokens_norm[i:end_idx]).to(device)
        B_curr = batch_tok.shape[0]
        
        with torch.no_grad():
            flat_tok = batch_tok.view(-1, 64)
            p_wav, p_ax, p_pos, p_dur = model(flat_tok)
            
            # Denormalize and Move to CPU
            p_wav_np = p_wav.cpu().numpy() * wav_std[None, :] + wav_mean[None, :]
            p_wav_np = p_wav_np.reshape(B_curr, 192, 32)
            
            p_ax_idx = p_ax.argmax(dim=-1).cpu().numpy().reshape(B_curr, 192)
            p_pos_np = p_pos.cpu().numpy().reshape(B_curr, 192)
            p_dur_np = p_dur.cpu().numpy().reshape(B_curr, 192)
            
            for b in range(B_curr):
                all_waveforms[i+b] = stitch_patches_chain(
                    p_wav_np[b], p_ax_idx[b], p_pos_np[b], p_dur_np[b],
                    window_len=WINDOW_LEN
                )
    
    # Apply Gravity
    print("Applying per-class gravity ...")
    if args.gravity_raw_dir and os.path.exists(args.gravity_raw_dir):
        import glob as _glob
        cls_grav_dc = {}
        for path in sorted(_glob.glob(os.path.join(args.gravity_raw_dir, "Data_MeLabel_*.h5"))):
            with h5py.File(path, "r") as f:
                grav_raw = f["x_gravity"][:].astype(np.float32)
                lbls     = f["window_label"][:].astype(int)
            for cls_id in range(6):
                mask = lbls == cls_id
                if mask.sum() > 0:
                    cls_grav_dc.setdefault(cls_id, []).append(grav_raw[mask].mean(axis=(0,1)))
        
        for cls_id in cls_grav_dc:
            dc = np.mean(cls_grav_dc[cls_id], axis=0)
            mask = labels == cls_id
            if mask.sum() > 0:
                all_waveforms[mask] += dc[None, None, :]
    elif os.path.exists(args.class_gravity):
        # Fallback to precomputed means if raw dir is not provided
        class_gravity = np.load(args.class_gravity, allow_pickle=True).item()
        for cid in range(6):
            mask = labels == cid
            if mask.sum() > 0:
                all_waveforms[mask] += class_gravity.get(cid, np.array([0, 9.8, 0]))

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "synthetic_waveforms.hdf5")
    with h5py.File(out_path, "w") as f:
        f.create_dataset("waveforms", data=all_waveforms, compression="gzip")
        f.create_dataset("labels",    data=labels.astype(np.int32))
    
    print(f"✅ Saved: {out_path}")
    
    # Preview Plot
    try:
        unique_cls = sorted(np.unique(labels))
        fig, axes_plt = plt.subplots(len(unique_cls), 3, figsize=(12, 2.5*len(unique_cls)))
        if len(unique_cls) == 1: axes_plt = [axes_plt]
        for row, cls_id in enumerate(unique_cls):
            sample = all_waveforms[labels == cls_id][0]
            for col, ax_name in enumerate(["X", "Y", "Z"]):
                axes_plt[row][col].plot(sample[:, col])
                axes_plt[row][col].set_title(f"{ACTIVITY_NAMES[cls_id]} — {ax_name}")
                axes_plt[row][col].grid(alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(args.out_dir, "synthetic_waveforms_preview.png")
        plt.savefig(plot_path, dpi=120)
        print(f"   Preview: {plot_path}")
    except Exception as e:
        print(f"  (Plot skipped: {e})")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--syn_tokens",  default="synthetic_tokens_v2_6class_calibrated/synthetic_tokens.hdf5")
    p.add_argument("--checkpoint",  default="reconstruction_code/patch_decoder_best.pt")
    p.add_argument("--class_gravity", default="results_wisdm_v2_6class/class_gravity_means.npy")
    p.add_argument("--gravity_raw_dir", default="preprocessed_wisdm_v2_6class")
    p.add_argument("--out_dir",     default="synthetic_waveforms_v2_6class_final")
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--device",      default="auto")
    return p.parse_args()

if __name__ == "__main__":
    decode(parse_args())