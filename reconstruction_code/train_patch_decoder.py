#!/usr/bin/env python3
"""
train_patch_decoder.py — Train a per-patch MLP that maps each Bio-PM token (64-d)
back to its original patch waveform (32-d) + axis + position + duration.

Why this works better than the sequence CNN:
  - token[i][j] was computed FROM patch[i][j] → direct one-to-one mapping
  - ~1.4M patch-level training pairs (vs 10,810 window pairs before)
  - Simpler problem: 64-d → 32-d per patch

Training data:
  tokens       ← results_week1/token_store.hdf5  (N, 192, 64)
  patches      ← preprocessed_biopm/Data_MeLabel_{s}.h5  x_acc_filt (N, 192, 38)
    cols 0:32  = normalized patch waveform (32 samples)
    col  32    = position in window (0..1)
    col  33    = axis (0=X, 1=Y, 2=Z)
    col  34    = original duration (length before normalisation)

Usage:
    python train_patch_decoder.py
"""

import os
import argparse
import glob
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

ACTIVITY_NAMES = {0:"Walking",1:"Jogging",2:"Upstairs",
                  3:"Downstairs",4:"Sitting",5:"Standing"}


def get_device(req="auto"):
    if req != "auto": return torch.device(req)
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")


# ══════════════════════════════════════════════════════════════════════════════
# DATASET — patch-level pairs
# ══════════════════════════════════════════════════════════════════════════════
class PatchPairDataset(Dataset):
    """
    One sample = one (token_64, patch_waveform_32, axis, position, duration).
    We expand all valid (non-NaN) patches across all windows.
    ~1.4M samples total.
    """

    def __init__(self, token_store: str, biopm_dir: str):
        print("Loading token store ...")
        with h5py.File(token_store, "r") as f:
            tokens      = f["tokens"][:].astype(np.float32)      # (N, 192, 64)
            subject_ids = f["subject_ids"][:].astype(int)
            window_ids  = f["window_ids"][:].astype(int)
            labels_w    = f["labels"][:].astype(int)

        N = len(tokens)
        print(f"  {N} windows, {tokens.shape[1]} max patches each")

        # Build per-subject offset (window_ids are global 0..N-1)
        unique_subjs = list(dict.fromkeys(subject_ids.tolist()))
        offset = {}
        run = 0
        for s in unique_subjs:
            offset[int(s)] = run
            run += int((subject_ids == s).sum())

        # Load patches per subject
        patches_by_subj = {}
        for path in sorted(glob.glob(os.path.join(biopm_dir, "Data_MeLabel_*.h5"))):
            s = int(os.path.basename(path).split("_")[-1].split(".")[0])
            with h5py.File(path, "r") as f:
                patches_by_subj[s] = f["x_acc_filt"][:].astype(np.float32)  # (N_s,192,38)

        # Expand to patch-level pairs
        print("Expanding to patch-level pairs ...")
        tok_list, wav_list, ax_list, pos_list, dur_list, lbl_list = [], [], [], [], [], []

        for i in range(N):
            s       = int(subject_ids[i])
            local_w = int(window_ids[i]) - offset.get(s, 0)
            if s not in patches_by_subj or local_w >= len(patches_by_subj[s]):
                continue
            p38 = patches_by_subj[s][local_w]   # (192, 38)
            tok = tokens[i]                       # (192, 64)

            for j in range(p38.shape[0]):
                row = p38[j]
                if np.isnan(row).any():
                    break  # NaN rows = padding, stop
                tok_list.append(tok[j])            # (64,)
                wav_list.append(row[:32])          # patch waveform
                pos_list.append(row[32])           # position
                ax_list.append(int(row[33]))       # axis 0/1/2
                dur_list.append(row[34])           # original length
                lbl_list.append(labels_w[i])

        self.tokens   = np.array(tok_list,  dtype=np.float32)   # (M, 64)
        self.patches  = np.array(wav_list,  dtype=np.float32)   # (M, 32)
        self.positions= np.array(pos_list,  dtype=np.float32)   # (M,)
        self.axes     = np.array(ax_list,   dtype=np.int64)     # (M,)
        self.durations= np.array(dur_list,  dtype=np.float32)   # (M,)
        self.labels   = np.array(lbl_list,  dtype=np.int64)     # (M,)

        M = len(self.tokens)
        print(f"  Total patch pairs: {M:,}")
        print(f"  Avg patches/window: {M/N:.1f}")

        # Normalisation
        self.tok_mean = self.tokens.mean(axis=0)
        self.tok_std  = self.tokens.std(axis=0) + 1e-8
        self.wav_mean = self.patches.mean(axis=0)
        self.wav_std  = self.patches.std(axis=0) + 1e-8
        self.dur_mean = self.durations.mean()
        self.dur_std  = self.durations.std() + 1e-8

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tok = (self.tokens[idx] - self.tok_mean) / self.tok_std
        wav = (self.patches[idx] - self.wav_mean) / self.wav_std
        dur = (self.durations[idx] - self.dur_mean) / self.dur_std
        return (torch.from_numpy(tok),
                torch.from_numpy(wav),
                torch.tensor(self.axes[idx]),
                torch.tensor(self.positions[idx]),
                torch.tensor(dur),
                self.labels[idx])


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════
class PatchDecoder(nn.Module):
    """
    Per-token MLP decoder. Shared backbone → 4 heads:
      - patch waveform (32-d)
      - axis (3-class)
      - position (scalar 0..1)
      - duration (scalar)
    """

    def __init__(self, token_dim=64, patch_dim=32, hidden=256):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(token_dim, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden),    nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden),    nn.GELU(),
        )

        self.patch_head = nn.Sequential(
            nn.Linear(hidden, 128), nn.GELU(),
            nn.Linear(128, patch_dim),
        )
        self.axis_head  = nn.Linear(hidden, 3)
        self.pos_head   = nn.Sequential(
            nn.Linear(hidden, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid()
        )
        self.dur_head   = nn.Sequential(
            nn.Linear(hidden, 32), nn.GELU(), nn.Linear(32, 1)
        )

    def forward(self, tok):
        h = self.backbone(tok)
        return (self.patch_head(h),
                self.axis_head(h),
                self.pos_head(h).squeeze(-1),
                self.dur_head(h).squeeze(-1))


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN
# ══════════════════════════════════════════════════════════════════════════════
def train(args):
    device = get_device(args.device)
    print(f"\nDevice: {device}")

    ds = PatchPairDataset(args.token_store, args.biopm_dir)
    n_val   = max(1, int(0.1 * len(ds)))
    n_train = len(ds) - n_val
    from torch.utils.data import random_split
    tr_ds, va_ds = random_split(ds, [n_train, n_val],
                                generator=torch.Generator().manual_seed(42))
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"\nTrain: {len(tr_ds):,}  Val: {len(va_ds):,}")

    model = PatchDecoder(hidden=256).to(device)
    print(f"PatchDecoder parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    os.makedirs(args.out_dir, exist_ok=True)
    best_val = float("inf")

    print(f"\n{'Epoch':>6} {'Tr wav':>9} {'Val wav':>9} {'Val ax%':>9} {'Val pos':>9}")
    print("-" * 48)

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        tr_wav = 0.0
        for tok, wav, ax, pos, dur, _ in tr_dl:
            tok = tok.to(device); wav = wav.to(device)
            ax  = ax.to(device);  pos = pos.to(device); dur = dur.to(device)

            p_wav, p_ax, p_pos, p_dur = model(tok)

            # Patch waveform: MSE + cosine
            cos = 1 - F.cosine_similarity(p_wav, wav, dim=1).mean()
            l   = F.mse_loss(p_wav, wav) + 0.3 * cos
            # Metadata heads
            l  += 0.3 * F.cross_entropy(p_ax, ax)
            l  += 0.2 * F.mse_loss(p_pos, pos)
            l  += 0.2 * F.mse_loss(p_dur, dur)

            opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_wav += F.mse_loss(p_wav, wav).item()
        tr_wav /= len(tr_dl)

        # Val
        model.eval()
        va_wav, va_ax_acc, va_pos = 0.0, 0.0, 0.0
        with torch.no_grad():
            for tok, wav, ax, pos, dur, _ in va_dl:
                tok = tok.to(device); wav = wav.to(device)
                ax  = ax.to(device);  pos = pos.to(device); dur = dur.to(device)
                p_wav, p_ax, p_pos, p_dur = model(tok)
                va_wav    += F.mse_loss(p_wav, wav).item()
                va_ax_acc += (p_ax.argmax(1) == ax).float().mean().item()
                va_pos    += F.mse_loss(p_pos, pos).item()
        va_wav    /= len(va_dl)
        va_ax_acc /= len(va_dl)
        va_pos    /= len(va_dl)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>6} {tr_wav:>9.4f} {va_wav:>9.4f} "
                  f"{100*va_ax_acc:>8.1f}% {va_pos:>9.4f}")

        if va_wav < best_val:
            best_val = va_wav
            torch.save({
                "epoch": epoch, "state_dict": model.state_dict(),
                "val_wav_mse": va_wav,
                "tok_mean": ds.tok_mean, "tok_std": ds.tok_std,
                "wav_mean": ds.wav_mean, "wav_std": ds.wav_std,
                "dur_mean": ds.dur_mean, "dur_std": ds.dur_std,
            }, os.path.join(args.out_dir, "patch_decoder_best.pt"))

    print(f"\nBest val patch MSE: {best_val:.4f}")
    print(f"Checkpoint: {args.out_dir}/patch_decoder_best.pt")

    # Sanity plot — compare real vs predicted patches
    model.eval()
    tok, wav, ax, pos, dur, lbl = next(iter(va_dl))
    # ── Full window reconstruction sanity check ───────────────────────────────
    # Reconstruct 3 full windows from val data and compare real vs predicted
    from scipy.interpolate import interp1d as _interp1d

    def _resample(patch32, tgt_len):
        tgt_len = max(1, int(round(tgt_len)))
        if tgt_len == 32:
            return patch32
        x_old = np.linspace(0, 1, 32)
        x_new = np.linspace(0, 1, tgt_len)
        return _interp1d(x_old, patch32, kind="linear")(x_new)

    def _stitch(wavs, axes, positions, durations, win_len=300):
        out    = np.zeros((win_len, 3), dtype=np.float32)
        counts = np.zeros((win_len, 3), dtype=np.float32)
        for j in range(len(wavs)):
            ax  = int(np.clip(int(axes[j]), 0, 2))
            dur = max(1, min(win_len, int(round(float(durations[j])))))
            pos = float(np.clip(float(positions[j]), 0, 1))
            start = int(pos * win_len)
            end   = min(start + dur, win_len)
            if end <= start:
                continue
            seg = _resample(wavs[j], end - start)
            out[start:end, ax]    += seg
            counts[start:end, ax] += 1
        m = counts > 0
        out[m] /= counts[m]
        return out

    # Pick 3 windows from val split — need to grab window-level data
    # Re-load a small sample of windows directly from the HDF5 files
    import glob as _glob
    import random as _random
    _random.seed(0)

    # Load a handful of token windows + their real patch data
    with h5py.File(args.token_store, "r") as f:
        _all_tokens  = f["tokens"][:]       # (N, 192, 64)
        _all_sids    = f["subject_ids"][:].astype(int)
        _all_wids    = f["window_ids"][:].astype(int)
        _all_labels  = f["labels"][:].astype(int)

    # Build offset map
    _uniq = list(dict.fromkeys(_all_sids.tolist()))
    _off  = {}; _run = 0
    for _s in _uniq:
        _off[int(_s)] = _run
        _run += int((_all_sids == _s).sum())

    # Load patches per subject (just for the 3 we need)
    _patch_by_subj = {}
    for _path in sorted(_glob.glob(os.path.join(args.biopm_dir, "Data_MeLabel_*.h5"))):
        _s = int(os.path.basename(_path).split("_")[-1].split(".")[0])
        with h5py.File(_path, "r") as _f:
            _patch_by_subj[_s] = _f["x_acc_filt"][:].astype(np.float32)

    # Pick 3 random valid window indices
    _valid = [i for i in range(len(_all_tokens))
              if _all_sids[i] in _patch_by_subj
              and (_all_wids[i] - _off.get(int(_all_sids[i]), 0)) < len(_patch_by_subj[_all_sids[i]])]
    _picks = _random.sample(_valid, min(3, len(_valid)))

    model.eval()
    fig, axes_grid = plt.subplots(3, 3, figsize=(13, 8))
    ax_names = ["X", "Y", "Z"]

    for row, idx in enumerate(_picks):
        _s      = int(_all_sids[idx])
        _lw     = int(_all_wids[idx]) - _off.get(_s, 0)
        _p38    = _patch_by_subj[_s][_lw]         # (192, 38)
        _tok_w  = _all_tokens[idx]                 # (192, 64)
        _lbl    = _all_labels[idx]

        # Real waveform from actual patches
        _n_valid = (~np.isnan(_p38).any(axis=1)).sum()
        _real_wavs = _p38[:_n_valid, :32]          # real patch waveforms
        _real_axes = _p38[:_n_valid, 33].astype(int)
        _real_pos  = _p38[:_n_valid, 32]
        _real_dur  = _p38[:_n_valid, 34]
        real_win = _stitch(_real_wavs, _real_axes, _real_pos, _real_dur)

        # Predicted waveform via patch decoder
        _tok_norm = (_tok_w[:_n_valid] - ds.tok_mean) / ds.tok_std
        with torch.no_grad():
            _t = torch.from_numpy(_tok_norm).to(device)
            _p_wav, _p_ax, _p_pos, _p_dur = model(_t)
        _p_wav  = _p_wav.cpu().numpy() * ds.wav_std + ds.wav_mean
        _p_axes = _p_ax.argmax(1).cpu().numpy()
        _p_pos  = _p_pos.cpu().numpy()
        _p_dur  = _p_dur.cpu().numpy() * ds.dur_std + ds.dur_mean
        pred_win = _stitch(_p_wav, _p_axes, _p_pos, _p_dur)

        for col in range(3):
            axes_grid[row][col].plot(real_win[:, col], label="Real", alpha=0.85)
            axes_grid[row][col].plot(pred_win[:, col], label="Pred",
                                     alpha=0.85, linestyle="--", color="orange")
            axes_grid[row][col].set_title(
                f"Sample {row+1} — {ACTIVITY_NAMES.get(_lbl, '')} — {ax_names[col]}")
            axes_grid[row][col].legend(fontsize=7)
            axes_grid[row][col].grid(alpha=0.3)

    plt.suptitle("Window Reconstruction: Real vs Predicted (PatchDecoder)", fontsize=11)
    plt.tight_layout()
    win_plot = os.path.join(args.out_dir, "window_reconstruction.png")
    plt.savefig(win_plot, dpi=120)
    print(f"Window reconstruction plot: {win_plot}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--token_store", default="results_week1/token_store.hdf5")
    p.add_argument("--biopm_dir",   default="preprocessed_biopm")
    p.add_argument("--out_dir",     default="decoder_checkpoints")
    p.add_argument("--epochs",      type=int,   default=60)
    p.add_argument("--batch_size",  type=int,   default=512)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--device",      default="auto")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
