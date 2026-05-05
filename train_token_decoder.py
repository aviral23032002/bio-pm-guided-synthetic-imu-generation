#!/usr/bin/env python3
"""
train_token_decoder.py — Train a 1D CNN that maps
    (Bio-PM acc tokens, gravity vector) → raw IMU waveform

Pipeline:
    acc_tokens  (192, 64)  ┐
                            ├→  GravityTokenDecoder CNN  →  window_acc_raw (300, 3)
    gravity_vec (64,)      ┘

Paired training data:
    tokens + gravity_vecs  ← results_week1/token_store.hdf5
    raw waveforms          ← preprocessed_biopm/Data_AccLabel_{subj}.h5 (window_acc_raw)

Run extract_gravity_vectors.py FIRST to add gravity_vecs to token_store.

Usage:
    python train_token_decoder.py
    python train_token_decoder.py --epochs 100 --lr 5e-4
"""

import os
import argparse
import glob
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

ACTIVITY_NAMES = {
    0: "Walking", 1: "Jogging", 2: "Upstairs",
    3: "Downstairs", 4: "Sitting", 5: "Standing"
}


# ══════════════════════════════════════════════════════════════════════════════
# DEVICE
# ══════════════════════════════════════════════════════════════════════════════
def get_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════
class TokenWaveformDataset(Dataset):
    """
    Loads paired (acc_tokens, gravity_vec, raw_waveform) for every real window.

    Target: window_acc_raw (300, 3) — full raw signal (movement + gravity combined)
    Inputs: acc tokens (192, 64) + gravity_vec (64,) from encoder_gravity

    window_ids in token_store are GLOBAL (0..N-1):
        local_w = global_w - offset_map[subject]
    """

    def __init__(self, token_store_path: str, biopm_dir: str):
        print(f"Loading token store: {token_store_path}")
        with h5py.File(token_store_path, "r") as f:
            self.tokens      = f["tokens"][:].astype(np.float32)        # (N, 192, 64)
            self.gravity_vecs= f["gravity_vecs"][:].astype(np.float32)  # (N, 64)
            self.labels      = f["labels"][:].astype(int)
            self.subject_ids = f["subject_ids"][:].astype(int)
            self.window_ids  = f["window_ids"][:].astype(int)

        print(f"  Tokens:       {self.tokens.shape}")
        print(f"  Gravity vecs: {self.gravity_vecs.shape}")

        # Load raw waveforms (window_acc_raw — movement + gravity combined)
        print(f"Loading raw waveforms from: {biopm_dir}")
        self.raw_by_subj = {}
        for path in sorted(glob.glob(os.path.join(biopm_dir, "Data_AccLabel_*.h5"))):
            subj = int(os.path.basename(path).split("_")[-1].split(".")[0])
            with h5py.File(path, "r") as f:
                raw = f["window_acc_raw"][:].astype(np.float32)  # (N_s, 300, 3)
            self.raw_by_subj[subj] = raw
            print(f"  Subject {subj:>2}: {raw.shape}")

        # Normalisation stats
        all_raw   = np.concatenate(list(self.raw_by_subj.values()), axis=0)
        self.raw_mean = all_raw.mean(axis=(0, 1), keepdims=True)   # (1,1,3)
        self.raw_std  = all_raw.std(axis=(0, 1), keepdims=True) + 1e-8

        tok_flat = self.tokens.reshape(-1, 64)
        self.tok_mean = tok_flat.mean(axis=0)
        self.tok_std  = tok_flat.std(axis=0) + 1e-8

        grav_flat = self.gravity_vecs
        self.grav_mean = grav_flat.mean(axis=0)
        self.grav_std  = grav_flat.std(axis=0) + 1e-8

        # Build per-subject offset map (window_ids are global 0..N-1)
        unique_subjs = list(dict.fromkeys(self.subject_ids.tolist()))
        self.subj_offset = {}
        running = 0
        for s in unique_subjs:
            self.subj_offset[int(s)] = running
            running += int((self.subject_ids == s).sum())

        valid = []
        for i in range(len(self.tokens)):
            s       = int(self.subject_ids[i])
            w       = int(self.window_ids[i])
            local_w = w - self.subj_offset.get(s, 0)
            if s in self.raw_by_subj and 0 <= local_w < len(self.raw_by_subj[s]):
                valid.append(i)
        self.valid_indices = np.array(valid)
        print(f"  Valid paired samples: {len(self.valid_indices)} / {len(self.tokens)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i       = self.valid_indices[idx]
        s       = int(self.subject_ids[i])
        local_w = int(self.window_ids[i]) - self.subj_offset.get(s, 0)

        tok  = (self.tokens[i] - self.tok_mean) / self.tok_std          # (192, 64)
        grav = (self.gravity_vecs[i] - self.grav_mean) / self.grav_std  # (64,)
        raw  = self.raw_by_subj[s][local_w]                              # (300, 3)
        raw  = (raw - self.raw_mean[0]) / self.raw_std[0]

        return (torch.from_numpy(tok),
                torch.from_numpy(grav),
                torch.from_numpy(raw),
                self.labels[i])


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════
class FiLMLayer(nn.Module):
    """Conditions the 1D feature maps on the global gravity vector."""
    def __init__(self, grav_dim, channels):
        super().__init__()
        # Predicts a scale (gamma) and shift (beta) for each channel
        self.proj = nn.Linear(grav_dim, channels * 2)
        # Initialize to identity (gamma=0, beta=0) for stable early training
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, grav):
        # x: (B, C, L), grav: (B, grav_dim)
        film_params = self.proj(grav)                     # (B, C*2)
        gamma, beta = film_params.chunk(2, dim=-1)        # 2 x (B, C)
        gamma = gamma.unsqueeze(-1)                       # (B, C, 1)
        beta = beta.unsqueeze(-1)                         # (B, C, 1)
        return x * (1.0 + gamma) + beta


class ResBlock1D(nn.Module):
    def __init__(self, channels, kernel=5, dilation=1):
        super().__init__()
        # Calculate padding to maintain sequence length
        pad = (kernel - 1) * dilation // 2
        
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class GravityTokenDecoder(nn.Module):
    """
    Maps (acc_tokens (B,192,64), gravity_vec (B,64)) → raw waveform (B,300,3).

    Gravity conditioning:
        Gravity vector is injected via FiLM modulation before each ResBlock.
    
    Gravity carries the static orientation/posture (DC offset of raw signal).
    Acc tokens carry movement patterns. Together they reconstruct the full signal.
    """

    def __init__(self, token_dim=64, grav_dim=64, out_len=300, out_channels=3, hidden=128):
        super().__init__()
        self.out_len = out_len

        # Initial projection of tokens: (B, 64, 192) -> (B, hidden, 192)
        self.input_proj = nn.Sequential(
            nn.Conv1d(token_dim, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # We inject gravity using FiLM before each ResBlock stage
        self.film1 = FiLMLayer(grav_dim, hidden)
        self.res1  = ResBlock1D(hidden, kernel=5, dilation=1)
        
        self.film2 = FiLMLayer(grav_dim, hidden)
        self.res2  = ResBlock1D(hidden, kernel=5, dilation=2)  # Expanded receptive field
        
        self.film3 = FiLMLayer(grav_dim, hidden)
        self.res3  = ResBlock1D(hidden, kernel=5, dilation=4)  # Expanded receptive field
        
        self.film4 = FiLMLayer(grav_dim, hidden)
        self.res4  = ResBlock1D(hidden, kernel=5, dilation=8)  # Maximum context

        # Output head
        self.out_head = nn.Sequential(
            nn.Conv1d(hidden, 32, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv1d(32, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, tokens, gravity_vec):
        # tokens: (B, 192, 64) -> Permute to (B, C, L) -> (B, 64, 192)
        x = tokens.permute(0, 2, 1)
        
        x = self.input_proj(x)                  # (B, hidden, 192)

        # Apply FiLM conditioning and Dilated ResBlocks at native resolution
        x = self.film1(x, gravity_vec)
        x = self.res1(x)
        
        x = self.film2(x, gravity_vec)
        x = self.res2(x)
        
        x = self.film3(x, gravity_vec)
        x = self.res3(x)
        
        x = self.film4(x, gravity_vec)
        x = self.res4(x)

        # Late Upsampling: Interpolate just before the final mapping
        x = F.interpolate(x, size=self.out_len, mode="linear", align_corners=False) # (B, hidden, 300)
        
        x = self.out_head(x)                    # (B, 3, 300)
        return x.permute(0, 2, 1)               # (B, 300, 3)


# Keep old name as alias for decode_synthetic_tokens.py compatibility
TokenDecoder = GravityTokenDecoder


# ══════════════════════════════════════════════════════════════════════════════
# LOSS
# ══════════════════════════════════════════════════════════════════════════════
def pearson_loss(pred, target):
    """1 - mean Pearson correlation. Encourages waveform shape matching."""
    p = pred   - pred.mean(dim=1, keepdim=True)
    t = target - target.mean(dim=1, keepdim=True)
    num = (p * t).sum(dim=1)
    den = (p.pow(2).sum(dim=1).sqrt() * t.pow(2).sum(dim=1).sqrt()).clamp(min=1e-8)
    return 1.0 - (num / den).mean()


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def train(args):
    device = get_device(args.device)
    print(f"\nDevice: {device}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = TokenWaveformDataset(args.token_store, args.biopm_dir)
    n_val   = max(1, int(0.15 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True,  num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size,
                          shuffle=False, num_workers=0, pin_memory=False)
    print(f"\nTrain: {len(train_ds)}  Val: {len(val_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = GravityTokenDecoder(token_dim=64, grav_dim=64, out_len=300,
                                out_channels=3, hidden=128)
    model = model.to(device)
    print(f"GravityTokenDecoder parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    os.makedirs(args.out_dir, exist_ok=True)
    best_val = float("inf")
    train_losses, val_losses = [], []

    print(f"\nTraining for {args.epochs} epochs ...\n")
    print(f"{'Epoch':>6} {'Train MSE':>12} {'Val MSE':>12} {'Val Pearson':>13}")
    print("-" * 48)

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_mse = 0.0
        for tok, grav, raw, _ in train_dl:
            tok  = tok.to(device)   # (B, 192, 64)
            grav = grav.to(device)  # (B, 64)
            raw  = raw.to(device)   # (B, 300, 3)

            pred = model(tok, grav)

            # ── Split loss ────────────────────────────────────────────────────
            # raw_mse:  gets gravity DC right (model knows gravity from grav input)
            # zm_mse:   gets movement shape right (zero-mean = gravity removed)
            # pearson:  gets waveform correlation right
            pred_zm = pred   - pred.mean(dim=1, keepdim=True)
            raw_zm  = raw    - raw.mean(dim=1, keepdim=True)
            raw_mse = F.mse_loss(pred, raw)
            zm_mse  = F.mse_loss(pred_zm, raw_zm)
            pear    = pearson_loss(pred, raw)
            loss    = 0.4 * raw_mse + 0.4 * zm_mse + 0.2 * pear
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_mse += F.mse_loss(pred, raw).item()
        train_mse /= len(train_dl)

        # Val
        model.eval()
        val_mse, val_pear = 0.0, 0.0
        with torch.no_grad():
            for tok, grav, raw, _ in val_dl:
                tok  = tok.to(device)
                grav = grav.to(device)
                raw  = raw.to(device)
                pred = model(tok, grav)
                val_mse  += F.mse_loss(pred, raw).item()
                val_pear += (1.0 - pearson_loss(pred, raw).item())
        val_mse  /= len(val_dl)
        val_pear /= len(val_dl)

        train_losses.append(train_mse)
        val_losses.append(val_mse)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>6} {train_mse:>12.4f} {val_mse:>12.4f} {val_pear:>13.4f}")

        if val_mse < best_val:
            best_val = val_mse
            torch.save({
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "val_mse":    val_mse,
                "tok_mean":   dataset.tok_mean,
                "tok_std":    dataset.tok_std,
                "grav_mean":  dataset.grav_mean,
                "grav_std":   dataset.grav_std,
                "raw_mean":   dataset.raw_mean,
                "raw_std":    dataset.raw_std,
            }, os.path.join(args.out_dir, "decoder_best.pt"))

    print(f"\nBest val MSE: {best_val:.4f}")
    print(f"Checkpoint:  {args.out_dir}/decoder_best.pt")

    # Loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train MSE")
    plt.plot(val_losses,   label="Val MSE")
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.legend(); plt.grid(alpha=0.3)
    plt.title("GravityTokenDecoder Training Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "decoder_loss.png"), dpi=120)
    print(f"Loss curve: {args.out_dir}/decoder_loss.png")

    # Reconstruction plot
    model.eval()
    tok, grav, raw, lbl = next(iter(val_dl))
    with torch.no_grad():
        pred = model(tok.to(device), grav.to(device)).cpu()

    fig, axes = plt.subplots(3, 3, figsize=(12, 7))
    for row in range(3):
        for col, ax_name in enumerate(["X", "Y", "Z"]):
            axes[row][col].plot(raw[row, :, col].numpy(), label="Real", alpha=0.8)
            axes[row][col].plot(pred[row, :, col].numpy(), label="Pred",
                                alpha=0.8, linestyle="--")
            axes[row][col].set_title(
                f"Sample {row+1} — {ACTIVITY_NAMES.get(lbl[row].item(),'')} — {ax_name}")
            axes[row][col].legend(fontsize=7)
            axes[row][col].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "decoder_reconstruction.png"), dpi=120)
    print(f"Reconstruction plot: {args.out_dir}/decoder_reconstruction.png")


# ══════════════════════════════════════════════════════════════════════════════
# ARGS
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--token_store", default="results_week1/token_store.hdf5")
    p.add_argument("--biopm_dir",   default="preprocessed_biopm")
    p.add_argument("--out_dir",     default="decoder_checkpoints")
    p.add_argument("--epochs",      type=int,   default=80)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--device",      default="auto",
                   choices=["auto", "mps", "cuda", "cpu"])
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())