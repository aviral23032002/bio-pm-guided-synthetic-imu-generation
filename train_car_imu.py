#!/usr/bin/env python3
"""
train_car_imu.py — Train the CAR-IMU decoder on Bio-PM token sequences.

WHAT THIS DOES:
    Loads token_store.hdf5 (from Week 1), trains the CARIMUDecoder to predict
    the next Bio-PM token given all previous tokens + the activity class label.

    This is standard "teacher forcing" training — we feed the model the REAL
    token sequence and ask it to predict one step ahead at every position.

TRAINING DETAILS:
    - Loss:      MSE between predicted and real next token (in embedding space)
    - Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
    - Schedule:  Cosine annealing LR decay
    - Gradient clipping at 1.0 (prevents exploding gradients)
    - Saves best checkpoint based on validation loss
    - Logs per-class loss so you can see which activities are hardest to learn

EXPECTED TRAINING TIME:
    CPU:  ~10-20 minutes for 50 epochs
    MPS:  ~2-4  minutes for 50 epochs  ← MacBook M1/M2/M3/M4
    CUDA: ~1-2  minutes for 50 epochs

USAGE:
    # Auto-detects MPS on Apple Silicon (recommended for M4 MacBook):
    python train_car_imu.py

    # Force a specific device:
    python train_car_imu.py --device mps
    python train_car_imu.py --device cpu
    python train_car_imu.py --device cuda

    Other flags:
    --token_store   path to token_store.hdf5   (default: results_week1/token_store.hdf5)
    --out_dir       where to save checkpoints  (default: car_imu_checkpoints/)
    --epochs        number of training epochs  (default: 50)
    --batch_size    (default: 128 on MPS, 64 on CPU)
    --lr            learning rate              (default: 1e-4)
    --val_split     fraction held out for val  (default: 0.1)
"""

import os
import sys
import time
import argparse
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import our model
from car_imu_decoder import CARIMUDecoder, TOKEN_DIM, SEQ_LEN, NUM_CLASSES, ACTIVITY_NAMES

# MPS has some ops that need float32 — ensure no float64 slips in
torch.set_default_dtype(torch.float32)


def get_best_device(requested: str = "auto") -> torch.device:
    """
    Pick the best available device.
      auto → MPS (Apple Silicon) > CUDA > CPU
      mps  → force MPS  (M1/M2/M3/M4 Mac GPU)
      cuda → force CUDA (NVIDIA GPU)
      cpu  → force CPU
    """
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")



# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# Loads token sequences from token_store.hdf5
# Each item = one 10-second window → (192, 64) token matrix + class label
# ══════════════════════════════════════════════════════════════════════════════
class TokenDataset(Dataset):
    """
    Loads Bio-PM token sequences from token_store.hdf5.

    Each item:
        tokens:      (192, 64) float32  — full token sequence Z for one window
        class_label: int                — activity class 0-5

    The model's forward() prepends [CLS_c] internally, so we don't do it here.
    """
    def __init__(self, token_store_path: str):
        super().__init__()
        print(f"Loading token store: {token_store_path}")
        with h5py.File(token_store_path, "r") as f:
            # Load everything into RAM — ~10810 × 192 × 64 × 4 bytes ≈ 500MB
            # If this is too large, you can memory-map it, but RAM is fine here
            self.tokens  = torch.from_numpy(f["tokens"][:].astype(np.float32))
            self.labels  = torch.from_numpy(f["labels"][:].astype(np.int64))
            self.subject_ids = f["subject_ids"][:].astype(np.int32)

        print(f"  Loaded {len(self.tokens)} windows")
        print(f"  Token shape per window: {self.tokens.shape[1:]}")

        # Print class distribution
        print(f"  Class distribution:")
        for cls_id in range(NUM_CLASSES):
            count = (self.labels == cls_id).sum().item()
            pct   = 100 * count / len(self.labels)
            flag  = " ⚠ MINORITY" if pct < 10 else ""
            print(f"    {ACTIVITY_NAMES[cls_id]:<12} {count:>5} ({pct:.1f}%){flag}")

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx], self.labels[idx]


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, device, epoch, log_interval=50):
    """Run one full pass through the training data."""
    model.train()
    total_loss = 0.0
    n_batches  = 0
    start_time = time.time()

    for batch_idx, (tokens, labels) in enumerate(loader):
        tokens = tokens.to(device)    # (B, 192, 64)
        labels = labels.to(device)    # (B,)

        # Forward pass
        # logits shape: (B, 193, 64) — position 0 is the [CLS_c] output
        logits = model(tokens, labels)

        # Compute loss:
        # - logits[:, 1:, :] are predictions at positions 1..192
        #   (what comes after [CLS_c], and after each real token)
        # - tokens[:, :, :] are the targets (the actual token sequences)
        # So we're asking: given [CLS_c] + z_1..z_{t-1}, predict z_t
        pred   = logits[:, 1:, :]    # (B, 192, 64) — drop [CLS_c] prediction
        target = tokens              # (B, 192, 64)
        loss   = F.mse_loss(pred, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

        if (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch:3d} | batch {batch_idx+1:4d}/{len(loader)} | "
                  f"loss {loss.item():.4f} | {elapsed:.1f}s")

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, device):
    """Compute validation loss + per-class loss breakdown."""
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    # Track per-class loss
    class_loss  = {i: 0.0 for i in range(NUM_CLASSES)}
    class_count = {i: 0   for i in range(NUM_CLASSES)}

    for tokens, labels in loader:
        tokens = tokens.to(device)
        labels = labels.to(device)

        logits = model(tokens, labels)
        pred   = logits[:, 1:, :]
        loss   = F.mse_loss(pred, tokens)

        total_loss += loss.item()
        n_batches  += 1

        # Per-class breakdown
        for cls_id in range(NUM_CLASSES):
            mask = (labels == cls_id)
            if mask.sum() > 0:
                cls_loss = F.mse_loss(pred[mask], tokens[mask]).item()
                class_loss[cls_id]  += cls_loss
                class_count[cls_id] += 1

    avg_loss = total_loss / max(n_batches, 1)

    # Average per-class loss
    per_class = {}
    for cls_id in range(NUM_CLASSES):
        if class_count[cls_id] > 0:
            per_class[cls_id] = class_loss[cls_id] / class_count[cls_id]
        else:
            per_class[cls_id] = float("nan")

    return avg_loss, per_class


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Train CAR-IMU decoder")
    p.add_argument("--token_store", type=str,
                   default="results_week1/token_store.hdf5")
    p.add_argument("--out_dir",     type=str,
                   default="car_imu_checkpoints")
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch_size",  type=int,   default=None,
                   help="Batch size (default: 128 on MPS/CUDA, 64 on CPU)")
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--val_split",   type=float, default=0.1,
                   help="Fraction of data held out for validation")
    p.add_argument("--device",      type=str,   default="auto",
                   choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--log_interval",type=int,   default=50,
                   help="Print loss every N batches")
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    device = get_best_device(args.device)

    # Set batch size based on device if not specified
    if args.batch_size is None:
        args.batch_size = 128 if device.type in ("mps", "cuda") else 64

    print("=" * 60)
    print("CAR-IMU Decoder — Training")
    print("=" * 60)
    print(f"  Token store: {args.token_store}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  LR:          {args.lr}")
    print(f"  Device:      {device}  ", end="")
    if device.type == "mps":
        print("← Apple Silicon GPU 🍎")
    elif device.type == "cuda":
        print(f"← NVIDIA GPU ({torch.cuda.get_device_name(0)})")
    else:
        print("← CPU")
    print()

    # ── Dataset + DataLoaders ─────────────────────────────────────────────────
    dataset = TokenDataset(args.token_store)

    # Train / val split (random, not by subject — we're training a generator)
    n_val   = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                               shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size,
                               shuffle=False, num_workers=0, pin_memory=False)

    print(f"\n  Train: {n_train} windows | Val: {n_val} windows")
    print(f"  Batches per epoch: {len(train_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CARIMUDecoder(
        num_classes = NUM_CLASSES,
        d_model     = TOKEN_DIM,   # 64
        n_layers    = 4,
        n_heads     = 4,
        ffn_dim     = 128,         # 2 × D
        dropout     = 0.1,
    ).to(device)

    print(f"\n  Model parameters: {model.count_parameters():,}")

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),    # slightly higher β2 for stability
    )
    # Cosine annealing: LR decays from lr → 0 over all epochs
    # This helps convergence in the later epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    print(f"\n{'─'*60}")
    print(f"Starting training...")
    print(f"{'─'*60}\n")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, args.log_interval)

        # Validate
        val_loss, per_class_loss = evaluate(model, val_loader, device)

        # Step LR scheduler
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        epoch_time = time.time() - epoch_start

        # ── Epoch summary ─────────────────────────────────────────────────────
        print(f"\n{'─'*60}")
        print(f"Epoch {epoch:3d}/{args.epochs}  |  "
              f"Train MSE: {train_loss:.4f}  |  Val MSE: {val_loss:.4f}  |  "
              f"{epoch_time:.1f}s  |  LR: {scheduler.get_last_lr()[0]:.2e}")

        # Per-class val loss
        print(f"  Per-class val MSE:")
        for cls_id, cls_name in ACTIVITY_NAMES.items():
            flag = " ⚠" if cls_id in [2, 3, 4, 5] else ""   # minority classes
            print(f"    {cls_name:<12} {per_class_loss[cls_id]:.4f}{flag}")

        # ── Save best model ────────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(args.out_dir, "best_model.pt")
            torch.save({
                "epoch":      epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss":   val_loss,
                "per_class_loss": per_class_loss,
                "args":       vars(args),
            }, ckpt_path)
            print(f"  ✅ New best val loss {val_loss:.4f} — saved to {ckpt_path}")

        print()

    # ── Save final loss curves ─────────────────────────────────────────────────
    np.save(os.path.join(args.out_dir, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(args.out_dir, "val_losses.npy"),   np.array(val_losses))

    # ── Plot loss curves ───────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(train_losses, label="Train MSE", color="#4361ee")
        ax.plot(val_losses,   label="Val MSE",   color="#f72585")
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
        ax.set_title("CAR-IMU Training Loss")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        curve_path = os.path.join(args.out_dir, "loss_curve.png")
        plt.savefig(curve_path, dpi=150)
        plt.close()
        print(f"Loss curve saved to {curve_path}")
    except ImportError:
        pass

    # ── Final summary ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"  Best val MSE:  {best_val_loss:.4f}")
    print(f"  Final train MSE: {train_losses[-1]:.4f}")
    print(f"  Checkpoint:    {args.out_dir}/best_model.pt")
    print()
    print("Next step: run  python generate_synthetic.py")


if __name__ == "__main__":
    main()
