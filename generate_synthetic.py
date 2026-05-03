#!/usr/bin/env python3
"""
generate_synthetic.py — Generate synthetic Bio-PM token sequences using CAR-IMU.

Strategies:
  upsample  — bring every class up to majority count (original)
  ratio     — add fixed % to minority classes (original)
  mixup     — phase-amplitude mixup from real tokens only (new)
  combined  — CAR-IMU (ratio) + mixup (ratio) concatenated (new)

New flags:
  --strategy   {upsample,ratio,mixup,combined}
  --car_ratio  float  fraction for CAR-IMU branch in combined (default 0.25)
  --mixup_ratio float fraction for mixup branch in combined (default 0.25)
  --spectral_filter       enable post-generation spectral quality filter
  --spectral_threshold    cosine-sim threshold (default 0.7)
"""

import os
import sys
import argparse
import numpy as np
import h5py
import torch

from car_imu_decoder import CARIMUDecoder, TOKEN_DIM, SEQ_LEN, NUM_CLASSES, ACTIVITY_NAMES


# ── Device selection ──────────────────────────────────────────────────────────
def get_best_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ══════════════════════════════════════════════════════════════════════════════
# CHANGE 2A — Phase-Amplitude Mixup
# ══════════════════════════════════════════════════════════════════════════════
def phase_amplitude_mixup(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    lambda_a: float = None,
    lambda_p: float = None,
) -> np.ndarray:
    """
    Mix two same-class token sequences (192, 64) in the frequency domain.

    lambda_a: amplitude mix weight (default Uniform(0.7, 1.0))
    lambda_p: phase mix weight — higher = stay closer to anchor (default Uniform(0.9, 1.0))

    Returns: float32 ndarray shape (192, 64)
    """
    if lambda_a is None:
        lambda_a = np.random.uniform(0.7, 1.0)
    if lambda_p is None:
        lambda_p = np.random.uniform(0.9, 1.0)

    # FFT along time axis (axis=0): shape (97, 64) for rfft of 192-point signal
    F_a = np.fft.rfft(seq_a, axis=0)
    F_b = np.fft.rfft(seq_b, axis=0)

    A_a, P_a = np.abs(F_a), np.angle(F_a)
    A_b, P_b = np.abs(F_b), np.angle(F_b)

    # Mix amplitude
    A_mix = lambda_a * A_a + (1.0 - lambda_a) * A_b

    # Mix phase along shortest angular path
    delta = P_b - P_a
    delta = (delta + np.pi) % (2 * np.pi) - np.pi   # wrap to [-π, π]
    P_mix = P_a + (1.0 - lambda_p) * delta

    # Reconstruct
    F_mix = A_mix * np.exp(1j * P_mix)
    result = np.fft.irfft(F_mix, n=SEQ_LEN, axis=0)
    return result.astype(np.float32)


def generate_mixup_tokens(
    real_tokens: np.ndarray,   # (N, 192, 64)
    real_labels: np.ndarray,   # (N,)
    gen_counts:  dict,         # cls_id -> int
) -> tuple:
    """
    Generate mixup synthetic tokens for each class in gen_counts.

    Returns (syn_tokens (N_total, 192, 64), syn_labels (N_total,))
    """
    all_tokens = []
    all_labels = []

    for cls_id in range(NUM_CLASSES):
        n = gen_counts.get(cls_id, 0)
        if n == 0:
            continue

        cls_tokens = real_tokens[real_labels == cls_id]
        M = len(cls_tokens)
        print(f"  [{ACTIVITY_NAMES[cls_id]}] generating {n} mixup sequences "
              f"from {M} real samples")

        for _ in range(n):
            idx_a = np.random.randint(0, M)
            idx_b = np.random.randint(0, M)
            mixed = phase_amplitude_mixup(cls_tokens[idx_a], cls_tokens[idx_b])
            all_tokens.append(mixed)
            all_labels.append(cls_id)

    if not all_tokens:
        return (np.empty((0, SEQ_LEN, TOKEN_DIM), dtype=np.float32),
                np.empty(0, dtype=int))

    return (np.stack(all_tokens, axis=0).astype(np.float32),
            np.array(all_labels, dtype=int))


# ══════════════════════════════════════════════════════════════════════════════
# LOAD TRAINED MODEL
# ══════════════════════════════════════════════════════════════════════════════
def load_model(checkpoint_path: str, device: torch.device) -> CARIMUDecoder:
    """Load trained CAR-IMU from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = CARIMUDecoder(
        num_classes=NUM_CLASSES,
        d_model=TOKEN_DIM,
        n_layers=4,
        n_heads=4,
        ffn_dim=128,
        dropout=0.0,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"  Trained for {ckpt['epoch']} epochs")
    print(f"  Best val MSE: {ckpt['val_loss']:.4f}")
    print(f"  Per-class val MSE at best checkpoint:")
    for cls_id, name in ACTIVITY_NAMES.items():
        mse = ckpt['per_class_loss'].get(cls_id, float('nan'))
        print(f"    {name:<12} {mse:.4f}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE HOW MANY TO GENERATE PER CLASS
# ══════════════════════════════════════════════════════════════════════════════
def compute_generation_counts(
    real_labels: np.ndarray,
    strategy: str,
    syn_ratio: float,
) -> dict:
    """
    Decide how many synthetic windows to generate per class.

    strategy='upsample': bring every class up to majority count.
    strategy='ratio':    add syn_ratio × real_count for minority classes (<10%).
    """
    total  = len(real_labels)
    counts = {cls_id: int((real_labels == cls_id).sum())
              for cls_id in range(NUM_CLASSES)}

    print(f"\n  Real class counts:")
    for cls_id, name in ACTIVITY_NAMES.items():
        pct  = 100 * counts[cls_id] / total
        flag = " ⚠ MINORITY" if pct < 10 else ""
        print(f"    {name:<12} {counts[cls_id]:>5} ({pct:.1f}%){flag}")

    gen_counts = {}

    if strategy == "upsample":
        target = max(counts.values())
        print(f"\n  Strategy: upsample to {target} (majority class count)")
        for cls_id in range(NUM_CLASSES):
            gen_counts[cls_id] = max(0, target - counts[cls_id])

    elif strategy in ("ratio", "mixup", "combined"):
        print(f"\n  Strategy: add {syn_ratio:.0%} synthetic for minority classes")
        for cls_id in range(NUM_CLASSES):
            pct = counts[cls_id] / total
            gen_counts[cls_id] = (int(counts[cls_id] * syn_ratio)
                                  if pct < 0.10 else 0)

    print(f"\n  Synthetic windows to generate:")
    total_gen = 0
    for cls_id, name in ACTIVITY_NAMES.items():
        n = gen_counts.get(cls_id, 0)
        total_gen += n
        if n > 0:
            print(f"    {name:<12} +{n}")
    print(f"    {'TOTAL':<12} +{total_gen}")

    return gen_counts


# ══════════════════════════════════════════════════════════════════════════════
# GENERATE CAR-IMU TOKENS (batched autoregressive)
# ══════════════════════════════════════════════════════════════════════════════
def generate_for_class(
    model:       CARIMUDecoder,
    class_id:    int,
    n_samples:   int,
    temperature: float,
    device:      torch.device,
    batch_size:  int = 64,
) -> np.ndarray:
    """
    Generate n_samples synthetic token sequences for one activity class.
    Returns: (n_samples, SEQ_LEN, TOKEN_DIM) numpy array
    """
    model.eval()
    all_tokens = []
    generated  = 0

    with torch.no_grad():
        while generated < n_samples:
            B = min(batch_size, n_samples - generated)

            labels  = torch.tensor([class_id] * B, dtype=torch.long, device=device)
            context = model.class_emb(labels).unsqueeze(1)   # (B, 1, D)

            step_tokens = []

            for step in range(SEQ_LEN):
                ctx_pe = model.pos_enc(context, labels, seq_len=context.size(1))

                h = ctx_pe
                for layer in model.layers:
                    h = layer(h)

                h   = model.ln_final(h)
                nxt = model.out_head(h[:, -1, :])   # (B, D)

                if temperature > 0.0:
                    nxt = nxt + torch.randn_like(nxt) * temperature

                step_tokens.append(nxt)

                nxt_proj = model.input_proj(nxt).unsqueeze(1)
                context  = torch.cat([context, nxt_proj], dim=1)

            batch_out = torch.stack(step_tokens, dim=1).cpu().numpy()
            all_tokens.append(batch_out)
            generated += B
            print(f"    Generated {generated}/{n_samples} [{ACTIVITY_NAMES[class_id]}]")

    return np.concatenate(all_tokens, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# CHANGE 4 — SPECTRAL QUALITY FILTER
# ══════════════════════════════════════════════════════════════════════════════
def spectral_filter(
    syn_tokens:  np.ndarray,   # (N_syn, 192, 64)
    syn_labels:  np.ndarray,
    real_tokens: np.ndarray,   # (N_real, 192, 64)
    real_labels: np.ndarray,
    threshold:   float = 0.7,
) -> tuple:
    """
    Keep only synthetic sequences whose spectral profile cosine-similarity
    to the real class mean spectrum is >= threshold.

    Returns filtered (syn_tokens, syn_labels).
    """
    print(f"\n  Spectral filter (threshold={threshold:.2f}):")
    keep_idx = []

    for cls_id in range(NUM_CLASSES):
        real_mask = real_labels == cls_id
        syn_mask  = syn_labels  == cls_id
        if real_mask.sum() == 0 or syn_mask.sum() == 0:
            continue

        # Mean power spectrum of real sequences: average over N_real and 64 dims
        # rfft along time axis (axis=1) → shape (N, 97, 64), then mean → (97,)
        real_fft = np.abs(np.fft.rfft(real_tokens[real_mask], axis=1))
        real_ref  = real_fft.mean(axis=(0, 2))   # (97,)

        syn_indices = np.where(syn_mask)[0]
        kept = 0
        for idx in syn_indices:
            seq     = syn_tokens[idx]                           # (192, 64)
            syn_fft = np.abs(np.fft.rfft(seq, axis=0)).mean(axis=1)  # (97,)
            denom   = (np.linalg.norm(syn_fft) * np.linalg.norm(real_ref) + 1e-8)
            cos_sim = np.dot(syn_fft, real_ref) / denom
            if cos_sim >= threshold:
                keep_idx.append(idx)
                kept += 1

        total = syn_mask.sum()
        pct   = 100 * kept / total if total > 0 else 0
        print(f"    {ACTIVITY_NAMES[cls_id]:<12} kept {kept}/{total} "
              f"({pct:.0f}% pass rate)")

    if not keep_idx:
        print("  ⚠ Spectral filter removed ALL samples — returning unfiltered.")
        return syn_tokens, syn_labels

    keep_idx = np.array(keep_idx)
    return syn_tokens[keep_idx], syn_labels[keep_idx]


# ══════════════════════════════════════════════════════════════════════════════
# QUALITY CHECK — cosine sim + autocorrelation + variance ratio
# ══════════════════════════════════════════════════════════════════════════════
def quality_check(
    real_tokens: np.ndarray,
    real_labels: np.ndarray,
    syn_tokens:  np.ndarray,
    syn_labels:  np.ndarray,
):
    """
    Per-class quality metrics:
      - Cosine similarity between real and synthetic centroids
      - L2 distance between centroids
      - Lag-1 temporal autocorrelation ratio (syn/real)
      - Per-token variance ratio (syn/real)
    """
    print("\n  Quality Check — Real vs Synthetic Token Distributions:")
    print(f"  {'Class':<12} {'Cos Sim':>9} {'L2':>8} {'AutoCorr':>10} {'VarRatio':>10} {'Status'}")
    print("  " + "-" * 68)

    real_mean = real_tokens.mean(axis=1)
    syn_mean  = syn_tokens.mean(axis=1)

    for cls_id in range(NUM_CLASSES):
        real_mask = real_labels == cls_id
        syn_mask  = syn_labels  == cls_id
        if real_mask.sum() == 0 or syn_mask.sum() == 0:
            continue

        real_c = real_mean[real_mask]
        syn_c  = syn_mean[syn_mask]

        real_cent = real_c.mean(axis=0)
        syn_cent  = syn_c.mean(axis=0)

        cos_sim = (np.dot(real_cent, syn_cent) /
                   (np.linalg.norm(real_cent) * np.linalg.norm(syn_cent) + 1e-8))
        l2_dist = np.linalg.norm(real_cent - syn_cent)

        # Lag-1 autocorrelation: mean over samples and token dims
        # sequence shape: (N, 192, 64) — correlate along time axis
        def lag1_autocorr(seqs):
            # seqs: (N, L, D)
            mu   = seqs.mean(axis=1, keepdims=True)        # (N,1,D)
            s    = seqs - mu
            num  = (s[:, :-1, :] * s[:, 1:, :]).mean()
            denom = (s ** 2).mean() + 1e-8
            return float(num / denom)

        r_ac = lag1_autocorr(real_tokens[real_mask])
        s_ac = lag1_autocorr(syn_tokens[syn_mask])
        ac_ratio = s_ac / (r_ac + 1e-8)

        # Variance ratio
        real_var = real_tokens[real_mask].var(axis=1).mean()
        syn_var  = syn_tokens[syn_mask].var(axis=1).mean()
        var_ratio = float(syn_var / (real_var + 1e-8))

        status = "✅ Good" if cos_sim > 0.85 else ("⚠️  Mod" if cos_sim > 0.6 else "❌ Poor")
        print(f"  {ACTIVITY_NAMES[cls_id]:<12} {cos_sim:>9.4f} {l2_dist:>8.4f} "
              f"{ac_ratio:>10.4f} {var_ratio:>10.4f}  {status}")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE SYNTHETIC TOKEN STORE
# ══════════════════════════════════════════════════════════════════════════════
def save_synthetic(
    out_dir:    str,
    syn_tokens: np.ndarray,
    syn_labels: np.ndarray,
) -> str:
    """Save synthetic tokens in the same HDF5 format as token_store.hdf5."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "synthetic_tokens.hdf5")

    with h5py.File(path, "w") as f:
        f.create_dataset("tokens",      data=syn_tokens.astype(np.float32),
                         compression="gzip")
        f.create_dataset("mean_tokens", data=syn_tokens.mean(axis=1).astype(np.float32),
                         compression="gzip")
        f.create_dataset("labels",      data=syn_labels.astype(np.float32))
        f.create_dataset("subject_ids", data=np.full(len(syn_labels), -1,
                                                      dtype=np.float32))

    print(f"\n  Saved: {path}")
    print(f"    tokens:  {syn_tokens.shape}")
    print(f"    labels:  {syn_labels.shape}")
    for cls_id, name in ACTIVITY_NAMES.items():
        n = (syn_labels == cls_id).sum()
        if n > 0:
            print(f"      {name:<12} {n}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic tokens with CAR-IMU")
    p.add_argument("--checkpoint",  type=str,
                   default="car_imu_checkpoints/best_model.pt")
    p.add_argument("--token_store", type=str,
                   default="results_week1/token_store.hdf5")
    p.add_argument("--out_dir",     type=str, default="synthetic_tokens")
    p.add_argument("--temperature", type=float, default=0.5,
                   help="Sampling noise: 0=greedy, 0.5=recommended, 1.0=max")
    p.add_argument("--strategy",    type=str, default="upsample",
                   choices=["upsample", "ratio", "mixup", "combined"],
                   help="Generation strategy")
    p.add_argument("--syn_ratio",   type=float, default=0.5,
                   help="Ratio for upsample/ratio/mixup strategies")
    p.add_argument("--car_ratio",   type=float, default=0.25,
                   help="CAR-IMU ratio for combined strategy")
    p.add_argument("--mixup_ratio", type=float, default=0.25,
                   help="Mixup ratio for combined strategy")
    p.add_argument("--device",      type=str, default="auto",
                   choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--spectral_filter",    action="store_true", default=False,
                   help="Enable post-generation spectral quality filter")
    p.add_argument("--spectral_threshold", type=float, default=0.7,
                   help="Cosine-sim threshold for spectral filter")
    return p.parse_args()


def main():
    args   = parse_args()
    device = get_best_device(args.device)

    print("=" * 60)
    print("CAR-IMU — Synthetic Token Generation")
    print("=" * 60)
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Token store: {args.token_store}")
    print(f"  Strategy:    {args.strategy}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Device:      {device}")

    # ── Load real tokens ──────────────────────────────────────────────────────
    print(f"\nLoading real tokens from {args.token_store} ...")
    with h5py.File(args.token_store, "r") as f:
        real_tokens = f["tokens"][:]
        real_labels = f["labels"][:].astype(int)

    all_syn_tokens = []
    all_syn_labels = []

    # ── Strategy: mixup only ──────────────────────────────────────────────────
    if args.strategy == "mixup":
        print("\nRunning phase-amplitude mixup (no CAR-IMU required) ...")
        gen_counts = compute_generation_counts(real_labels, "ratio", args.syn_ratio)
        mx_tok, mx_lab = generate_mixup_tokens(real_tokens, real_labels, gen_counts)
        all_syn_tokens.append(mx_tok)
        all_syn_labels.append(mx_lab)

    # ── Strategy: combined (CAR-IMU + mixup) ─────────────────────────────────
    elif args.strategy == "combined":
        # Branch 1: CAR-IMU
        model = load_model(args.checkpoint, device)
        print(f"\n[Combined] CAR-IMU branch (ratio={args.car_ratio}) ...")
        car_counts = compute_generation_counts(real_labels, "ratio", args.car_ratio)
        for cls_id in range(NUM_CLASSES):
            n = car_counts.get(cls_id, 0)
            if n == 0:
                continue
            print(f"\n  [{ACTIVITY_NAMES[cls_id]}] generating {n} CAR-IMU sequences ...")
            syn = generate_for_class(model, cls_id, n, args.temperature,
                                     device, args.batch_size)
            all_syn_tokens.append(syn)
            all_syn_labels.append(np.full(n, cls_id, dtype=int))

        # Branch 2: Mixup — also save separately
        print(f"\n[Combined] Mixup branch (ratio={args.mixup_ratio}) ...")
        mix_counts = compute_generation_counts(real_labels, "ratio", args.mixup_ratio)
        mx_tok, mx_lab = generate_mixup_tokens(real_tokens, real_labels, mix_counts)
        if len(mx_tok) > 0:
            all_syn_tokens.append(mx_tok)
            all_syn_labels.append(mx_lab)
            # Save mixup-only separately for evaluate_augmented.py --mixup_tokens
            mixup_dir = os.path.join(args.out_dir, "mixup_only")
            print(f"\n  Saving mixup-only tokens to {mixup_dir} ...")
            save_synthetic(mixup_dir, mx_tok, mx_lab)

    # ── Strategy: upsample / ratio (original CAR-IMU only) ───────────────────
    else:
        model      = load_model(args.checkpoint, device)
        gen_counts = compute_generation_counts(real_labels, args.strategy, args.syn_ratio)
        print(f"\nGenerating synthetic tokens (temperature={args.temperature}) ...")
        for cls_id in range(NUM_CLASSES):
            n = gen_counts.get(cls_id, 0)
            if n == 0:
                continue
            print(f"\n  [{ACTIVITY_NAMES[cls_id]}] generating {n} sequences ...")
            syn = generate_for_class(model, cls_id, n, args.temperature,
                                     device, args.batch_size)
            all_syn_tokens.append(syn)
            all_syn_labels.append(np.full(n, cls_id, dtype=int))

    if not all_syn_tokens or all(len(t) == 0 for t in all_syn_tokens):
        print("Nothing to generate — all classes already balanced.")
        return

    syn_tokens = np.concatenate(all_syn_tokens, axis=0)
    syn_labels = np.concatenate(all_syn_labels, axis=0)

    # ── Quality check ─────────────────────────────────────────────────────────
    quality_check(real_tokens, real_labels, syn_tokens, syn_labels)

    # ── Spectral filter (optional) ────────────────────────────────────────────
    if args.spectral_filter:
        syn_tokens, syn_labels = spectral_filter(
            syn_tokens, syn_labels, real_tokens, real_labels,
            threshold=args.spectral_threshold)

    # ── Save combined output ──────────────────────────────────────────────────
    syn_path = save_synthetic(args.out_dir, syn_tokens, syn_labels)

    print("\n" + "=" * 60)
    print("✅  Synthetic token generation complete!")
    print("=" * 60)
    print(f"  Real tokens:      {len(real_tokens):,} windows")
    print(f"  Synthetic tokens: {len(syn_tokens):,} windows")
    print(f"  Combined total:   {len(real_tokens) + len(syn_tokens):,} windows")
    print()
    print("Next step: run the augmented HAR classifier evaluation:")
    print()
    print("  python evaluate_augmented.py \\")
    print(f"      --real_tokens   {args.token_store} \\")
    print(f"      --syn_tokens    {syn_path} \\")
    if args.strategy == "combined":
        mixup_path = os.path.join(args.out_dir, "mixup_only", "synthetic_tokens.hdf5")
        print(f"      --mixup_tokens  {mixup_path} \\")
    print( "      --out_dir       results_week2")


if __name__ == "__main__":
    main()
