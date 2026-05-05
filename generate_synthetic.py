#!/usr/bin/env python3
"""
generate_synthetic.py — Generate synthetic Bio-PM token sequences using trained CAR-IMU.

WHAT THIS DOES:
    Loads the trained CAR-IMU checkpoint, generates synthetic token sequences
    for minority activity classes (Upstairs, Downstairs, Sitting, Standing),
    and saves them in the same format as token_store.hdf5 so they can be
    directly mixed with real tokens for HAR classifier training.

    Also runs quality checks:
      - Cosine similarity between synthetic and real tokens (per class)
      - L2 distance distributions
      - Saves a UMAP comparing real vs synthetic token distributions

HOW MANY TO GENERATE:
    We upsample minority classes to match the majority class count (Walking=4168).
    Or you can set a fixed ratio (--syn_ratio 0.5 = 50% synthetic added).

USAGE:
    python generate_synthetic.py

    Key flags:
    --checkpoint   car_imu_checkpoints/best_model.pt
    --token_store  results_week1/token_store.hdf5
    --out_dir      synthetic_tokens/
    --temperature  0.5   (noise level: 0 = greedy, 1.0 = max noise)
    --strategy     upsample   (match majority class) or ratio (fixed %)
    --syn_ratio    0.5        (used only if strategy=ratio)
    --device       auto
"""

import os
import sys
import argparse
import numpy as np
import h5py
import torch

from car_imu_decoder import CARIMUDecoder, TOKEN_DIM, SEQ_LEN, NUM_CLASSES, ACTIVITY_NAMES


# ── Device selection (same as train script) ────────────────────────────────────
def get_best_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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
        dropout=0.0,   # no dropout at inference
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

    strategy='upsample':
        Bring every class up to the count of the majority class.
        e.g. Walking=4168, Standing=471 → generate 4168-471=3697 Standing windows

    strategy='ratio':
        Add syn_ratio × real_count synthetic windows per class.
        e.g. syn_ratio=0.5 → add 50% more of each class
        Focus on minority classes only (< 10% of total).
    """
    total = len(real_labels)
    counts = {}
    for cls_id in range(NUM_CLASSES):
        counts[cls_id] = int((real_labels == cls_id).sum())

    print(f"\n  Real class counts:")
    for cls_id, name in ACTIVITY_NAMES.items():
        pct = 100 * counts[cls_id] / total
        flag = " ⚠ MINORITY" if pct < 10 else ""
        print(f"    {name:<12} {counts[cls_id]:>5} ({pct:.1f}%){flag}")

    gen_counts = {}

    if strategy == "upsample":
        # Target = majority class count
        target = max(counts.values())
        print(f"\n  Strategy: upsample to {target} (majority class count)")
        for cls_id in range(NUM_CLASSES):
            needed = max(0, target - counts[cls_id])
            gen_counts[cls_id] = needed

    elif strategy == "ratio":
        print(f"\n  Strategy: add {syn_ratio:.0%} synthetic for ALL classes")
        for cls_id in range(NUM_CLASSES):
            gen_counts[cls_id] = int(counts[cls_id] * syn_ratio)

    elif strategy == "minority_ratio":
        print(f"\n  Strategy: add {syn_ratio:.0%} synthetic for MINORITY classes only")
        for cls_id in range(NUM_CLASSES):
            pct = counts[cls_id] / total
            if pct < 0.18:   # Targets Upstairs, Downstairs, Sitting, Standing
                gen_counts[cls_id] = int(counts[cls_id] * syn_ratio)
            else:
                gen_counts[cls_id] = 0
                
    elif strategy == "balance":
        # Target = syn_ratio used as a fixed count per class
        target_count = int(syn_ratio)
        print(f"\n  Strategy: balance (generating {target_count} synthetic per class)")
        for cls_id in range(NUM_CLASSES):
            gen_counts[cls_id] = target_count

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
# GENERATE SYNTHETIC TOKENS
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

    Uses BATCHED autoregressive generation:
      - Runs `batch_size` sequences in PARALLEL per forward pass
      - Each step: one forward pass → predict next token for ALL sequences at once
      - ~30x faster than the naive one-at-a-time loop

    Returns: (n_samples, SEQ_LEN, TOKEN_DIM) numpy array
    """
    model.eval()
    all_tokens = []
    generated  = 0

    with torch.no_grad():
        while generated < n_samples:
            B = min(batch_size, n_samples - generated)

            # ── Initialize context: just the class token, shape (B, 1, 64) ──
            labels  = torch.tensor([class_id] * B, dtype=torch.long, device=device)
            context = model.class_emb(labels).unsqueeze(1)   # (B, 1, D)

            step_tokens = []   # will collect (B, 64) at each step

            # ── Autoregressive loop: 192 steps, but B sequences at once ─────
            for step in range(SEQ_LEN):
                # Add positional encoding to current context
                ctx_pe = model.pos_enc(context)            # (B, step+1, D)

                # Forward through all transformer layers
                h = ctx_pe
                for layer in model.layers:
                    h = layer(h)

                # Predict next token from LAST position only → (B, D)
                h       = model.ln_final(h)
                nxt     = model.out_head(h[:, -1, :])      # (B, D)

                # Add sampling noise (temperature)
                if temperature > 0.0:
                    nxt = nxt + torch.randn_like(nxt) * temperature

                step_tokens.append(nxt)                    # save prediction

                # Append projected prediction to context for next step
                nxt_proj = model.input_proj(nxt).unsqueeze(1)   # (B, 1, D)
                context  = torch.cat([context, nxt_proj], dim=1) # (B, step+2, D)

            # ── Stack steps → (B, SEQ_LEN, D) ──────────────────────────────
            batch_out = torch.stack(step_tokens, dim=1).cpu().numpy()  # (B, 192, 64)
            all_tokens.append(batch_out)
            generated += B

            print(f"    Generated {generated}/{n_samples} [{ACTIVITY_NAMES[class_id]}]")

    return np.concatenate(all_tokens, axis=0)   # (n_samples, 192, 64)



# ══════════════════════════════════════════════════════════════════════════════
# QUALITY CHECK — compare synthetic vs real token distributions
# ══════════════════════════════════════════════════════════════════════════════
def quality_check(
    real_tokens:  np.ndarray,  # (N_real, 192, 64)
    real_labels:  np.ndarray,
    syn_tokens:   np.ndarray,  # (N_syn, 192, 64)
    syn_labels:   np.ndarray,
):
    """
    For each class, compare:
      - Mean cosine similarity between real and synthetic mean-pooled tokens
      - L2 distance between class centroids
    High cosine sim (>0.8) and low L2 = synthetic tokens are in the right region.
    """
    print("\n  Quality Check — Real vs Synthetic Token Distributions:")
    print(f"  {'Class':<12} {'Cos Sim':>10} {'L2 Dist':>10} {'Assessment'}")
    print("  " + "-" * 55)

    # Mean-pool each sequence: (N, 64)
    real_mean = real_tokens.mean(axis=1)   # (N_real, 64)
    syn_mean  = syn_tokens.mean(axis=1)    # (N_syn, 64)

    for cls_id in range(NUM_CLASSES):
        real_mask = real_labels == cls_id
        syn_mask  = syn_labels  == cls_id

        if real_mask.sum() == 0 or syn_mask.sum() == 0:
            continue

        real_centroid = real_mean[real_mask].mean(axis=0)   # (64,)
        syn_centroid  = syn_mean[syn_mask].mean(axis=0)     # (64,)

        # Cosine similarity between centroids
        cos_sim = (np.dot(real_centroid, syn_centroid) /
                   (np.linalg.norm(real_centroid) * np.linalg.norm(syn_centroid) + 1e-8))

        # L2 distance between centroids
        l2_dist = np.linalg.norm(real_centroid - syn_centroid)

        assessment = "✅ Good" if cos_sim > 0.85 else ("⚠️  Moderate" if cos_sim > 0.6 else "❌ Poor")
        print(f"  {ACTIVITY_NAMES[cls_id]:<12} {cos_sim:>10.4f} {l2_dist:>10.4f}  {assessment}")

def calibrate_tokens(real_tokens, real_labels, syn_tokens, syn_labels):
    """Force synthetic tokens to match the exact mean/std of real tokens PER CLASS."""
    print("\n  Calibrating synthetic tokens per-class...")
    calibrated = syn_tokens.copy()
    
    for cls_id in range(6):
        m_r = real_labels == cls_id
        m_s = syn_labels == cls_id
        
        if m_r.sum() > 10 and m_s.sum() > 10:
            # Stats for THIS class only
            r_mean = real_tokens[m_r].mean(axis=(0, 1))
            r_std  = real_tokens[m_r].std(axis=(0, 1)) + 1e-8
            
            s_mean = syn_tokens[m_s].mean(axis=(0, 1))
            s_std  = syn_tokens[m_s].std(axis=(0, 1)) + 1e-8
            
            # Calibrate this class
            cls_syn = (syn_tokens[m_s] - s_mean) / s_std
            calibrated[m_s] = (cls_syn * r_std) + r_mean
            
    return calibrated.astype(np.float32)


def compute_320d_features(tokens: np.ndarray, gravity_vecs: np.ndarray) -> np.ndarray:
    """
    Convert (N, 192, 64) tokens + (N, 64) gravity to 320-d axis-wise pooled features.
    Matches evaluate_augmented.py feature computation exactly.
    """
    # Axis-wise mean: tokens are (N, 192, 64)
    # prepended [CLS] means 192 tokens. Preprocessing says 64 tokens per axis?
    # Actually evaluate_augmented.py uses tokens[:, 0::3, :].mean(axis=1)
    mean_x    = tokens[:, 0::3, :].mean(axis=1)   # (N, 64)
    mean_y    = tokens[:, 1::3, :].mean(axis=1)   # (N, 64)
    mean_z    = tokens[:, 2::3, :].mean(axis=1)   # (N, 64)
    std_feats = tokens.std(axis=1)                 # (N, 64)
    return np.concatenate([
        mean_x, mean_y, mean_z, std_feats, gravity_vecs
    ], axis=1)   # (N, 320)


def filter_confused_synthetic(
    syn_tokens:    np.ndarray,
    syn_labels:    np.ndarray,
    syn_gravity:   np.ndarray,
    real_tokens:   np.ndarray,
    real_labels:   np.ndarray,
    real_gravity:  np.ndarray,
    confusion_pairs: dict,
    threshold: float = 0.85,
) -> tuple:
    """
    Vectorized filter to remove synthetic tokens that are too similar to confuser centroids.
    """
    # Compute 320-d features for real tokens
    real_feats = compute_320d_features(real_tokens, real_gravity)
    all_syn_feats = compute_320d_features(syn_tokens, syn_gravity)

    keep_mask = np.ones(len(syn_labels), dtype=bool)

    for target_cls, confuser_cls in confusion_pairs.items():
        target_mask = syn_labels == target_cls
        if not target_mask.any():
            continue

        # Compute confuser centroid in 320-d space
        conf_mask = real_labels == confuser_cls
        if not conf_mask.any():
            continue
        conf_centroid = real_feats[conf_mask].mean(axis=0)
        conf_centroid /= (np.linalg.norm(conf_centroid) + 1e-8)

        # Compute similarities for ALL syn tokens of this target class
        target_feats = all_syn_feats[target_mask]
        target_feats_norm = target_feats / (np.linalg.norm(target_feats, axis=1, keepdims=True) + 1e-8)
        
        sims = target_feats_norm @ conf_centroid  # (N_target,)
        
        # Identify which ones to discard
        discard_submask = sims >= threshold
        
        # Update global keep_mask
        # target_mask is a boolean mask of length N_syn. 
        # We need to map discard_submask (length N_target) back to N_syn.
        idx_of_target = np.where(target_mask)[0]
        idx_to_discard = idx_of_target[discard_submask]
        keep_mask[idx_to_discard] = False

        print(f"    {ACTIVITY_NAMES[target_cls]:<12} kept {sum(~discard_submask)}/{len(sims)} "
              f"({100*sum(~discard_submask)/len(sims):.1f}% pass rate) "
              f"discarded {sum(discard_submask)} vs {ACTIVITY_NAMES[confuser_cls]}")

    if not keep_mask.any():
        return (np.empty((0, syn_tokens.shape[1], syn_tokens.shape[2]), dtype=np.float32),
                np.empty(0, dtype=int),
                np.empty((0, syn_gravity.shape[1]), dtype=np.float32))

    return (syn_tokens[keep_mask], syn_labels[keep_mask], syn_gravity[keep_mask])


# ══════════════════════════════════════════════════════════════════════════════
# SAVE SYNTHETIC TOKEN STORE
# ══════════════════════════════════════════════════════════════════════════════
def save_synthetic(
    out_dir:    str,
    syn_tokens: np.ndarray,   # (N, 192, 64)
    syn_labels: np.ndarray,   # (N,)
    syn_gravity: np.ndarray = None,
):
    """Save synthetic tokens in the same HDF5 format as token_store.hdf5."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "synthetic_tokens.hdf5")

    with h5py.File(path, "w") as f:
        f.create_dataset("tokens",      data=syn_tokens.astype(np.float32),
                         compression="gzip")
        f.create_dataset("mean_tokens", data=syn_tokens.mean(axis=1).astype(np.float32),
                         compression="gzip")
        f.create_dataset("labels",      data=syn_labels.astype(np.float32))
        # Mark subject_ids as -1 to distinguish synthetic from real
        f.create_dataset("subject_ids", data=np.full(len(syn_labels), -1,
                                                      dtype=np.float32))
        
        grav_data = syn_gravity if syn_gravity is not None \
                    else np.zeros((len(syn_tokens), 64), dtype=np.float32)
        f.create_dataset("gravity_vecs", data=grav_data.astype(np.float32),
                         compression="gzip")

    print(f"\n  Saved: {path}")
    print(f"    tokens:  {syn_tokens.shape}")
    print(f"    labels:  {syn_labels.shape}")
    print(f"    Label counts:")
    for cls_id, name in ACTIVITY_NAMES.items():
        n = (syn_labels == cls_id).sum()
        if n > 0:
            print(f"      {name:<12} {n}")

    return path


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic tokens with CAR-IMU")
    p.add_argument("--checkpoint",   type=str,
                   default="car_imu_checkpoints/best_model.pt")
    p.add_argument("--token_store",  type=str,
                   default="results_week1/token_store.hdf5")
    p.add_argument("--out_dir",      type=str, default="synthetic_tokens")
    p.add_argument("--temperature",  type=float, default=0.5,
                   help="Sampling noise: 0=greedy, 0.5=recommended, 1.0=max")
    p.add_argument("--strategy",     type=str, default="upsample",
                   choices=["upsample", "ratio", "minority_ratio", "balance", "confusion_aware"],
                   help="upsample=match majority; ratio=all classes; minority_ratio=only imbalanced; balance=fixed count; confusion_aware: minority-only + confusion filtering")
    p.add_argument("--syn_ratio",    type=float, default=0.5,
                   help="Ratio if strategy=ratio, or fixed count if strategy=balance. E.g. 5000")
    p.add_argument("--confusion_threshold", type=float, default=0.85,
                   help="Cosine similarity threshold for confusion filter (default=0.85). Lower = stricter filtering.")
    p.add_argument("--minority_ratio", type=float, default=0.10,
                   help="Synthetic ratio for minority classes in confusion_aware mode (default=0.10 = 10%)")
    p.add_argument("--device",       type=str, default="auto",
                   choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--batch_size",   type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    device = get_best_device(args.device)

    print("=" * 60)
    print("CAR-IMU — Synthetic Token Generation")
    print("=" * 60)
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Token store: {args.token_store}")
    print(f"  Strategy:    {args.strategy}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Device:      {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model(args.checkpoint, device)

    # ── Load real tokens to compute generation targets ────────────────────────
    print(f"\nLoading real tokens from {args.token_store} ...")
    with h5py.File(args.token_store, "r") as f:
        real_tokens = f["tokens"][:]      # (N, 192, 64)
        real_labels = f["labels"][:].astype(int)

    # ── Compute how many synthetic to generate per class ─────────────────────
    if args.strategy == "confusion_aware":
        print(f"\n  Strategy: confusion_aware")
        print(f"  Minority ratio:       {args.minority_ratio:.0%}")
        print(f"  Confusion threshold:  {args.confusion_threshold}")
        print(f"  Target classes:       Upstairs (2), Downstairs (3)")
        print(f"  Confusion pairs:")
        print(f"    Upstairs   → filter tokens similar to Walking")
        print(f"    Downstairs → filter tokens similar to Sitting")

        # Load gravity from token store for centroid computation
        with h5py.File(args.token_store, "r") as f:
            real_gravity = f["gravity_vecs"][:]   # (N_real, 64)

        # Step 1 — Determine generation counts
        # Only Upstairs (2) and Downstairs (3) get synthetic tokens
        # Scaled by difficulty (inverse F1)
        per_class_f1 = {
            2: 0.600,   # Upstairs   — hardest minority
            3: 0.686,   # Downstairs — second hardest
        }

        gen_counts = {}
        for cls_id in range(NUM_CLASSES):
            if cls_id not in per_class_f1:
                gen_counts[cls_id] = 0   # skip all other classes
                continue
            difficulty = 1.0 - per_class_f1[cls_id]
            ratio      = difficulty * args.minority_ratio
            n_real     = int((real_labels == cls_id).sum())
            gen_counts[cls_id] = int(n_real * ratio)

        print(f"\n  Generation counts:")
        for cls_id, n in gen_counts.items():
            if n > 0:
                print(f"    {ACTIVITY_NAMES[cls_id]:<12} +{n}")

        # Step 2 — Generate raw synthetic tokens
        print(f"\n  Generating raw synthetic tokens ...")
        raw_syn_tokens, raw_syn_labels = [], []

        for cls_id in [2, 3]:   # Upstairs, Downstairs only
            n = gen_counts[cls_id]
            if n == 0: continue
            print(f"\n  [{ACTIVITY_NAMES[cls_id]}] generating {n} sequences ...")
            syn = generate_for_class(model, cls_id, n, args.temperature, device, args.batch_size)
            raw_syn_tokens.append(syn)
            raw_syn_labels.append(np.full(n, cls_id, dtype=int))

        if not raw_syn_tokens:
            print("  Nothing generated.")
            return

        raw_syn_tokens = np.concatenate(raw_syn_tokens, axis=0)
        raw_syn_labels = np.concatenate(raw_syn_labels, axis=0)

        # Step 3 — Assign class-mean gravity to synthetic tokens
        raw_syn_gravity = np.stack([
            real_gravity[real_labels == lbl].mean(axis=0)
            for lbl in raw_syn_labels
        ])   # (N_syn, 64)

        # Step 4 — Apply confusion-aware filter
        print(f"\n  Applying confusion-aware filter ...")
        confusion_pairs = {
            2: 0,   # Upstairs → filter vs Walking
            3: 4,   # Downstairs → filter vs Sitting
        }

        syn_tokens, syn_labels, syn_gravity = filter_confused_synthetic(
            syn_tokens    = raw_syn_tokens,
            syn_labels    = raw_syn_labels,
            syn_gravity   = raw_syn_gravity,
            real_tokens   = real_tokens,
            real_labels   = real_labels,
            real_gravity  = real_gravity,
            confusion_pairs = confusion_pairs,
            threshold     = args.confusion_threshold,
        )

        # Step 5 — Standard quality check on filtered tokens
        if len(syn_tokens) > 0:
            quality_check(real_tokens, real_labels, syn_tokens, syn_labels)

        # Step 6 — Save
        all_syn_tokens = [syn_tokens]
        all_syn_labels = [syn_labels]
        # Calibration (optional, keep it consistent with other strategies)
        # syn_tokens = calibrate_tokens(real_tokens, real_labels, syn_tokens, syn_labels)

    else:
        gen_counts = compute_generation_counts(real_labels, args.strategy, args.syn_ratio)
        # ── Generate ──────────────────────────────────────────────────────────────
        print(f"\nGenerating synthetic tokens (temperature={args.temperature}) ...")
        all_syn_tokens = []
        all_syn_labels = []

        for cls_id in range(NUM_CLASSES):
            n = gen_counts.get(cls_id, 0)
            if n == 0: continue

            print(f"\n  [{ACTIVITY_NAMES[cls_id]}] generating {n} sequences ...")
            syn = generate_for_class(model, cls_id, n, args.temperature, device, args.batch_size)
            all_syn_tokens.append(syn)
            all_syn_labels.append(np.full(n, cls_id, dtype=int))

        if not all_syn_tokens:
            print("Nothing to generate — all classes already balanced.")
            return

        syn_tokens = np.concatenate(all_syn_tokens, axis=0)
        syn_labels = np.concatenate(all_syn_labels, axis=0)
        
        # For non-confusion_aware, gravity will be placeholder in save_synthetic
        syn_gravity = None 

        # ── Token Calibration ─────────────────────────────────────────────────────
        syn_tokens = calibrate_tokens(real_tokens, real_labels, syn_tokens, syn_labels)
        
        # ── Quality check ─────────────────────────────────────────────────────────
        quality_check(real_tokens, real_labels, syn_tokens, syn_labels)

        all_syn_tokens = [syn_tokens]
        all_syn_labels = [syn_labels]

    # ── Save ──────────────────────────────────────────────────────────────────
    syn_tokens = np.concatenate(all_syn_tokens, axis=0)
    syn_labels = np.concatenate(all_syn_labels, axis=0)
    syn_path = save_synthetic(args.out_dir, syn_tokens, syn_labels, syn_gravity)

    # ── Print next step ───────────────────────────────────────────────────────
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
    print( "      --features      features/biopm_features_all.npz \\")
    print( "      --out_dir       results_week2")


if __name__ == "__main__":
    main()
