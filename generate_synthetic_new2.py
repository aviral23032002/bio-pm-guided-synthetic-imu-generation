#!/usr/bin/env python3
"""
generate_synthetic.py — Generate synthetic Bio-PM token sequences using trained CAR-IMU.

STRATEGIES:
    upsample:        bring all classes to majority class count
    ratio:           add fixed % synthetic for all classes
    minority_ratio:  add fixed % synthetic for minority classes only
    balance:         generate fixed count per class
    confusion_aware: minority-only + confusion filtering (previous strategy)
    boundary_aware:  ALL confused classes + bidirectional filtering
                     scaled by confusion severity from confusion matrix
                     Recommended for balanced datasets with representation overlap.

BOUNDARY_AWARE STRATEGY:
    Instead of targeting classes by F1 difficulty or minority status,
    targets classes by confusion severity — how much they spill into
    other classes in the confusion matrix. Generates synthetic tokens
    for BOTH sides of each confused pair, filtered to remove tokens
    that look like the confuser class. Reinforces decision boundaries
    from both directions simultaneously.

    Confusion pairs (from confusion matrix, smartwatch dataset):
      Walking    <-> Upstairs    (0.27 / 0.22 spillover)
      Downstairs <-> Sitting     (0.23 / 0.16 spillover)
      Downstairs <-> Standing    (0.13 / 0.17 spillover)
      Jogging    ->  Walking     (0.03 spillover — below threshold, skipped)

    Key finding: no_calibrate=True gives better results (default).
    Uncalibrated tokens preserve natural boundary diversity that
    calibration collapses (v5 empirical finding, +0.013 vs +0.004).

PIPELINE (boundary_aware):
    1. Compute generation counts from confusion_severity x base_ratio
    2. Generate raw tokens autoregressively
    3. Assign class-mean gravity to synthetic tokens
    4. [Optional] Calibrate — disabled by default
    5. Confusion filter in 320-d space — bidirectional, multi-confuser
    6. Quality gate in 320-d space — per-class thresholds
    7. Quality check report
    8. Save with gravity_vecs

USAGE:
    python generate_synthetic.py \\
        --checkpoint  car_imu_checkpoints_6class/best_model.pt \\
        --token_store results_wisdm_v2_6class/token_store.hdf5 \\
        --out_dir     synthetic_tokens_boundary_aware \\
        --strategy    boundary_aware \\
        --base_ratio  0.15 \\
        --confusion_threshold 0.70 \\
        --min_quality 0.70 \\
        --temperature 0.3 \\
        --no_calibrate
"""

import os
import argparse
import numpy as np
import h5py
import torch

from car_imu_decoder import CARIMUDecoder, TOKEN_DIM, SEQ_LEN, NUM_CLASSES, ACTIVITY_NAMES


# ══════════════════════════════════════════════════════════════════════════════
# BOUNDARY-AWARE CONFIGURATION
# Derived from confusion matrix (condition A, 320-d real only, smartwatch)
# ══════════════════════════════════════════════════════════════════════════════

# Confusion severity per class — fraction misclassified into other classes
CONFUSION_SEVERITY = {
    0: 0.27,   # Walking    -> Upstairs (0.27)
    1: 0.03,   # Jogging    -> Walking  (0.03) — below threshold, skip
    2: 0.22,   # Upstairs   -> Walking  (0.22)
    3: 0.36,   # Downstairs -> Sitting (0.23) + Standing (0.13)
    4: 0.16,   # Sitting    -> Downstairs (0.16)
    5: 0.17,   # Standing   -> Downstairs (0.17)
}

# Minimum confusion severity to be included as generation target
SEVERITY_THRESHOLD = 0.10

# Bidirectional confusion pairs — each class filtered against its confusers
# Format: {target_cls: int | [int, ...]}
BOUNDARY_CONFUSION_PAIRS = {
    0: [2],      # Walking    -> filter vs Upstairs
    2: [0],      # Upstairs   -> filter vs Walking
    3: [4, 5],   # Downstairs -> filter vs Sitting AND Standing
    4: [3],      # Sitting    -> filter vs Downstairs
    5: [3],      # Standing   -> filter vs Downstairs
}

# Per-class quality thresholds (320-d cosine similarity after filtering)
# Downstairs: lenient threshold due to known lower generation quality
BOUNDARY_QUALITY_THRESHOLDS = {
    0: 0.70,   # Walking    — moderate
    2: 0.70,   # Upstairs   — moderate
    3: 0.55,   # Downstairs — lenient (cos_sim historically 0.52-0.69)
    4: 0.70,   # Sitting    — moderate
    5: 0.70,   # Standing   — moderate
}


# ══════════════════════════════════════════════════════════════════════════════
# DEVICE SELECTION
# ══════════════════════════════════════════════════════════════════════════════
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
    print(f"  Per-class val MSE:")
    for cls_id, name in ACTIVITY_NAMES.items():
        mse = ckpt['per_class_loss'].get(cls_id, float('nan'))
        print(f"    {name:<12} {mse:.4f}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# GENERATION COUNT COMPUTATION (non-boundary_aware strategies)
# ══════════════════════════════════════════════════════════════════════════════
def compute_generation_counts(
    real_labels: np.ndarray,
    strategy:    str,
    syn_ratio:   float,
) -> dict:
    total  = len(real_labels)
    counts = {cls_id: int((real_labels == cls_id).sum())
              for cls_id in range(NUM_CLASSES)}

    print(f"\n  Real class counts:")
    for cls_id, name in ACTIVITY_NAMES.items():
        pct  = 100 * counts[cls_id] / total
        flag = " ⚠ MINORITY" if pct < 18 else ""
        print(f"    {name:<12} {counts[cls_id]:>5} ({pct:.1f}%){flag}")

    gen_counts = {}

    if strategy == "upsample":
        target = max(counts.values())
        print(f"\n  Strategy: upsample to {target}")
        for cls_id in range(NUM_CLASSES):
            gen_counts[cls_id] = max(0, target - counts[cls_id])

    elif strategy == "ratio":
        print(f"\n  Strategy: add {syn_ratio:.0%} for ALL classes")
        for cls_id in range(NUM_CLASSES):
            gen_counts[cls_id] = int(counts[cls_id] * syn_ratio)

    elif strategy == "minority_ratio":
        print(f"\n  Strategy: add {syn_ratio:.0%} for MINORITY classes only")
        for cls_id in range(NUM_CLASSES):
            pct = counts[cls_id] / total
            gen_counts[cls_id] = int(counts[cls_id] * syn_ratio) if pct < 0.18 else 0

    elif strategy == "balance":
        target_count = int(syn_ratio)
        print(f"\n  Strategy: balance ({target_count} per class)")
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
# AUTOREGRESSIVE GENERATION
# ══════════════════════════════════════════════════════════════════════════════
def generate_for_class(
    model:       CARIMUDecoder,
    class_id:    int,
    n_samples:   int,
    temperature: float,
    device:      torch.device,
    batch_size:  int = 64,
) -> np.ndarray:
    """Batched autoregressive generation for one activity class."""
    model.eval()
    all_tokens = []
    generated  = 0

    with torch.no_grad():
        while generated < n_samples:
            B       = min(batch_size, n_samples - generated)
            labels  = torch.tensor([class_id] * B, dtype=torch.long, device=device)
            context = model.class_emb(labels).unsqueeze(1)
            step_tokens = []

            for step in range(SEQ_LEN):
                ctx_pe = model.pos_enc(context)
                h = ctx_pe
                for layer in model.layers:
                    h = layer(h)
                h   = model.ln_final(h)
                nxt = model.out_head(h[:, -1, :])
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
# CALIBRATION (optional — disabled by default for boundary_aware)
# ══════════════════════════════════════════════════════════════════════════════
def calibrate_tokens(
    real_tokens: np.ndarray,
    real_labels: np.ndarray,
    syn_tokens:  np.ndarray,
    syn_labels:  np.ndarray,
) -> np.ndarray:
    """
    Force synthetic tokens to match real class mean/std per class.

    NOTE: For boundary_aware strategy, --no_calibrate is recommended.
    Empirical finding (v2 vs v5): uncalibrated tokens (+0.013 MLP delta)
    outperform calibrated tokens (+0.004 MLP delta) because natural
    generation diversity helps place tokens near class boundaries.
    """
    print("\n  Calibrating synthetic tokens per-class...")
    calibrated = syn_tokens.copy()

    for cls_id in range(NUM_CLASSES):
        m_r = real_labels == cls_id
        m_s = syn_labels  == cls_id
        if m_r.sum() < 10 or m_s.sum() < 10:
            continue

        r_mean = real_tokens[m_r].mean(axis=(0, 1))
        r_std  = real_tokens[m_r].std(axis=(0, 1))  + 1e-8
        s_mean = syn_tokens[m_s].mean(axis=(0, 1))
        s_std  = syn_tokens[m_s].std(axis=(0, 1))   + 1e-8

        cls_syn         = (syn_tokens[m_s] - s_mean) / s_std
        calibrated[m_s] = (cls_syn * r_std) + r_mean
        print(f"    {ACTIVITY_NAMES[cls_id]:<12} calibrated "
              f"(Δmean={abs(r_mean-s_mean).mean():.4f} "
              f"Δstd={abs(r_std-s_std).mean():.4f})")

    return calibrated.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 320-D FEATURE COMPUTATION
# Must match evaluate_augmented.py exactly
# ══════════════════════════════════════════════════════════════════════════════
def compute_320d_features(
    tokens:       np.ndarray,   # (N, 192, 64)
    gravity_vecs: np.ndarray,   # (N, 64)
) -> np.ndarray:
    """
    Axis-wise mean pooling + global std + gravity = 320-d.
    Tokens interleaved by axis: 0::3=x, 1::3=y, 2::3=z (64 tokens each).
    Matches evaluate_augmented.py feature computation exactly.
    """
    mean_x    = tokens[:, 0::3, :].mean(axis=1)   # (N, 64)
    mean_y    = tokens[:, 1::3, :].mean(axis=1)   # (N, 64)
    mean_z    = tokens[:, 2::3, :].mean(axis=1)   # (N, 64)
    std_feats = tokens.std(axis=1)                 # (N, 64)
    return np.concatenate(
        [mean_x, mean_y, mean_z, std_feats, gravity_vecs], axis=1
    )   # (N, 320)


# ══════════════════════════════════════════════════════════════════════════════
# CONFUSION FILTER — vectorized, bidirectional, multi-confuser
# ══════════════════════════════════════════════════════════════════════════════
def filter_confused_synthetic(
    syn_tokens:      np.ndarray,
    syn_labels:      np.ndarray,
    syn_gravity:     np.ndarray,
    real_tokens:     np.ndarray,
    real_labels:     np.ndarray,
    real_gravity:    np.ndarray,
    confusion_pairs: dict,
    threshold:       float = 0.70,
) -> tuple:
    """
    Vectorized confusion filter in 320-d feature space.

    For each target class:
      Compute cosine similarity to each confuser centroid in 320-d space.
      Discard any token exceeding threshold for ANY confuser.
      Keep only tokens genuinely distinct from all confusers.

    Bidirectional: Walking filtered vs Upstairs AND Upstairs filtered vs Walking.
    Multi-confuser: Downstairs filtered vs both Sitting AND Standing.
    """
    real_feats    = compute_320d_features(real_tokens, real_gravity)
    all_syn_feats = compute_320d_features(syn_tokens,  syn_gravity)
    keep_mask     = np.ones(len(syn_labels), dtype=bool)

    print(f"\n  Confusion filter (threshold={threshold}, space=320-d):")

    for target_cls, confuser_spec in confusion_pairs.items():
        confuser_list = ([confuser_spec] if isinstance(confuser_spec, int)
                         else list(confuser_spec))

        target_mask = syn_labels == target_cls
        if not target_mask.any():
            continue

        target_feats      = all_syn_feats[target_mask]
        target_feats_norm = target_feats / (
            np.linalg.norm(target_feats, axis=1, keepdims=True) + 1e-8)

        discard_submask = np.zeros(target_mask.sum(), dtype=bool)

        for confuser_cls in confuser_list:
            conf_mask = real_labels == confuser_cls
            if not conf_mask.any():
                continue
            conf_centroid  = real_feats[conf_mask].mean(axis=0)
            conf_centroid /= np.linalg.norm(conf_centroid) + 1e-8
            sims            = target_feats_norm @ conf_centroid
            discard_submask |= (sims >= threshold)

        idx_of_target = np.where(target_mask)[0]
        keep_mask[idx_of_target[discard_submask]] = False

        n_kept     = int((~discard_submask).sum())
        n_total    = len(discard_submask)
        conf_names = [ACTIVITY_NAMES[c] for c in confuser_list]
        print(f"    {ACTIVITY_NAMES[target_cls]:<12} "
              f"kept {n_kept}/{n_total} "
              f"({100*n_kept/n_total:.1f}% pass) "
              f"discarded {n_total-n_kept} "
              f"vs {conf_names}")

    if not keep_mask.any():
        return (np.empty((0, syn_tokens.shape[1], syn_tokens.shape[2]),
                         dtype=np.float32),
                np.empty(0, dtype=int),
                np.empty((0, syn_gravity.shape[1]), dtype=np.float32))

    return (syn_tokens[keep_mask],
            syn_labels[keep_mask],
            syn_gravity[keep_mask])


# ══════════════════════════════════════════════════════════════════════════════
# QUALITY GATE — 320-d space, per-class thresholds
# ══════════════════════════════════════════════════════════════════════════════
def apply_quality_gate(
    syn_tokens:        np.ndarray,
    syn_labels:        np.ndarray,
    syn_gravity:       np.ndarray,
    real_tokens:       np.ndarray,
    real_labels:       np.ndarray,
    real_gravity:      np.ndarray,
    min_quality:       float = 0.70,
    per_class_quality: dict  = None,
) -> tuple:
    """
    Drops entire classes whose post-filter cosine similarity in 320-d space
    falls below per-class threshold. Evaluated in 320-d for consistency
    with confusion filter. Downstairs uses lenient threshold (0.55) due to
    known lower generation quality.
    """
    real_feats = compute_320d_features(real_tokens, real_gravity)
    syn_feats  = compute_320d_features(syn_tokens,  syn_gravity)

    classes_in_syn = np.unique(syn_labels)
    keep_mask      = np.zeros(len(syn_labels), dtype=bool)

    print(f"\n  Quality gate (space=320-d):")
    print(f"  {'Class':<12} {'Threshold':>10} {'Cos Sim':>10} {'Status':>10}")
    print("  " + "-" * 48)

    for cls_id in classes_in_syn:
        real_mask = real_labels == cls_id
        syn_mask  = syn_labels  == cls_id
        if real_mask.sum() == 0 or syn_mask.sum() == 0:
            continue

        threshold = (per_class_quality.get(cls_id, min_quality)
                     if per_class_quality else min_quality)

        real_centroid = real_feats[real_mask].mean(axis=0)
        syn_centroid  = syn_feats[syn_mask].mean(axis=0)
        cos_sim       = (np.dot(real_centroid, syn_centroid) /
                         (np.linalg.norm(real_centroid) *
                          np.linalg.norm(syn_centroid) + 1e-8))

        if cos_sim >= threshold:
            keep_mask[syn_mask] = True
            status = "✅ KEEP"
        else:
            status = "❌ DROP"

        print(f"  {ACTIVITY_NAMES[cls_id]:<12} {threshold:>10.2f} "
              f"{cos_sim:>10.4f} {status:>10}")

    filtered_tokens  = syn_tokens[keep_mask]
    filtered_labels  = syn_labels[keep_mask]
    filtered_gravity = syn_gravity[keep_mask]

    dropped = [ACTIVITY_NAMES[c] for c in classes_in_syn
               if not keep_mask[syn_labels == c].any()]
    if dropped:
        print(f"\n  Dropped classes: {dropped}")
    else:
        print(f"\n  All classes passed quality gate ✅")

    return filtered_tokens, filtered_labels, filtered_gravity


# ══════════════════════════════════════════════════════════════════════════════
# QUALITY CHECK — final distribution report
# ══════════════════════════════════════════════════════════════════════════════
def quality_check(
    real_tokens: np.ndarray,
    real_labels: np.ndarray,
    syn_tokens:  np.ndarray,
    syn_labels:  np.ndarray,
) -> None:
    """Cosine similarity and L2 distance in 64-d mean-pooled space."""
    print("\n  Quality Check — Real vs Synthetic (64-d mean-pooled):")
    print(f"  {'Class':<12} {'Cos Sim':>10} {'L2 Dist':>10} {'Assessment'}")
    print("  " + "-" * 55)

    real_mean = real_tokens.mean(axis=1)
    syn_mean  = syn_tokens.mean(axis=1)

    for cls_id in range(NUM_CLASSES):
        r_mask = real_labels == cls_id
        s_mask = syn_labels  == cls_id
        if r_mask.sum() == 0 or s_mask.sum() == 0:
            continue

        r_cent  = real_mean[r_mask].mean(axis=0)
        s_cent  = syn_mean[s_mask].mean(axis=0)
        cos_sim = (np.dot(r_cent, s_cent) /
                   (np.linalg.norm(r_cent) * np.linalg.norm(s_cent) + 1e-8))
        l2_dist = np.linalg.norm(r_cent - s_cent)
        assess  = ("✅ Good" if cos_sim > 0.85 else
                   "⚠️  Moderate" if cos_sim > 0.6 else "❌ Poor")
        print(f"  {ACTIVITY_NAMES[cls_id]:<12} {cos_sim:>10.4f} "
              f"{l2_dist:>10.4f}  {assess}")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
def save_synthetic(
    out_dir:     str,
    syn_tokens:  np.ndarray,
    syn_labels:  np.ndarray,
    syn_gravity: np.ndarray = None,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "synthetic_tokens.hdf5")

    grav_data = (syn_gravity if syn_gravity is not None
                 else np.zeros((len(syn_tokens), 64), dtype=np.float32))

    with h5py.File(path, "w") as f:
        f.create_dataset("tokens",
                         data=syn_tokens.astype(np.float32),
                         compression="gzip")
        f.create_dataset("mean_tokens",
                         data=syn_tokens.mean(axis=1).astype(np.float32),
                         compression="gzip")
        f.create_dataset("labels",
                         data=syn_labels.astype(np.float32))
        f.create_dataset("subject_ids",
                         data=np.full(len(syn_labels), -1, dtype=np.float32))
        f.create_dataset("gravity_vecs",
                         data=grav_data.astype(np.float32),
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
# ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="Generate synthetic Bio-PM token sequences with CAR-IMU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint",   type=str,
                   default="car_imu_checkpoints/best_model.pt")
    p.add_argument("--token_store",  type=str,
                   default="results_week1/token_store.hdf5")
    p.add_argument("--out_dir",      type=str,
                   default="synthetic_tokens")
    p.add_argument("--temperature",  type=float, default=0.3,
                   help="Sampling noise per step (0=greedy, 0.3=default)")
    p.add_argument("--strategy",     type=str, default="boundary_aware",
                   choices=["upsample", "ratio", "minority_ratio",
                            "balance", "confusion_aware", "boundary_aware"],
                   help="Generation strategy (default: boundary_aware)")
    p.add_argument("--syn_ratio",    type=float, default=0.5,
                   help="Ratio for ratio/minority_ratio, or count for balance")
    p.add_argument("--minority_ratio", type=float, default=0.25,
                   help="Synthetic ratio for confusion_aware strategy")
    p.add_argument("--base_ratio",   type=float, default=0.15,
                   help="Base ratio for boundary_aware, scaled by confusion "
                        "severity. (default=0.15) "
                        "Final count = n_real x severity x base_ratio")
    p.add_argument("--confusion_threshold", type=float, default=0.70,
                   help="Cosine sim threshold for confusion filter (default=0.70). "
                        "Lower = stricter. Tokens above threshold are discarded.")
    p.add_argument("--min_quality",  type=float, default=0.70,
                   help="Default min cos_sim (320-d) to keep a class after "
                        "filtering (default=0.70). "
                        "Per-class overrides set in BOUNDARY_QUALITY_THRESHOLDS.")
    p.add_argument("--no_calibrate", action="store_true", default=False,
                   help="Disable calibration. RECOMMENDED for boundary_aware. "
                        "Uncalibrated tokens preserve boundary diversity "
                        "(empirical: +0.013 vs +0.004 with calibration).")
    p.add_argument("--device",       type=str, default="auto",
                   choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--batch_size",   type=int, default=32)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    args   = parse_args()
    device = get_best_device(args.device)

    print("=" * 65)
    print("CAR-IMU — Synthetic Token Generation")
    print("=" * 65)
    print(f"  Checkpoint:   {args.checkpoint}")
    print(f"  Token store:  {args.token_store}")
    print(f"  Strategy:     {args.strategy}")
    print(f"  Temperature:  {args.temperature}")
    print(f"  Device:       {device}")
    print(f"  Calibrate:    {'disabled (recommended)' if args.no_calibrate else 'enabled'}")

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model(args.checkpoint, device)

    # ── Load real tokens + gravity ────────────────────────────────────────────
    print(f"\nLoading real tokens from {args.token_store} ...")
    with h5py.File(args.token_store, "r") as f:
        real_tokens  = f["tokens"][:]
        real_labels  = f["labels"][:].astype(int)
        real_gravity = (f["gravity_vecs"][:]
                        if "gravity_vecs" in f
                        else np.zeros((len(real_labels), 64), dtype=np.float32))

    total = len(real_labels)
    print(f"  {total:,} windows loaded")
    print(f"  Class distribution:")
    for cls_id, name in ACTIVITY_NAMES.items():
        n   = (real_labels == cls_id).sum()
        pct = 100 * n / total
        print(f"    {name:<12} {n:>5} ({pct:.1f}%)")

    # ══════════════════════════════════════════════════════════════════════════
    # BOUNDARY AWARE STRATEGY
    # ══════════════════════════════════════════════════════════════════════════
    if args.strategy == "boundary_aware":

        print(f"\n  Strategy: boundary_aware")
        print(f"  Base ratio:            {args.base_ratio:.2f}")
        print(f"  Confusion threshold:   {args.confusion_threshold}")
        print(f"  Min quality (320-d):   {args.min_quality}")
        print(f"  Severity threshold:    {SEVERITY_THRESHOLD}")
        print(f"  Calibration:           {'disabled' if args.no_calibrate else 'enabled'}")
        print(f"\n  Target classes (confusion_severity >= {SEVERITY_THRESHOLD}):")
        print(f"  {'Class':<12} {'Severity':>10} {'Ratio':>8} "
              f"{'N_gen':>8}  Confusers  Quality Gate")
        print("  " + "-" * 72)

        # ── Step 1: Generation counts from confusion severity ─────────────────
        gen_counts = {}
        for cls_id in range(NUM_CLASSES):
            severity = CONFUSION_SEVERITY.get(cls_id, 0.0)
            if severity < SEVERITY_THRESHOLD:
                gen_counts[cls_id] = 0
                continue

            ratio              = severity * args.base_ratio
            n_real             = int((real_labels == cls_id).sum())
            gen_counts[cls_id] = int(n_real * ratio)

            conf_spec  = BOUNDARY_CONFUSION_PAIRS.get(cls_id, [])
            conf_list  = ([conf_spec] if isinstance(conf_spec, int)
                          else list(conf_spec))
            conf_names = [ACTIVITY_NAMES[c] for c in conf_list]
            q_thresh   = BOUNDARY_QUALITY_THRESHOLDS.get(cls_id, args.min_quality)

            print(f"  {ACTIVITY_NAMES[cls_id]:<12} {severity:>10.2f} "
                  f"{ratio:>8.3f} {gen_counts[cls_id]:>8}  "
                  f"{str(conf_names):<25} {q_thresh:.2f}")

        total_gen = sum(gen_counts.values())
        exp_after = int(total_gen * 0.6)
        print(f"\n  Total raw to generate: {total_gen}")
        print(f"  Expected after filter: ~{exp_after} (~60% pass rate)")
        print(f"  Expected ratio:        ~{exp_after/total:.1%} of real data")

        # ── Step 2: Autoregressive generation ────────────────────────────────
        print(f"\n  Generating raw synthetic tokens (temp={args.temperature}) ...")
        raw_tokens_list = []
        raw_labels_list = []

        for cls_id in sorted(CONFUSION_SEVERITY.keys()):
            n = gen_counts.get(cls_id, 0)
            if n == 0:
                continue
            print(f"\n  [{ACTIVITY_NAMES[cls_id]}] generating {n} sequences ...")
            syn = generate_for_class(
                model, cls_id, n, args.temperature, device, args.batch_size)
            raw_tokens_list.append(syn)
            raw_labels_list.append(np.full(n, cls_id, dtype=int))

        if not raw_tokens_list:
            print("  Nothing to generate.")
            return

        raw_syn_tokens = np.concatenate(raw_tokens_list, axis=0)
        raw_syn_labels = np.concatenate(raw_labels_list, axis=0)
        print(f"\n  Generated {len(raw_syn_tokens)} raw sequences")

        # ── Step 3: Class-mean gravity ────────────────────────────────────────
        raw_syn_gravity = np.stack([
            real_gravity[real_labels == lbl].mean(axis=0)
            for lbl in raw_syn_labels
        ])

        # ── Step 3b: Optional calibration ────────────────────────────────────
        if not args.no_calibrate:
            print("\n  ⚠ Calibration enabled.")
            print("    Note: --no_calibrate recommended for boundary_aware.")
            print("    Uncalibrated tokens gave +0.013 vs +0.004 with calibration.")
            raw_syn_tokens = calibrate_tokens(
                real_tokens, real_labels, raw_syn_tokens, raw_syn_labels)

        # ── Step 4: Confusion filter (bidirectional, multi-confuser) ─────────
        syn_tokens, syn_labels, syn_gravity = filter_confused_synthetic(
            syn_tokens      = raw_syn_tokens,
            syn_labels      = raw_syn_labels,
            syn_gravity     = raw_syn_gravity,
            real_tokens     = real_tokens,
            real_labels     = real_labels,
            real_gravity    = real_gravity,
            confusion_pairs = BOUNDARY_CONFUSION_PAIRS,
            threshold       = args.confusion_threshold,
        )

        if len(syn_tokens) == 0:
            print("  ⚠ All tokens filtered out.")
            print("  Try: lower --confusion_threshold")
            return

        print(f"\n  After confusion filter: {len(syn_tokens)} tokens "
              f"({len(syn_tokens)/total:.1%} of real)")

        # ── Step 5: Quality gate (320-d, per-class thresholds) ────────────────
        syn_tokens, syn_labels, syn_gravity = apply_quality_gate(
            syn_tokens        = syn_tokens,
            syn_labels        = syn_labels,
            syn_gravity       = syn_gravity,
            real_tokens       = real_tokens,
            real_labels       = real_labels,
            real_gravity      = real_gravity,
            min_quality       = args.min_quality,
            per_class_quality = BOUNDARY_QUALITY_THRESHOLDS,
        )

        if len(syn_tokens) == 0:
            print("  ⚠ No tokens survived quality gate.")
            print("  Try: lower --confusion_threshold or lower --min_quality")
            return

        print(f"\n  After quality gate: {len(syn_tokens)} tokens "
              f"({len(syn_tokens)/total:.1%} of real)")

        # ── Step 6: Quality check ─────────────────────────────────────────────
        quality_check(real_tokens, real_labels, syn_tokens, syn_labels)

    # ══════════════════════════════════════════════════════════════════════════
    # CONFUSION AWARE STRATEGY (kept for comparison with boundary_aware)
    # ══════════════════════════════════════════════════════════════════════════
    elif args.strategy == "confusion_aware":

        per_class_f1 = {
            2: 0.667,   # Upstairs
            3: 0.690,   # Downstairs
            4: 0.784,   # Sitting
        }
        confusion_pairs_ca = {
            2: [0],
            3: [4, 5],
            4: [3],
        }
        per_class_quality_ca = {
            2: args.min_quality,
            3: 0.55,
            4: args.min_quality,
        }

        print(f"\n  Strategy: confusion_aware")
        print(f"  Minority ratio:      {args.minority_ratio:.0%}")
        print(f"  Confusion threshold: {args.confusion_threshold}")

        gen_counts = {}
        for cls_id in range(NUM_CLASSES):
            if cls_id not in per_class_f1:
                gen_counts[cls_id] = 0
                continue
            diff               = 1.0 - per_class_f1[cls_id]
            n_real             = int((real_labels == cls_id).sum())
            gen_counts[cls_id] = int(n_real * diff * args.minority_ratio)

        print(f"\n  Generation counts:")
        for cls_id, n in gen_counts.items():
            if n > 0:
                print(f"    {ACTIVITY_NAMES[cls_id]:<12} +{n}")

        raw_tokens_list, raw_labels_list = [], []
        for cls_id in sorted(per_class_f1.keys()):
            n = gen_counts[cls_id]
            if n == 0:
                continue
            print(f"\n  [{ACTIVITY_NAMES[cls_id]}] generating {n} sequences ...")
            syn = generate_for_class(
                model, cls_id, n, args.temperature, device, args.batch_size)
            raw_tokens_list.append(syn)
            raw_labels_list.append(np.full(n, cls_id, dtype=int))

        if not raw_tokens_list:
            print("  Nothing generated.")
            return

        raw_syn_tokens  = np.concatenate(raw_tokens_list, axis=0)
        raw_syn_labels  = np.concatenate(raw_labels_list, axis=0)
        raw_syn_gravity = np.stack([
            real_gravity[real_labels == lbl].mean(axis=0)
            for lbl in raw_syn_labels
        ])

        if not args.no_calibrate:
            raw_syn_tokens = calibrate_tokens(
                real_tokens, real_labels, raw_syn_tokens, raw_syn_labels)

        syn_tokens, syn_labels, syn_gravity = filter_confused_synthetic(
            syn_tokens=raw_syn_tokens, syn_labels=raw_syn_labels,
            syn_gravity=raw_syn_gravity, real_tokens=real_tokens,
            real_labels=real_labels, real_gravity=real_gravity,
            confusion_pairs=confusion_pairs_ca,
            threshold=args.confusion_threshold)

        if len(syn_tokens) == 0:
            print("  ⚠ All filtered. Lower --confusion_threshold.")
            return

        syn_tokens, syn_labels, syn_gravity = apply_quality_gate(
            syn_tokens=syn_tokens, syn_labels=syn_labels,
            syn_gravity=syn_gravity, real_tokens=real_tokens,
            real_labels=real_labels, real_gravity=real_gravity,
            min_quality=args.min_quality,
            per_class_quality=per_class_quality_ca)

        if len(syn_tokens) == 0:
            print("  ⚠ No tokens survived quality gate.")
            return

        quality_check(real_tokens, real_labels, syn_tokens, syn_labels)

    # ══════════════════════════════════════════════════════════════════════════
    # ALL OTHER STRATEGIES
    # ══════════════════════════════════════════════════════════════════════════
    else:
        gen_counts = compute_generation_counts(
            real_labels, args.strategy, args.syn_ratio)

        print(f"\nGenerating (temperature={args.temperature}) ...")
        all_tokens_list, all_labels_list = [], []

        for cls_id in range(NUM_CLASSES):
            n = gen_counts.get(cls_id, 0)
            if n == 0:
                continue
            print(f"\n  [{ACTIVITY_NAMES[cls_id]}] generating {n} sequences ...")
            syn = generate_for_class(
                model, cls_id, n, args.temperature, device, args.batch_size)
            all_tokens_list.append(syn)
            all_labels_list.append(np.full(n, cls_id, dtype=int))

        if not all_tokens_list:
            print("Nothing to generate.")
            return

        syn_tokens  = np.concatenate(all_tokens_list, axis=0)
        syn_labels  = np.concatenate(all_labels_list, axis=0)
        syn_gravity = None

        if not args.no_calibrate:
            syn_tokens = calibrate_tokens(
                real_tokens, real_labels, syn_tokens, syn_labels)

        quality_check(real_tokens, real_labels, syn_tokens, syn_labels)

    # ── Save ──────────────────────────────────────────────────────────────────
    syn_path = save_synthetic(
        args.out_dir,
        syn_tokens,
        syn_labels,
        syn_gravity if args.strategy in ["boundary_aware", "confusion_aware"] else None,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("✅  Synthetic token generation complete!")
    print("=" * 65)
    print(f"  Real tokens:      {len(real_tokens):,} windows")
    print(f"  Synthetic tokens: {len(syn_tokens):,} windows")
    print(f"  Ratio:            {len(syn_tokens)/len(real_tokens):.1%} of real")
    print(f"  Combined total:   {len(real_tokens)+len(syn_tokens):,} windows")
    print()
    print("Next step:")
    print(f"  python evaluate_augmented.py \\")
    print(f"      --real_tokens {args.token_store} \\")
    print(f"      --syn_tokens  {syn_path} \\")
    print(f"      --out_dir     results_boundary_aware \\")
    print(f"      --classifier  both")


if __name__ == "__main__":
    main()