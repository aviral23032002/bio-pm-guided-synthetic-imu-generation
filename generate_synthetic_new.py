#!/usr/bin/env python3
"""
generate_synthetic.py — Generate synthetic Bio-PM token sequences using trained CAR-IMU.

WHAT THIS DOES:
    Loads the trained CAR-IMU checkpoint, generates synthetic token sequences
    for minority activity classes, and saves them in the same HDF5 format as
    token_store.hdf5 for direct mixing with real tokens in HAR classifier training.

PIPELINE (confusion_aware strategy):
    1. Generate raw synthetic tokens autoregressively
    2. Calibrate — shift/scale to match real class mean/std  ← BEFORE filter
    3. Confusion filter — discard tokens too similar to confuser class centroids
    4. Quality gate — drop entire classes below cos_sim threshold (320-d space)
    5. Quality check — report final distribution metrics
    6. Save

KEY CHANGES FROM PREVIOUS VERSION:
    - Calibration moved BEFORE confusion filter (not after)
    - Quality gate now uses 320-d axis-wise features (consistent with filter)
    - Multi-confuser support: Downstairs filtered vs Sitting AND Standing
    - Downstairs added back with per-class lenient quality threshold (0.55)
    - Per-class F1 updated to reflect smartwatch 320-d condition A results
    - Per-class quality thresholds configurable via --per_class_quality flag

USAGE:
    python generate_synthetic.py \
        --checkpoint  car_imu_checkpoints_6class/best_model.pt \
        --token_store results_wisdm_v2_6class/token_store.hdf5 \
        --out_dir     synthetic_tokens_v5 \
        --strategy    confusion_aware \
        --minority_ratio 0.25 \
        --confusion_threshold 0.70 \
        --min_quality 0.75 \
        --temperature 0.3
"""

import os
import argparse
import numpy as np
import h5py
import torch

from car_imu_decoder import CARIMUDecoder, TOKEN_DIM, SEQ_LEN, NUM_CLASSES, ACTIVITY_NAMES


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
    print(f"  Per-class val MSE at best checkpoint:")
    for cls_id, name in ACTIVITY_NAMES.items():
        mse = ckpt['per_class_loss'].get(cls_id, float('nan'))
        print(f"    {name:<12} {mse:.4f}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# GENERATION COUNT COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════
def compute_generation_counts(
    real_labels: np.ndarray,
    strategy: str,
    syn_ratio: float,
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
        print(f"\n  Strategy: upsample to {target} (majority class count)")
        for cls_id in range(NUM_CLASSES):
            gen_counts[cls_id] = max(0, target - counts[cls_id])

    elif strategy == "ratio":
        print(f"\n  Strategy: add {syn_ratio:.0%} synthetic for ALL classes")
        for cls_id in range(NUM_CLASSES):
            gen_counts[cls_id] = int(counts[cls_id] * syn_ratio)

    elif strategy == "minority_ratio":
        print(f"\n  Strategy: add {syn_ratio:.0%} synthetic for MINORITY classes only")
        for cls_id in range(NUM_CLASSES):
            pct = counts[cls_id] / total
            gen_counts[cls_id] = int(counts[cls_id] * syn_ratio) if pct < 0.18 else 0

    elif strategy == "balance":
        target_count = int(syn_ratio)
        print(f"\n  Strategy: balance ({target_count} synthetic per class)")
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
    model:      CARIMUDecoder,
    class_id:   int,
    n_samples:  int,
    temperature: float,
    device:     torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Batched autoregressive generation for one activity class."""
    model.eval()
    all_tokens = []
    generated  = 0

    with torch.no_grad():
        while generated < n_samples:
            B      = min(batch_size, n_samples - generated)
            labels = torch.tensor([class_id] * B, dtype=torch.long, device=device)
            context = model.class_emb(labels).unsqueeze(1)   # (B, 1, D)
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
# CALIBRATION — match real class mean/std  (run BEFORE filter)
# ══════════════════════════════════════════════════════════════════════════════
def calibrate_tokens(
    real_tokens: np.ndarray,
    real_labels: np.ndarray,
    syn_tokens:  np.ndarray,
    syn_labels:  np.ndarray,
) -> np.ndarray:
    """
    Force synthetic tokens to match the exact mean and std of real tokens
    per class. Applied BEFORE confusion filter so the filter evaluates
    calibrated tokens against real centroids.
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

        cls_syn = (syn_tokens[m_s] - s_mean) / s_std
        calibrated[m_s] = (cls_syn * r_std) + r_mean
        print(f"    {ACTIVITY_NAMES[cls_id]:<12} calibrated "
              f"(Δmean={abs(r_mean - s_mean).mean():.4f} "
              f"Δstd={abs(r_std - s_std).mean():.4f})")

    return calibrated.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 320-D FEATURE COMPUTATION (must match evaluate_augmented.py exactly)
# ══════════════════════════════════════════════════════════════════════════════
def compute_320d_features(
    tokens:       np.ndarray,   # (N, 192, 64)
    gravity_vecs: np.ndarray,   # (N, 64)
) -> np.ndarray:
    """
    Axis-wise mean pooling + global std + gravity = 320-d.
    Tokens interleaved by axis: 0::3=x, 1::3=y, 2::3=z.
    """
    mean_x    = tokens[:, 0::3, :].mean(axis=1)   # (N, 64)
    mean_y    = tokens[:, 1::3, :].mean(axis=1)   # (N, 64)
    mean_z    = tokens[:, 2::3, :].mean(axis=1)   # (N, 64)
    std_feats = tokens.std(axis=1)                 # (N, 64)
    return np.concatenate(
        [mean_x, mean_y, mean_z, std_feats, gravity_vecs], axis=1
    )   # (N, 320)


# ══════════════════════════════════════════════════════════════════════════════
# CONFUSION FILTER — vectorized, multi-confuser support
# ══════════════════════════════════════════════════════════════════════════════
def filter_confused_synthetic(
    syn_tokens:      np.ndarray,
    syn_labels:      np.ndarray,
    syn_gravity:     np.ndarray,
    real_tokens:     np.ndarray,
    real_labels:     np.ndarray,
    real_gravity:    np.ndarray,
    confusion_pairs: dict,        # {target_cls: int | list[int]}
    threshold:       float = 0.70,
) -> tuple:
    """
    Vectorized filter operating in 320-d feature space.
    Removes synthetic tokens that are too similar to their confuser class
    centroids. Supports multiple confusers per target class.

    A token is discarded if its cosine similarity to ANY confuser centroid
    exceeds the threshold.
    """
    real_feats    = compute_320d_features(real_tokens, real_gravity)
    all_syn_feats = compute_320d_features(syn_tokens,  syn_gravity)
    keep_mask     = np.ones(len(syn_labels), dtype=bool)

    for target_cls, confuser_spec in confusion_pairs.items():
        # Support single int or list of ints
        confuser_list = ([confuser_spec]
                         if isinstance(confuser_spec, int)
                         else list(confuser_spec))

        target_mask = syn_labels == target_cls
        if not target_mask.any():
            continue

        target_feats      = all_syn_feats[target_mask]
        target_feats_norm = target_feats / (
            np.linalg.norm(target_feats, axis=1, keepdims=True) + 1e-8)

        # Discard if too similar to ANY confuser
        discard_submask = np.zeros(target_mask.sum(), dtype=bool)

        for confuser_cls in confuser_list:
            conf_mask = real_labels == confuser_cls
            if not conf_mask.any():
                continue
            conf_centroid  = real_feats[conf_mask].mean(axis=0)
            conf_centroid /= np.linalg.norm(conf_centroid) + 1e-8

            sims = target_feats_norm @ conf_centroid   # (N_target,)
            discard_submask |= (sims >= threshold)

        idx_of_target  = np.where(target_mask)[0]
        idx_to_discard = idx_of_target[discard_submask]
        keep_mask[idx_to_discard] = False

        n_kept      = int((~discard_submask).sum())
        n_total     = len(discard_submask)
        conf_names  = [ACTIVITY_NAMES[c] for c in confuser_list]
        print(f"    {ACTIVITY_NAMES[target_cls]:<12} "
              f"kept {n_kept}/{n_total} "
              f"({100*n_kept/n_total:.1f}% pass rate) "
              f"discarded vs {conf_names}")

    if not keep_mask.any():
        empty_tok = np.empty(
            (0, syn_tokens.shape[1], syn_tokens.shape[2]), dtype=np.float32)
        empty_lbl = np.empty(0, dtype=int)
        empty_grv = np.empty((0, syn_gravity.shape[1]), dtype=np.float32)
        return empty_tok, empty_lbl, empty_grv

    return (syn_tokens[keep_mask],
            syn_labels[keep_mask],
            syn_gravity[keep_mask])


# ══════════════════════════════════════════════════════════════════════════════
# QUALITY GATE — per-class thresholds, evaluated in 320-d space
# ══════════════════════════════════════════════════════════════════════════════
def apply_quality_gate(
    syn_tokens:        np.ndarray,
    syn_labels:        np.ndarray,
    syn_gravity:       np.ndarray,
    real_tokens:       np.ndarray,
    real_labels:       np.ndarray,
    real_gravity:      np.ndarray,
    min_quality:       float = 0.75,
    per_class_quality: dict  = None,   # {cls_id: threshold} overrides min_quality
) -> tuple:
    """
    Drops entire classes whose post-filter cosine similarity (in 320-d space)
    falls below the threshold. Supports per-class thresholds for classes like
    Downstairs that have known lower generation quality.
    """
    # Evaluate in 320-d feature space (consistent with confusion filter)
    real_feats = compute_320d_features(real_tokens, real_gravity)
    syn_feats  = compute_320d_features(syn_tokens,  syn_gravity)

    classes_in_syn = np.unique(syn_labels)
    keep_mask      = np.zeros(len(syn_labels), dtype=bool)

    print(f"\n  Quality gate (default min_cos_sim={min_quality}, space=320-d):")
    print(f"  {'Class':<12} {'Threshold':>10} {'Cos Sim':>10} {'Status':>15}")
    print("  " + "-" * 52)

    for cls_id in classes_in_syn:
        real_mask = real_labels == cls_id
        syn_mask  = syn_labels  == cls_id

        if real_mask.sum() == 0 or syn_mask.sum() == 0:
            continue

        # Per-class threshold if provided
        threshold = (per_class_quality.get(cls_id, min_quality)
                     if per_class_quality else min_quality)

        real_centroid = real_feats[real_mask].mean(axis=0)
        syn_centroid  = syn_feats[syn_mask].mean(axis=0)

        cos_sim = (
            np.dot(real_centroid, syn_centroid) /
            (np.linalg.norm(real_centroid) *
             np.linalg.norm(syn_centroid) + 1e-8)
        )

        if cos_sim >= threshold:
            keep_mask[syn_mask] = True
            status = "✅ KEEP"
        else:
            status = f"❌ DROP"

        print(f"  {ACTIVITY_NAMES[cls_id]:<12} {threshold:>10.2f} "
              f"{cos_sim:>10.4f} {status:>15}")

    filtered_tokens  = syn_tokens[keep_mask]
    filtered_labels  = syn_labels[keep_mask]
    filtered_gravity = syn_gravity[keep_mask]

    dropped = [ACTIVITY_NAMES[c] for c in classes_in_syn
               if not keep_mask[syn_labels == c].any()]
    if dropped:
        print(f"\n  Dropped: {dropped}")
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
    """Cosine similarity and L2 distance between real and synthetic centroids."""
    print("\n  Quality Check — Real vs Synthetic Token Distributions:")
    print(f"  {'Class':<12} {'Cos Sim':>10} {'L2 Dist':>10} {'Assessment'}")
    print("  " + "-" * 55)

    real_mean = real_tokens.mean(axis=1)
    syn_mean  = syn_tokens.mean(axis=1)

    for cls_id in range(NUM_CLASSES):
        r_mask = real_labels == cls_id
        s_mask = syn_labels  == cls_id
        if r_mask.sum() == 0 or s_mask.sum() == 0:
            continue

        r_cent = real_mean[r_mask].mean(axis=0)
        s_cent = syn_mean[s_mask].mean(axis=0)

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
        f.create_dataset("tokens",      data=syn_tokens.astype(np.float32),
                         compression="gzip")
        f.create_dataset("mean_tokens", data=syn_tokens.mean(axis=1).astype(np.float32),
                         compression="gzip")
        f.create_dataset("labels",      data=syn_labels.astype(np.float32))
        f.create_dataset("subject_ids", data=np.full(len(syn_labels), -1,
                                                      dtype=np.float32))
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
# ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="Generate synthetic Bio-PM token sequences with CAR-IMU")

    p.add_argument("--checkpoint",   type=str,
                   default="car_imu_checkpoints/best_model.pt")
    p.add_argument("--token_store",  type=str,
                   default="results_week1/token_store.hdf5")
    p.add_argument("--out_dir",      type=str,
                   default="synthetic_tokens")
    p.add_argument("--temperature",  type=float, default=0.3,
                   help="Sampling noise per autoregressive step "
                        "(0=greedy, 0.3=recommended, 1.0=max)")
    p.add_argument("--strategy",     type=str, default="confusion_aware",
                   choices=["upsample", "ratio", "minority_ratio",
                            "balance", "confusion_aware"],
                   help="Generation strategy")
    p.add_argument("--syn_ratio",    type=float, default=0.5,
                   help="Ratio for ratio/minority_ratio strategies, "
                        "or fixed count for balance strategy")
    p.add_argument("--minority_ratio", type=float, default=0.25,
                   help="Synthetic ratio for confusion_aware targets "
                        "(default=0.25 = 25%% of real class count)")
    p.add_argument("--confusion_threshold", type=float, default=0.70,
                   help="Cosine similarity threshold for confusion filter "
                        "(default=0.70). Lower = stricter.")
    p.add_argument("--min_quality",  type=float, default=0.75,
                   help="Default minimum cos_sim (320-d) to keep a class "
                        "after filtering (default=0.75)")
    p.add_argument("--no_calibrate", action="store_true",
                   help="Disable per-class token calibration")
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

    print("=" * 60)
    print("CAR-IMU — Synthetic Token Generation")
    print("=" * 60)
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Token store: {args.token_store}")
    print(f"  Strategy:    {args.strategy}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Device:      {device}")
    print(f"  Calibrate:   {'no' if args.no_calibrate else 'yes'}")

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model(args.checkpoint, device)

    # ── Load real tokens ──────────────────────────────────────────────────────
    print(f"\nLoading real tokens from {args.token_store} ...")
    with h5py.File(args.token_store, "r") as f:
        real_tokens  = f["tokens"][:]
        real_labels  = f["labels"][:].astype(int)
        real_gravity = f["gravity_vecs"][:] if "gravity_vecs" in f \
                       else np.zeros((len(real_labels), 64), dtype=np.float32)

    # ══════════════════════════════════════════════════════════════════════════
    # CONFUSION AWARE STRATEGY
    # ══════════════════════════════════════════════════════════════════════════
    if args.strategy == "confusion_aware":

        # ── Configuration ────────────────────────────────────────────────────
        # Target classes: Upstairs, Downstairs, Sitting
        # Per-class F1 from condition A 320-d smartwatch results
        per_class_f1 = {
            2: 0.667,   # Upstairs   — hardest minority
            3: 0.690,   # Downstairs — moderate, add back with lenient gate
            4: 0.784,   # Sitting    — confused with Downstairs
        }

        # Confusion pairs — supports multiple confusers per class
        # Format: {target_cls: int | [int, int, ...]}
        confusion_pairs = {
            2: 0,       # Upstairs   → filter vs Walking only
            3: [4, 5],  # Downstairs → filter vs Sitting AND Standing
            4: 3,       # Sitting    → filter vs Downstairs only
        }

        # Per-class quality thresholds (override --min_quality)
        # Downstairs has known lower generation quality → lenient threshold
        per_class_quality = {
            2: args.min_quality,   # Upstairs   — use default (0.75)
            3: 0.55,               # Downstairs — lenient (known poor quality)
            4: args.min_quality,   # Sitting    — use default (0.75)
        }

        print(f"\n  Strategy: confusion_aware")
        print(f"  Minority ratio:        {args.minority_ratio:.0%}")
        print(f"  Confusion threshold:   {args.confusion_threshold}")
        print(f"  Default min quality:   {args.min_quality}")
        print(f"  Target classes:")
        for cls_id, f1 in per_class_f1.items():
            difficulty = 1.0 - f1
            ratio      = difficulty * args.minority_ratio
            n_real     = int((real_labels == cls_id).sum())
            n_gen      = int(n_real * ratio)
            q_thresh   = per_class_quality[cls_id]
            conf       = confusion_pairs[cls_id]
            conf_names = ([ACTIVITY_NAMES[conf]] if isinstance(conf, int)
                          else [ACTIVITY_NAMES[c] for c in conf])
            print(f"    {ACTIVITY_NAMES[cls_id]:<12} "
                  f"F1={f1:.3f} difficulty={difficulty:.3f} "
                  f"→ +{n_gen} tokens  "
                  f"quality_gate={q_thresh}  "
                  f"filter_vs={conf_names}")

        # ── Step 1: Compute generation counts ─────────────────────────────────
        gen_counts = {}
        for cls_id in range(NUM_CLASSES):
            if cls_id not in per_class_f1:
                gen_counts[cls_id] = 0
                continue
            difficulty         = 1.0 - per_class_f1[cls_id]
            ratio              = difficulty * args.minority_ratio
            n_real             = int((real_labels == cls_id).sum())
            gen_counts[cls_id] = int(n_real * ratio)

        # ── Step 2: Generate raw tokens ────────────────────────────────────────
        print(f"\n  Generating raw synthetic tokens ...")
        raw_syn_tokens_list, raw_syn_labels_list = [], []

        for cls_id in sorted(per_class_f1.keys()):
            n = gen_counts[cls_id]
            if n == 0:
                continue
            print(f"\n  [{ACTIVITY_NAMES[cls_id]}] generating {n} sequences ...")
            syn = generate_for_class(
                model, cls_id, n, args.temperature, device, args.batch_size)
            raw_syn_tokens_list.append(syn)
            raw_syn_labels_list.append(np.full(n, cls_id, dtype=int))

        if not raw_syn_tokens_list:
            print("  Nothing to generate.")
            return

        raw_syn_tokens = np.concatenate(raw_syn_tokens_list, axis=0)
        raw_syn_labels = np.concatenate(raw_syn_labels_list, axis=0)

        # ── Step 3: Assign class-mean gravity ─────────────────────────────────
        raw_syn_gravity = np.stack([
            real_gravity[real_labels == lbl].mean(axis=0)
            for lbl in raw_syn_labels
        ])   # (N_syn, 64)

        # ── Step 3b: CALIBRATE before filtering ──────────────────────────────
        # Critical: calibrate first so filter evaluates distribution-aligned tokens
        if not args.no_calibrate:
            raw_syn_tokens = calibrate_tokens(
                real_tokens, real_labels,
                raw_syn_tokens, raw_syn_labels
            )

        # ── Step 4: Confusion filter (on calibrated tokens) ───────────────────
        print(f"\n  Applying confusion-aware filter (threshold={args.confusion_threshold}) ...")
        syn_tokens, syn_labels, syn_gravity = filter_confused_synthetic(
            syn_tokens      = raw_syn_tokens,
            syn_labels      = raw_syn_labels,
            syn_gravity     = raw_syn_gravity,
            real_tokens     = real_tokens,
            real_labels     = real_labels,
            real_gravity    = real_gravity,
            confusion_pairs = confusion_pairs,
            threshold       = args.confusion_threshold,
        )

        if len(syn_tokens) == 0:
            print("  ⚠ All tokens filtered out. Lower --confusion_threshold.")
            return

        # ── Step 4b: Quality gate (320-d, per-class thresholds) ───────────────
        syn_tokens, syn_labels, syn_gravity = apply_quality_gate(
            syn_tokens        = syn_tokens,
            syn_labels        = syn_labels,
            syn_gravity       = syn_gravity,
            real_tokens       = real_tokens,
            real_labels       = real_labels,
            real_gravity      = real_gravity,
            min_quality       = args.min_quality,
            per_class_quality = per_class_quality,
        )

        if len(syn_tokens) == 0:
            print("  ⚠ No tokens survived quality gate.")
            print("  Try: lower --confusion_threshold or lower --min_quality")
            return

        # ── Step 5: Quality check ─────────────────────────────────────────────
        quality_check(real_tokens, real_labels, syn_tokens, syn_labels)

    # ══════════════════════════════════════════════════════════════════════════
    # ALL OTHER STRATEGIES
    # ══════════════════════════════════════════════════════════════════════════
    else:
        gen_counts = compute_generation_counts(
            real_labels, args.strategy, args.syn_ratio)

        print(f"\nGenerating synthetic tokens (temperature={args.temperature}) ...")
        all_syn_tokens_list, all_syn_labels_list = [], []

        for cls_id in range(NUM_CLASSES):
            n = gen_counts.get(cls_id, 0)
            if n == 0:
                continue
            print(f"\n  [{ACTIVITY_NAMES[cls_id]}] generating {n} sequences ...")
            syn = generate_for_class(
                model, cls_id, n, args.temperature, device, args.batch_size)
            all_syn_tokens_list.append(syn)
            all_syn_labels_list.append(np.full(n, cls_id, dtype=int))

        if not all_syn_tokens_list:
            print("Nothing to generate.")
            return

        syn_tokens  = np.concatenate(all_syn_tokens_list, axis=0)
        syn_labels  = np.concatenate(all_syn_labels_list, axis=0)
        syn_gravity = None

        if not args.no_calibrate:
            syn_tokens = calibrate_tokens(
                real_tokens, real_labels, syn_tokens, syn_labels)

        quality_check(real_tokens, real_labels, syn_tokens, syn_labels)

    # ── Save ──────────────────────────────────────────────────────────────────
    syn_path = save_synthetic(
        args.out_dir, syn_tokens, syn_labels,
        syn_gravity if args.strategy == "confusion_aware" else None
    )

    print("\n" + "=" * 60)
    print("✅  Synthetic token generation complete!")
    print("=" * 60)
    print(f"  Real tokens:      {len(real_tokens):,} windows")
    print(f"  Synthetic tokens: {len(syn_tokens):,} windows")
    print(f"  Ratio:            {len(syn_tokens)/len(real_tokens):.1%} of real")
    print(f"  Combined total:   {len(real_tokens) + len(syn_tokens):,} windows")
    print()
    print("Next step:")
    print(f"  python evaluate_augmented.py \\")
    print(f"      --real_tokens {args.token_store} \\")
    print(f"      --syn_tokens  {syn_path} \\")
    print(f"      --out_dir     results_v5 \\")
    print(f"      --classifier  both")


if __name__ == "__main__":
    main()