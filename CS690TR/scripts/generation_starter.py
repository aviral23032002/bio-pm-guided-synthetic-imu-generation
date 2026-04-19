#!/usr/bin/env python3
"""
generation_starter.py — EXPERIMENTAL scaffold for movement generation.

===========================================================================
  ⚠️  THIS IS EXPERIMENTAL AND NOT A VALIDATED GENERATION PIPELINE  ⚠️
===========================================================================

BioPM's encoder_acc is a BIDIRECTIONAL masked transformer (BERT-style).
It was trained to reconstruct masked movement elements from context, NOT
to generate sequences autoregressively.  This means:

  ✗  It is NOT a proper autoregressive generator (like GPT)
  ✗  It cannot natively produce novel movement sequences from scratch
  ✗  Generated outputs have not been validated for scientific correctness

What this script provides instead:

  ✓  A "masked infilling" experiment: mask some movement elements in a
     real window and let the model reconstruct them from context
  ✓  A scaffold for iterative generation (mask → predict → unmask → repeat)
  ✓  Clear documentation of limitations

This may be useful for:
  - Understanding what the model has learned
  - Exploring the model's internal representations
  - Course projects that want to experiment with generation ideas
  - Prototyping augmentation strategies

It is NOT suitable for:
  - Publishing generated movement data as realistic
  - Any claim that BioPM is a generative model

=== USAGE ===

  python scripts/generation_starter.py \
      --data_dir     /path/to/preprocessed \
      --checkpoint   checkpoints/checkpoint.pt \
      --mask_ratio   0.5 \
      --device       cpu
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models.biopm import load_pretrained_encoder, ConvEncode
from src.data.preprocessing import load_preprocessed_h5


def parse_args():
    p = argparse.ArgumentParser(
        description="Experimental: masked infilling with BioPM")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Directory with preprocessed Data_MeLabel_*.h5 files")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to 50MR checkpoint.pt")
    p.add_argument("--mask_ratio", type=float, default=0.5,
                   help="Fraction of ME patches to mask (default: 0.5)")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--sample_idx", type=int, default=0,
                   help="Index of sample to experiment with")
    return p.parse_args()


def masked_infilling_experiment(model, X, pos_info, add_emb, mask_ratio,
                                device="cpu"):
    """
    Mask a fraction of movement-element patches and run the encoder.

    This demonstrates the model's ability to reconstruct masked patches
    from surrounding context, which is what it was trained to do.

    Args:
        model:      BioPMModel instance
        X:          (1, L, 32) movement element patches
        pos_info:   (1, L) fractional positions
        add_emb:    (1, L, K) additional embeddings
        mask_ratio: fraction of valid patches to mask
        device:     computation device

    Returns:
        original_tokens:     (L, 64) encoder output with no masking
        reconstructed_tokens: (L, 64) encoder output with masking applied
        mask:                (L,) which patches were masked
    """
    X_t = torch.from_numpy(X).float().to(device)
    pos_t = torch.from_numpy(pos_info).float().to(device)
    add_t = torch.from_numpy(add_emb).float().to(device)

    # Find valid (non-NaN) patches
    valid_mask = ~torch.isnan(X_t).any(dim=-1).squeeze(0)  # (L,)
    valid_indices = torch.where(valid_mask)[0]

    if len(valid_indices) == 0:
        print("No valid patches in this sample!")
        return None, None, None

    # Randomly select patches to mask
    n_mask = max(1, int(len(valid_indices) * mask_ratio))
    perm = torch.randperm(len(valid_indices))[:n_mask]
    mask_indices = valid_indices[perm]

    # Create mask tensor: 1.0 = masked, 0.0 = visible
    mask_no = torch.zeros(1, X_t.shape[1], device=device)
    mask_yes = torch.zeros(1, X_t.shape[1], device=device)
    mask_yes[0, mask_indices] = 1.0

    with torch.no_grad():
        original_tokens = model.encoder_acc(X_t, pos_t, mask_no, add_t)
        reconstructed_tokens = model.encoder_acc(X_t, pos_t, mask_yes, add_t)

    return (original_tokens.squeeze(0).cpu().numpy(),
            reconstructed_tokens.squeeze(0).cpu().numpy(),
            mask_yes.squeeze(0).cpu().numpy())


def main():
    args = parse_args()

    print("=" * 60)
    print("EXPERIMENTAL: BioPM Masked Infilling")
    print("=" * 60)
    print()
    print("⚠️  This is NOT a validated generation pipeline.")
    print("⚠️  BioPM is a bidirectional encoder, not an autoregressive generator.")
    print("⚠️  Outputs are for exploration only.")
    print()

    (X, pos_info, add_emb, labels, pids,
     X_grav, raw_acc) = load_preprocessed_h5(args.data_dir)

    model = load_pretrained_encoder(args.checkpoint, device=args.device)

    idx = args.sample_idx
    if idx >= len(X):
        print(f"Sample index {idx} out of range (max {len(X) - 1})")
        sys.exit(1)

    sample_X = X[idx:idx + 1]
    sample_pos = pos_info[idx:idx + 1]
    sample_add = add_emb[idx:idx + 1]

    print(f"Sample {idx}: label={int(labels[idx])}, subject={int(pids[idx])}")
    print(f"  ME patches shape: {sample_X.shape}")
    n_valid = int((~np.isnan(sample_X)).any(axis=-1).sum())
    print(f"  Valid patches: {n_valid}")
    print(f"  Mask ratio: {args.mask_ratio}")
    print()

    orig, recon, mask = masked_infilling_experiment(
        model, sample_X, sample_pos, sample_add,
        args.mask_ratio, args.device)

    if orig is None:
        return

    masked_indices = np.where(mask > 0.5)[0]
    print(f"Masked {len(masked_indices)} patches: {masked_indices.tolist()}")
    print()

    # Compare original vs reconstructed at masked positions
    for i in masked_indices[:5]:
        cos_sim = np.dot(orig[i], recon[i]) / (
            np.linalg.norm(orig[i]) * np.linalg.norm(recon[i]) + 1e-8)
        l2_dist = np.linalg.norm(orig[i] - recon[i])
        print(f"  Patch {i}: cosine_sim={cos_sim:.4f}, L2_dist={l2_dist:.4f}")

    print()
    print("Interpretation:")
    print("  - High cosine similarity means the model can reconstruct")
    print("    masked patches well from surrounding context.")
    print("  - This suggests the model has learned meaningful")
    print("    movement-element representations.")
    print()
    print("For iterative generation experiments, you could:")
    print("  1. Start with a partial sequence (e.g. first few real patches)")
    print("  2. Mask the remaining positions")
    print("  3. Run encoder to get predictions at masked positions")
    print("  4. 'Unmask' the most confident predictions")
    print("  5. Repeat until all positions are filled")
    print()
    print("⚠️  This is a research exploration idea, not a validated method.")


if __name__ == "__main__":
    main()
