"""
Feature extraction from the pretrained BioPM / 50MR model.

This module provides the main function students will use:
  extract_features(data_root, checkpoint_path, ...) → features, labels, pids

The extracted feature vector per window is:
  [mean_pool(64-d) | std_pool(64-d) | gravity_flat(900-d)] = 1028-d

  - mean_pool / std_pool: mean and standard deviation across all valid
    movement-element tokens from encoder_acc (the transformer stream)
  - gravity_flat: the raw low-pass gravity signal interpolated to 300
    timesteps × 3 axes = 900 values, flattened

This 1028-d vector is a good general-purpose representation of a single
movement window.  Students can use it for classification, regression,
clustering, retrieval, visualisation, or biomarker discovery.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..models.biopm import (
    BioPMModel, masked_mean_std, load_pretrained_encoder,
)
from ..data.dataset import MovementElementDataset
from ..data.preprocessing import load_preprocessed_h5


# Target length for gravity interpolation (300 timesteps × 3 axes = 900-d)
TARGET_GRAV_LEN = 300


def load_biopm_for_inference(checkpoint_path: str, n_classes: int = 11,
                             device: str = "cpu") -> BioPMModel:
    """Load BioPM with pretrained encoder_acc, in eval mode."""
    return load_pretrained_encoder(checkpoint_path, n_classes, device)


def extract_features(data_root: str, checkpoint_path: str,
                     batch_size: int = 32, device: str = "cpu",
                     num_workers: int = 0):
    """
    End-to-end feature extraction.

    Args:
        data_root:       directory containing Data_MeLabel_*.h5 files
        checkpoint_path: path to submovement_transformer_50MR/checkpoint.pt
        batch_size:      inference batch size
        device:          'cpu' or 'cuda'
        num_workers:     DataLoader workers

    Returns:
        features:  (N, 1028) numpy array of BioPM embeddings
        labels:    (N,) numpy array of integer labels
        pids:      (N,) numpy array of subject IDs
    """
    # --- Load data ---
    (X, pos_info, add_emb, labels, pids,
     X_grav, raw_acc) = load_preprocessed_h5(data_root)

    print(f"Loaded {X.shape[0]} windows from {data_root}")
    print(f"  ME patches:  {X.shape}")
    print(f"  Gravity:     {X_grav.shape if X_grav is not None else 'None'}")

    dataset = MovementElementDataset(
        X=X, X_grav=raw_acc, y=labels, pos_info=pos_info,
        additional_embedding=add_emb, pid=pids,
        name="feature_extract", is_label=True,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)

    if X_grav is not None:
        X_grav_tensor = torch.from_numpy(X_grav).float()
    else:
        X_grav_tensor = None

    # --- Load model ---
    model = load_biopm_for_inference(checkpoint_path, device=device)

    # --- Extract ---
    all_features, all_labels, all_pids = [], [], []
    global_idx = 0

    with torch.no_grad():
        for batch in loader:
            my_X, my_Y, my_PID, raw_batch, my_pos, my_add = batch
            bs = my_X.shape[0]

            my_X = my_X.to(device, dtype=torch.float)
            my_pos = my_pos.to(device, dtype=torch.float)
            my_add = my_add.to(device, dtype=torch.float)

            # No masking at inference time
            mask = torch.zeros(bs, my_X.shape[1], device=device)

            # Encoder_acc forward → (B, L, 64)
            acc_tokens = model.encoder_acc(my_X, my_pos, mask, my_add)

            # Pool: mean + std → (B, 128)
            acc_pooled = masked_mean_std(acc_tokens)

            # Gravity stream
            if X_grav_tensor is not None:
                grav_batch = X_grav_tensor[global_idx:global_idx + bs].to(device)
                g = grav_batch.transpose(1, 2)  # (B, 3, T)
                g = torch.where(torch.isnan(g), torch.zeros_like(g), g)
                if g.shape[-1] != TARGET_GRAV_LEN:
                    g = F.interpolate(g, size=TARGET_GRAV_LEN,
                                      mode="linear", align_corners=False)
                g_flat = g.reshape(bs, -1)  # (B, 900)
            else:
                g_flat = torch.zeros(bs, TARGET_GRAV_LEN * 3, device=device)

            # Fused feature: (B, 128 + 900) = (B, 1028)
            fused = torch.cat([acc_pooled, g_flat], dim=-1)

            all_features.append(fused.cpu().numpy())
            all_labels.append(my_Y.numpy() if isinstance(my_Y, torch.Tensor)
                              else np.array(my_Y))
            all_pids.append(my_PID.numpy() if isinstance(my_PID, torch.Tensor)
                            else np.array(my_PID))
            global_idx += bs

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    pids = np.concatenate(all_pids, axis=0)

    print(f"Extracted features: {features.shape}")
    return features, labels, pids
