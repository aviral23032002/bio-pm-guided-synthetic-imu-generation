"""
PyTorch Dataset for BioPM movement-element data.

Expects preprocessed HDF5 files produced by the preprocessing pipeline.
Each HDF5 file contains per-subject data with keys:
  - x_acc_filt:     (N_windows, L_patches, 32+metadata) movement elements
  - x_gravity:      (N_windows, T_gravity, 3) low-pass gravity signal
  - window_acc_raw: (N_windows, T_raw, 3) raw accelerometer windows
  - window_label:   (N_windows,) integer activity labels
"""

import torch
import numpy as np


class MovementElementDataset:
    """
    Dataset that yields one window at a time for BioPM inference.

    Each __getitem__ returns:
        sample         : (L, 32) normalized movement-element patches
        label          : int activity label
        subject_id     : int subject identifier
        gravity        : (T, 3) raw gravity window (for GravityCNNEncoder)
        pos_info       : (L,) fractional position of each patch in the window
        additional_emb : (L, K) axis code, duration, etc.
    """

    def __init__(self, X, X_grav, y, pos_info, additional_embedding, pid,
                 name="", is_label=True, transform=None):
        self.X = torch.from_numpy(X).float()
        self.X_grav = torch.from_numpy(X_grav).float()
        self.y = y
        self.is_label = is_label
        self.transform = transform
        self.pid = pid
        self.pos_info = pos_info
        self.additional_embedding = additional_embedding
        if name:
            print(f"{name} set: {len(self.X)} samples")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.X[idx]
        sample_grav = self.X_grav[idx]
        y = self.y[idx] if self.is_label else []
        if self.transform:
            sample_grav = self.transform(sample_grav)
        return (sample, y, self.pid[idx], sample_grav,
                self.pos_info[idx], self.additional_embedding[idx])
