#!/usr/bin/env python3
"""
Minimal preprocessing example — shows how raw data becomes BioPM input.

This is a self-contained demo.  It creates synthetic 3-axis accelerometer
data, runs it through the preprocessing pipeline, and shows the output
shapes at each step.

Run from the CS690TR directory:
    python examples/example_preprocessing.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.preprocessing import (
    resample_to_target_fs,
    bandpass_filter,
    lowpass_filter,
    detect_zero_crossings,
)


def main():
    print("=" * 60)
    print("Preprocessing Pipeline Demo (synthetic data)")
    print("=" * 60)

    # --- Step 1: Create synthetic 3-axis accelerometer data ---
    # Simulates 20 seconds of data at 50 Hz with sinusoidal motion + noise
    ori_fs = 50  # Hz
    duration = 20  # seconds
    N = ori_fs * duration
    t = np.arange(N) / ori_fs

    # Synthetic acceleration: sine waves + random noise (in g)
    np.random.seed(42)
    acc_raw = np.column_stack([
        0.5 * np.sin(2 * np.pi * 1.5 * t) + 0.1 * np.random.randn(N),  # X
        0.3 * np.sin(2 * np.pi * 2.0 * t + 0.5) + 0.1 * np.random.randn(N),  # Y
        1.0 + 0.2 * np.sin(2 * np.pi * 0.5 * t) + 0.05 * np.random.randn(N),  # Z (gravity ~1g)
    ])
    labels = np.ones(N)  # dummy label = 1 for all samples

    print(f"\n1. Raw data:  shape={acc_raw.shape}, fs={ori_fs} Hz, "
          f"duration={duration}s")

    # --- Step 2: Resample to 30 Hz ---
    target_fs = 30
    acc_res, t_res, labels_res = resample_to_target_fs(t, acc_raw, labels,
                                                        target_fs)
    print(f"2. Resampled: shape={acc_res.shape}, fs={target_fs} Hz")

    # --- Step 3: Filter ---
    acc_filt = bandpass_filter(acc_res, low_hz=0.5, high_hz=12,
                               fs=target_fs, order=6)
    acc_grav = lowpass_filter(acc_res, cutoff_hz=0.5, fs=target_fs, order=6)
    print(f"3. Filtered:  body_acc={acc_filt.shape}, gravity={acc_grav.shape}")

    # --- Step 4: Window + movement element extraction ---
    config = {
        'target_FS': target_fs,
        'normalize_size_target': 32,
        'normalize_size_assign': 32,
        'pad_size': 192,
        'WS': 10,
    }

    ws = int(config['WS'] * target_fs)
    window_filt = acc_filt[:ws]
    window_time = t_res[:ws]

    try:
        (_, _, me_list, me_norm, me_info, me_norm_pad,
         _, pos_info, _, _) = detect_zero_crossings(
            window_filt, window_time, config)

        print(f"\n4. Movement elements found: {len(me_list)}")
        print(f"   Normalized ME shape:     ({len(me_list)}, "
              f"{config['normalize_size_target']})")
        print(f"   ME metadata columns:     {list(me_info.columns)}")
        print(f"   Position info shape:     {pos_info.shape}")
        print(f"   Padded output shape:     {me_norm_pad.shape}")

    except Exception as e:
        print(f"\n4. Zero-crossing detection failed: {e}")
        print("   This can happen with very smooth synthetic data.")
        print("   Real sensor data typically has enough crossings.")
        return

    # --- Step 5: Show what goes into the model ---
    print(f"\n5. What the model receives per window:")
    print(f"   x_acc_filt:  ({config['pad_size']}, "
          f"{32 + 1 + 5}) = patches + pos + metadata")
    print(f"                first 32 cols = normalised ME values")
    print(f"                col 32 = fractional position in window")
    print(f"                cols 33+ = axis, duration, min, max, direction")
    print(f"   x_gravity:   ({ws}, 3) = raw gravity window")
    print(f"   window_label: scalar integer label")

    print(f"\n{'=' * 60}")
    print("See scripts/preprocess_data.py for the full preprocessing script.")


if __name__ == "__main__":
    main()
