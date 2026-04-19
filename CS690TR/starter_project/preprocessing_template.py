#!/usr/bin/env python3
"""
Preprocessing template — adapt this for YOUR dataset.

This template walks you through each step of converting your raw sensor
data into the HDF5 format BioPM expects.  It is a simplified version of
scripts/preprocess_data.py with clearer placeholders.

=== STEPS ===

  1. LOAD:     Read your raw data file
  2. CLEAN:    Handle NaN, convert units to g
  3. RESAMPLE: Interpolate to 30 Hz
  4. FILTER:   Bandpass → body accel; lowpass → gravity
  5. WINDOW:   Sliding windows (10s default)
  6. EXTRACT:  Zero-crossing movement elements per window
  7. SAVE:     Write HDF5 files for BioPM

=== USAGE ===

  python starter_project/preprocessing_template.py \
      --input  /path/to/your/raw_data.csv \
      --output /path/to/output_dir \
      --subject_id 1
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import h5py
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.preprocessing import (
    resample_to_target_fs,
    bandpass_filter,
    lowpass_filter,
    detect_zero_crossings,
    assign_zero_crossings,
)

# ===================================================================
# STEP 1: Define your dataset parameters
# ===================================================================

# YOUR DATASET PARAMETERS:
ORIGINAL_SAMPLE_RATE = 50    # Hz — your data's original sample rate
TARGET_SAMPLE_RATE = 30      # Hz — BioPM expects 30 Hz
WINDOW_SEC = 10              # seconds per window
SLIDE_SEC = 5                # seconds overlap
PAD_SIZE = 192               # max movement elements per 10s window
NORMALIZE_SIZE = 32          # normalized ME length (do not change)
SKIP_LABELS = {0}            # labels to skip (e.g., null/unlabeled)


# ===================================================================
# STEP 2: Load your raw data
# ===================================================================
def load_my_data(file_path):
    """
    EDIT THIS FUNCTION for your data format.

    Must return:
        acc:    (N, 3) array — 3-axis acceleration in g (not m/s²!)
        labels: (N,) array  — integer label per sample
        time:   (N,) array  — timestamps in seconds
    """
    # Example: CSV with columns time, acc_x, acc_y, acc_z, label
    df = pd.read_csv(file_path)

    # EDIT: select your acceleration columns
    acc = df[['acc_x', 'acc_y', 'acc_z']].values

    # EDIT: convert units if needed
    # If your data is in m/s², uncomment:
    # acc = acc / 9.80665

    # EDIT: select your label column
    labels = df['label'].values.astype(int)

    # EDIT: timestamps — if you have a timestamp column:
    # time = df['timestamp'].values
    # If not, create synthetic timestamps:
    time = np.arange(len(acc)) / ORIGINAL_SAMPLE_RATE

    # Clean NaN values
    acc_df = pd.DataFrame(acc, columns=['x', 'y', 'z'])
    acc_df = acc_df.apply(pd.to_numeric, errors='coerce').interpolate()
    acc = acc_df.values

    return acc, labels, time


# ===================================================================
# STEP 3: Remap your labels
# ===================================================================
def remap_my_labels(labels):
    """
    EDIT THIS FUNCTION for your label mapping.

    Map raw labels → contiguous 0,1,2,...,K-1
    """
    unique = sorted(set(int(l) for l in np.unique(labels)) - SKIP_LABELS)
    mapping = {old: new for new, old in enumerate(unique)}
    return mapping


# ===================================================================
# STEP 4: Run the full pipeline
# ===================================================================
def preprocess(file_path, subject_id, output_dir):
    print(f"Loading {file_path} ...")
    acc, labels, time = load_my_data(file_path)
    print(f"  Raw data: {acc.shape}, {ORIGINAL_SAMPLE_RATE} Hz")

    # Resample
    acc_res, time_res, labels_res = resample_to_target_fs(
        time, acc, labels, TARGET_SAMPLE_RATE)
    print(f"  Resampled: {acc_res.shape}, {TARGET_SAMPLE_RATE} Hz")

    # Filter
    acc_filt = bandpass_filter(acc_res, 0.5, 12, TARGET_SAMPLE_RATE, order=6)
    acc_grav = lowpass_filter(acc_res, 0.5, TARGET_SAMPLE_RATE, order=6)

    # Window
    ws = int(WINDOW_SEC * TARGET_SAMPLE_RATE)
    step = int(SLIDE_SEC * TARGET_SAMPLE_RATE)
    label_map = remap_my_labels(labels_res)

    config = {
        'target_FS': TARGET_SAMPLE_RATE,
        'normalize_size_target': NORMALIZE_SIZE,
        'normalize_size_assign': NORMALIZE_SIZE,
        'pad_size': PAD_SIZE,
        'WS': WINDOW_SEC,
    }

    win_raw, win_x_acc, win_x_grav, win_labels = [], [], [], []
    start = 0
    n_skipped = 0

    while start + ws < acc_filt.shape[0]:
        w_labels = labels_res[start:start + ws]
        try:
            mode_label = statistics.mode(w_labels.astype(int))
        except Exception:
            start += step
            continue

        if mode_label in SKIP_LABELS:
            start += step
            n_skipped += 1
            continue

        w_filt = acc_filt[start:start + ws]
        w_grav = acc_grav[start:start + ws]
        w_raw = acc_res[start:start + ws]
        w_time = time_res[start:start + ws]

        try:
            (_, _, me_list, me_norm, me_info, _, _,
             pos_info, zc_list, zc_time) = detect_zero_crossings(
                w_filt, w_time, config)
        except Exception:
            start += step
            n_skipped += 1
            continue

        if len(me_list) == 0:
            start += step
            n_skipped += 1
            continue

        # Build x_acc_filt
        x_acc = np.concatenate([
            me_norm,
            pos_info.reshape(-1, 1),
            me_info[['axis', 'len', 'min', 'max', 'dirct']].values,
        ], axis=1)

        if x_acc.shape[0] < PAD_SIZE:
            pad = np.full((PAD_SIZE - x_acc.shape[0], x_acc.shape[1]), np.nan)
            x_acc = np.vstack([x_acc, pad])
        else:
            x_acc = x_acc[:PAD_SIZE]

        mapped = label_map.get(mode_label, mode_label)
        win_raw.append(w_raw.astype(np.float32))
        win_x_acc.append(x_acc.astype(np.float32))
        win_x_grav.append(w_grav.astype(np.float32))
        win_labels.append(float(mapped))
        start += step

    print(f"  Windows: {len(win_labels)} valid, {n_skipped} skipped")

    if not win_labels:
        print("  WARNING: no valid windows produced!")
        return

    # Save
    os.makedirs(output_dir, exist_ok=True)
    h5_path = os.path.join(output_dir, f"Data_MeLabel_{subject_id}.h5")
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("window_acc_raw",
                         data=np.array(win_raw, dtype=np.float32))
        f.create_dataset("x_acc_filt",
                         data=np.array(win_x_acc, dtype=np.float32))
        f.create_dataset("x_gravity",
                         data=np.array(win_x_grav, dtype=np.float32))
        f.create_dataset("window_label",
                         data=np.array(win_labels, dtype=np.float32))

    # Also save Data_AccLabel for completeness
    h5_acc_path = os.path.join(output_dir, f"Data_AccLabel_{subject_id}.h5")
    with h5py.File(h5_acc_path, "w") as f:
        f.create_dataset("window_acc_raw",
                         data=np.array(win_raw, dtype=np.float32))
        f.create_dataset("window_label",
                         data=np.array(win_labels, dtype=np.float32))

    print(f"  Saved: {h5_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True, help="Raw data file")
    p.add_argument("--output", type=str, required=True, help="Output directory")
    p.add_argument("--subject_id", type=int, default=1, help="Subject ID")
    args = p.parse_args()

    preprocess(args.input, args.subject_id, args.output)
    print("\nDone! Next step: run extract_features.py on the output directory.")


if __name__ == "__main__":
    main()
