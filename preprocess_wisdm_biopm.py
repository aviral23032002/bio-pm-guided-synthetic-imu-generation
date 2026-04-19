#!/usr/bin/env python3
"""
preprocess_wisdm_biopm.py — Convert WISDM v1.1 raw data to Bio-PM HDF5 format.

WISDM specifics:
  - Sample rate: 20 Hz
  - Units: m/s²  (must divide by 9.80665 → g)
  - Labels: string activity names (Walking, Jogging, Upstairs, Downstairs, Sitting, Standing)
  - All subjects in one file; z column has trailing semicolons
  - 36 subjects (IDs 1–36)

Output (per subject):
  preprocessed_biopm/
    Data_MeLabel_{subject_id}.h5
      x_acc_filt   (W, 192, 38)   — movement-element patches + metadata
      x_gravity    (W, 300, 3)    — gravity signal
      window_acc_raw (W, 300, 3)  — raw acceleration
      window_label   (W,)         — integer label (0-5)

    Data_AccLabel_{subject_id}.h5
      window_acc_raw          (W, 300, 3)
      window_acc_filt_gravity (W, 300, 6)
      window_label            (W,)

Usage:
    python preprocess_wisdm_biopm.py \
        --raw  WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt \
        --out  preprocessed_biopm
"""

import os
import sys
import argparse
import statistics
import warnings
import numpy as np
import pandas as pd
import h5py

warnings.filterwarnings('ignore')

# ── Add CS690TR to path so we can import Bio-PM preprocessing utilities ──────
BIOPM_DIR = os.path.join(os.path.dirname(__file__), "CS690TR")
sys.path.insert(0, BIOPM_DIR)

from src.data.preprocessing import (
    resample_to_target_fs,
    bandpass_filter,
    lowpass_filter,
    detect_zero_crossings,
    assign_zero_crossings,
)

# ── WISDM constants ───────────────────────────────────────────────────────────
ORI_FS       = 20       # WISDM sample rate (Hz)
TARGET_FS    = 30       # Bio-PM target rate (Hz)
WINDOW_SEC   = 10       # seconds per window
SLIDE_SEC    = 5        # hop size (50% overlap)
PAD_SIZE     = 192      # max ME patches per window  (= WS * 192/10)
NORM_SIZE    = 32       # normalised ME length (fixed by Bio-PM)
HIGH_F1      = 12.0     # bandpass upper cutoff (Hz)
LOW_F1       = 0.5      # bandpass lower / lowpass cutoff (Hz)
FILTER_ORDER = 6

# Label mapping: string → int
ACTIVITY_MAP = {
    'Walking':    0,
    'Jogging':    1,
    'Upstairs':   2,
    'Downstairs': 3,
    'Sitting':    4,
    'Standing':   5,
}

CONFIG = {
    'target_FS':              TARGET_FS,
    'WS':                     WINDOW_SEC,
    'pad_size':               PAD_SIZE,
    'normalize_size_target':  NORM_SIZE,
    'normalize_size_assign':  NORM_SIZE,
}


def load_wisdm(raw_path: str) -> pd.DataFrame:
    """Load and clean the WISDM raw text file."""
    print(f"Loading {raw_path} ...")
    columns = ['user', 'activity', 'timestamp', 'x', 'y', 'z']
    df = pd.read_csv(raw_path, header=None, names=columns, on_bad_lines='skip')

    # Strip trailing semicolons from z column
    df['z'] = df['z'].astype(str).str.replace(';', '', regex=False)

    # Convert sensor columns to numeric, drop bad rows
    for col in ['x', 'y', 'z']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Convert m/s² → g
    for col in ['x', 'y', 'z']:
        df[col] = df[col] / 9.80665

    # Map activity strings → integers
    df['label_int'] = df['activity'].map(ACTIVITY_MAP)
    df.dropna(subset=['label_int'], inplace=True)
    df['label_int'] = df['label_int'].astype(int)

    print(f"  Total samples: {len(df):,}")
    print(f"  Subjects: {sorted(df['user'].unique())}")
    return df


def class_balance_report(df: pd.DataFrame):
    """Print and flag class distribution."""
    print("\n── Class Balance Audit ──────────────────────────────────────────")
    counts = df['activity'].value_counts()
    total  = len(df)
    print(f"{'Activity':<15} {'Samples':>10} {'%':>8}")
    print("-" * 38)
    minority = []
    for act, cnt in counts.items():
        pct = 100 * cnt / total
        flag = " ⚠ MINORITY" if pct < 10 else ""
        print(f"{act:<15} {cnt:>10,} {pct:>7.2f}%{flag}")
        if pct < 10:
            minority.append(act)
    print(f"\n[FLAG] Minority classes (<10%): {minority}")
    print("─" * 38)


def preprocess_subject(subj_df: pd.DataFrame, subject_id: int,
                        output_dir: str, config: dict):
    """Run full Bio-PM preprocessing pipeline for one subject."""
    # Sort by timestamp to ensure chronological order
    subj_df = subj_df.sort_values('timestamp').reset_index(drop=True)

    acc_raw = subj_df[['x', 'y', 'z']].values.astype(np.float64)
    labels  = subj_df['label_int'].values.astype(np.float64)
    time_arr = np.arange(len(acc_raw)) / ORI_FS  # synthetic timestamps

    # ── Resample 20 Hz → 30 Hz ───────────────────────────────────────────────
    acc_res, time_res, labels_res = resample_to_target_fs(
        time_arr, acc_raw, labels, TARGET_FS)

    # ── Filter ───────────────────────────────────────────────────────────────
    acc_filt = bandpass_filter(acc_res, LOW_F1, HIGH_F1, TARGET_FS,
                               order=FILTER_ORDER)
    acc_grav = lowpass_filter(acc_res, LOW_F1, TARGET_FS, order=FILTER_ORDER)

    # ── Sliding windows ───────────────────────────────────────────────────────
    ws   = int(WINDOW_SEC * TARGET_FS)   # 300 samples
    step = int(SLIDE_SEC  * TARGET_FS)   # 150 samples

    win_acc_raw, win_x_acc, win_x_grav, win_labels = [], [], [], []
    win_acc_filt_grav = []
    n_skipped = 0

    start = 0
    while start + ws < acc_filt.shape[0]:
        w_labels = labels_res[start:start + ws]
        try:
            mode_label = int(statistics.mode(w_labels.astype(int)))
        except Exception:
            start += step
            n_skipped += 1
            continue

        w_raw  = acc_res[start:start + ws]
        w_filt = acc_filt[start:start + ws]
        w_grav = acc_grav[start:start + ws]
        w_time = time_res[start:start + ws]

        # ── Zero-crossing movement-element extraction ──────────────────────
        try:
            (_, _, me_list, me_norm, me_info, _, _,
             pos_info, zc_list, zc_time_list) = detect_zero_crossings(
                w_filt, w_time, config)

            (_, _, _, grav_norm, grav_info,
             _, _, _) = assign_zero_crossings(
                w_grav, w_time, zc_list, zc_time_list, config)
        except Exception:
            start += step
            n_skipped += 1
            continue

        if len(me_list) == 0:
            start += step
            n_skipped += 1
            continue

        # ── Build x_acc_filt: [ME_norm(32) | pos(1) | axis,len,min,max,dirct(5)] ─
        x_acc = np.concatenate([
            me_norm,
            pos_info.reshape(-1, 1),
            me_info[['axis', 'len', 'min', 'max', 'dirct']].values,
        ], axis=1)  # shape: (n_me, 38)

        # Pad / truncate to PAD_SIZE
        if x_acc.shape[0] < PAD_SIZE:
            pad = np.full((PAD_SIZE - x_acc.shape[0], x_acc.shape[1]), np.nan)
            x_acc = np.vstack([x_acc, pad])
        else:
            x_acc = x_acc[:PAD_SIZE]

        win_acc_raw.append(w_raw.astype(np.float32))
        win_x_acc.append(x_acc.astype(np.float32))
        win_x_grav.append(w_grav.astype(np.float32))
        win_labels.append(float(mode_label))
        win_acc_filt_grav.append(
            np.concatenate([w_filt, w_grav], axis=1).astype(np.float32))
        start += step

    print(f"  Subject {subject_id}: {len(win_labels)} windows, {n_skipped} skipped")

    if not win_labels:
        print(f"  WARNING: no valid windows for subject {subject_id} — skipping")
        return

    # ── Stack and save ────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    arr_raw      = np.array(win_acc_raw,      dtype=np.float32)
    arr_x_acc    = np.array(win_x_acc,        dtype=np.float32)
    arr_x_grav   = np.array(win_x_grav,       dtype=np.float32)
    arr_labels   = np.array(win_labels,        dtype=np.float32)
    arr_filtgrav = np.array(win_acc_filt_grav, dtype=np.float32)

    # Data_MeLabel — needed by extract_features.py
    me_path = os.path.join(output_dir, f"Data_MeLabel_{subject_id}.h5")
    with h5py.File(me_path, "w") as f:
        f.create_dataset("window_acc_raw", data=arr_raw)
        f.create_dataset("x_acc_filt",     data=arr_x_acc)
        f.create_dataset("x_gravity",      data=arr_x_grav)
        f.create_dataset("window_label",   data=arr_labels)

    # Data_AccLabel — for completeness / alternative pipelines
    acc_path = os.path.join(output_dir, f"Data_AccLabel_{subject_id}.h5")
    with h5py.File(acc_path, "w") as f:
        f.create_dataset("window_acc_raw",          data=arr_raw)
        f.create_dataset("window_acc_filt_gravity", data=arr_filtgrav)
        f.create_dataset("window_label",            data=arr_labels)

    print(f"  → Saved: {me_path}  (x_acc_filt {arr_x_acc.shape}, "
          f"x_gravity {arr_x_grav.shape})")


def main():
    p = argparse.ArgumentParser(description="WISDM → Bio-PM HDF5 preprocessing")
    p.add_argument("--raw", type=str,
                   default="WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt",
                   help="Path to WISDM_ar_v1.1_raw.txt")
    p.add_argument("--out", type=str,
                   default="preprocessed_biopm",
                   help="Output directory for per-subject HDF5 files")
    p.add_argument("--subjects", type=str, default="all",
                   help="Comma-separated subject IDs to process, or 'all'")
    args = p.parse_args()

    df = load_wisdm(args.raw)
    class_balance_report(df)

    # Determine subjects to process
    all_subjects = sorted(df['user'].unique())
    if args.subjects == "all":
        subjects = all_subjects
    else:
        subjects = [int(s) for s in args.subjects.split(",")]

    print(f"\n── Preprocessing {len(subjects)} subjects ───────────────────────")
    for subj_id in subjects:
        subj_df = df[df['user'] == subj_id].copy()
        if len(subj_df) < 1000:
            print(f"  Subject {subj_id}: too few samples ({len(subj_df)}), skipping")
            continue
        preprocess_subject(subj_df, subj_id, args.out, CONFIG)

    print(f"\n✅ Done! Bio-PM HDF5 files saved to: {args.out}/")
    print("\nNext steps:")
    print("  # Extract Bio-PM features:")
    print(f"  python CS690TR/scripts/extract_features.py \\")
    print(f"      --data_dir {args.out} \\")
    print(f"      --checkpoint CS690TR/checkpoints/checkpoint.pt \\")
    print(f"      --output features/biopm_features.npz")


if __name__ == "__main__":
    main()
