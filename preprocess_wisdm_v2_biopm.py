#!/usr/bin/env python3
"""
preprocess_wisdm_v2_biopm.py — Convert WISDM v2 (18 activities) raw data to Bio-PM HDF5 format.

WISDM v2 specifics:
  - Sample rate: 20 Hz
  - Units: m/s² (must divide by 9.80665 → g)
  - Labels: single characters (A-S, skipping N)
  - Files: per-subject files in raw/watch/accel/
  - Format: Subject-id, Activity Label, Timestamp, x, y, z;

Output (per subject):
  preprocessed_wisdm_v2/
    Data_MeLabel_{subject_id}.h5
    Data_AccLabel_{subject_id}.h5
"""

import os
import sys
import argparse
import statistics
import warnings
import numpy as np
import pandas as pd
import h5py
from glob import glob

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

# ── WISDM v2 constants ────────────────────────────────────────────────────────
ORI_FS       = 20       # WISDM sample rate (Hz)
TARGET_FS    = 30       # Bio-PM target rate (Hz)
WINDOW_SEC   = 10       # seconds per window
SLIDE_SEC    = 5        # hop size (50% overlap)
PAD_SIZE     = 192      # max ME patches per window
NORM_SIZE    = 32       # normalised ME length
HIGH_F1      = 12.0     # bandpass upper cutoff (Hz)
LOW_F1       = 0.5      # bandpass lower / lowpass cutoff (Hz)
FILTER_ORDER = 6

# Activity mapping (Mobility only: A-E) to match Week 1 IDs
ACTIVITY_MAP = {
    'A': 0,  # walking
    'B': 1,  # jogging
    'C': 2,  # upstairs
    'D': 3,  # downstairs
    'E': 4,  # sitting
    'F': 5,  # standing
}

CONFIG = {
    'target_FS':              TARGET_FS,
    'WS':                     WINDOW_SEC,
    'pad_size':               PAD_SIZE,
    'normalize_size_target':  NORM_SIZE,
    'normalize_size_assign':  NORM_SIZE,
}

def load_wisdm_file(file_path: str) -> pd.DataFrame:
    """Load and clean a single WISDM v2 raw text file."""
    columns = ['user', 'activity', 'timestamp', 'x', 'y', 'z']
    df = pd.read_csv(file_path, header=None, names=columns, on_bad_lines='skip')
    
    # Strip trailing semicolons from z column
    df['z'] = df['z'].astype(str).str.replace(';', '', regex=False)
    
    # Convert sensor columns to numeric
    for col in ['x', 'y', 'z']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    
    # Convert m/s² → g
    for col in ['x', 'y', 'z']:
        df[col] = df[col] / 9.80665
        
    # Map activity chars → integers
    df['label_int'] = df['activity'].map(ACTIVITY_MAP)
    df.dropna(subset=['label_int'], inplace=True)
    df['label_int'] = df['label_int'].astype(int)
    
    return df

def preprocess_subject(subj_df: pd.DataFrame, subject_id: int, output_dir: str, config: dict):
    """Run full Bio-PM preprocessing pipeline for one subject."""
    subj_df = subj_df.sort_values('timestamp').reset_index(drop=True)
    
    acc_raw = subj_df[['x', 'y', 'z']].values.astype(np.float64)
    labels  = subj_df['label_int'].values.astype(np.float64)
    time_arr = np.arange(len(acc_raw)) / ORI_FS
    
    acc_res, time_res, labels_res = resample_to_target_fs(time_arr, acc_raw, labels, TARGET_FS)
    
    acc_filt = bandpass_filter(acc_res, LOW_F1, HIGH_F1, TARGET_FS, order=FILTER_ORDER)
    acc_grav = lowpass_filter(acc_res, LOW_F1, TARGET_FS, order=FILTER_ORDER)
    
    ws   = int(WINDOW_SEC * TARGET_FS)
    step = int(SLIDE_SEC  * TARGET_FS)
    
    win_acc_raw, win_x_acc, win_x_grav, win_labels = [], [], [], []
    win_acc_filt_grav = []
    
    start = 0
    while start + ws < acc_filt.shape[0]:
        w_labels = labels_res[start:start + ws]
        try:
            mode_label = int(statistics.mode(w_labels.astype(int)))
        except Exception:
            start += step
            continue
            
        w_raw  = acc_res[start:start + ws]
        w_filt = acc_filt[start:start + ws]
        w_grav = acc_grav[start:start + ws]
        w_time = time_res[start:start + ws]
        
        try:
            (_, _, me_list, me_norm, me_info, _, _, pos_info, zc_list, zc_time_list) = detect_zero_crossings(w_filt, w_time, config)
            (_, _, _, grav_norm, grav_info, _, _, _) = assign_zero_crossings(w_grav, w_time, zc_list, zc_time_list, config)
        except Exception:
            start += step
            continue
            
        if len(me_list) == 0:
            start += step
            continue
            
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
            
        win_acc_raw.append(w_raw.astype(np.float32))
        win_x_acc.append(x_acc.astype(np.float32))
        win_x_grav.append(w_grav.astype(np.float32))
        win_labels.append(float(mode_label))
        win_acc_filt_grav.append(np.concatenate([w_filt, w_grav], axis=1).astype(np.float32))
        start += step
        
    if not win_labels:
        return
        
    os.makedirs(output_dir, exist_ok=True)
    arr_raw      = np.array(win_acc_raw,      dtype=np.float32)
    arr_x_acc    = np.array(win_x_acc,        dtype=np.float32)
    arr_x_grav   = np.array(win_x_grav,       dtype=np.float32)
    arr_labels   = np.array(win_labels,        dtype=np.float32)
    arr_filtgrav = np.array(win_acc_filt_grav, dtype=np.float32)
    
    me_path = os.path.join(output_dir, f"Data_MeLabel_{subject_id}.h5")
    with h5py.File(me_path, "w") as f:
        f.create_dataset("window_acc_raw", data=arr_raw)
        f.create_dataset("x_acc_filt",     data=arr_x_acc)
        f.create_dataset("x_gravity",      data=arr_x_grav)
        f.create_dataset("window_label",   data=arr_labels)
        
    acc_path = os.path.join(output_dir, f"Data_AccLabel_{subject_id}.h5")
    with h5py.File(acc_path, "w") as f:
        f.create_dataset("window_acc_raw",          data=arr_raw)
        f.create_dataset("window_acc_filt_gravity", data=arr_filtgrav)
        f.create_dataset("window_label",            data=arr_labels)
        
    print(f"  Subject {subject_id}: {len(win_labels)} windows saved.")

def main():
    p = argparse.ArgumentParser(description="WISDM v2 → Bio-PM HDF5 preprocessing")
    p.add_argument("--raw_dir", type=str,
                   default="/Users/danny/Downloads/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset/wisdm-dataset/raw/watch/accel",
                   help="Directory containing WISDM v2 raw watch accel files")
    p.add_argument("--out", type=str,
                   default="preprocessed_wisdm_v2",
                   help="Output directory")
    args = p.parse_args()
    
    files = sorted(glob(os.path.join(args.raw_dir, "data_*_accel_watch.txt")))
    print(f"Found {len(files)} subject files.")
    
    for f in files:
        subject_id = os.path.basename(f).split("_")[1]
        print(f"Processing subject {subject_id}...")
        df = load_wisdm_file(f)
        preprocess_subject(df, subject_id, args.out, CONFIG)
        
    print(f"\n✅ Done! Files saved to: {args.out}/")

if __name__ == "__main__":
    main()
