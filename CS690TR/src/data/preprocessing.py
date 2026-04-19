"""
Preprocessing utilities for BioPM.

This module provides the signal-processing functions needed to convert raw
3-axis accelerometer data into the movement-element (ME) representation
that BioPM expects.

Pipeline overview (using mHealth as the reference):
  1. Resample to target sample rate (default 30 Hz)
  2. Bandpass filter → body-acceleration (movement) signal
  3. Lowpass filter  → gravity signal
  4. Sliding window  → fixed-length windows (e.g. 10 s)
  5. Zero-crossing detection per axis → movement elements
  6. Normalize each ME to fixed length (32 samples)
  7. Pack into (x_acc_filt, x_gravity, window_label) and save as HDF5

Students: the functions in this file are *general-purpose*.  The
dataset-specific choices (column indices, label mapping, file format)
live in the scripts that call these functions.
"""

import os
import re
import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.optimize import brentq
from scipy.signal import butter, filtfilt, find_peaks


# ===================================================================
# 1. Resampling
# ===================================================================

def resample_to_target_fs(time_array, acc_raw, label, target_fs):
    """
    Linearly resample 3-axis accelerometer + nearest-neighbour resample
    labels to a new sample rate.

    Args:
        time_array: (N,) timestamps in seconds
        acc_raw:    (N, 3) raw acceleration
        label:      (N,) integer labels
        target_fs:  desired output sample rate in Hz

    Returns:
        acc_resampled:  (M, 3)
        time_resampled: (M,)
        label_resampled:(M,)
    """
    t_new = np.linspace(time_array[0], time_array[-1],
                        int(time_array[-1] * target_fs))
    axes = []
    for i in range(acc_raw.shape[1]):
        f = interp1d(time_array, acc_raw[:, i], kind="linear")
        axes.append(f(t_new).reshape(-1, 1))
    acc_resampled = np.concatenate(axes, axis=1)
    label_resampled = interp1d(time_array, label, kind="nearest")(t_new)
    return acc_resampled, t_new, label_resampled


# ===================================================================
# 2. Filtering
# ===================================================================

def bandpass_filter(data, low_hz, high_hz, fs, order=6):
    """Butterworth bandpass filter applied along axis 0."""
    nyq = fs / 2.0
    b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype='band')
    return filtfilt(b, a, data, axis=0)


def lowpass_filter(data, cutoff_hz, fs, order=6):
    """Butterworth lowpass filter (extracts gravity component)."""
    nyq = fs / 2.0
    b, a = butter(order, cutoff_hz / nyq, btype='low')
    return filtfilt(b, a, data, axis=0)


def highpass_filter(data, cutoff_hz, fs, order=6):
    """Butterworth highpass filter."""
    nyq = fs / 2.0
    b, a = butter(order, cutoff_hz / nyq, btype='high')
    return filtfilt(b, a, data, axis=0)


# ===================================================================
# 3. Zero-crossing-based movement element extraction
# ===================================================================

def detect_zero_crossings(vel, time_index, config):
    """
    Detect zero-crossings on each axis of the filtered acceleration signal
    and extract normalised movement elements (MEs).

    Args:
        vel:        (T, 3) bandpass-filtered acceleration window
        time_index: (T,) timestamps for that window
        config:     dict with keys:
                      target_FS, normalize_size_target, pad_size, WS

    Returns:
        Tuple of:
          resampled_vel, time_index,
          me_list,                          raw ME segments (list)
          me_normalize_list,                (n_me, normalize_size) normalised
          me_normalizeInfo_list,            DataFrame of metadata per ME
          me_normalize_padding_list,        zero-padded version
          me_normalizeInfo_padding_list,    padded metadata
          pos_info,                         (n_me,) fractional positions
          zero_crossings_list,              per-axis crossing indices
          zero_crossings_time_list          per-axis crossing times
    """
    me_list = []
    me_normalize_list = np.empty((0, config['normalize_size_target']))
    me_normalizeInfo_list = pd.DataFrame([])
    resampled_vel = vel.copy()
    pos_info = []
    zero_crossings_list = []
    zero_crossings_time_list = []

    for i in range(vel.shape[1]):
        c_axis = i
        spline = UnivariateSpline(time_index, vel[:, i], s=0)

        zc_time = []
        for ii in range(len(time_index) - 1):
            if spline(time_index[ii]) * spline(time_index[ii + 1]) < 0:
                root = brentq(spline, time_index[ii], time_index[ii + 1])
                zc_time.append(root)

        zc_idx = []
        for t in zc_time:
            idx = np.searchsorted(time_index, t) - 1
            if 0 <= idx < len(time_index) - 1:
                zc_idx.append(idx)
        zc_idx = np.array(zc_idx)

        # Remove crossings too close together
        filt_zc = [zc_idx[0]]
        filt_zc_time = [zc_time[0]]
        for aa in range(1, len(zc_idx)):
            if zc_idx[aa] - filt_zc[-1] > round(config['target_FS'] * 0.05):
                filt_zc.append(zc_idx[aa])
                filt_zc_time.append(zc_time[aa])

        # Merge small-amplitude MEs
        amp_mv = []
        for ii in range(len(filt_zc) - 1):
            c_vel = vel[filt_zc[ii]:filt_zc[ii + 1] + 2, i]
            amp_mv.append(max(c_vel))
        amp_mv = np.array(amp_mv)
        mask = amp_mv < 0.01
        indices = np.where(mask)[0]
        if len(indices) > 0:
            groups = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
            drop_groups = [g[1:] for g in groups if len(g) > 1]
            if drop_groups:
                drop_indices = np.concatenate(drop_groups)
                filt_zc = np.delete(filt_zc, drop_indices)
                filt_zc_time = np.delete(filt_zc_time, drop_indices)
            else:
                filt_zc = np.array(filt_zc)
                filt_zc_time = np.array(filt_zc_time)
        else:
            filt_zc = np.array(filt_zc)
            filt_zc_time = np.array(filt_zc_time)

        zero_crossings = filt_zc
        zero_crossings_time = filt_zc_time
        zero_crossings_list.append(zero_crossings)
        zero_crossings_time_list.append(zero_crossings_time)

        for ii in range(zero_crossings[:-1].shape[0]):
            c_vel = vel[zero_crossings[ii]:zero_crossings[ii + 1] + 2, i]
            start_time = zero_crossings_time[ii]
            end_time = zero_crossings_time[ii + 1]
            num_samples = len(c_vel)
            resampled_t = np.linspace(start_time, end_time, num_samples)
            resampled_c_vel = spline(resampled_t)
            resampled_vel[zero_crossings[ii] + 1:zero_crossings[ii + 1] + 1, i] = resampled_c_vel[1:-1]

            resampled_t2 = np.linspace(start_time, end_time,
                                       config['normalize_size_target'])
            resampled_c_vel2 = spline(resampled_t2)

            direct_val = 1 if np.mean(resampled_c_vel2) > 0 else -1
            resampled_c_vel3 = resampled_c_vel2 * (1 if direct_val == 1 else -1)

            min_val = resampled_c_vel3.min()
            max_val = resampled_c_vel3.max()
            ori_len = len(c_vel)
            norm_vel = (resampled_c_vel3 - min_val) / (max_val - min_val + 1e-10)
            peaks, _ = find_peaks(norm_vel, height=0, prominence=0.3)

            c_pos = ((zero_crossings[ii] + zero_crossings[ii + 1]) / 2) / \
                    (config['WS'] * config['target_FS'])
            pos_info.append(c_pos)

            me_list.append(resampled_c_vel)
            me_normalize_list = np.concatenate(
                (me_normalize_list, resampled_c_vel2.reshape(1, -1)))

            row = pd.DataFrame(
                [[c_axis, zero_crossings[ii], zero_crossings[ii + 1],
                  ori_len, min_val, max_val, direct_val, len(peaks)]],
                columns=['axis', 'start_point', 'end_point', 'len',
                         'min', 'max', 'dirct', 'peaks'])
            me_normalizeInfo_list = pd.concat(
                [me_normalizeInfo_list, row], axis=0)

    if len(me_list) > config['pad_size']:
        me_list = me_list[:config['pad_size']]
        me_normalize_list = me_normalize_list[:config['pad_size']]
        me_normalizeInfo_list = me_normalizeInfo_list[:config['pad_size']]
        pos_info = pos_info[:config['pad_size']]

    pad_rows = config['pad_size'] - me_normalize_list.shape[0]
    pad_data = np.full((pad_rows, me_normalize_list.shape[1]), -100.0)
    me_normalize_padding = np.vstack((me_normalize_list, pad_data))

    pad_info = pd.DataFrame(
        np.full((pad_rows, me_normalizeInfo_list.shape[1]), -100.0),
        columns=me_normalizeInfo_list.columns)
    me_normalizeInfo_padding = pd.concat(
        [me_normalizeInfo_list, pad_info], ignore_index=True)

    return (resampled_vel, time_index, me_list, me_normalize_list,
            me_normalizeInfo_list, me_normalize_padding,
            me_normalizeInfo_padding, np.array(pos_info),
            zero_crossings_list, zero_crossings_time_list)


def assign_zero_crossings(vel, time_index, zero_crossings_list,
                          zero_crossings_time_list, config):
    """
    Re-use zero-crossing positions from the body-acceleration signal to
    segment the gravity signal into matching movement elements.

    Same interface as detect_zero_crossings but takes pre-computed crossings.
    """
    me_list = []
    nsize = config.get('normalize_size_assign', config['normalize_size_target'])
    me_normalize_list = np.empty((0, nsize))
    me_normalizeInfo_list = pd.DataFrame([])
    resampled_vel = vel.copy()
    pos_info = []

    for i in range(vel.shape[1]):
        c_axis = i
        spline = UnivariateSpline(time_index, vel[:, i], s=0)
        zero_crossings = zero_crossings_list[i]
        zero_crossings_time = zero_crossings_time_list[i]

        for ii in range(zero_crossings[:-1].shape[0]):
            c_vel = vel[zero_crossings[ii]:zero_crossings[ii + 1] + 2, i]
            start_time = zero_crossings_time[ii]
            end_time = zero_crossings_time[ii + 1]
            num_samples = len(c_vel)
            resampled_t = np.linspace(start_time, end_time, num_samples)
            resampled_c_vel = spline(resampled_t)
            resampled_vel[zero_crossings[ii] + 1:zero_crossings[ii + 1] + 1, i] = \
                resampled_c_vel[1:-1]

            resampled_t2 = np.linspace(start_time, end_time, nsize)
            resampled_c_vel2 = spline(resampled_t2)

            direct_val = 1 if np.mean(resampled_c_vel2) > 0 else -1
            resampled_c_vel3 = resampled_c_vel2 * (1 if direct_val == 1 else -1)
            min_val = resampled_c_vel3.min()
            max_val = resampled_c_vel3.max()
            ori_len = len(c_vel)
            norm_vel = (resampled_c_vel3 - min_val) / (max_val - min_val + 1e-10)
            peaks, _ = find_peaks(norm_vel, height=0, prominence=0.3)

            c_pos = ((zero_crossings[ii] + zero_crossings[ii + 1]) / 2) / \
                    (config['WS'] * config['target_FS'])
            pos_info.append(c_pos)
            me_list.append(resampled_c_vel)
            me_normalize_list = np.concatenate(
                (me_normalize_list, resampled_c_vel2.reshape(1, -1)))
            row = pd.DataFrame(
                [[c_axis + 2, zero_crossings[ii], zero_crossings[ii + 1],
                  ori_len, min_val, max_val, len(peaks)]],
                columns=['axis', 'start_point', 'end_point', 'len',
                         'min', 'max', 'peaks'])
            me_normalizeInfo_list = pd.concat(
                [me_normalizeInfo_list, row], axis=0)

    if len(me_list) > config['pad_size']:
        me_list = me_list[:config['pad_size']]
        me_normalize_list = me_normalize_list[:config['pad_size']]
        me_normalizeInfo_list = me_normalizeInfo_list[:config['pad_size']]
        pos_info = pos_info[:config['pad_size']]

    pad_rows = config['pad_size'] - me_normalize_list.shape[0]
    pad_data = np.full((pad_rows, me_normalize_list.shape[1]), -100.0)
    me_normalize_padding = np.vstack((me_normalize_list, pad_data))

    pad_info = pd.DataFrame(
        np.full((pad_rows, me_normalizeInfo_list.shape[1]), -100.0),
        columns=me_normalizeInfo_list.columns)
    me_normalizeInfo_padding = pd.concat(
        [me_normalizeInfo_list, pad_info], ignore_index=True)

    return (resampled_vel, time_index, me_list, me_normalize_list,
            me_normalizeInfo_list, me_normalize_padding,
            me_normalizeInfo_padding, np.array(pos_info))


# ===================================================================
# 4. Loading preprocessed HDF5 data
# ===================================================================

def load_preprocessed_h5(data_root):
    """
    Load all preprocessed Data_MeLabel_*.h5 files from a directory.

    This is what the model expects at inference time.  The HDF5 files
    are produced by the preprocessing scripts.

    Returns:
        acc_filt:              (N, L, 32) normalised ME patches
        pos_info:              (N, L) fractional patch positions
        additional_embedding:  (N, L, K) axis + duration + metadata
        labels:                (N,) integer labels
        subject_ids:           (N,) subject identifiers
        gravity:               (N, T, 3) gravity windows (or None)
        raw_acc:               (N, T_raw, 3) raw accelerometer windows
    """
    pattern = re.compile(r'Data_MeLabel_.*\.h5')
    matched = []
    for root, _, fnames in os.walk(data_root):
        for f in fnames:
            if pattern.match(f):
                matched.append(os.path.join(root, f))
    if not matched:
        raise FileNotFoundError(f"No Data_MeLabel_*.h5 files in {data_root}")

    all_acc, all_label, all_pid, all_grav, all_raw = [], [], [], [], []
    for path in sorted(matched):
        parts = path.replace('.h5', '').split('_')
        subject_id = int(parts[-1])
        with h5py.File(path, "r") as hf:
            all_acc.append(np.array(hf['x_acc_filt']))
            all_raw.append(np.array(hf['window_acc_raw']))
            all_label.append(np.array(hf['window_label']))
            if 'gravity_window_40hz' in hf:
                all_grav.append(np.array(hf['gravity_window_40hz']))
            elif 'x_gravity' in hf:
                all_grav.append(np.array(hf['x_gravity']))
        n = len(all_label[-1])
        all_pid.append(np.full(n, subject_id))

    all_acc = np.concatenate(all_acc)
    all_raw = np.concatenate(all_raw)
    all_label = np.concatenate(all_label)
    all_pid = np.concatenate(all_pid)
    all_grav = np.concatenate(all_grav) if all_grav else None

    NORM_SIZE = 32
    acc_filt = all_acc[:, :, :NORM_SIZE]
    pos_info = all_acc[:, :, NORM_SIZE]
    additional_embedding = all_acc[:, :, NORM_SIZE + 1:]

    return (acc_filt, pos_info, additional_embedding,
            all_label, all_pid, all_grav, all_raw)
