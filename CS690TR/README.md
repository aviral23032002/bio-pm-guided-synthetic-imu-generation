# CS690T — BioPM Feature Extraction Toolkit

A student-friendly package for using the pretrained **BioPM** (Biological Primitives Model, also known as **50MR**) to extract movement representations from wearable accelerometer data.

## What is BioPM?

BioPM is a dual-stream transformer model that learns representations of human movement from 3-axis accelerometer data. It decomposes raw acceleration into biologically meaningful **movement elements** (zero-crossing segments) and processes them with:

1. **Accelerometer stream** — A transformer encoder that takes movement-element patches (normalised to 32 samples each) along with axis identity, duration, and temporal position, and produces contextual 64-d token embeddings.
2. **Gravity stream** — A 1-D CNN that encodes the low-pass gravity component of the window into a 64-d vector.

The model was pretrained using a **masked reconstruction** objective (BERT-style): random movement elements are masked, and the encoder learns to reconstruct them from surrounding context. The pretrained encoder (called **50MR** for its 50% masking rate) can then be used to extract general-purpose features from any accelerometer data.

## What can you do with it?

- **Classification** — activity recognition, gesture detection, health status
- **Clustering** — discover movement patterns
- **Visualisation** — t-SNE / UMAP plots of movement spaces
- **Regression** — predict continuous outcomes from movement features
- **Biomarker discovery** — find movement signatures of conditions
- **Retrieval** — find similar movement segments
- **Transfer learning** — use BioPM features as input to your own model

---

## Quick Start

### 1. Setup

```bash
cd CS690TR
pip install -r requirements.txt
```

### 2. Checkpoint (already included)

The pretrained 50MR checkpoint is already at:
```
CS690TR/checkpoints/checkpoint.pt    (5.6 MB)
```
No action needed — the scripts default to this path.

### 3. Preprocess your data

```bash
python scripts/preprocess_data.py \
    --raw_data_dir /path/to/your/raw/data \
    --output_dir   preprocessed/ \
    --window_sec   10 \
    --slide_sec    5 \
    --ori_fs       50 \
    --target_fs    30
```

You will need to edit the `load_raw_data()` function in `scripts/preprocess_data.py` to match your data format. Look for lines marked with `# STUDENT:`.

### 4. Extract features

```bash
python scripts/extract_features.py \
    --data_dir     preprocessed/ \
    --checkpoint   checkpoints/checkpoint.pt \
    --output       features/biopm_features.npz \
    --device       cpu
```

### 5. Use features in your project

```python
import numpy as np
data = np.load('features/biopm_features.npz')
X = data['features']   # (N, 1028) BioPM embeddings
y = data['labels']      # (N,) activity labels
pids = data['pids']     # (N,) subject IDs
```

---

## Folder Structure

```
CS690TR/
├── README.md                           ← you are here
├── requirements.txt                    ← Python dependencies
├── checkpoints/
│   ├── README.md                       ← where to place checkpoint.pt
│   └── checkpoint.pt                   ← (you add this)
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── biopm.py                    ← full BioPM model architecture
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py                  ← PyTorch Dataset for ME data
│   │   └── preprocessing.py            ← signal processing utilities
│   ├── inference/
│   │   ├── __init__.py
│   │   └── feature_extractor.py        ← core feature extraction logic
│   └── utils/
│       └── __init__.py
├── scripts/
│   ├── preprocess_data.py              ← main preprocessing script
│   ├── extract_features.py             ← main feature extraction script
│   └── generation_starter.py           ← experimental generation scaffold
├── examples/
│   ├── example_preprocessing.py        ← preprocessing shape demo
│   ├── example_feature_extraction.py   ← feature extraction shape demo
│   └── example_downstream.py           ← classification example
└── starter_project/
    ├── my_project.py                   ← template for your final project
    └── preprocessing_template.py       ← simplified preprocessing template
```

---

## Preprocessing Your Own Data

BioPM expects data in a specific HDF5 format. The preprocessing pipeline converts raw accelerometer data through these steps:

### Expected Raw Data

Your raw data should contain:
- **3-axis acceleration** (X, Y, Z) — can be in m/s² (will be converted to g) or already in g
- **Activity labels** — integer labels per sample (or per time window)
- **Sample rate** — you need to know your data's original sample rate

### What the Preprocessing Does

| Step | Description | What it does |
|------|-------------|-------------|
| 1 | **Resample** | Interpolate to 30 Hz (BioPM's expected rate) |
| 2 | **Bandpass filter** | 0.5–12 Hz → body acceleration (removes gravity + high-freq noise) |
| 3 | **Lowpass filter** | < 0.5 Hz → gravity component |
| 4 | **Window** | 10-second sliding windows with 5-second overlap |
| 5 | **Movement elements** | Per-axis zero-crossing detection → segments between sign changes |
| 6 | **Normalise** | Each ME resampled to 32 points, with metadata (axis, duration, position) |
| 7 | **Save** | HDF5 per subject with `x_acc_filt`, `x_gravity`, `window_label` |

### What You Need to Modify

In `scripts/preprocess_data.py` or `starter_project/preprocessing_template.py`, edit:

1. **`load_raw_data()` / `load_my_data()`** — change to read your file format and select the correct columns
2. **Unit conversion** — divide by 9.80665 if your data is in m/s²
3. **`ORIGINAL_SAMPLE_RATE`** — set to your data's actual sample rate
4. **`remap_labels()` / `remap_my_labels()`** — map your labels to contiguous integers
5. **`SKIP_LABELS`** — labels to exclude (e.g., null/transition classes)

### Output Files

Preprocessing produces per-subject HDF5 files:

- `Data_MeLabel_{subject_id}.h5` — used by `extract_features.py`
  - `x_acc_filt`: `(W, 192, 37+)` — movement elements + metadata
  - `x_gravity`: `(W, 300, 3)` — gravity windows
  - `window_acc_raw`: `(W, 300, 3)` — raw acceleration
  - `window_label`: `(W,)` — integer labels

---

## Feature Extraction Details

### The 1028-d Feature Vector

Each window produces a 1028-dimensional feature vector:

| Dimensions | Source | Description |
|-----------|--------|-------------|
| `[0:64]` | encoder_acc → mean pool | Mean of 64-d transformer tokens over valid patches |
| `[64:128]` | encoder_acc → std pool | Std of 64-d transformer tokens over valid patches |
| `[128:1028]` | gravity signal | Raw gravity interpolated to 300×3, flattened to 900 |

### Why These Features?

- The **transformer tokens** capture the temporal structure and relationships between movement elements within the window. Mean+std pooling provides a fixed-size summary regardless of how many movement elements exist.
- The **gravity component** captures the orientation and slow postural changes of the sensor, which is informative for activity recognition.

### Batch Processing

For large datasets, you can increase `--batch_size` (default 32). If you have a GPU, use `--device cuda` for faster extraction.

---

## Experimental: Movement Generation

`scripts/generation_starter.py` provides a scaffold for exploring masked infilling with BioPM.

**Important limitations:**
- BioPM is a **bidirectional** encoder (BERT-style), **not** an autoregressive generator
- It cannot natively produce novel movement sequences from scratch
- The masked infilling experiment shows what the model reconstructs when patches are hidden, which is what it was trained to do
- Generated outputs have **not been validated** for scientific correctness

This is provided as an **optional** exploration path for students interested in generative approaches. It should not interfere with the main preprocessing + feature extraction workflow.

---

## Known Limitations and Assumptions

1. **Sample rate**: The pipeline assumes resampling to 30 Hz. If your data is already at 30 Hz, set `--ori_fs 30` and `--target_fs 30`.

2. **3-axis accelerometer only**: BioPM was designed for single-location 3-axis accelerometer data. Gyroscope, magnetometer, or multi-location data would require architectural changes.

3. **Window size**: The default is 10 seconds. Changing this affects the maximum number of movement elements (pad_size = WS × 192/10). Very short windows may have too few MEs for meaningful representation.

4. **Movement elements**: Zero-crossing detection assumes the signal has enough dynamic content. Very still periods may produce few or no MEs and will be skipped.

5. **Gravity separation**: The lowpass filter at 0.5 Hz assumes gravity changes slowly relative to body movement. This is standard for wearable accelerometer analysis.

6. **Checkpoint compatibility**: The checkpoint must match the architecture in `src/models/biopm.py`. The provided architecture matches the 50MR (d_model=64, 5 layers, 4 heads) variant.

7. **No fine-tuning support**: This package is for **inference/feature extraction only**. Fine-tuning the model requires the full training pipeline from the original repository.

---

## Troubleshooting

### "No Data_MeLabel_*.h5 files found"
Your preprocessing step didn't produce output, or you're pointing to the wrong directory. Check that preprocessing ran successfully and produced `.h5` files.

### "Checkpoint not found"
Place the `checkpoint.pt` file in `checkpoints/`. See `checkpoints/README.md`.

### "No valid windows for subject X"
The preprocessing skipped all windows. This can happen if:
- All labels are in the skip set
- The signal is too flat (no zero crossings detected)
- The data is too short for a full window

### "CUDA out of memory"
Reduce `--batch_size` or use `--device cpu`.

### Zero-crossing detection fails
This happens with very smooth or constant signals. The bandpass filter should remove DC and very low frequency components, but if your data has unusual characteristics, you may need to adjust the filter parameters in `DEFAULT_CONFIG`.

### Import errors
Make sure you run scripts from the `CS690TR/` directory, or that your `PYTHONPATH` includes it.

---

## Dependencies

- Python 3.8+
- PyTorch >= 1.10
- NumPy, SciPy, pandas, h5py, scikit-learn

Install all with:
```bash
pip install -r requirements.txt
```
