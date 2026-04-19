# CAR-IMU: Class-Conditional Autoregressive IMU Generation

**CS690R Final Project** — Bio-PM Guided Synthetic IMU Generation

We propose **CAR-IMU**, a conditional autoregressive generator that operates in the representation space of Bio-PM (a pretrained biosignal encoder) to generate synthetic IMU token sequences for data augmentation in Human Activity Recognition (HAR).

---

## Project Overview

The core idea:
1. Take real IMU sensor windows → run through frozen **Bio-PM encoder** → get token embeddings Z
2. Train a **causal transformer decoder** (CAR-IMU) to generate Z conditioned on activity class
3. Use synthetic Z to augment training data → improve HAR classifier F1, especially on minority classes

```
Raw IMU  →  Bio-PM Encoder (frozen)  →  Token sequences Z
                                              ↓
                                       CAR-IMU Decoder (trained)
                                              ↓
                               Synthetic tokens for rare activities
                                              ↓
                                  HAR Classifier → Beat baseline F1
```

---

## Dataset

We use **WISDM Activity Recognition Dataset v1.1**
- 36 subjects, 6 activities: Walking, Jogging, Upstairs, Downstairs, Sitting, Standing
- 20 Hz, 3-axis accelerometer
- ~1 million samples total

**Download:**
```
https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz
```
Extract so you have:
```
WISDM_ar_v1.1/
    WISDM_ar_v1.1_raw.txt
    WISDM_ar_v1.1_raw_about.txt
    WISDM_ar_v1.1_trans_about.txt
    WISDM_ar_v1.1_transformed.arff
    readme.txt
```

---

## Bio-PM Model (Pretrained Encoder)

The `CS690TR/` folder contains the Bio-PM model code. You need to place the pretrained checkpoint manually (not tracked in git due to size):

1. Get `checkpoint.pt` from your course materials / instructor
2. Place it at: `CS690TR/checkpoints/checkpoint.pt`

---

## Setup

### 1. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate       # Mac/Linux
# venv\Scripts\activate        # Windows
```

### 2. Install dependencies
```bash
cd CS690TR
pip install -r requirements.txt
cd ..
pip install umap-learn matplotlib seaborn scikit-learn
```

---

## Reproducing Week 1 Results (Full Pipeline)

Run the commands below **in order** from the project root directory.

### Step 1 — Preprocess WISDM → Bio-PM HDF5 format
Converts raw sensor data for all 36 subjects into the format Bio-PM expects.
Takes ~15–20 minutes on CPU.

```bash
python preprocess_wisdm_biopm.py \
    --raw  WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt \
    --out  preprocessed_biopm
```

**Output:** `preprocessed_biopm/Data_MeLabel_1.h5` ... `Data_MeLabel_36.h5`
Each file contains movement-element patches, gravity signal, and labels per window.

To test on just one subject first (faster, ~1 min):
```bash
python preprocess_wisdm_biopm.py --subjects 33 --out preprocessed_biopm_test
```

---

### Step 2 — Extract Bio-PM features for all subjects
Runs the frozen Bio-PM encoder on every window → saves 1028-d embeddings.
Takes ~5–10 minutes on CPU.

```bash
python CS690TR/scripts/extract_features.py \
    --data_dir   preprocessed_biopm \
    --checkpoint CS690TR/checkpoints/checkpoint.pt \
    --output     features/biopm_features_all.npz
```

**Output:** `features/biopm_features_all.npz`
Contains `features (10810, 1028)`, `labels (10810,)`, `pids (10810,)`.

---

### Step 3 — Run Week 1 Analysis
Extracts full token matrices, runs sanity checks, generates UMAP plots, computes subject embeddings, and runs the baseline HAR classifier.

```bash
python week1_analysis.py \
    --data_dir   preprocessed_biopm \
    --checkpoint CS690TR/checkpoints/checkpoint.pt \
    --features   features/biopm_features_all.npz \
    --out_dir    results_week1
```

If you already ran this once and just want to regenerate plots (skip the slow Bio-PM pass):
```bash
python week1_analysis.py \
    --data_dir   preprocessed_biopm \
    --checkpoint CS690TR/checkpoints/checkpoint.pt \
    --features   features/biopm_features_all.npz \
    --out_dir    results_week1 \
    --skip_tokens
```

---

## Week 1 Results

| Metric | Value |
|--------|-------|
| Total windows | 10,810 |
| NN same-class fraction | **90%** ✅ |
| Subject separation | **0%** — activity-only conditioning used |
| **LOSO Baseline (Linear Probe)** | **0.691 ± 0.177** |
| **LOSO Baseline (MLP)** | **0.689 ± 0.207** |

**CAR-IMU must beat: 0.689 Macro-F1**

Minority classes (target for augmentation):
| Class | Windows | % of data |
|-------|---------|-----------|
| Standing | 471 | 4.4% |
| Sitting | 592 | 5.5% |
| Downstairs | 994 | 9.2% |

---

## Output Files (Week 1)

```
results_week1/
    token_store.hdf5            ← Full Z matrices (10810 × 192 × 64) — CAR-IMU training input
    subject_embeddings.npy      ← Subject-style vectors (36 × 64)
    subject_embedding_ids.npy   ← Subject ID mapping
    baseline_results.npy        ← LOSO F1 scores per fold
    umap_activity.png           ← Token space coloured by activity class
    umap_subjects.png           ← Subject-style embedding space
```

> **Note:** `token_store.hdf5` is ~500MB and not tracked in git. Regenerate by running Step 3 above.

---

## Project Structure

```
Project/
├── CS690TR/                        ← Bio-PM model code (provided)
│   ├── checkpoints/checkpoint.pt   ← NOT in git — get from course materials
│   ├── src/models/biopm.py         ← Bio-PM encoder architecture
│   ├── src/inference/              ← Feature extraction utilities
│   └── scripts/extract_features.py ← Feature extraction entry point
│
├── WISDM_ar_v1.1/                  ← NOT in git — download separately
│
├── preprocessed_biopm/             ← NOT in git — generated by Step 1
├── features/                       ← NOT in git — generated by Step 2
├── results_week1/                  ← Partially in git (plots + small .npy files)
│
├── preprocess_wisdm_biopm.py       ← Step 1 script
├── week1_analysis.py               ← Step 3 script
└── .gitignore
```

---

## Week 2 (Coming Next)

- [ ] `car_imu_decoder.py` — 4-layer causal transformer, D=64, activity-conditional
- [ ] `train_car_imu.py` — Teacher forcing, MSE loss on token sequences
- [ ] `generate_synthetic.py` — Autoregressive rollout per activity class
- [ ] Augmented HAR classifier — real + synthetic tokens, beat F1 = 0.689

---

## Team

CS690R — Spring 2026
