# CAR-IMU: Class-Conditional Autoregressive IMU Generation

**CS690R Final Project** — Bio-PM Guided Synthetic IMU Generation

We propose **CAR-IMU**, a conditional autoregressive generator that operates in the representation space of Bio-PM (a pretrained biosignal encoder) to generate synthetic IMU token sequences for data augmentation in Human Activity Recognition (HAR).

---

## Project Overview

```
Raw IMU  →  Bio-PM Encoder (frozen)  →  Token sequences Z (192 × 64)
                                               ↓
                                        CAR-IMU Decoder (trained)
                                        [4-layer causal transformer]
                                               ↓
                              Synthetic tokens for rare activity classes
                                               ↓
                                 HAR Classifier → Beat baseline F1
```

---

## Quick Start — Notebooks (Local or Colab)

| Notebook | What it covers |
|---|---|
| `week1_pipeline.ipynb` | Data download → preprocessing → Bio-PM features → baseline |
| `week2_car_imu.ipynb` | Train CAR-IMU → generate synthetic tokens → evaluate augmentation |

**On Google Colab:** Open notebook → File → Open from GitHub → paste repo URL → select notebook → switch runtime to T4 GPU.

---

## Dataset

**WISDM Activity Recognition Dataset v1.1**
- 36 subjects, 6 activities: Walking, Jogging, Upstairs, Downstairs, Sitting, Standing
- 20 Hz, 3-axis accelerometer, ~1 million samples

**Download:**
```
https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz
```
Extract so you have `WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt`

---

## Bio-PM Pretrained Encoder

The `CS690TR/` folder contains Bio-PM model code (provided by course).
Place the pretrained checkpoint manually (not in git due to size):

```
CS690TR/checkpoints/checkpoint.pt   ← get from course materials / instructor
```

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux

cd CS690TR && pip install -r requirements.txt && cd ..
pip install umap-learn matplotlib seaborn scikit-learn
```

---

## ━━━ WEEK 1 ━━━ Preprocessing → Features → Baseline

### Step 1 — Preprocess WISDM → Bio-PM HDF5 format

Converts raw sensor data for all 36 subjects into the HDF5 format Bio-PM expects.
Resamples 20 Hz → 30 Hz, extracts zero-crossing movement elements.

```bash
python preprocess_wisdm_biopm.py \
    --raw  WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt \
    --out  preprocessed_biopm
```

**Output:** `preprocessed_biopm/Data_MeLabel_1.h5` … `Data_MeLabel_36.h5`

Single-subject test (1 min):
```bash
python preprocess_wisdm_biopm.py --subjects 33 --out preprocessed_biopm_test
```

---

### Step 2 — Extract Bio-PM Features

Runs the frozen Bio-PM encoder → 1028-d embeddings per window.

```bash
python CS690TR/scripts/extract_features.py \
    --data_dir   preprocessed_biopm \
    --checkpoint CS690TR/checkpoints/checkpoint.pt \
    --output     features/biopm_features_all.npz
```

**Output:** `features/biopm_features_all.npz` — shape `(10810, 1028)`

---

### Step 3 — Week 1 Analysis

Extracts full token matrices, NN sanity check, UMAP plots, subject embeddings, LOSO baseline.

```bash
python week1_analysis.py \
    --data_dir   preprocessed_biopm \
    --checkpoint CS690TR/checkpoints/checkpoint.pt \
    --features   features/biopm_features_all.npz \
    --out_dir    results_week1
```

Skip Bio-PM re-extraction if already ran (just redo plots/classifier):
```bash
python week1_analysis.py ... --skip_tokens
```

---

### Week 1 Results

| Metric | Value |
|---|---|
| Total windows | 10,810 |
| NN same-class accuracy | **90%** ✅ |
| Subject separation | **0%** → activity-only conditioning chosen |
| **LOSO Baseline (Linear Probe)** | **0.691 Macro-F1** |
| **LOSO Baseline (MLP)** | **0.689 Macro-F1** |

Class distribution (imbalanced — minority classes are the target):

| Class | Windows | % |
|---|---|---|
| Walking | 4,168 | 38.6% |
| Jogging | 3,361 | 31.1% |
| Upstairs | 1,224 | 11.3% |
| Downstairs | 994 | 9.2% |
| Sitting ⚠ | 592 | 5.5% |
| Standing ⚠ | 471 | 4.4% |

**Week 1 outputs:**
```
results_week1/
    token_store.hdf5          ← (10810 × 192 × 64) — CAR-IMU training input [NOT in git, ~500MB]
    subject_embeddings.npy    ← Subject-style vectors (36 × 64)
    umap_activity.png         ← Token space coloured by activity
    umap_subjects.png         ← Subject embedding space
    baseline_results.npy      ← LOSO F1 scores per fold
```

---

## ━━━ WEEK 2 ━━━ CAR-IMU Training → Generation → Evaluation

### Step 4 — Architecture Sanity Check

```bash
python car_imu_decoder.py
```

Expected: 137,472 parameters, MSE ~1.0 at random init, all 6 classes sample correctly.

---

### Step 5 — Train the CAR-IMU Decoder

4-layer causal transformer trained with teacher forcing + MSE loss on Bio-PM token sequences.

```bash
python train_car_imu.py \
    --token_store results_week1/token_store.hdf5 \
    --out_dir     car_imu_checkpoints \
    --epochs      50 \
    --device      auto          # auto-detects MPS (Apple Silicon) > CUDA > CPU
```

**Expected training time:** ~10 min on MPS (M4 Mac) / ~5 min CUDA / ~20 min CPU

**Result achieved:** Val MSE = **0.0157** after 50 epochs (↓ from 1.02 at random init)

Per-class val MSE after training:
| Class | MSE |
|---|---|
| Downstairs | 0.0126 |
| Walking | 0.0137 |
| Upstairs | 0.0139 |
| Jogging | 0.0166 |
| Sitting | 0.0235 |
| Standing | 0.0239 |

---

### Step 6 — Generate Synthetic Tokens

Uses trained CAR-IMU to autoregressively generate synthetic Bio-PM token sequences.

**Quick test (ratio strategy — ~2 min on MPS):**
```bash
python generate_synthetic.py \
    --checkpoint  car_imu_checkpoints/best_model.pt \
    --token_store results_week1/token_store.hdf5 \
    --out_dir     synthetic_tokens \
    --strategy    ratio \
    --syn_ratio   0.5 \
    --temperature 0.3
```

**Full upsample (balances all classes to 4,168 — ~20 min on MPS):**
```bash
python generate_synthetic.py \
    --strategy    upsample \
    --temperature 0.5
```

Quality check output (cosine similarity real vs synthetic centroids — should be > 0.85):
```
Jogging      0.865  ✅ Good
Upstairs     0.926  ✅ Good
Downstairs   0.896  ✅ Good
Sitting      0.959  ✅ Good
Standing     0.939  ✅ Good
```

---

### Step 7 — Evaluate Augmented HAR Classifier

LOSO cross-validation comparing real-only vs real + synthetic training data.

```bash
python evaluate_augmented.py \
    --real_tokens results_week1/token_store.hdf5 \
    --syn_tokens  synthetic_tokens/synthetic_tokens.hdf5 \
    --out_dir     results_week2
```

---

### Week 2 Results

| Condition | LR Macro-F1 | MLP Macro-F1 |
|---|---|---|
| [REF] Week 1 real-only (1028-d) | 0.691 | **0.689** |
| [A] Real only (64-d tokens) | 0.646 | 0.661 |
| [B] Real + Synthetic (CAR-IMU) | 0.567 | 0.637 |
| **Δ = [B] − [A]** | -0.079 | **-0.024** |

Per-class F1 (MLP, upsample strategy):

| Class | Real only | Augmented | Δ |
|---|---|---|---|
| Walking | 0.790 | 0.775 | -0.015 |
| Jogging | 0.952 | 0.923 | -0.029 |
| Upstairs | 0.564 | 0.561 | -0.003 |
| Downstairs | 0.533 | 0.497 | -0.036 |
| Sitting | 0.780 | 0.716 | -0.064 |
| **Standing** ⚠ | 0.694 | **0.708** | **+0.014** ✅ |

**Finding:** Full upsample (14K synthetic) slightly hurt overall F1 — too many synthetic samples relative to real (14K synthetic vs 10K real). Standing (rarest class, 471 windows) is the only class that improved. Next step: ratio strategy with lower temperature.

---

## Project Structure

```
Project/
├── CS690TR/                          ← Bio-PM starter code (provided)
│   ├── checkpoints/checkpoint.pt     ← NOT in git — get from course materials
│   ├── src/models/biopm.py           ← Bio-PM encoder architecture
│   └── scripts/extract_features.py  ← Feature extraction entry point
│
├── WISDM_ar_v1.1/                    ← NOT in git — download separately
├── preprocessed_biopm/               ← NOT in git — generated by Step 1
├── features/                         ← NOT in git — generated by Step 2
├── car_imu_checkpoints/              ← NOT in git — generated by Step 5
├── synthetic_tokens/                 ← NOT in git — generated by Step 6
│
├── results_week1/                    ← Partially tracked (plots, small .npy)
├── results_week2/                    ← Tracked (small .npy evaluation results)
│
├── preprocess_wisdm_biopm.py         ← Week 1, Step 1
├── week1_analysis.py                 ← Week 1, Step 3
├── car_imu_decoder.py                ← Week 2, model architecture
├── train_car_imu.py                  ← Week 2, Step 5
├── generate_synthetic.py             ← Week 2, Step 6
├── evaluate_augmented.py             ← Week 2, Step 7
│
├── week1_pipeline.ipynb              ← Colab notebook: Week 1
├── week2_car_imu.ipynb               ← Colab notebook: Week 2
└── .gitignore
```

---

## Team

CS690R — Spring 2026
