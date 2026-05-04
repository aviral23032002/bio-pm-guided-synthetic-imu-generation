# Experimentation Log: CAR-IMU

This document tracks the different strategies, hyperparameter settings, and evaluation results tried during the project.

## Summary of Trials

| ID | Strategy | Calibration | Temperature | Result / Finding |
|---|---|---|---|---|
| **0** | **Baseline** | N/A | N/A | **Macro-F1: 0.689** (MLP on real-only tokens) |
| **1** | **Full Upsample** | No | 0.5 | **Macro-F1: 0.637** (↓ -0.024). Too many synthetic samples diluted the real signal. |
| **2** | **Ratio Strategy (0.5)** | No | 0.3 | Improved minority classes (Standing +0.014 F1). |
| **3** | **Calibrated Tokens** | **Yes** | 0.3 | **Highest Fidelity.** Alignment of mean/std between real/synthetic tokens improved cluster overlap. |
| **4** | **Gravity-Aware Recon** | Yes | 0.3 | Successfully reconstructed total acceleration waveforms using class-mean gravity vectors. |

---

## 📈 Visualizing Fidelity (Trial 3: Calibrated)

The calibration step ensures that synthetic tokens reside within the same manifold as the real tokens.

### T-SNE Overlay (Calibrated)
![T-SNE Overlay](full_pipeline_report_calibrated/pipeline_tsne_overlay.png)
*Figure 1: T-SNE distribution of real vs. calibrated synthetic tokens across all 6 classes.*

### UMAP Overlay (Calibrated)
![UMAP Overlay](full_pipeline_report_calibrated/pipeline_umap_overlay.png)
*Figure 2: UMAP cluster analysis showing high overlap between synthetic and real activity distributions.*

---

## 🌊 Waveform Reconstruction Gallery

Comparison of real-world rectified intensity profiles (blue) vs. synthetic profiles (red).

![Waveform Gallery](synthetic_waveforms_v2_6class_calibrated/synthetic_reconstruction_gallery.png)
*Figure 3: Rectified mean intensity profiles for Walking, Jogging, Upstairs, Downstairs, Sitting, and Standing.*

---

## 📋 Quantitative Metrics (Final Report)

| Activity | Cos Sim | Spec Corr | ZCR Err | SMA Err | Waveform RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Walking** | 0.9636 | 0.9884 | 0.0592 | 0.2198 | **0.084** |
| **Jogging** | 0.8348 | 0.8929 | 0.0088 | 0.4143 | **0.125** |
| **Upstairs** | 0.9613 | 0.9964 | 0.0457 | 0.1688 | **0.072** |
| **Downstairs** | 0.9123 | 0.9850 | 0.0604 | 0.1441 | **0.091** |
| **Sitting** | 0.9437 | 0.9812 | 0.0439 | 0.3626 | **0.065** |
| **Standing** | 0.9525 | 0.9872 | 0.0261 | 0.2077 | **0.068** |

### Key Takeaways
1. **Calibration is critical**: Without calibration, the domain shift between the autoregressive output and the original Bio-PM tokens led to drops in F1 performance.
2. **Frequency Alignment**: Spectral Correlation remains extremely high (>0.98 for most classes), proving that the Bio-PM patches successfully capture the rhythmic nature of human movement.
3. **Imbalance Mitigation**: The project successfully improved the performance of the **Standing** class (+1.4% F1), which was the rarest class in the original WISDM dataset.
