# Bone Age Estimation — Results Summary

**Author:** Keerthi P R
**Registration:** 252BDA04
**Institution:** St Joseph's University, Bengaluru
**Date:** September 2026

---

## 🎯 Headline Numbers

| Metric | Value |
|--------|-------|
| **Test MAE** | **17.07 months** |
| **Test RMSE** | 21.69 months |
| **R² Score** | **0.7292** |
| **Pearson r** | 0.8546 |
| **Mean Bias** | −0.08 months |

---

## 📊 Comprehensive Test Set Metrics

### Regression Metrics

- **MAE (Mean Absolute Error):** 17.07 months (~1.42 years)
- **MSE (Mean Squared Error):** 470.57
- **RMSE (Root Mean Squared Error):** 21.69 months (~1.81 years)
- **R² (Coefficient of Determination):** 0.7292
- **Pearson Correlation Coefficient:** 0.8546
- **Mean Residual:** −0.08 months (essentially zero bias)
- **Standard Deviation of Residuals:** 21.69 months

### Bland-Altman Method Agreement Analysis

- **Bias (Mean Difference):** −0.08 months
- **Upper 95% Limit of Agreement:** +42.44 months
- **Lower 95% Limit of Agreement:** −42.59 months
- **Empirical Coverage:** 94.6% within 95% LoA

---

## 📈 Age-Stratified Performance

| Age Group | Sample Count | MAE (months) | Mean Residual |
|-----------|--------------|--------------|---------------|
| 0–2 years | 15 | 19.96 | +19.04 |
| 2–5 years | 92 | 22.80 | +18.51 |
| 5–9 years | 306 | 17.63 | +12.62 |
| **9–13 years (BEST)** | **525** | **14.19** | **−1.53** |
| 13–17 years | 309 | 18.23 | −14.48 |
| 17–19 years (WORST) | 15 | 44.59 | −44.59 |

**Key observation:** Model performance strongly correlates with training sample count per age group — best performance in most-represented groups (9-13 years), degrading at age extremes (regression toward the mean).

---

## 🏗️ Model Configuration

### Architecture
- **Base:** ResNet50 (frozen, ImageNet pre-trained)
- **Head:** GAP → Dense(256, ReLU) → Dropout(0.3) → Dense(64, ReLU) → Dropout(0.2) → Dense(1, Linear)
- **Total Parameters:** 24,128,769
- **Trainable Parameters:** 541,057
- **Non-Trainable (Frozen) Parameters:** 23,587,712

### Training
- **Loss Function:** Mean Absolute Error (MAE)
- **Optimizer:** Adam
- **Initial Learning Rate:** 0.001
- **Batch Size:** 32
- **Epochs Completed:** 15 (EarlyStopping triggered)
- **Best Validation MAE:** 17.01 months

### Data
- **Total Images:** 12,611
- **Training Set:** 10,089 (80.0%)
- **Validation Set:** 1,260 (10.0%)
- **Test Set:** 1,262 (10.0%)
- **Input Image Size:** 224 × 224 × 3 (resized from ~973 × 1321 grayscale)

### Callbacks
- **EarlyStopping:** patience=5, restore_best_weights=True
- **ReduceLROnPlateau:** factor=0.5, patience=3, min_lr=1e-6
- **ModelCheckpoint:** save_best_only=True

---

## 📊 Clinical Context

| Benchmark | MAE (months) |
|-----------|--------------|
| Random guessing | ~35 |
| **Our model (this project)** | **17.07** |
| Radiologist inter-observer agreement | ~7–10 |
| State-of-the-art (fine-tuned, ensemble) | ~4–6 |

Our model captures approximately 50% of the gap between random guessing and expert radiologist performance using only frozen transfer learning without fine-tuning — a solid baseline for a one-week undergraduate project.

---

## 🖼️ Generated Visualizations

The following plots were generated during evaluation:

1. **age_distribution.png** — Histogram of bone ages with mean/median lines + sex breakdown
2. **sample_xrays.png** — Grid of 6 random X-rays with age/sex labels
3. **training_curves.png** — Training vs validation MAE and MSE over epochs
4. **predictions_vs_actuals.png** — Scatter of predictions vs actuals + residual plot
5. **bland_altman.png** — Bland-Altman method agreement plot with 95% LoA
6. **age_stratified_analysis.png** — MAE by age group + sample distribution

---

## 🎯 Key Takeaways

1. **The model works, but has known limitations** — 17-month MAE is a solid baseline but not clinically deployable
2. **The near-zero bias (−0.08) is a strong result** — indicates no systematic over/under-prediction
3. **Age-stratified errors reveal a clear pattern** — regression toward the mean at age extremes due to data imbalance
4. **Bland-Altman analysis** shows the model is currently suitable for screening/triage but not radiologist replacement
5. **The approach is sound and extendable** — fine-tuning ResNet50 layers and adding sex as a feature could bring MAE down to ~13 months

---

## 📚 References

- **Dataset:** RSNA Pediatric Bone Age Challenge (2017), hosted on Kaggle
- **Base Model:** He, K. et al. "Deep Residual Learning for Image Recognition." CVPR 2016.
- **Method Agreement:** Bland, J.M. & Altman, D.G. "Statistical methods for assessing agreement between two methods of clinical measurement." Lancet 1986.

---

*Complete methodology and analysis: see `BoneAge_StudyGuide.pdf`*
