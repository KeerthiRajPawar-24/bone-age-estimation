# 🦴 Bone Age Estimation from Pediatric Hand X-rays

**A Deep Learning Regression Approach using Transfer Learning with ResNet50**

---

## 📋 Project Overview

Automated estimation of pediatric bone age (in months) from hand X-ray images using deep learning. Bone age assessment is a critical clinical procedure used by pediatric endocrinologists to diagnose growth disorders such as growth hormone deficiency and precocious puberty. Traditional manual assessment (Greulich-Pyle atlas) takes 15–20 minutes per case and shows significant inter-observer variability (~7–10 months).

This project frames the problem as a **regression task** (rather than the more common classification approach), enabling continuous, clinically precise predictions.

---

## 👤 Author

**Keerthi P R**
- MSc Big Data Analytics, St Joseph's University, Bengaluru
- 📧 keerthirajpawar11@gmail.com
- 🔗 [LinkedIn](https://linkedin.com/in/keerthirajpawar) | [GitHub](https://github.com/KeerthiRajPawar-24)

---

## 📊 Dataset

- **Name:** RSNA Pediatric Bone Age Challenge Dataset
- **Source:** Radiological Society of North America (RSNA), released 2017
- **Hosted on:** [Kaggle](https://www.kaggle.com/datasets/kmader/rsna-bone-age)
- **Size:** 12,611 training images (~9.3 GB)
- **Image format:** Grayscale PNG, ~973 × 1321 pixels
- **Labels:** `boneage` (in months, 1–228) and `male` (boolean)
- **Age range:** 0.1 to 19 years
- **Sex distribution:** 54% male, 46% female

*Note: This repository does NOT include the dataset. Download it directly from Kaggle using the provided notebook.*

---

## 🎯 Results

### Test Set Performance

| Metric | Value |
|--------|-------|
| **Mean Absolute Error (MAE)** | **17.07 months** (~1.42 years) |
| **Root Mean Squared Error (RMSE)** | 21.69 months (~1.81 years) |
| **R² Score** | **0.7292** |
| **Pearson Correlation (r)** | 0.8546 |
| **Mean Residual (Bias)** | −0.08 months |

### Bland-Altman Analysis (Clinical Method Agreement)

| Metric | Value |
|--------|-------|
| **Bias** | −0.08 months |
| **95% Limits of Agreement** | [−42.59, +42.44] months |
| **Points within 95% LoA** | 94.6% |

### Age-Stratified Analysis

| Age Group | Count | MAE (months) | Mean Residual |
|-----------|-------|--------------|---------------|
| 0–2 years | 15 | 19.96 | +19.04 |
| 2–5 years | 92 | 22.80 | +18.51 |
| 5–9 years | 306 | 17.63 | +12.62 |
| **9–13 years (best)** | **525** | **14.19** | **−1.53** |
| 13–17 years | 309 | 18.23 | −14.48 |
| 17–19 years (worst) | 15 | 44.59 | −44.59 |

---

## 🏗️ Model Architecture

- **Base:** ResNet50 (frozen, ImageNet pre-trained) — 23.5M frozen parameters
- **Head:** Global Average Pooling → Dense(256, ReLU) → Dropout(0.3) → Dense(64, ReLU) → Dropout(0.2) → Dense(1, Linear)
- **Trainable parameters:** 541K
- **Loss function:** Mean Absolute Error (MAE)
- **Optimizer:** Adam (lr=0.001)
- **Callbacks:** EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

---

## 🧪 Methodology

1. **Data Exploration** — Statistical analysis of age distribution, skewness, sex balance
2. **Preprocessing** — 80/10/10 train/val/test split, resize to 224×224, ResNet50 normalization
3. **Augmentation** — Rotation, shifts, zoom, horizontal flip (training only)
4. **Transfer Learning** — Frozen ResNet50 base + trainable regression head
5. **Training** — 20 epochs max, EarlyStopping at epoch 15
6. **Evaluation** — MAE, RMSE, R², Pearson r, Bland-Altman analysis, age-stratified errors

---

## 📁 Repository Structure

```
bone-age-estimation/
├── code/
│   ├── BoneAge_Estimation.ipynb    # Main Colab notebook (executable)
│   └── utils.py                     # Reusable utility functions
├── docs/
│   ├── BoneAge_StudyGuide.pdf       # Comprehensive study guide
│   └── Results_Summary.md           # Metrics summary
├── results/
│   └── (plots go here after running)
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Files to exclude from Git
└── README.md                        # This file
```

---

## 🚀 How to Run

### Prerequisites
- Google Colab account
- Kaggle account (for dataset download)

### Steps

1. **Open the notebook in Google Colab**
   - Upload `code/BoneAge_Estimation.ipynb` to Colab

2. **Enable GPU**
   - Runtime → Change runtime type → T4 GPU

3. **Get your Kaggle API credentials**
   - Go to Kaggle → Settings → API → Create Legacy API Key
   - This downloads `kaggle.json`

4. **Run cells top to bottom**
   - Upload `kaggle.json` when prompted
   - Dataset will download automatically (~10 min)
   - Training takes ~30 min on Colab T4 GPU

5. **Download outputs immediately after training**
   - The notebook includes a final cell to download all artifacts

---

## 📈 Key Insights

- **Transfer learning was the right call** — with only ~10K images, training from scratch would badly overfit
- **Regression, not classification** — bone age is a continuous variable; regression preserves clinical precision
- **Heteroscedasticity in residuals** — errors are small in the middle age range and grow at extremes (confirmed via residual plots)
- **Regression toward the mean bias** — the model over-predicts young children and under-predicts older teens due to age imbalance in the dataset
- **Bland-Altman analysis** shows the model is not yet a radiologist replacement (LoA ±42 vs radiologist ±10-12) but suitable for screening/triage

---

## 🔮 Future Improvements

- **Fine-tune ResNet50 layers** — unfreeze top layers with small LR (expected MAE reduction to ~13 months)
- **Add sex as a feature** — sex is a known biological predictor of skeletal maturation
- **Weighted sampling** — oversample underrepresented age groups (infants, older teens)
- **Ensemble methods** — average predictions from multiple models
- **Uncertainty quantification** — Monte Carlo Dropout or Bayesian methods
- **Grad-CAM explainability** — visualize where the model looks
- **External validation** — test on Indian pediatric datasets

---

## 🏥 Clinical Interpretation

The current model achieves 17.07 months MAE — cutting the error range from random guessing (~35 months) roughly in half, without reaching radiologist inter-observer agreement (~10 months). This positions the model as suitable for:

1. **Screening / triage** — flag cases for radiologist review
2. **Quality assurance** — cross-check radiologist assessments
3. **Research** — large-scale bone age studies

Full clinical deployment as a radiologist substitute would require fine-tuning, external validation, uncertainty estimation, regulatory approval (FDA / CDSCO), and human-in-the-loop workflows.

---

## 📚 Documentation

See `docs/BoneAge_StudyGuide.pdf` for a comprehensive 16-section walkthrough covering:
- Problem statement and clinical background
- Complete data exploration and preprocessing rationale
- Model architecture with layer-by-layer explanation
- Every metric with math formula and interpretation
- Bland-Altman analysis theory and results
- Age-stratified error analysis
- Deployment considerations
- Drawbacks and limitations
- Future work
- Interview Q&A preparation

---

## 🛠️ Tools & Libraries

- **Python 3.13**
- **TensorFlow / Keras** — model building and training
- **NumPy, Pandas** — data manipulation
- **Matplotlib** — visualization
- **Scikit-learn** — train/test split
- **Google Colab (Tesla T4 GPU)** — training environment

See `requirements.txt` for exact versions.

---

## 📝 License

Academic project submitted for MSc Big Data Analytics coursework at St Joseph's University, Bengaluru.

---

## 🙏 Acknowledgments

- **RSNA** for releasing this valuable public dataset
- **Kaggle** for hosting the challenge and dataset
- **TensorFlow / Keras** team for the excellent deep learning framework
