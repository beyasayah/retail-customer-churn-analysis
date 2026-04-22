# 🛍️ Retail Customer Behaviour — ML Workshop

> **Machine Learning Workshop** | Retail Gift E-commerce Customer Behaviour Analysis  
> Prepared by **Beya Sayah** — Academic Year 2025-2026

---

## 📌 Project Overview

This project applies supervised and unsupervised machine learning techniques to a retail gift e-commerce dataset to:

- **Predict customer churn** (classification)
- **Predict average customer spending** — `MonetaryAvg` (regression)
- **Segment customers** by negative behaviour patterns (clustering)

The pipeline covers the full ML lifecycle: raw data exploration → cleaning → preprocessing → feature engineering → model training → evaluation.

---

## 📁 Project Structure

```
project_ml_retail/
│
├── data/
│   ├── raw/
│   │   └── retail_customers_COMPLETE_CATEGORICAL.csv
│   ├── processed/
│   │   └── df_clean_start.csv
│   └── train_test/
│       ├── tree/          # X_train/test for tree models (unscaled)
│       └── others/        # X_train/test for linear models (scaled)
│
├── models/
│   └── linear/
│       └── best_model.pkl
│
├── reports/
│   ├── exploration_report.xlsx
│   └── model_comparison.png
│
└── notebooks/
    ├── 01_exploration_preparation.ipynb
    ├── 02_preprocessing.ipynb
    ├── 02_preprocessing-linear-.ipynb
    ├── 03_linearreg_training.ipynb
    └── 03_trees_training.ipynb
```

---

## 📓 Notebooks

### `01_exploration_preparation.ipynb` — Data Exploration & Quality Audit
- Loads the raw CSV dataset and inspects shape, dtypes, and descriptive statistics
- Identifies and visualises **missing values** (heatmap)
- Detects **outliers** using the IQR method with boxplots
- Analyses **correlations** and flags multicollinearity between features
- Flags constant/low-variance columns
- Exports a quality report to `reports/exploration_report.xlsx`
- Saves the clean starting point to `data/processed/df_clean_start.csv`

---

### `02_preprocessing.ipynb` — Preprocessing for Tree/Classification Models
- Converts numeric columns stored as `object` dtype
- Replaces error-code values (`-1`, `999`, `99`) with `NaN`
- Parses `RegistrationDate` into `RegYear` and `RegMonth` features
- Performs **feature engineering** on `LastLoginIP` (GeoIP flag)
- Applies a `ColumnTransformer` pipeline:
  - `SimpleImputer` for missing values
  - `StandardScaler` for numerics
  - `OneHotEncoder` / `OrdinalEncoder` for categoricals
- Conducts **feature selection** using Mutual Information against the churn target
- Drops leaky/weak features (`TenureRatio`, `UniqueCountries`)
- Saves train/test splits to `data/train_test/tree/`

---

### `02_preprocessing-linear-.ipynb` — Preprocessing for Linear/Regression Models
- Same cleaning steps as above, adapted for the regression target (`MonetaryAvg`)
- Investigates negative values in `MonetaryAvg` and handles them
- Extended EDA: boxplots of key features (`FavoriteSeason`, `PreferredTimeOfDay`, `BasketSizeCategory`, `Region`) against the target
- Applies `StandardScaler` (required for linear models)
- Performs **PCA analysis**:
  - Explained variance curve to determine optimal components
  - 2D PCA scatter plot coloured by `MonetaryAvg`
  - Feature loadings for PC1 and PC2
- Saves scaled train/test splits to `data/train_test/others/`

---

### `03_linearreg_training.ipynb` — Linear Regression Models
**Target:** `MonetaryAvg` (average customer spend in £)

| Model | CV R² | Test R² | Test RMSE | Test MAE |
|---|---|---|---|---|
| Linear Regression | ~0.39 | 0.4157 | ~34 | ~16 |
| Ridge (L2) | tuned | tuned | — | — |
| Lasso (L1) | tuned | tuned | — | — |

- Trains baseline `LinearRegression`, then compares with `Ridge` and `Lasso`
- Uses `RidgeCV` / `LassoCV` for automatic alpha selection (5-fold CV)
- Plots **top 15 feature coefficients** for the best Ridge model
- Applies **outlier capping** (p1/p99), which reduced RMSE by ~34 points on the test set
- Saves the best model to `models/linear/best_model.pkl`

> **Key finding:** ~40% of spending variance is explained by behavioural/demographic features. The remaining 60% is attributed to unobserved factors (promotions, product categories, personal decisions).

---

### `03_trees_training.ipynb` — Tree-Based Classification Models
**Target:** Customer churn (binary classification)

Models trained and compared:
- `DecisionTreeClassifier` (with and without SMOTE)
- `RandomForestClassifier` (with `GridSearchCV` tuning)
- `XGBClassifier` (with `RandomizedSearchCV` + early stopping)

Key steps:
- Drops leaky columns (`Recency`, `TenureRatio`, `RFMSegment`) before training
- Applies **SMOTE** to balance the churn class
- **Negative Behaviour Clustering** using KMeans on (`ReturnRatio`, `NegativeQuantityCount`, `CancelRate`):
  - Elbow method + silhouette scores to pick K=3
  - PCA 2D visualisation of clusters
- Final model comparison ranked by **F1-weighted** and **AUC-ROC**
- Saves comparison chart to `reports/model_comparison.png`

---

## ⚙️ Setup & Installation

### Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn joblib openpyxl
```

### Run Order

```
01_exploration_preparation.ipynb
        ↓
02_preprocessing.ipynb          (→ tree models)
02_preprocessing-linear-.ipynb  (→ linear models)
        ↓
03_trees_training.ipynb
03_linearreg_training.ipynb
```

---

## 🔑 Key Features Used

| Feature | Description |
|---|---|
| `MonetaryAvg` / `MonetaryTotal` | Spending metrics |
| `Frequency` | Purchase frequency |
| `ReturnRatio`, `CancelRate` | Negative behaviour indicators |
| `SatisfactionScore` | Customer satisfaction (1–5) |
| `FavoriteSeason`, `PreferredTimeOfDay` | Temporal preferences |
| `BasketSizeCategory`, `ProductDiversity` | Shopping habits |
| `Region`, `GeoIP` | Geographic features |
| `RegYear`, `RegMonth` | Registration date features |

---

## 📊 Results Summary

| Task | Best Model | Best Metric |
|---|---|---|
| Spend Prediction | Ridge Regression | R² ≈ 0.41, MAE ≈ £16 |
| Churn Classification | XGBoost (tuned) | AUC-ROC: see `model_comparison.png` |
| Behaviour Segmentation | KMeans (K=3) | Silhouette ≈ 0.3+ |

---


