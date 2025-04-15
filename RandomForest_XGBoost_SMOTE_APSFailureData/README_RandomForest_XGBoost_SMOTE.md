# ⚙️ Imbalanced Classification with Random Forest, XGBoost, and SMOTE

This project addresses classification of rare failures in trucks using advanced tree-based methods including Random Forest, XGBoost with L1-penalized logistic regression, and SMOTE oversampling. It is based on the APS Failure dataset and was developed for Homework 6 of the DSCI 552 (Machine Learning for Data Science) course at USC.

---

## 📁 Dataset Overview

- **Source**: [UCI APS Failure at Scania Trucks](https://archive.ics.uci.edu/ml/datasets/APS+Failure+at+Scania+Trucks)
- **Samples**: 60,000 training examples (only ~1,000 positives)
- **Features**: 170 numeric attributes + 1 binary class label
- **Problem**: Highly imbalanced binary classification (positive class ≈ 1.6%)

---

## 🧪 Project Workflow

### 1️⃣ Data Preparation
- **Missing Value Handling**: Applied imputation strategy (e.g., mean or constant value).
- **Exploration**:
  - Calculated Coefficient of Variation (CV) for all features.
  - Plotted a correlation matrix.
  - Visualized top √170 ≈ 13 features via scatter and box plots.
- **Class Imbalance Check**: Identified significant imbalance in class distribution.

---

## 🌲 Random Forest Experiments

### 🔹 Baseline Random Forest (No Compensation)
- Trained on original data
- Evaluation: Confusion Matrix, ROC Curve, AUC, Misclassification Rate
- Computed Out-of-Bag (OOB) error and compared with test error

### 🔹 Balanced Random Forest
- Used class weighting to compensate for imbalance
- Re-ran experiments and compared metrics with baseline

---

## ⚡ XGBoost with L1-Penalized Logistic Regression

- Used XGBoost with linear logistic regression model at each node
- Applied cross-validation (e.g., 5-fold) to determine regularization parameter α
- Evaluated:
  - Confusion Matrix
  - ROC Curve and AUC
  - Cross-validation vs. test set performance

---

## 🔁 SMOTE (Synthetic Minority Oversampling Technique)

- Applied SMOTE to address imbalance before model training
- Trained XGBoost (same as above) on SMOTE-augmented dataset
- Compared performance with uncompensated version
- Ensured correct handling of CV (no data leakage from SMOTE)

---

## 📈 Results Summary

| Model                  | Class Compensation | CV Method | AUC (Test) | Comments |
|-----------------------|--------------------|-----------|------------|----------|
| Random Forest         | ❌ No              | —         | —          | Baseline |
| Random Forest         | ✅ Class Weights   | —         | —          | Improved balance |
| XGBoost               | ❌ No              | 5-Fold    | —          | Baseline boosting |
| XGBoost + SMOTE       | ✅ SMOTE           | 5-Fold    | —          | Best performance |

*(See notebook for actual numeric metrics.)*

---

## 🛠️ Tools & Libraries

- Python 3
- pandas, NumPy
- scikit-learn (RandomForestClassifier, metrics, imputation)
- imbalanced-learn (SMOTE)
- XGBoost (`xgboost.XGBClassifier`)
- matplotlib, seaborn (visualizations)

---

## 📂 File Structure

```
RandomForest_XGBoost_SMOTE.ipynb      # Full notebook with preprocessing, modeling, and evaluation
README.md                             # Project summary (this file)
aps_failure_training_set.csv          # APS training data (if included)
```

---

## 📫 Contact

If you're interested in learning more or collaborating, feel free to connect via [LinkedIn](https://www.linkedin.com/in/navya-bhat).
