# 🏙️ Crime Rate Prediction using Regression, LASSO, and Boosting Models

This project explores various regression techniques—including linear regression, ridge regression, LASSO, principal component regression (PCR), and gradient boosting—for predicting violent crime rates using the **Communities and Crime** dataset. 

---

## 📁 Dataset Overview

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime)
- **Objective**: Predict violent crime per population across 1994 US communities
- **Features**: Socioeconomic, law enforcement, demographic, and policy-related data (~100+ predictors)

---

## 🧪 Project Workflow

### 1️⃣ Data Preparation
- Used first 1495 rows as training set, remaining as test set.
- Removed non-predictive attributes (as documented).
- Handled missing values using imputation techniques.

### 2️⃣ Exploratory Analysis
- Visualized correlation matrix to identify strongly correlated features.
- Calculated **Coefficient of Variation (CV)** for each feature.
- Selected top √128 ≈ 11 features with highest CV for closer inspection via scatter and box plots.

### 3️⃣ Regression Models

#### 🔹 Linear Regression
- Fit on training data using least squares
- Computed test error (Mean Squared Error)

#### 🔹 Ridge Regression
- Used cross-validation to select λ (regularization strength)
- Fit model and reported test MSE

#### 🔹 LASSO Regression
- Fit model using cross-validation to select λ
- Repeated both with:
  - Raw features
  - Standardized features
- Reported test MSE and selected variables

#### 🔹 Principal Component Regression (PCR)
- Applied PCA on feature set
- Chose number of components (M) via cross-validation
- Fit final model and computed test error

---

## 🚀 Gradient Boosting with XGBoost

- Trained **L1-penalized gradient boosting trees**
- Used cross-validation to tune the regularization parameter α
- Compared model performance with previous regression approaches

---

## 📈 Results Summary

| Model                  | Regularization / Tuning        | Notes |
|-----------------------|-------------------------------|-------|
| Linear Regression     | None                          | Baseline |
| Ridge Regression      | λ via CV                      | Regularized |
| LASSO                 | λ via CV                      | Sparse model |
| PCR                   | M via CV                      | Reduced dimensionality |
| XGBoost (Boosting)    | α via CV                      | Nonlinear model |

*(Please refer to notebook for exact numerical results)*

---

## 🛠️ Tools & Libraries

- Python 3
- pandas, NumPy
- scikit-learn (LinearRegression, RidgeCV, LassoCV, PCA)
- matplotlib, seaborn
- XGBoost for boosting trees

---

## 📂 File Structure

```
Ridge_Lasso_Boosting.ipynb            # Main notebook with all regression models and boosting
README.md                             # Summary of the project (this file)
communities.csv                 # Cleaned and preprocessed dataset (if included)
```

---

## 📫 Contact

Feel free to connect via [LinkedIn](https://www.linkedin.com/in/navya-bhat) if you'd like to learn more or collaborate.
