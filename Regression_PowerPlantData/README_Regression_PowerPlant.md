# ⚡ Power Plant Energy Output Prediction using Regression Models

This project explores the use of both **linear regression** and **K-nearest neighbors (KNN) regression** to predict the net hourly electrical energy output of a Combined Cycle Power Plant. The dataset is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant).

---

## 📁 Dataset Description

The dataset includes 9568 rows and 5 columns:
- **Features**:
  - Ambient Temperature (T) in °C
  - Ambient Pressure (AP) in millibar
  - Relative Humidity (RH) in %
  - Exhaust Vacuum (V) in cm Hg
- **Target Variable**:
  - Electrical Energy Output (EP) in MW

Data represents operational measurements taken from a Combined Cycle Power Plant over 6 years.

---

## 🧪 Project Workflow

### 1️⃣ Exploratory Data Analysis (EDA)
- Count of observations and features.
- Pairwise scatterplots between features and target variable.
- Summary statistics: mean, median, range, quartiles, IQR.
- Detection and visualization of outliers.

### 2️⃣ Simple Linear Regression
- Fitted separate regression models for each predictor to assess their individual association with EP.
- Visualized residuals and checked statistical significance (p-values).

### 3️⃣ Multiple Linear Regression
- Modeled EP using all predictors together.
- Assessed which predictors remain significant when others are included.
- Plotted simple vs. multiple regression coefficients to analyze multicollinearity effects.

### 4️⃣ Polynomial Regression (Nonlinear Associations)
- Explored cubic polynomial fits for each predictor.
- Evaluated if higher-degree terms (X², X³) improve predictive performance.

### 5️⃣ Interaction Effects
- Included pairwise interactions between all predictors.
- Analyzed statistical significance of interaction terms.
- Built an optimized model with only significant interactions and nonlinear terms.

---

## 🔁 Model Evaluation

- Split data: 70% train, 30% test
- Compared:
  - Linear model with only raw predictors
  - Full model with interaction and polynomial terms (after pruning insignificant ones)
- Metrics: Train and Test Mean Squared Error (MSE)

---

## 🤖 KNN Regression

- Applied K-Nearest Neighbors Regression (with and without feature normalization)
- Searched for optimal k in the range 1 to 100
- Plotted Train and Test Errors vs. 1/k
- Compared KNN performance with best linear regression model

---

## 📈 Results Summary

- Multiple linear regression captured much of the variability in EP using raw predictors.
- Including nonlinear and interaction terms improved performance marginally.
- KNN regression with normalized features performed competitively, especially at low values of k.
- Final comparison provided insights into model flexibility vs. interpretability tradeoffs.

---

## 🛠️ Tools & Libraries

- Python 3
- pandas, NumPy
- matplotlib, seaborn
- scikit-learn (LinearRegression, KNN, PolynomialFeatures, train_test_split)

---

## 📂 File Structure

```
Regression_PowerPlant.ipynb           # Main notebook with analysis
README.md                             # Project summary (this file)
Folds5x2_pp.xlsx                      # Raw data file 
```

---

## 📫 Contact

For questions or collaboration, feel free to connect via [LinkedIn](https://www.linkedin.com/in/navya-bhat).
