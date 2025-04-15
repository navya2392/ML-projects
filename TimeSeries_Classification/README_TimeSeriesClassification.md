# 📊 Time Series Classification using Logistic Regression and Feature Engineering

This project focuses on classifying human activity based on time series sensor data from the [AReM dataset](https://archive.ics.uci.edu/ml/datasets/Activity+Recognition+system+based+on+Multisensor+data+fusion+%28AReM%29). 

I used both **binary** and **multi-class** classification techniques with **extensive feature extraction**, including:
- Time-domain statistics
- Feature selection via p-values and L1-regularization
- Recursive Feature Elimination (RFE)
- Stratified Cross-Validation and performance evaluation with ROC and AUC

## 📁 Dataset Overview

- 7 activities: `bending1`, `bending2`, `cycling`, `lying`, `sitting`, `standing`, `walking`
- Each activity folder contains multiple instances of 6 time series:
  - `avg_rss12`, `var_rss12`, `avg_rss13`, `var_rss13`, `vg_rss23`, `ar_rss23`
- Each time series: 480 time steps
- Training/Test split:
  - Use first 2 instances from `bending1` and `bending2`, and first 3 instances from other activities as **test set**
  - Remaining data is used as **training set**

## 🧪 Feature Engineering

For each of the 6 time series per instance, we compute:
- Minimum
- Maximum
- Mean
- Median
- Standard Deviation
- First Quartile (25th percentile)
- Third Quartile (75th percentile)

Features are also extracted after splitting each time series into `l` segments (`l = 1 to 20`) to improve granularity.

## 🔍 Classification Experiments

### 1️⃣ Binary Classification: Bending vs. Other Activities
- **Model:** Logistic Regression
- **Feature Selection:** 
  - Using p-values from regression coefficients
  - Recursive Feature Elimination (RFE)
- **Evaluation:**
  - Confusion Matrix, ROC Curve, AUC
  - Cross-validation accuracy vs. Test accuracy
- **Case-Control Adjustment:** Applied if class imbalance observed

### 2️⃣ Binary Classification with L1-Regularization
- **Model:** L1-Penalized Logistic Regression
- **Cross-validated parameters:** Number of segments `l` and regularization strength `C`

### 3️⃣ Multi-class Classification
- **Models:** 
  - Multinomial Logistic Regression with L1-penalty
  - Naïve Bayes (Gaussian and Multinomial)
- **Evaluation:**
  - Accuracy on test set
  - Multiclass ROC and Confusion Matrix analysis

## 📈 Results Summary

- Feature extraction significantly improved classification performance.
- L1-penalized logistic regression was more scalable and easier to automate than p-value based selection.
- Multinomial logistic regression with L1-penalty showed robust performance across activities.

## 🛠️ Tools & Libraries

- Python 3
- NumPy, pandas, scikit-learn
- Matplotlib, seaborn (for visualization)
- Bootstrap for confidence intervals
- StratifiedKFold for cross-validation

## 📂 File Structure

```
TimeSeriesClassification.ipynb        # Main notebook with all experiments
README.md                             # Project summary (this file)
data/AReM                             # time series data
```
