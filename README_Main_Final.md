# 📚 ML Projects Portfolio – DSCI 552

This repository contains machine learning projects developed as part of the DSCI 552 (Machine Learning for Data Science) course at USC. Each folder represents a unique assignment or project focused on applying different machine learning models, techniques, and evaluation strategies.

---

## 🔍 Project Summaries

### 1️⃣ Random Forest, XGBoost, and SMOTE
**Objective**: Classify rare truck failures using a highly imbalanced dataset from Scania trucks.  
**Models and Methodologies**: Trained Random Forests with and without class balancing, applied SMOTE for oversampling, and used XGBoost with L1-penalized logistic regression. Evaluated model performance with AUC, confusion matrices, and out-of-bag error rates.  
**Tools Used**: `scikit-learn`, `imbalanced-learn`, `XGBoost`, `pandas`, `NumPy`, `matplotlib`.  
📂 Folder: `RandomForest_XGBoost_SMOTE/`

---

### 2️⃣ Ridge, LASSO, and Boosting
**Objective**: Predict violent crime rates in US communities using regression models.  
**Models and Methodologies**: Applied linear regression, ridge regression, LASSO (with and without standardization), and principal component regression (PCR). Final model comparison included XGBoost as a nonlinear benchmark.  
**Tools Used**: `scikit-learn`, `XGBoost`, `pandas`, `NumPy`, `matplotlib`, `seaborn`.  
📂 Folder: `Ridge_Lasso_Boosting/`

---

### 3️⃣ SVM Classification and K-Means Clustering
**Objective**: Perform multi-label classification and clustering on Anuran Calls using MFCC features.  
**Models and Methodologies**: Trained SVMs with Gaussian and L1-penalized kernels using binary relevance for multi-label tasks; applied SMOTE for imbalance. K-means clustering was evaluated over 50 simulations using Hamming scores.  
**Tools Used**: `scikit-learn`, `SMOTE`, `pandas`, `NumPy`, `matplotlib`, `seaborn`.  
📂 Folder: `SVM_KMeansClustering/`

---

### 4️⃣ Time Series Classification
**Objective**: Classify human activities based on sensor time series data from the AReM dataset.  
**Models and Methodologies**: Extracted time-domain features and trained logistic regression models with feature selection (p-values, L1-regularization). Extended to multi-class classification using multinomial logistic regression and Naive Bayes.  
**Tools Used**: `scikit-learn`, `pandas`, `NumPy`, `matplotlib`, `seaborn`.  
📂 Folder: `TimeSeriesClassification/`

---

### 5️⃣ Decision Trees for Medical Diagnosis
**Objective**: Diagnose inflammation types using interpretable decision trees.  
**Models and Methodologies**: Built and visualized full decision trees, extracted IF-THEN rules, and applied cost-complexity pruning for simplicity. Focused on clinical interpretability through rule-based modeling.  
**Tools Used**: `scikit-learn`, `pandas`, `matplotlib`, `NumPy`.  
📂 Folder: `DecisionTrees_Inflammation/`

---

### 6️⃣ Regression on Power Plant Data
**Objective**: Predict energy output of a Combined Cycle Power Plant from environmental variables.  
**Models and Methodologies**: Used simple and multiple linear regression, explored polynomial and interaction terms, and compared performance with KNN regression. Analyzed coefficients, residuals, and MSE metrics.  
**Tools Used**: `scikit-learn`, `pandas`, `NumPy`, `matplotlib`, `seaborn`.  
📂 Folder: `Regression_PowerPlant/`

---

### 7️⃣ KNN Classification
**Objective**: Predict vertebral condition (normal vs abnormal) using biomechanical measurements.  
**Models and Methodologies**: Implemented unweighted and weighted KNN models, evaluated performance with different distance metrics and training sizes. Built learning curves and analyzed confusion matrix, F1-score, and precision-recall.  
**Tools Used**: `scikit-learn`, `pandas`, `NumPy`, `matplotlib`, `seaborn`.  
📂 Folder: `KNNClassification/`

---

## 🛠️ Common Tools & Libraries

- Python 3
- pandas, NumPy
- scikit-learn
- matplotlib, seaborn
- XGBoost, imbalanced-learn
- SMOTE, PCA, RFE

---

## 📫 Contact

Created by [Navya Bhat](https://www.linkedin.com/in/navya-bhat)  
For questions or collaboration inquiries, feel free to connect!
