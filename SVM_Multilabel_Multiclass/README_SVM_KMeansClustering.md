# 🐸 Multi-Label Classification and Clustering on Anuran Calls Dataset

This project explores both **multi-label classification** using Support Vector Machines (SVMs) and **unsupervised learning** using K-Means clustering on the Anuran Calls (MFCCs) dataset. It was developed as part of Homework 7 for the DSCI 552 (Machine Learning for Data Science) course at USC.

---

## 📁 Dataset Overview

- **Source**: [UCI Anuran Calls (MFCCs)](https://archive.ics.uci.edu/ml/datasets/Anuran+Calls+%28MFCCs%29)
- **Labels**: Each instance has 3 labels — `Family`, `Genus`, and `Species`
- **Goal**: Evaluate models for multi-class, multi-label classification and clustering performance

---

## 🧪 Part 1: Multi-Label SVM Classification

### 🔹 Tasks Performed
- Used **binary relevance strategy**: trained one SVM classifier per label
- Applied **Gaussian kernel SVMs** using one-vs-all strategy
- Evaluated models using:
  - Exact Match
  - Hamming Score / Hamming Loss

### 🔹 Extensions
- Repeated experiments using **L1-penalized linear SVMs**
- Used **SMOTE** to address class imbalance for each label
- 10-fold cross-validation used to tune hyperparameters (kernel width, penalty)
- (Optional practice: experimented with classifier chains and multi-label confusion matrices)

---

## 📊 Part 2: K-Means Clustering

### 🔹 Procedure
- Performed K-Means clustering on the entire dataset (unsupervised)
- Automatically determined `k ∈ {1, 2, ..., 50}` using **CH index** or silhouette analysis
- Assigned each cluster a label triplet (Family, Genus, Species) based on majority voting

### 🔹 Evaluation
- Compared predicted vs. actual multi-labels using:
  - **Hamming Distance**
  - **Hamming Loss**
  - **Hamming Score**
- Repeated the simulation 50 times to compute **average and standard deviation** of evaluation metrics

---

## 📈 Results Summary

- Gaussian SVMs performed well for each label when tuned with cross-validation.
- SMOTE helped improve performance for minority classes.
- Clustering-based label assignment produced non-trivial but weaker performance compared to supervised methods.
- Monte Carlo simulations provided robust metric estimates for clustering effectiveness.

---

## 🛠️ Tools & Libraries

- Python 3
- pandas, NumPy
- scikit-learn (SVC, KMeans, metrics, GridSearchCV)
- imbalanced-learn (SMOTE)
- matplotlib, seaborn

---

## 📂 File Structure

```
SVM_KMeansClustering.ipynb            # Main notebook for classification and clustering
README.md                             # This summary file
anuran_calls_data.csv                 # Original dataset (if included)
```

---

## 📫 Contact

Developed by [Navya Bhat](https://www.linkedin.com/in/navya-bhat).  
Feel free to connect for collaboration or questions!
