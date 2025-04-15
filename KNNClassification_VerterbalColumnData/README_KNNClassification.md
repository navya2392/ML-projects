# 🔍 KNN Classification on Vertebral Column Data

This project involves applying and analyzing the performance of **K-Nearest Neighbors (KNN)** on the Vertebral Column Data Set from the UCI Machine Learning Repository. 
Source: https://archive.ics.uci.edu/ml/datasets/Vertebral+Column


---

## 📁 Dataset: Vertebral Column

The dataset includes six biomechanical features derived from spinal shape and orientation:

- Pelvic incidence
- Pelvic tilt
- Lumbar lordosis angle
- Sacral slope
- Pelvic radius
- Grade of spondylolisthesis

### Class Labels:
- **NO (Normal)** → 0
- **AB (Abnormal)** → 1

Only binary classification (NO vs. AB) is considered in this work.

---

## 🧪 Project Workflow

### 1️⃣ Data Preprocessing and EDA
- Scatterplots of features with class-wise coloring.
- Boxplots to observe feature distribution across classes.
- Data split: First 70 samples from Class 0 and first 140 from Class 1 used for training; remaining used for testing.

### 2️⃣ KNN Classification (Unweighted)
- Implemented KNN using Euclidean distance.
- Evaluated train and test errors across multiple values of `k` from 208 to 1.
- Identified best performing `k*`, and computed:
  - Confusion Matrix
  - True Positive Rate, True Negative Rate
  - Precision, F1-Score

### 3️⃣ Learning Curve
- Explored the effect of training set size on test error.
- For each N ∈ {10, 20, ..., 210}, chose the best `k` from {1, 6, ..., N-4}.
- Plotted test error vs training set size.

---

## 🔁 Distance Metric Variants

Tested KNN with alternative distance metrics:

| Distance Metric      | Notes |
|----------------------|-------|
| **Manhattan (p=1)** | Special case of Minkowski |
| **Minkowski (log10(p) ∈ [0.1, 1])** | Custom tuning |
| **Chebyshev (p → ∞)** | Max coordinate difference |
| **Mahalanobis** | Accounts for feature correlation (with pseudoinverse where needed) |

---

## ⚖️ Weighted KNN

Weighted KNN replaces majority voting with weights ∝ 1/distance:

- Tested with Euclidean, Manhattan, and Chebyshev distances.
- Compared best test errors over a range of `k`.

---

## 🧠 Results Summary

- **Best k** identified via train/test error analysis.
- **Mahalanobis** distance showed improvement in some cases due to feature correlation.
- **Weighted KNN** consistently improved performance for most metrics.
- Learning curve showed diminishing returns after ~150 training samples.
- Lowest training error rate and best-performing configuration are highlighted in the notebook.

---

## 🛠️ Tools & Libraries

- Python 3
- pandas, NumPy, matplotlib, seaborn
- scikit-learn (KNN, distance metrics, metrics API)

---

## 📂 File Structure

```
KNNClassification.ipynb               # Main notebook
README.md                             # Project summary (this file)
column_2C.dat                         # Raw data
```

---

## 📫 Contact

For questions, feel free to reach out via [LinkedIn](https://www.linkedin.com/in/navya-bhat).
