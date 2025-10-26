This repository contains **Machine Learning** and **Deep Learning** projects developed as part of the **DSCI 552 (Machine Learning for Data Science)** course at the **University of Southern California (USC)**.  
Each folder represents a unique project focusing on various ML/DL models, techniques, and evaluation strategies.

---

## 🔍 Project Summaries

### 1️⃣ Deep Learning Image Classification
**Objective**: Build and evaluate deep convolutional neural networks (CNNs) for multi-class image classification using transfer learning.  
**Models Used**: VGG16, ResNet50, ResNet101, EfficientNetB0  
**Approach**:  
- Used **transfer learning** from ImageNet weights.  
- Applied **data augmentation**, **early stopping**, and **model checkpointing**.  
- Compared models on accuracy, validation loss, and convergence time.ResNet architectures performed best overall.  
**Tools Used**: `TensorFlow`, `Keras`, `NumPy`, `Pandas`, `Matplotlib`, `Seaborn`  
📂 Folder: `DeepLearning_Classification/`

---

### 2️⃣ Random Forest, XGBoost, and SMOTE
**Objective**: Classify rare truck failures using a highly imbalanced dataset from Scania trucks.  
**Models and Methodologies**: Random Forests (with/without class balancing), SMOTE oversampling, and XGBoost with L1-penalized logistic regression.  
**Tools Used**: `scikit-learn`, `imbalanced-learn`, `XGBoost`, `pandas`, `NumPy`, `matplotlib`.  
📂 Folder: `RandomForest_XGBoost_SMOTE_TruckData/`

---

### 3️⃣ Ridge, LASSO, and Boosting
**Objective**: Predict violent crime rates in US communities using regression models.  
**Approach**: Applied linear regression, ridge regression, LASSO, PCR, and XGBoost comparison.  
**Tools Used**: `scikit-learn`, `XGBoost`, `pandas`, `NumPy`, `matplotlib`, `seaborn`.  
📂 Folder: `Ridge_Lasso_Boosting_CrimeData/`

---

### 4️⃣ SVM Classification and K-Means Clustering
**Objective**: Perform multi-label classification and clustering on Anuran Calls using MFCC features.  
**Approach**:  
- Trained SVMs with Gaussian & L1-penalized kernels using binary relevance.  
- Applied SMOTE for imbalance and K-means clustering for pattern discovery.  
**Tools Used**: `scikit-learn`, `SMOTE`, `pandas`, `NumPy`, `matplotlib`, `seaborn`.  
📂 Folder: `SVM_Multilabel_Multiclass/`

---

### 5️⃣ Time Series Classification
**Objective**: Classify human activities based on time-series data from the AReM dataset.  
**Approach**:  
- Extracted temporal features, applied logistic regression with feature selection.  
- Compared multinomial logistic regression vs Naive Bayes.  
**Tools Used**: `scikit-learn`, `pandas`, `NumPy`, `matplotlib`, `seaborn`.  
📂 Folder: `TimeSeries_Classification/`

---

### 6️⃣ Decision Trees for Medical Diagnosis
**Objective**: Diagnose inflammation types using interpretable decision trees.  
**Approach**:  
- Constructed full trees, extracted IF-THEN rules, and pruned for interpretability.  
**Tools Used**: `scikit-learn`, `pandas`, `matplotlib`, `NumPy`.  
📂 Folder: `DecisionTrees_InflammationData/`

---

### 7️⃣ Regression on Power Plant Data
**Objective**: Predict energy output from environmental variables.  
**Approach**:  
- Applied multiple and polynomial regression, interaction terms, and KNN regression.  
**Tools Used**: `scikit-learn`, `pandas`, `NumPy`, `matplotlib`, `seaborn`.  
📂 Folder: `Regression_PowerPlantData/`

---

### 8️⃣ KNN Classification
**Objective**: Predict vertebral condition (normal vs abnormal) using biomechanical features.  
**Approach**:  
- Compared unweighted and weighted KNN models with different distance metrics.  
- Evaluated with F1-score, precision, recall, and learning curves.  
**Tools Used**: `scikit-learn`, `pandas`, `NumPy`, `matplotlib`, `seaborn`.  
📂 Folder: `KNNClassification_VertebralColumn/`

---

## 🧠 Tech Stack & Tools

- **Languages**: Python 3  
- **ML/DL Frameworks**: TensorFlow, Keras, scikit-learn, XGBoost  
- **Libraries**: pandas, NumPy, matplotlib, seaborn  
- **Additional Tools**: imbalanced-learn, SMOTE, PCA, RFE  
- **Environment**: Jupyter Notebook, Google Colab

- 
