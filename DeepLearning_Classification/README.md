# 🧠 Deep Learning Image Classification

This project applies **Convolutional Neural Networks (CNNs)** and **Transfer Learning** techniques to a multi-class image classification problem using **TensorFlow** and **Keras**.  
The notebook explores multiple architectures including **VGG16**, **ResNet50**, **ResNet101**, and **EfficientNetB0**, comparing their performance on the same dataset.

---

## 🎯 Project Overview
The goal of this project is to build and evaluate deep learning models capable of accurately classifying images into multiple categories.  
The notebook walks through:
- Data preprocessing and augmentation  
- Model architecture design  
- Training, validation, and hyperparameter tuning  
- Evaluation and insights from model performance  

---

## 📁 Folder Contents
- **DeepLearning_ImageClassification.ipynb** — Main notebook (data processing, model training, evaluation).  
- **Final Project.pdf** — Project report summarizing methodology and results.  
- **DeepLearning_ImageClassification_Outputs.txt** — Key output metrics extracted from the notebook.  
- **README.md** — This documentation file.

---

## ⚙️ Methodology

### **1. Dataset**
- Images organized into training, validation, and test sets.
- Preprocessing includes **resizing**, **normalization**, and **augmentation** (rotation, zoom, flip).
- Data loaded using `ImageDataGenerator` for efficient batch processing.


### ***2. 🧰 Tech Stack Used**
| Category | Technologies |
|-----------|--------------|
| **Language** | Python |
| **Deep Learning Frameworks** | TensorFlow, Keras |
| **Data Processing** | NumPy, Pandas, scikit-learn |
| **Visualization Tools** | Matplotlib, Seaborn |
| **Environment** | Jupyter Notebook |
| **Hardware** | GPU-enabled environment (Google Colab)

### **3. Model Architectures**
- **Baseline CNN:** Custom-built with 3 convolutional layers + pooling and fully connected dense layers.  
- **Transfer Learning Models:**  
  - VGG16  
  - ResNet50  
  - ResNet101  
  - EfficientNetB0  
- Pre-trained on ImageNet; top layers fine-tuned for classification task.  
- **Optimizer:** Adam  
- **Loss Function:** Categorical Crossentropy  
- **Metric:** Accuracy  

### **4. Training**
- Early stopping and model checkpointing used to prevent overfitting.  
- Trained for 20–50 epochs depending on model convergence.  
- Validation accuracy used to determine best-performing checkpoint.  

---

## 📊 Evaluation

- **Metrics Used:** Accuracy, Precision, Recall, F1-score, Confusion Matrix  
- **Analysis:** Confusion matrices reveal misclassifications mostly between visually similar classes.  
- **Visualization:** Training and validation curves show early convergence for transfer learning models.  

---

## 📈 Summary of Results

Test accuracy across all models is fairly close, with **ResNet50** performing slightly better overall.  
Results vary slightly across runs due to stochastic training behavior.

| Model | Test Accuracy (Range) |
|:------|:----------------------|
| VGG16 | 74–75% |
| ResNet50 | **78–80%** |
| ResNet101 | 76–78% |
| EfficientNetB0 | 74–76% |

**Observations:**
- ResNet architectures consistently outperform others due to deeper residual learning.  
- EfficientNet shows good efficiency but slightly lower accuracy on this dataset.  
- All models generalize well after augmentation and regularization.  

---

## 💡 Key Takeaways
- Transfer learning improves accuracy by leveraging pre-trained weights.  
- Early stopping and dropout effectively control overfitting.  
- ResNet-based models achieve the best trade-off between depth and performance.  
- Future improvements could include data balancing or model ensembling for marginal gains.  

---

## 🧩 Tools & Libraries
- **Frameworks:** TensorFlow, Keras  
- **Data Handling:** NumPy, Pandas, scikit-learn  
- **Visualization:** Matplotlib, Seaborn  
- **Environment:** Jupyter Notebook  

---
