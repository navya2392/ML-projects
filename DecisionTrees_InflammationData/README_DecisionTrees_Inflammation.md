# 🌡️ Acute Inflammations Classification using Decision Trees

This project applies decision tree classifiers to the **Acute Inflammations** dataset to demonstrate interpretable modeling using rule-based systems. 

---

## 📁 Dataset Overview

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Acute+Inflammations)
- **Objective**: Predict whether a patient has one or more of the following conditions:
  - Acute Inflammation of Urinary Bladder
  - Acute Nephritis

- **Features**: Binary medical symptoms (e.g., Temperature > 37.8°C, Lumbar pain, Micturition problems, etc.)
- **Labels**: Multi-label (converted into multiclass using label powerset strategy)

---

## 🌲 Decision Tree Modeling

### 1️⃣ Full Tree
- Trained a decision tree on the full dataset (no train/test split).
- Visualized the full tree using `sklearn.tree.plot_tree`.

### 2️⃣ IF-THEN Rule Extraction
- Translated tree decision paths into **interpretable IF-THEN rules**.
- These rules allow clinicians to understand how symptoms lead to diagnoses.

### 3️⃣ Cost-Complexity Pruning
- Applied **cost-complexity pruning** to simplify the model.
- Selected `ccp_alpha` using cross-validation to trade off between accuracy and interpretability.
- Generated a **pruned tree** and corresponding simplified IF-THEN rules.

---

## 🧠 Key Takeaways

- Decision trees provide intuitive, rule-based interpretability that aligns well with diagnostic reasoning.
- Pruning helps generalize better by reducing overfitting while preserving interpretability.
- Label powerset transformation was used to handle multi-label classification.

---

## 🛠️ Tools & Libraries

- Python 3
- scikit-learn (DecisionTreeClassifier, plot_tree, cost-complexity pruning)
- matplotlib (for visualization)
- pandas, NumPy

---

## 📂 File Structure

```
DecisionTrees_Inflammation.ipynb      # Notebook with decision tree models and visualizations
README.md                             # Summary of the Decision Tree component (this file)
diagnosis.data                        # Raw data file
```

---

## 📫 Contact

Feel free to connect via [LinkedIn](https://www.linkedin.com/in/navya-bhat) if you'd like to learn more or collaborate!
