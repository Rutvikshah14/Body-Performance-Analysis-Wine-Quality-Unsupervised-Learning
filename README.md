# Body-Performance-Analysis-Wine-Quality-Unsupervised-Learning

# 🤖 Machine Learning Project: Clustering & Ensemble Analysis

## 👤 Author
**Rutvik Shah**

---

## 📌 Overview

This project investigates the effect of **dimensionality reduction** on clustering and classification performance using two real-world datasets. It integrates:

- **Clustering algorithms**: KMeans, Agglomerative
- **Dimensionality reduction**: PCA, UMAP
- **Ensemble classifiers**: AdaBoost, Random Forest

---

## 📁 Datasets

### 1. Body Performance
🔗 [Dataset Link](https://www.kaggle.com/datasets/kukuroo3/body-performance-data)

- 13,394 participants
- 12 attributes (e.g., age, gender, weight, blood pressure, flexibility, strength)
- Target: Performance Class (A–D)

### 2. White Wine Quality  
🔗 [Dataset Link](https://www.kaggle.com/datasets/piyushagni5/white-wine-quality)

- 4,898 samples
- 12 physicochemical features
- Target: Wine Quality (0–10 → Binarized)

---

## 🔍 Preprocessing Summary

### Shared Steps:
- Duplicate removal
- Categorical encoding (e.g., gender → binary)
- 10% random sampling for efficiency
- Feature scaling via `RobustScaler`

### Specific to Wine Dataset:
- Quality transformed into binary class (good/bad)
- Heatmap used for feature correlation visualization

---

## 🔄 Dimensionality Reduction

### PCA (Principal Component Analysis)
- Scaled input before PCA
- Used elbow plots to determine component count

### UMAP (Uniform Manifold Approximation & Projection)
- Visualizes similar data closer in low-dimensional space

---

## 📊 Clustering Results

### Best Number of Clusters

| Dataset | Method      | Original | PCA | UMAP |
|---------|-------------|----------|-----|------|
| Body    | KMeans      | 3        | 2   | 3    |
| Body    | Agglomerative | 2      | 3   | 3    |
| Wine    | KMeans      | 3        | 3   | 3    |
| Wine    | Agglomerative | 3      | 3   | 4    |

---

## 🌲 Ensemble Models & Performance

### 🔶 Body Performance Dataset

| Transformation | Model        | Accuracy | Precision | Recall | F1 Score |
|----------------|--------------|----------|-----------|--------|----------|
| Original       | AdaBoost     | 54%      | 0.56      | 0.54   | 0.53     |
| Original       | Random Forest| **61%**  | 0.60      | 0.61   | 0.60     |
| PCA            | AdaBoost     | 56%      | 0.57      | 0.56   | 0.56     |
| PCA            | Random Forest| 57%      | 0.57      | 0.57   | 0.55     |
| UMAP           | AdaBoost     | 45%      | 0.48      | 0.45   | 0.43     |
| UMAP           | Random Forest| 45%      | 0.45      | 0.45   | 0.44     |

### 🔷 White Wine Quality Dataset

| Transformation | Model        | Accuracy | Precision | Recall | F1 Score |
|----------------|--------------|----------|-----------|--------|----------|
| Original       | AdaBoost     | **68%**  | 0.67      | 0.68   | 0.65     |
| Original       | Random Forest| 67%      | 0.66      | 0.67   | 0.66     |
| PCA            | AdaBoost     | 63%      | 0.61      | 0.63   | 0.61     |
| PCA            | Random Forest| **74%**  | 0.73      | 0.74   | 0.72     |
| UMAP           | AdaBoost     | 67%      | 0.66      | 0.67   | 0.66     |
| UMAP           | Random Forest| 69%      | 0.68      | 0.69   | 0.68     |

---

## ⚙️ Hyperparameter Tuning

### AdaBoost
- Best Parameters:
  - `n_estimators`: 50 or 100
  - `learning_rate`: 0.1 or 1.0
- Tuned with `GridSearchCV`

### Random Forest
- Best Parameters:
  - `n_estimators`: 100
  - `max_depth`: None or 20
  - `min_samples_split`: 2 or 10
  - `min_samples_leaf`: 1

---

## ✅ Conclusions

- **Random Forest outperforms AdaBoost** overall in most scenarios.
- **PCA** often improves model performance slightly.
- **UMAP** is excellent for visualizing clusters but sometimes hurts model accuracy.
- **AdaBoost performs best on original white wine data** (68%).
- **Random Forest on PCA-transformed wine data** yields the highest overall accuracy (74%).

---

## 📚 References

- [Body Performance Dataset](https://www.kaggle.com/datasets/kukuroo3/body-performance-data)  
- [White Wine Quality Dataset](https://www.kaggle.com/datasets/piyushagni5/white-wine-quality)  
- [Scikit-learn Docs](https://scikit-learn.org/stable/)  
- [KMeans - W3Schools](https://www.w3schools.com/python/python_ml_k-means.asp)

---

> 🧠 *This project blends unsupervised and supervised learning with data transformation techniques to evaluate both cluster structures and classification models.*
