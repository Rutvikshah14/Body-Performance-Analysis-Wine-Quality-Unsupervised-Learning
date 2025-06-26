# Body-Performance-Analysis-Wine-Quality-Unsupervised-Learning

# 🤖 Machine Learning Project: Clustering & Ensemble Analysis

## 📌 Author
**Rutvik Shah**

---

## 🧠 Project Overview

This project presents a comprehensive machine learning analysis using **clustering algorithms**, **dimensionality reduction techniques**, and **ensemble models**. The goal was to evaluate the effects of dimensionality reduction on clustering quality and predictive performance using two real-world datasets.

---

## 📁 Datasets

### 1. Body Performance Dataset
- 📍 [Link](https://www.kaggle.com/datasets/kukuroo3/body-performance-data)
- **Participants:** 13,394
- **Attributes:** Age, gender, height, weight, blood pressure, body fat %, grip strength, flexibility, core strength, and leg power.
- **Target:** Performance class (A–D)

### 2. White Wine Quality Dataset
- 📍 [Link](https://www.kaggle.com/datasets/piyushagni5/white-wine-quality)
- **Observations:** 4,898
- **Features:** Alcohol, sugar, acidity, pH, sulfates, etc.
- **Target:** Quality score (0–10)

---

## 🔍 Exploratory Data Analysis & Preprocessing

### Common Steps
- Removed duplicates
- Encoded categorical features
- Normalized data using `RobustScaler`
- Random 10% sampling for performance optimization

### White Wine Dataset Specific
- Transformed quality into binary class (good vs not good)
- Correlation matrix visualized via heatmap

---

## 🔄 Dimensionality Reduction Techniques

### 🔷 PCA (Principal Component Analysis)
- Scikit-learn implementation
- Variance explained via elbow plot
- Retained 2–3 principal components

### 🔶 UMAP (Uniform Manifold Approximation and Projection)
- Applied on both training and test sets
- Effective 2D visualization of complex relationships

---

## 📊 Clustering Methods

### ✅ KMeans
- Body Dataset: Optimal clusters = 3 (original), 2 (PCA), 3 (UMAP)
- Wine Dataset: Optimal clusters = 3 (all cases)

### ✅ Agglomerative Clustering
- Body Dataset: 2 (original), 3 (PCA), 3 (UMAP)
- Wine Dataset: 3 (original & PCA), 4 (UMAP)

### 📌 Insight:
UMAP generally preserved cluster structure better, especially in the wine dataset.

---

## 🌲 Ensemble Modeling

### ✅ AdaBoost
- Tuned using `GridSearchCV`
- Best parameters: `n_estimators=100`, `learning_rate=0.1`
- Accuracy (Wine): ~66%–68%

### ✅ Random Forest
- Tuned using `GridSearchCV`
- Best parameters: `n_estimators=100`, `max_depth=None`, `min_samples_split=10`
- Accuracy (Wine): Up to **71.43%**

---

## 📈 Results Summary

| Dataset          | Model        | Accuracy | Precision | Recall | F1 Score |
|------------------|--------------|----------|-----------|--------|----------|
| Body (Original)  | RF           | 61%      | 0.60      | 0.61   | 0.60     |
| Body (PCA)       | RF           | 57%      | 0.57      | 0.57   | 0.55     |
| Body (UMAP)      | RF           | 45%      | 0.45      | 0.45   | 0.44     |
| Wine (Original)  | RF           | 67%      | 0.66      | 0.67   | 0.66     |
| Wine (PCA)       | RF           | **74%**  | 0.73      | 0.74   | 0.72     |
| Wine (UMAP)      | RF           | 69%      | 0.68      | 0.69   | 0.68     |

---

## 🔍 Key Takeaways

- **Random Forest consistently outperformed AdaBoost**, especially on PCA-transformed data.
- **UMAP** revealed hidden cluster structures in wine quality data not visible through PCA.
- **Dimensionality reduction (PCA/UMAP)** helped visualize data and slightly improved performance in some cases.
- **Clustering quality varied by method and dataset**, with Agglomerative Clustering showing robustness.

---

## 📚 References

- [Body Performance Data](https://www.kaggle.com/datasets/kukuroo3/body-performance-data)
- [White Wine Quality Data](https://www.kaggle.com/datasets/piyushagni5/white-wine-quality)
- [KMeans - W3Schools](https://www.w3schools.com/python/python_ml_k-means.asp)
- [Agglomerative Clustering - Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)

---

## ✅ Future Work

- Explore other dimensionality reduction techniques like t-SNE.
- Try other ensemble methods: Gradient Boosting, XGBoost.
- Perform feature importance and interpretability analysis.

---

> 📌 *This project demonstrates the synergy between dimensionality reduction and ensemble learning techniques for both classification and clustering tasks.*
