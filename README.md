# Face Recognition using Dimensionality Reduction (PCA & LDA)

This repository contains the implementation of a Face Recognition System using Matrix Factorization and Dimensionality Reduction techniques: **Principal Component Analysis (PCA)** and **Linear Discriminant Analysis (LDA)**[cite: 3]. This project was completed as part of the **Introduction to Machine Learning** curriculum at Alexandria National University[cite: 3].

## 📌 Project Overview
The objective is to correctly identify subjects from facial images using the **ORL Database of Faces** (40 subjects, 10 images per subject, grayscale, $92 \times 112$ pixels)[cite: 3]. Due to the high-dimensional nature of image data ($10,304$ features per image), PCA and LDA are leveraged to project the data into a lower-dimensional subspace before classification[cite: 3].

---

## 🔬 Core Implementations

### 1. Principal Component Analysis (PCA)
* **Objective:** Maximize the total variance of the data projected onto a lower-dimensional subspace[cite: 3].
* **Approach:** Centered the data matrix, computed the covariance matrix, and extracted eigenvalues/eigenvectors[cite: 3].
* **Variance Threshold ($\alpha$):** Evaluated performance across various total variance thresholds $\alpha = \{0.8, 0.85, 0.9, 0.95\}$ to dynamically choose the number of principal components ($r$)[cite: 3].

### 2. Linear Discriminant Analysis (LDA)
* **Objective:** Maximize between-class scatter ($S_b$) while minimizing within-class scatter ($S_w$) for multiclass classification[cite: 3].
* **Approach:** Handled multiclass LDA for 40 subjects, computing class-specific means and projection matrices utilizing the 39 dominant eigenvectors[cite: 3].

### 3. Classification & Tuning
* Used a **K-Nearest Neighbors (1-NN)** classifier as the base recognition engine[cite: 3].
* Tuned the $K$ hyperparameter ($K = \{1, 3, 5, 7\}$) to compare performance variations across both PCA and LDA reduced spaces[cite: 3].

---

## 📊 Experimental Setup & Methodology
* **Data Matrix Generation:** Flattened each $92 \times 112$ image into a $10,304$-dimensional vector, stacking them into a $400 \times 10,304$ Data Matrix ($D$)[cite: 3].
* **Train/Test Splitting:** 
  * *Default Split:* Kept odd rows for training and even rows for testing (5 instances per person for training, 5 for testing)[cite: 3].
  * *Alternative Split:* Evaluated a 70/30 split per subject (7 training instances, 3 testing instances) to assess the impact of data volume on accuracy[cite: 3].

---

## 🛠️ Tech Stack & Tools Used
* **Language:** Python[cite: 3]
* **Libraries:** NumPy, Pandas, Scikit-Learn, Matplotlib (for performance plotting)
* **Environment:** Jupyter Notebook

## 📈 Key Insights & Results
* Analyzed the explicit mathematical relationship between the variance threshold ($\alpha$) and classification accuracy in PCA[cite: 3].
* Compared the class-discriminative power of LDA against the descriptive power of PCA, demonstrating how LDA typically optimizes classification boundaries better for multi-class face tasks[cite: 3].
