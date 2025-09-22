# Churn Prediction System

This repository contains a machine learning pipeline for predicting **customer churn** using the Telco Customer Churn dataset. The notebook implements data cleaning, preprocessing, class imbalance handling, model training (Logistic Regression, Decision Tree, Neural Network), hyperparameter search, and evaluation with clear visualizations.

---

## 🔍 Problem

Telecom companies want to identify customers who are likely to cancel (churn) so they can take retention actions. This project aims to build models that accurately predict churn while balancing precision and recall so that marketing/retention teams can act effectively.

---

## 📂 Project Structure

* `telco-churn-notebook.ipynb` — main Jupyter Notebook with code, charts, and model outputs.

(The notebook contains the full pipeline shown in the shared code block.)

---

## 🛠️ Key Libraries Used

* `numpy`, `pandas` — data handling
* `matplotlib`, `seaborn` — visualization
* `scikit-learn` — preprocessing, model selection, metrics
* `imbalanced-learn` (`SMOTE`) — class imbalance handling
* `tensorflow` / `keras` (via `scikeras.wrappers.KerasClassifier`) — neural network model
* `GridSearchCV` — hyperparameter tuning

---

## 📥 Dataset

**File used:** `WA_Fn-UseC_-Telco-Customer-Churn.csv` (Telco Customer Churn dataset).

In the notebook the dataset is loaded from:

```python
"/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv"
```

---

## 🔁 Code Overview

1. **Load and inspect data**

   * Read CSV, check `head()`, `describe()`, and `info()` to understand features and missing values.
   * Convert `TotalCharges` to numeric and impute missing values with `MonthlyCharges * tenure`.
   * Drop `customerID` and map target `Churn` → `{Yes:1, No:0}`.

2. **Preprocessing**
   * Identify categorical and numerical columns.
   * Build a `ColumnTransformer`:

     * `StandardScaler` for numerical features
     * `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` for categorical features

3. **Train/test split & SMOTE**
   * Fit transformer on train and transform both sets.
   * Apply `SMOTE` to the training set to rebalance classes.

4. **Modeling**

   * **Logistic Regression** with `GridSearchCV` (tune `C`, `solver`).
   * **Decision Tree** with `GridSearchCV` (tune `max_depth`, `min_samples_split`, `criterion`).
   * **Neural Network** built with Keras, wrapped in `KerasClassifier` and tuned via `GridSearchCV`.

5. **Evaluation**

   * Collect metrics: **Accuracy, Precision, Recall, F1, ROC-AUC**.
   * Show **classification report** and **confusion matrix** heatmap.
   * Plot **feature importance / coefficients** for interpretable models.
   * Plot **ROC curves** and a comparison bar plot of metrics across models.

---

## 📊 Metrics & Objectives

* Because churn is often imbalanced, we prioritize **Recall** (catch potential churners) together with **Precision** (avoid too many false positives).
* ROC-AUC is used as a robust overall ranking metric for GridSearchCV.

---

## ✅ Results (summary)

* Models are compared on test set using the metrics above.
* The notebook stores each model's metrics in a `metrics_summary` dictionary and visualizes the comparison.
* The neural network is trained both as part of cross-validation and once outside the grid to visualize training history.

> Note: Exact numbers, best hyperparameters and plots are produced inside the notebook when executed and depend on random seeds and the SMOTE resampling.

---

## ▶️ How to run

1. Clone the repo (or download the notebook):

```bash
git clone https://github.com/helihathi/Customer-Churn-Prediction
cd Customer-Churn-Prediction
```

2. Open the notebook in Jupyter / VS Code / Colab and run all cells.

**Kaggle:** If you run on Kaggle, place the `WA_Fn-UseC_-Telco-Customer-Churn.csv` file in the dataset path used in the notebook and run all cells.

---

## 🧾 Files to include

* `customer-churn-prediction.ipynb` — Notebook with full code and visualizations
* `README.md` — this file

---
