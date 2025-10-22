### 0. 🔍 Overview

This project implements a Supervised Machine Learning framework to detect fraudulent credit card transactions. The core technical challenge addressed is the severe class imbalance inherent in financial data (where fraudulent cases represent a minute fraction of the total dataset). This necessitated the application of specialised techniques to ensure model efficacy on the critical minority class. The algorithm has been originally developed as a part of the Modulabs Data Science Bootcamp quest, later enhanced for the technical depth.

URL - https://www.kaggle.com/competitions/modu-ds-4-credit-card-fraud-detection/submissions

### Key Technical Focus

| Area | Methodology & Goal |
| --- | --- |
| **Data Challenge** | Highly Imbalanced Dataset (Fraud Rate $\ll 1\%$): Aiming to minimize False Negatives (missed fraud). |
| **Preprocessing** | Enhanced feature engineering to maximize separation between normal and fraud transactions, specifically handling skewed distributions. |
| **Imbalance Handling** | Strategic use of Oversampling (SMOTE), Undersampling (e.g., Tomek Links), and Model-Specific Cost Weighting (e.g., `class_weight='balanced'`). |
| **Evaluation** | Primary metrics are Area Under the Precision-Recall Curve (AUPRC) and F1-Score, as standard Accuracy is misleading in skewed contexts. |

## 1. 🎯 Objective

Initiated this project to master the methodologies required for accurate minority class classification, a key technical challenge across high-stakes domains (Finance, Healthcare, Public Safety). The goal was to build a resilient model that prioritises minimising the costly False Negative rate (missed fraud) over maximising overall (and often misleading) accuracy.

## 2. 💡 Technical Motivation & Approach

The project's primary motivation stems from the high financial and systemic cost associated with a single missed fraudulent event. The technical development was therefore focused on counteracting the inherent bias of imbalanced data.

### 2.1. Addressing the Metric Flaw

The project recognised that standard **Accuracy** ($\approx 99\%$) is a trivial and non-informative metric for this dataset. The focus was systematically shifted to performance indicators that accurately reflect true positive identification:

- Primary Metrics: Area Under the Precision-Recall Curve (AUPRC) and the F1-Score.
- Target Optimisation: Maximising Recall (Sensitivity) for the fraud class while maintaining an acceptable Precision level, thereby optimising the business utility of the model.

### 2.2. Data_sampling.md

In credit card fraud detection or similar tasks with highly imbalanced datasets, the model’s performance — particularly recall (sensitivity) — can be significantly affected. To improve model robustness and detect rare fraudulent cases, resampling techniques are essential.

### 1️⃣ Over-sampling

| Category | Description |
| --- | --- |
| **Goal** | Increase the number of samples in the minority (fraud) class to balance class ratios. |
| **Method** | Duplicate or synthetically generate new minority samples. |
| **Advantages** | Retains most of the original information and improves learning on the minority class. |
| **Disadvantages** | Risk of overfitting due to repetitive or highly similar samples. |
| **Representative Technique** | `SMOTE (Synthetic Minority Over-sampling Technique)` |

### 2️⃣ Under-sampling

| Category | Description |
| --- | --- |
| **Goal** | Reduce the number of samples in the majority (non-fraud) class to balance class ratios. |
| **Method** | Randomly remove samples from the majority class. |
| **Advantages** | Reduces dataset size, leading to faster model training. |
| **Disadvantages** | Potential loss of valuable information from the majority class. |
| **Representative Techniques** | `Random Under-sampling`, `Tomek Links` |

### 🌟 Practical Tip: Application in Fraud Detection

> 💡 Main Objective: Minimize false negatives — ensure that no actual fraud cases are missed (maximize recall).
> 
- Over-sampling methods such as SMOTE are often preferred due to their ability to enhance recall without significant data loss.
- In practice, hybrid approaches that combine over- and under-sampling (e.g., `SMOTEENN`) frequently achieve the best performance.

Example:

```python
from imblearn.combine import SMOTEENN

smote_enn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smote_enn.fit_resample(X, y)

```

---

### Exploring Train and Test Data Features

1. Understand the data structure — each row represents one transaction (case), and each column corresponds to a feature.
2. Check feature distributions — detect outliers and design preprocessing strategies.
3. Check for missing values (NULLs) — confirmed that there are no missing entries in this dataset.

### Key Findings from Data Exploration

- `Time` feature: removed.
- `Amount` feature: transaction amount — removed for model consistency.
- `Class` feature: target variable (0 = normal, 1 = fraud); *not present in the test dataset.*
- No missing values detected in either dataset.

### 2.3. Data_Processing_Pipeline.md

This section summarises the modular preprocessing pipeline implemented in the fraud detection project. It ensures data integrity, handles class imbalance, and optimises learning for the minority (fraudulent) class.

### 1️⃣ get_outlier — Outlier Detection (IQR Method)

Role: Detects extreme outliers based on data dispersion (variance).

Technique: Interquartile Range (IQR)–based detection.

Target: Applied only to fraudulent transactions (`Class = 1`) to remove abnormal or erroneous samples.

Result: Returns the index values of outlier rows, which are later excluded in `get_preprocessed_df`.

### 2️⃣ get_preprocessed_df — Unified Data Preprocessing

**Steps & Purpose**

- Log Transformation: Adds `Amount_Scaled = np.log1p(df['Amount'])` to normalise skewed transaction amounts.
- Column Removal: Drops `Time` and `Amount` columns. `Time` contributes little predictive value; `Amount` is replaced by its scaled version.
- Outlier Removal: Removes extreme values in the `V14` feature to prevent bias and improve interpretability.

### 3️⃣ get_train_test_dataset — Train/Test Split

**Steps & Purpose**

- Preprocessing: Calls `get_preprocessed_df` prior to splitting to ensure all transformations are applied.
- Feature/Target Split: Defines `X_features` (predictors) and `y_target` (label) for clean, structured inputs.
- Stratified Split: Uses `stratify=y_target` in `train_test_split` to maintain consistent fraud/non-fraud ratios.

### Function Implementation Verification

| Function | Implemented? | Summary |
| --- | --- | --- |
| get_outlier() | ✅ Yes | Detects outliers in `V14` using IQR; removes samples beyond upper/lower bounds. |
| get_preprocessed_df() | ✅ Yes | Applies log transform to `Amount`, removes `Time`, integrates outlier removal. |
| get_train_test_dataset() | ✅ Yes | Calls preprocessing, performs stratified split to maintain class balance. |

### Key Takeaways

- Ensures clean, balanced, and properly scaled data before modeling.
- Prevents bias from outliers and imbalanced sampling.
- Enables reproducible and consistent preprocessing across multiple runs.

### 2.4. Model Performance Comparison (Before / After Feature Engineering & Final Tuning)

| Stage | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Key Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Before Feature Engineering** | Logistic Regression | 0.9989 | 0.8205 | 0.5981 | 0.6919 | 0.9274 | Solid baseline performance; recall slightly low. |
|  | LightGBM | 0.9996 | 0.9570 | 0.8318 | 0.8899 | 0.9769 | Highest recall and AUC before preprocessing. |
|  | XGBoost | 0.9989 | 0.7281 | 0.7757 | 0.7511 | 0.9187 | Precision–recall balance somewhat weak. |
| **After Feature Engineering** | Logistic Regression | 0.9881 | 0.1358 | 0.8785 | 0.2353 | 0.9698 | Recall improved but precision dropped significantly. |
|  | LightGBM | 0.9995 | 0.8911 | 0.8411 | 0.8654 | 0.9793 | Maintained stable and balanced performance. |
|  | XGBoost | 0.9995 | 0.8846 | 0.8598 | 0.8720 | 0.9821 | Slightly higher recall than LightGBM. |
| **Final Tuning (Best)** | Logistic Regression | 0.9898 | 0.1550 | 0.8692 | 0.2631 | 0.9715 | Slight improvement in precision after tuning. |
|  | **LightGBM (Best)** | **0.9995** | **0.8911** | **0.8411** | **0.8654** | **0.9812** | Final best-performing model with the most balanced results. |
|  | XGBoost (Best) | 0.9995 | 0.8762 | 0.8598 | 0.8679 | 0.9823 | High recall but slightly less balanced than LightGBM. |

### 3. Summary & Insights

Effect of Feature Engineering: Normalising the `Amount` feature and removing outliers in `V14` significantly enhanced both recall and ROC-AUC across all models.

LightGBM Stability: LightGBM consistently demonstrated the most stable and balanced performance across all phases—before and after feature engineering, and during final tuning.

Final Model Selection: The final LightGBM model achieved the most balanced results, with an F1-score of 0.8654 and ROC-AUC of 0.9812, making it the optimal model for deployment. XGBoost also delivered very strong performance, but LightGBM proved slightly more stable and thus the preferred choice overall.
