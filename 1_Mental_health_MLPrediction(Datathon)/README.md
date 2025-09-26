# Mental Health ML Prediction (Datathon)

## 0. Abstract

This analysis applies machine learning techniques to predict depression using a large-scale survey dataset. The primary goal is to identify key sociodemographic, lifestyle, and psychosocial factors influencing mental health while evaluating the predictive performance of different models. Results show that ensemble approaches achieve superior accuracy and highlight important risk and protective factors, with implications for early screening and policy interventions.

---

## 1. Methods

### 1.1. Data

* **Source**: Kaggle Playground Series S4E11 (Train: 140,700; Test: 93,800).
* **Target Variable**: Depression (binary: 1 = Depressed, 0 = Not Depressed).
* **Class Distribution**: ~18.17% positive.

### 1.2. Preprocessing & Feature Engineering

* Age and CGPA discretized into bins.
* Gender, Role, Suicidal Thoughts, Family History encoded as binary.
* Profession, Degree, and Dietary Habits standardized using mapping rules.
* Sleep Duration parsed into numeric hours; missing values imputed.
* City mapped to contextual features (Green Space per capita, Population Density).
* Derived features: Age × Work Pressure, Performance Pressure, Performance Satisfaction.
* Target encoding for high-cardinality variables (City, Profession).

### 1.3. Modeling

* Outliers removed with IsolationForest (contamination = 0.04).
* Models: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost.
* Final: Stacking ensemble (CatBoost + XGBoost + Histogram GBM; meta-model = Logistic Regression).

### 1.4. Evaluation

* Validation: 5-fold CV.
* Metrics: Accuracy, Precision, Recall, F1, ROC AUC.

---

## 2. Results

### 2.1. Predictive Performance

* **5-Fold CV (Stacking)**: Accuracy = 0.9414 ± 0.0015.
* **Validation (Hold-out 20%)**:

  * ROC AUC = 0.9731
  * Accuracy = 0.94
  * Precision = 0.85 (positive class)
  * Recall = 0.79
  * F1 = 0.82

### 2.2. Feature Importance (CatBoost)

* Leading variables: Age, Gender, Profession (encoded), Work/Study Hours, Performance Pressure, Degree, Population Density, Sleep Hours, Student status, Age × Work Pressure.

### 2.3. Statistical Modeling

* Logistic regression confirmed Age and Performance Satisfaction as protective, while Financial Stress, Performance Pressure, and Student status were strong risk factors.

---

## 4. Conclusion

Stacked ensemble learning methods achieve high predictive performance in classifying depression from survey data. Stress-related variables, age, occupational/academic context, and lifestyle factors (sleep, diet) are key determinants. These insights support the potential of predictive models as early screening tools while underscoring the need for expert clinical judgment.

---

## 5. Limitations

* Target encoding applied outside cross-validation folds may cause data leakage.
* Recall can be further optimized via threshold tuning, PR-AUC maximization, or cost-sensitive methods.
* Fairness and bias assessments are needed prior to deployment.

---

## 6. Future Work

* Incorporate target encoding strictly within cross-validation folds.
* Conduct hyperparameter optimization (Optuna).
* Apply calibration to improve recall.
* Perform fairness and subgroup performance analysis.
* Build an interactive dashboard (e.g., Streamlit).

---

## 7. Citation

Kim, E. (2025). *Mental Health ML Prediction (Datathon).* GitHub Repository.

