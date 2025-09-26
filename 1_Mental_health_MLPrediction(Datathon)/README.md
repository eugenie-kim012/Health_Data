# Mental Health ML Prediction (Datathon)

## 0. Research Motivation

Depression poses a significant global health and socioeconomic burden. Early detection of risk factors is essential for shaping preventive measures and guiding resource allocation. This project applies advanced machine learning and econometric techniques to a large-scale survey dataset, showcasing how modern analytical methods can yield both predictive insights and interpretable findings relevant for health policy.

## 1. Research Questions

Which sociodemographic, lifestyle, and psychosocial factors are most strongly associated with depression?

How accurately can machine learning models predict depression compared to traditional approaches?

What are the policy and research implications of integrating predictive analytics into public health strategies?

## 2. Abstract

This analysis applies machine learning techniques to predict depression using a large-scale survey dataset. The primary goal is to identify key sociodemographic, lifestyle, and psychosocial factors influencing mental health while evaluating the predictive performance of different models. Results show that ensemble approaches achieve superior accuracy and highlight important risk and protective factors, with implications for early screening and policy interventions.

## 3. Methods

### 3.1 Data

* **Source**: Kaggle Playground Series S4E11 (Train: 140,700; Test: 93,800), URL: https://www.kaggle.com/competitions/playground-series-s4e11
* **Target Variable**: Depression (binary: 1 = Depressed, 0 = Not Depressed).
* **Class Distribution**: ~18.17% positive.

### 3.2 Preprocessing & Feature Engineering

* Age and CGPA discretized into bins.
* Gender, Role, Suicidal Thoughts, Family History encoded as binary.
* Profession, Degree, and Dietary Habits standardized using mapping rules.
* Sleep Duration parsed into numeric hours; missing values imputed.
* City mapped to contextual features (Green Space per capita, Population Density).
* Derived features: Age × Work Pressure, Performance Pressure, Performance Satisfaction.
* Target encoding for high-cardinality variables (City, Profession).

### 3.3 Modeling

* Outliers removed with IsolationForest (contamination = 0.04).
* Models: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost.
* Final: Stacking ensemble (CatBoost + XGBoost + Histogram GBM; meta-model = Logistic Regression).

### 3.4 Evaluation

* Validation: 5-fold CV.
* Metrics: Accuracy, Precision, Recall, F1, ROC AUC.

## 4. Results

### 4.1 Predictive Performance

* **5-Fold CV (Stacking)**: Accuracy = 0.9414 ± 0.0015.


<img width="417" height="247" alt="image" src="https://github.com/user-attachments/assets/208a6240-b852-41b5-8d79-e6d16b0e8fac" />


### 4.2 Validation Set (Final Submission)

* **ROC AUC Score**: 0.9731

| Class             | Precision | Recall | F1-score | Support |
| ----------------- | --------- | ------ | -------- | ------- |
| 0 (Not Depressed) | 0.95      | 0.97   | 0.96     | 23,027  |
| 1 (Depressed)     | 0.85      | 0.79   | 0.82     | 5,113   |

**Overall Metrics:**

* Accuracy: 0.94 (28,140 samples)
* Macro Average: Precision = 0.90, Recall = 0.88, F1 = 0.89
* Weighted Average: Precision = 0.94, Recall = 0.94, F1 = 0.94

**Interpretation:**

* The model demonstrates high discriminative power (ROC AUC = 0.9731).
* For the positive class (Depression=1): Precision = 0.85, Recall = 0.79, F1 = 0.82.
* Negative class (Not Depressed) achieves very strong performance across all metrics.

**Implications:**

* The ensemble model is reliable as a screening tool, showing robust performance under imbalanced conditions.
* Recall for the positive class could be further optimized (e.g., threshold adjustment, cost-sensitive methods).

### 4.3 Feature Importance (CatBoost)

The top 20 features influencing predictions are summarized below. Variables such as Age, Gender (Male), Profession, Work/Study Hours, Performance Pressure, Degree, and Population Density** rank highest in importance, with lifestyle and contextual features (e.g., Sleep Hours, Dietary Category, Green Space) also contributing meaningfully.


<img width="509" height="407" alt="image" src="https://github.com/user-attachments/assets/cecbdef7-2046-4a28-b9f0-3ded9529a5b2" />


### 4.4 ROC Curve (Stacking Ensemble)

The ROC curve illustrates the strong discriminative ability of the stacking model, with an area under the curve exceeding 0.97, reflecting excellent separation between positive and negative classes.

<img width="248" height="199" alt="image" src="https://github.com/user-attachments/assets/759764a9-553c-4179-8d96-f1422d8c765a" />

### 4.5 Logistic Regression Analysis

Logistic regression was conducted to provide interpretable insights into the relationships between predictors and depression outcomes.

* **Depression Model**: Age (-0.7804) and Performance Satisfaction (-0.4032) are protective factors. Financial Stress (+0.5687), Performance Pressure (+0.7463), and Student status (+1.3662) increase the likelihood of depression. Pseudo R² = 0.5787.
* **Multicollinearity (VIF)**: Most features show acceptable VIF (<10), though CGPA (~19.8) indicates some multicollinearity concerns.
* **Suicidal Thoughts Model**: Significant predictors include Age (-0.0566), Profession (-0.0278), Financial Stress (+0.1032), Student status (+0.2920), and Performance Pressure (+0.1123). Pseudo R² = 0.0302.
* **Interaction Terms**: Sleep Hours × Dietary Category was statistically significant (-0.0108), suggesting joint influence of sleep and diet on depression risk.

These regression findings complement the machine learning results, reinforcing the importance of stress, age, and lifestyle factors as determinants of mental health outcomes.


## 5. Policy Implications

The results of this analysis, combining machine learning and regression methods, provide several implications for public health policy and intervention design:

### 5.1 Stress Management as a Central Policy Priority

* **Work and Academic Pressure**: Strong positive associations with depression highlight the need for institutional policies addressing workload, performance pressure, and academic stress.
* **Financial Stress**: Consistently significant in both ML and regression results, pointing to the importance of socioeconomic support programmes, debt management counselling, and targeted subsidies.

### 5.2 Protective Lifestyle Interventions

* **Sleep and Diet**: Adequate sleep and healthy dietary categories are protective factors. Policies promoting sleep hygiene education, school/work schedule reform, and affordable nutrition access can contribute to improved mental health outcomes.
* **Interaction Effects**: The combined effect of poor sleep and unhealthy diet significantly raises depression risk, indicating that integrated lifestyle interventions may be more effective than siloed programmes.

### 5.3 Demographic Targeting for Early Screening

* **Age and Student Status**: Younger populations and students are at higher risk. Universities and secondary education institutions should integrate preventive screening tools and mental health counselling into existing structures.
* **Gender and Profession**: Differences across gender and occupational roles suggest tailoring outreach and workplace programmes.

### 5.4 Use of Predictive Models in Policy Context

* **Screening Tool**: While not a replacement for clinical assessment, predictive models could be used as low-cost triage tools to identify high-risk individuals for referral.
* **Data-Driven Prioritisation**: Governments and institutions can use such models to prioritise resources where the predicted burden is highest (e.g., student populations, financially stressed households).

### 5.5 Integration with National Health Systems

* **Digital Health Strategies**: Predictive analytics could be integrated into electronic health records (EHRs) to flag potential mental health risks early.
* **Policy Alignment**: Findings align with broader global health strategies such as the WHO Mental Health Action Plan and Sustainable Development Goals (SDGs), supporting early intervention and prevention.


<img width="589" height="487" alt="image" src="https://github.com/user-attachments/assets/1a25401d-1ad4-4f7e-b27d-650887153ad5" />


## 6. Conclusion

Stacked ensemble learning methods achieve high predictive performance in classifying depression from survey data. Stress-related variables, age, occupational/academic context, and lifestyle factors (sleep, diet) are key determinants. Logistic regression confirms these relationships, offering interpretable evidence for policy and intervention strategies.


## 7. Limitations

1. Target encoding applied outside cross-validation folds may cause data leakage.
2. Recall can be further optimized via threshold tuning, PR-AUC maximization, or cost-sensitive methods.
3. Fairness and bias assessments are needed prior to deployment.


## 8. Future Work

1. Integrate target encoding strictly within cross-validation folds.
2. Apply automated hyperparameter optimization (Optuna).
3. Explore calibration methods to improve recall.
4. Perform fairness and subgroup analysis.
5. Build an interactive dashboard (e.g., Streamlit).


## 9. Provenance and Acknowledgment

This analysis was originally conducted as part of the Datathon within the Data Science Programme at Modulabs in a team-based effort. The current refined version has been updated and extended by the author for inclusion in the professional portfolio submission. Enhancements include methodological improvements, expanded interpretation, and alignment with academic/reporting standards. This ensures recognition of the team collaboration while clarifying the author’s individual contributions in the final version.


## 9. Citation

Kim, Y. (2025). Mental Health ML Prediction (Datathon). GitHub Repository.
