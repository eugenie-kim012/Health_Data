# 📊 Employment Vulnerability, Education Outcomes and Health Expenditure 

*An Integrated Analysis Using World Development Indicators (WDI)*

## 📌 Abstract

This project investigates the dynamics between **employment vulnerability, education outcomes and health expenditure** in high-income countries, using cross-country panel data from the [Kaggle WDI dataset](https://www.kaggle.com/datasets/parsabahramsari/wdi-education-health-and-employment-2011-2021) – based on the World Bank's World Development Indicators (WDI) (2011–2021). The dataset covers 19 countries over 11 years, focusing on Europe, North America, and developed Asia. 

The analysis explores how levels of government health expenditure are associated with education attainment and employment statuses. The study interprets health expenditure as a form of **social protection investment**, highlighting fiscal pressures on health budgets, links between labour market vulnerability and economic performance, and long-term human capital accumulation. The objective is to demonstrate applied research skills in **data cleaning, empirical analysis, and policy-relevant interpretation**.

---

## 📌 Indicators Used

### 🎓 Education

* EDU_Bachelors_25Plus_Total
* EDU_LowerSecondary_25Plus_Total
* EDU_PostSecondary_25Plus_Total
* EDU_Primary_25Plus_Total
* EDU_ShortCycleTertiary_25Plus_Total

### 👷 Employment

* JOB_Unemp_Total
* JOB_Unemp_Youth_Total
* JOB_VulnerableEmployment_Total
* JOB_Part_Time_Total
* JOB_Self_employed_Total
* JOB_Contracters_Total

### 🏥 Health Expenditure

* GDP_Total_HealthExp
* GDP_Gov_HealthExp

---

## 📂 Repository Structure

```text
WDI-Education-Employment-EconDev/
│
├── README.md
├── WDI_V8_Final.ipynb                     # End-to-end codebook
│
├── notebooks/                             # Thematic notebooks
│   ├── 01_data_preparation.ipynb
│   ├── 02_education_econdev_analysis.ipynb
│   ├── 03_employment_econdev_analysis.ipynb
│   ├── 04_health_expenditure_linkages.ipynb (optional)
│   └── 05_summary_policy_implications.ipynb
│
├── data/
│   ├── WDI_Final_Processed.csv            # Processed dataset
│   └── (link to raw WDI source)
│
└── figures/
    ├── education_econdev_scatter.png
    ├── employment_econdev_trends.png
    └── regression_results.png
```

---

## 📊 Methods

1. **Data Preparation**

   * Cleaning missing values and harmonising indicators
   * Imputation for incomplete time series
   * Standardisation of variables for comparability

2. **Exploratory Analysis**

   * Descriptive mapping of education, employment and health expenditure
   * Visualisation of trends and regression outcomes

3. **Econometric Analysis**

   * Dependent variable: GDP_Gov_HealthExp
   * Explanatory variables: education & employment indicators
   * Panel fixed-effects regression with clustered / Driscoll–Kraay standard errors

---

## 📊 **Econometric Analysis**

## 📊 Two-way Fixed Effects Regression Results

This analysis examines how **labor market structures** and **education attainment** are associated with government health expenditure (% of GDP), using **two-way fixed effects with Driscoll–Kraay SE**.

---

### 🔹 Summary of Results

| DomainExplanatory VariableCoefficient (β)p-valueInterpretation |                               |         |        |                                                                                |
| -------------------------------------------------------------- | ----------------------------- | ------- | ------ | ------------------------------------------------------------------------------ |
| **Labor Market**                                               | Unemployment                  | 0.0661  | 0.0037 | Higher unemployment → ↑ Gov. health expenditure share (**significant**)        |
|                                                                | Vulnerable Employment         | 0.1304  | 0.331  | Not significant                                                                |
|                                                                | Part-time Employment          | 0.0208  | 0.060  | Weak positive effect (10% level)                                               |
|                                                                | Contract Employment           | 0.2195  | 0.021  | Higher contractors → ↑ Gov. health expenditure share (**significant**)         |
| **Education (contemporaneous)**                                | Secondary                     | -0.0108 | 0.134  | Not significant                                                                |
|                                                                | Post-secondary (non-tertiary) | -0.0264 | 0.071  | Marginal negative effect                                                       |
|                                                                | Tertiary                      | -0.0414 | 0.0049 | Higher tertiary attainment → ↓ Gov. health expenditure share (**significant**) |
| **Education (lagged 1y)**                                      | Secondary (t-1)               | -0.0152 | 0.035  | Significant delayed negative effect                                            |
|                                                                | Post-secondary (t-1)          | -0.0057 | 0.674  | Not significant                                                                |
|                                                                | Tertiary (t-1)                | -0.0051 | 0.762  | Not significant                                                                |


<img width="677" height="364" alt="image" src="https://github.com/user-attachments/assets/39449f0a-9f70-4740-bcc7-d5cb9b737b93" />

## ✅ Implications

### Labour market dynamics and fiscal pressure
Rising unemployment and contract-based work are strongly linked to higher government health expenditure share, suggesting that labour market insecurity directly increases public health financing needs.

This reflects the counter-cyclical nature of health budgets: governments tend to absorb more health costs when workers face unstable employment.

### Education attainment and fiscal reprioritisation
Higher tertiary education attainment correlates with a smaller share of GDP spent by government on health.

This may imply that as populations become more educated, health financing shifts relatively towards private sources (insurance, out-of-pocket, employer-based schemes) or governments reallocate budgets to other priorities.

### Temporal asymmetry
Education effects are mostly contemporaneous: tertiary attainment matters in the same year but not the following year.

Labour market effects are more persistent: unemployment and contractualisation drive government health expenditure up regardless of short-term volatility.

### Policy takeaways
- **Counter-cyclical safety nets**: Governments must anticipate rising health spending when labour markets weaken.  
- **Equity concerns**: Expanding tertiary education may unintentionally reduce the government’s relative fiscal role in health, raising risks of inequity in access if private financing dominates.  
- **Balanced strategy**: Integrating labour market protection with sustainable health financing policies is critical to avoid both fiscal shocks and social inequities.  

---

## 📌 Policy Implications

> **Labour market shocks and fiscal response**  
> • Rising unemployment and contract-based work significantly increase the share of government health expenditure in GDP, highlighting the counter-cyclical role of public budgets.  
>
> **Education-driven structural shifts**  
> • Higher tertiary education attainment is associated with a smaller government share in health financing, suggesting fiscal reallocation and stronger reliance on private sources.  
>
> **Equity and sustainability**  
> • Policymakers should anticipate fiscal pressures during labour market downturns while safeguarding equity as education-driven changes reshape the balance between public and private health financing.
