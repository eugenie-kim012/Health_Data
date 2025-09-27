# 🌍 WDI Employment, Health & Education Analysis

This project analyzes the relationship between **health expenditure, vulnerable employment, and education outcomes** using **World Development Indicators (WDI)** dataset.  

## 📌 Overview
- **Goal**: To explore how health spending (total and government) relates to **vulnerable employment** and **tertiary education outcomes** across countries.  
- **Approach**: Data preprocessing, visualization, correlation analysis, and regression models (OLS & Panel Fixed Effects).  
- **Scope**: 19 countries × 11 years (panel data).  
- **Data**: https://www.kaggle.com/datasets/parsabahramsari/wdi-education-health-and-employment-2011-2021/code


## Abstract
This study investigates the relationship between **health expenditure**, **employment vulnerability**, and **tertiary education outcomes** using panel data from the World Bank’s *World Development Indicators (WDI)*.  

We focus on 19 countries over 11 years, applying both **cross-sectional regression (OLS)** and **panel fixed-effects estimation** to explore whether changes in health spending are associated with improvements in employment or education outcomes.  

The analysis highlights the **limited short-term explanatory power** of aggregated health expenditure indicators, while suggesting directions for future empirical research in health economics and education policy.

## Research Questions
1. Does an increase in health expenditure (% of GDP) reduce **vulnerable employment**?  
2. Is there a measurable link between **health spending** and **tertiary education investment or attainment (BA+ share)**?  
3. How do these relationships hold under **panel fixed-effects estimation** that controls for country- and year-specific unobserved heterogeneity?  

## Data and Variables
**Source**: World Development Indicators (World Bank, 2025 release)  

- **Health expenditure**:  
  - Current health expenditure (% of GDP) — `SH.XPD.CHEX.GD.ZS`  
  - Domestic government health expenditure (% of GDP) — `SH.XPD.GHED.GD.ZS`  

- **Employment**:  
  - Vulnerable employment (% of total employment) — `SL.EMP.VULN.ZS`  
  - Unemployment, total (% of labor force) — `SL.UEM.TOTL.ZS`  
  - Unemployment, youth (% of youth labor force) — `SL.UEM.1524.ZS`  

- **Education**:  
  - Tertiary expenditure share (% of public education expenditure) — `SE.XPD.CTER.ZS`  
  - BA+ attainment (population 25+, %) — `SE.TER.CUAT.BA.ZS`  

## Methodology
1. **Preprocessing**
   - Converted indicators to numeric form, addressed missing values.  
   - Applied *winsorization* (1%–99%) to mitigate the influence of outliers.  

2. **Visualization**
   - Cross-country line charts (health and education spending over time).  
   - Scatter plots (health expenditure vs. employment vulnerability).  

3. **Econometric Analysis**
   - **OLS on first differences (Δ2019–2021)**:  
     Models short-term changes in tertiary outcomes relative to changes in health spending.  
   - **Panel OLS with fixed effects (entity + time)**:  
     Accounts for country-specific and year-specific unobservables.  
   - Robust standard errors (HC3) applied.  

## Results

| Model                               | Findings                                                                                           |
| ----------------------------------- | -------------------------------------------------------------------------------------------------- |
| **ΔTertiaryExpShare (OLS)**         | Health spending changes **not significant**; R² = 0.06.                                             |
| **ΔBA+ Attainment (OLS)**           | Weak, non-significant coefficients; R² = 0.12.                                                     |
| **VulnerableEmployment (Panel FE)** | Average ≈ 12%; no significant link with health spending. Robust SE confirms lack of significance.  |
| **TertiaryExpShare (Panel FE)**     | Explanatory power very low (Within R² = 0.01); model not significant.                              |

## Interpretation
- Health expenditure (both total and government share) shows **no consistent short-term impact** on employment vulnerability or higher education outcomes.  
- The weak explanatory power (low R² values) indicates that **other factors (institutional, fiscal, educational)** likely mediate these relationships.  
- While intuitive links between public spending, employment security, and education are often assumed, this dataset demonstrates the **limitations of using aggregate WDI indicators for causal inference**.


<img width="1017" height="446" alt="image" src="https://github.com/user-attachments/assets/e80ad07f-0321-453b-91a0-e8007e1558ce" />


<img width="1034" height="446" alt="image" src="https://github.com/user-attachments/assets/f0614a83-7a6e-493c-ab68-ea9f4401f123" />


<img width="1034" height="432" alt="image" src="https://github.com/user-attachments/assets/c06db3ff-9998-4f4f-9fe5-ade767749cca" />


<img width="1160" height="575" alt="image" src="https://github.com/user-attachments/assets/d8e8c36e-721f-4db8-b679-ac613deaf8a2" />


## Implications
- **Policy**: Simply raising health expenditure may not yield immediate improvements in labor market vulnerability or tertiary education attainment.  
- **Research**: Future work should extend the time horizon, incorporate additional variables (e.g., education expenditure disaggregation, governance quality), and explore **causal methods (IV, DiD)**.  
- **Portfolio relevance**: This analysis demonstrates the application of **data cleaning, visualization, panel econometrics, and critical interpretation** in a health economics context.  

## Technical Implementation
- **Language**: Python  
- **Libraries**: `pandas`, `plotly`, `statsmodels`, `linearmodels`  
- **Data**: World Development Indicators (World Bank, 2025 release)  
- **Notebook**: `WDI_V3.ipynb`  
