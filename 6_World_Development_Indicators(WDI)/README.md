# 📑 Employment Vulnerability, Education Outcomes, and Health Expenditure  
*An Integrated Analysis Using World Development Indicators (WDI)*

## Abstract

This project explores the interconnected dynamics between **employment vulnerability**, **education outcomes**, and **health expenditure** in high-income countries, using panel data from the World Bank’s *World Development Indicators (WDI)* (2011–2021) ([Kaggle WDI dataset](https://www.kaggle.com/datasets/parsabahramsari/wdi-education-health-and-employment-2011-2021)). .  

Rather than examining these dimensions in isolation, we frame them as part of an employment–education–health triangle:  
- **Employment vulnerability** may drive fiscal demand for health expenditure.  
- **Education outcomes** shape labor market vulnerability by influencing skill levels and employability.  
- **Health expenditure** and **education investment** both contribute to long-term human capital accumulation. 

The dataset covers 19 countries over 11 years, focusing on Europe, North America, and developed Asia. Using both OLS (first differences) and panel fixed-effects estimation, we assess short-term shocks (COVID-19) and structural patterns across economies.  

Findings show limited short-run explanatory power of aggregate WDI indicators, but highlight the importance of considering structural and institutional factors in how labor markets, education systems, and fiscal health policies interact. Future work could extend this analysis with longer-term panel data and a broader global sample to better capture long-run dynamics.  
 
## Research Questions
1. Does an increase in health expenditure (% of GDP) reduce the share of vulnerable employment?
2. Is there a measurable link between health spending and tertiary education investment or attainment? 
3. How do these relationships hold under panel fixed-effects estimation, which controls for unobserved heterogeneity across countries and years?

4. 1. Does an increase in health expenditure (% of GDP)** reduce the share of **vulnerable employment**?  
2. Is there a measurable link between **health spending** and **tertiary education investment and/or attainment**?  
3. How do these relationships hold under **panel fixed-effects estimation**, which controls for unobserved heterogeneity across countries and years?  


## Data
**Source**: [Kaggle WDI dataset](https://www.kaggle.com/datasets/parsabahramsari/wdi-education-health-and-employment-2011-2021)

- **Health Expenditure**  
  - Current health expenditure (% of GDP) — `SH.XPD.CHEX.GD.ZS`  
  - Government health expenditure (% of GDP) — `SH.XPD.GHED.GD.ZS`  

- **Employment**  
  - Vulnerable employment (% of total employment) — `SL.EMP.VULN.ZS`  
  - Unemployment, total (% of labor force) — `SL.UEM.TOTL.ZS`  
  - Unemployment, youth (% of youth labor force) — `SL.UEM.1524.ZS`  

- **Education**  
  - Tertiary expenditure share (% of education budget) — `SE.XPD.CTER.ZS`  
  - BA+ attainment (population 25+) — `SE.TER.CUAT.BA.ZS`  

## Methods
1. **Data Preprocessing**  
   - Converted indicators to numeric, handled missing values.  
   - Winsorization (1%–99%) applied to reduce outlier influence.  

2. **Visualization**  
   - Time-series line charts (cross-country health and education spending trends).  
   - Scatter plots linking expenditure to vulnerable employment. 

  
<img width="970" height="399" alt="image" src="https://github.com/user-attachments/assets/dbb51e72-e95c-4080-b478-aee35a1d9b78" />

<img width="1034" height="446" alt="image" src="https://github.com/user-attachments/assets/f0614a83-7a6e-493c-ab68-ea9f4401f123" />


<img width="1034" height="432" alt="image" src="https://github.com/user-attachments/assets/c06db3ff-9998-4f4f-9fe5-ade767749cca" />


<img width="1160" height="575" alt="image" src="https://github.com/user-attachments/assets/d8e8c36e-721f-4db8-b679-ac613deaf8a2" />



3. **Econometric Analysis**  
   - OLS (first differences, 2019–2021): ΔHealthExp and ΔGovHealthExp predicting changes in tertiary outcomes.  
   - Panel Fixed Effects (Entity + Time, Clustered SE): Controlling for unobserved heterogeneity across countries and years.  
   - Robust SE (HC3) for heteroscedasticity.  

---

## Results

| Model                               | Main Coefficients (ΔHealthExp / GovExp) | Robust SE | p-value | R² (within / overall) | Interpretation |
| ----------------------------------- | ---------------------------------------- | --------- | ------- | --------------------- | --------------- |
| **ΔTertiaryExpShare (OLS, 2019–21)** | β₁≈0.01, β₂≈-0.02                        | ~0.04–0.05 | 0.86    | R²=0.06               | Not significant |
| **ΔBA+ Attainment (OLS, 2019–21)**  | β₁≈-0.03, β₂≈0.01                        | ~0.05     | 0.34    | R²=0.12               | Not significant |
| **VulnerableEmployment (Panel FE)** | β₁≈-0.04, β₂≈+0.02                       | ~0.09     | 0.62    | Within=0.13 / Overall=0.03 | Weak fit, not robust |
| **TertiaryExpShare (Panel FE)**     | β₁≈0.01, β₂≈-0.01                        | ~0.03     | 0.87    | Within=0.01 / Overall=-0.47 | Very low explanatory power |

---

## Interpretation
- Health expenditure (both total and government share) shows no consistent short-term impact on employment vulnerability or tertiary education.  
- Very low R² values suggest that structural and institutional factors** are stronger determinants than aggregate spending measures.  
- Aggregate WDI indicators, while useful for cross-country comparison, may be insufficient for causal inference without richer policy or micro-level data.  

## Policy and Research Implications
- Policy: Increasing health expenditure alone is unlikely to yield immediate gains in labor security or tertiary attainment.  
- Research: Future studies should include lagged effects, education expenditure disaggregation, and causal designs (IV, DiD).  
- Portfolio relevance: This project demonstrates applied skills in data cleaning, visualization, and panel econometrics for health economics and education research.

## Limitations
- Aggregate indicators may mask heterogeneity across regions, policy regimes, and institutional quality.  
- Potential endogeneity between health and education spending not addressed here.  
- Short panel (2011–2021) limits ability to capture long-term lagged effects.  
- Future extensions: Instrumental Variables (IV), panel ARDL, or natural experiments (DiD).
  
