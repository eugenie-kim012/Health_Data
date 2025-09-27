# 🌍 WDI Health & Education Analysis

This project analyzes the relationship between **health expenditure, vulnerable employment, and education outcomes** using the World Bank’s **World Development Indicators (WDI)** dataset.  

---

## 📌 Overview
- **Goal**: To explore how health spending (total and government) relates to **vulnerable employment** and **tertiary education outcomes** across countries.  
- **Approach**: Data preprocessing, visualization, correlation analysis, and regression models (OLS & Panel Fixed Effects).  
- **Scope**: 19 countries × 11 years (panel data).  
- **Data**: https://www.kaggle.com/datasets/parsabahramsari/wdi-education-health-and-employment-2011-2021/code

---

## 📊 Key Indicators
- **HealthExpenditure (% of GDP)** → `SH.XPD.CHEX.GD.ZS`  
- **Government HealthExpenditure (% of GDP)** → `SH.XPD.GHED.GD.ZS`  
- **Vulnerable Employment (% of total employment)** → `SL.EMP.VULN.ZS`  
- **Unemployment (total, youth)** → `SL.UEM.TOTL.ZS`, `SL.UEM.1524.ZS`  
- **Tertiary Education Share** → `SE.XPD.CTER.ZS`  
- **BA+ Attainment (25+, %)** → `SE.TER.CUAT.BA.ZS`  

---

## 🔎 Methods
1. **Data Cleaning & Winsorization**  
   - Converted to numeric, handled missing values.  
   - Winsorized (1%–99%) to reduce outlier impact.  

2. **Visualization (Plotly)**  
   - Line charts for country-level health & education trends.  
   - Scatter plots for cross-country correlations.  

3. **Regression Analysis**  
   - **OLS** on cross-period differences (Δ2019–2021).  
   - **Panel OLS with Fixed Effects** (Entity + Time, clustered SE).  
   - Robust standard errors applied (HC3).  

---

## 📈 Results (Summary)

| Model                              | Findings                                                                                   |
| ---------------------------------- | ------------------------------------------------------------------------------------------ |
| **OLS (ΔTertiaryExpShare)**        | Health expenditure changes **not significantly related** to tertiary spending share (R²=0.06). |
| **OLS (ΔBA+ Attainment)**          | Weak, non-significant link between health spending and BA+ attainment (R²=0.12).              |
| **Panel FE (VulnerableEmployment)**| Average ~12% vulnerable employment; **no significant effect** of health expenditure.          |
| **Panel FE (TertiaryExpShare)**    | Very low explanatory power (R² < 0.02), model not significant.                               |

---

## 📝 Interpretation
- **No clear short-term relationship** between health expenditure and higher education outcomes.  
- **Vulnerable employment** remains high (~12%) but is **not significantly explained** by health spending shares.  
- Highlights the **limitations of short time windows and aggregated indicators**.  
- Suggests the need for **longer-term data** and additional controls (e.g., education spending, institutional factors).  

---

## ⚙️ Tech Stack
- **Python**: `pandas`, `plotly`, `statsmodels`, `linearmodels`  
- **Data**: World Bank WDI (Indicators Metadata + Main Data)  
- **Notebook**: `WDI_V3.ipynb`  

---

## 📂 Repository Structure
