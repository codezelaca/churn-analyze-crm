# Telco Customer Churn Analysis — EDA & Preprocessing

A structured exploratory data analysis (EDA) and data preprocessing pipeline built on the **IBM Telco Customer Churn** dataset. The cleaned, feature-engineered output is ready to feed directly into downstream churn-prediction models.

---

## Problem Statement

Customer churn is one of the most critical challenges in the telecom industry. Acquiring a new customer costs 5–10× more than retaining an existing one, making early identification of at-risk customers highly valuable.

This project answers:
- **Who** is likely to churn? (demographics, contract type, payment method)
- **When** is churn most likely? (tenure-based lifecycle stages)
- **What services** drive or prevent churn?
- **Which features** are the strongest predictors to feed into an ML model?

---

## Key Findings

| Insight | Detail |
|---|---|
| **Overall Churn Rate** | 26.5% — moderately imbalanced (2.77:1 ratio) |
| **Highest-Risk Segment** | New customers (0–12 months): **47.4% churn rate** |
| **Contract Type** | Month-to-month customers churn far more than annual/biannual contracts |
| **Internet Service** | Fiber optic users churn significantly more (Pearson r = +0.31) |
| **Payment Method** | Electronic check strongly associated with churn (r = +0.30) |
| **Tenure** | Strongest negative predictor — longer tenure = lower churn (r = −0.35) |
| **Online Security / Tech Support** | Customers lacking these services churn measurably more |

---

## Project Structure

```
practical02/
├── data/
│   ├── telco_customer_churn.csv        # Raw dataset (7043 rows × 21 cols)
│   └── telco_churn_processed.csv       # Cleaned & encoded output (7043 × 26)
├── notebooks/
│   └── eda_prac.ipynb                  # Full EDA + preprocessing notebook
├── requirements.txt                    # Python dependencies
└── README.md
```

---

## Dataset

**Source:** [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Field Group | Columns |
|---|---|
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Account Info | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| Services | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| Target | `Churn` (Yes / No) |

---

## Notebook Walkthrough — `eda_prac.ipynb`

| Section | What it does |
|---|---|
| **1. Data Loading & Inspection** | Load CSV, inspect dtypes, check shape and nulls |
| **2. Data Cleaning** | Coerce `TotalCharges` to numeric, fill 11 NaNs, drop `customerID`, remap `SeniorCitizen` |
| **3. Univariate & Bivariate EDA** | Churn distribution, histograms by churn, categorical breakdowns (demographics, services, account) |
| **4. Outlier Detection & Capping** | IQR boxplots on `tenure`, `MonthlyCharges`, `TotalCharges`; cap at [Q1−1.5×IQR, Q3+1.5×IQR] |
| **5. Feature Engineering** | `TenureGroup` (5 ordinal bands: 0-12 mo → 49-72 mo), `AvgMonthlySpend` (TotalCharges / tenure+1) |
| **6. Encoding** | Binary Yes/No → 0/1; "No internet/phone service" simplified to "No"; OHE for `InternetService`, `Contract`, `PaymentMethod` (drop_first) |
| **7. Feature Scaling** | StandardScaler on `tenure`, `MonthlyCharges`, `TotalCharges`, `AvgMonthlySpend` |
| **8. Correlation Analysis** | Pearson correlation ranking of all features vs `Churn_Numeric` |
| **9. Export** | Save `data/telco_churn_processed.csv` — fully numeric, training-ready |

---

## Environment Setup

### Option A — Conda (Recommended)

```bash
# Create environment
conda create -n churn-env python=3.10 -y
conda activate churn-env

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook notebooks/eda_prac.ipynb
```

### Option B — Python venv

```bash
# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook notebooks/eda_prac.ipynb
```

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

> Install all at once: `pip install -r requirements.txt`

---

## Output — Processed Dataset

`data/telco_churn_processed.csv` contains:

- **7043 rows × 26 columns** (25 features + 1 target)
- All features fully numeric (`int64` / `float64`)
- No missing or infinite values
- Continuous features z-score scaled
- All categorical features encoded

---

## Next Steps (Modelling)

1. **Train-Test Split** — `train_test_split(stratify=Churn_Numeric, test_size=0.2)`
2. **Handle Class Imbalance** — SMOTE on training set only, or `class_weight='balanced'`
3. **Baseline Model** — Logistic Regression with all 25 features
4. **Advanced Models** — Random Forest, XGBoost (leverage feature importances for selection)
5. **Feature Selection** — Drop `gender` (r = −0.009); run VIF analysis for multicollinearity

---

## Branch Structure

| Branch | Purpose |
|---|---|
| `main` | Stable, reviewed code |
| `eda_analyze` | EDA & preprocessing work (this branch) |
