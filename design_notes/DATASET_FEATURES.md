# CDIF Dataset Features & Preprocessing Guide

This document describes all datasets used in the PNKDIF evaluation, including feature splits and preprocessing.

---

## Dataset Summary

| Dataset | N | d_context | d_behavior | Anomaly % | Type |
|---------|---|-----------|------------|-----------|------|
| Syn-HighDim | 10,000 | 20 | 3 | 5% | Synthetic |
| Syn-Cluster | 10,000 | 2 | 3 | 5% | Synthetic |
| Adult | 30,162 | 7 | 6 | 5% | Real (injected) |
| Bank | 30,488 | 4 | 12 | 5% | Real (injected) |
| Cardio | 1,831 | 5 | 16 | 9.6% | Real (ODDS) |
| Thyroid | 3,772 | 3 | 3 | 2.5% | Real (ODDS) |
| SAML | 292,715 | 38 | 11 | 1.7% | Real (AML) |
| SAML (injected) | 292,715 | 38 | 11 | 3.7% | Real + injection |
| IEEE-CIS | 100,000 | 11 | 31 | 3.5% | Real (fraud) |
| PaySim | 100,000 | 3 | 7 | 0.1% | Real (fraud) |
| CreditCard | 284,807 | 2 | 28 | 0.17% | Real (fraud) |

---

## 1. Synthetic Datasets

### Syn-HighDim (High-Dimensional Context)

Tests robustness to irrelevant context dimensions.

| Type | Description |
|------|-------------|
| **Context** | 20 dimensions, only 2 are informative |
| **Behavior** | 3 dimensions, correlated with informative context |
| **Anomalies** | Shift in behavior relative to local peers |

### Syn-Cluster (Cluster-Specific Behavior)

Tests detection of cluster-conditioned anomalies.

| Type | Description |
|------|-------------|
| **Context** | 2D cluster assignment |
| **Behavior** | 3D, different mean/std per cluster |
| **Anomalies** | Points with behavior from wrong cluster |

---

## 2. UCI Datasets (with Injection)

### Adult Dataset (`adult.data`)

**Source:** UCI Adult/Census Income Dataset

| Type | Features |
|------|----------|
| **Context** | workclass, education, marital_status, relationship, race, sex, native_country (7) |
| **Behavior** | age, fnlwgt, education_num, capital_gain, capital_loss, hours_per_week (6) |
| **Injection** | Low capital → High capital behavior swap |

### Bank Dataset (`bank-additional-full.csv`)

**Source:** UCI Bank Marketing Dataset

| Type | Features |
|------|----------|
| **Context** | age, job, marital, education (4) |
| **Behavior** | duration, campaign, pdays, previous, balance, day, default, housing, loan, contact, month, poutcome (12) |
| **Injection** | Young → Senior behavior swap |

---

## 3. ODDS Datasets

### Cardio

**Source:** ODDS (Outlier Detection DataSets)

| Type | Features |
|------|----------|
| **Context** | First 5 features (demographics) |
| **Behavior** | Remaining 16 features (clinical) |
| **Anomalies** | Original labels (heart disease) |

### Thyroid

**Source:** ODDS

| Type | Features |
|------|----------|
| **Context** | First 3 features |
| **Behavior** | Last 3 features |
| **Anomalies** | Original labels (thyroid disease) |

---

## 4. Fraud Detection Datasets

### SAML-D (Anti-Money Laundering)

**Source:** Kaggle Synthetic AML Dataset
**Level:** Account-level (aggregated from 9.5M transactions)

| Type | Features | Count |
|------|----------|-------|
| **Context** | Distribution profiles: loc_{country}, curr_{currency}, ptype_{type} | 38 |
| **Behavior** | tx_count, tx_sum, tx_mean, tx_std, tx_max, tx_min, tx_range, unique_receivers, active_days, avg_tx_per_day, avg_receivers_per_day | 11 |
| **Anomalies (real)** | Is_laundering = 1 | 1.7% |
| **Anomalies (injected)** | Domestic accounts with cross-border behavior | +2% |

**Injection Strategy:** Accounts with low location entropy (domestic) are given behavior statistics of high-entropy accounts (cross-border). See `DATASET_INJECTION.md`.

### IEEE-CIS Fraud Detection

**Source:** Kaggle IEEE-CIS Competition

| Type | Features | Count |
|------|----------|-------|
| **Context** | ProductCD, card1-6, addr1-2, P_emaildomain, R_emaildomain | 11 |
| **Behavior** | TransactionAmt, TransactionDT, C1-C14, D1-D15 | 31 |
| **Anomalies** | isFraud = 1 | 3.5% |

### PaySim

**Source:** Kaggle PaySim Mobile Money

| Type | Features | Count |
|------|----------|-------|
| **Context** | type, step, nameOrig_hash | 3 |
| **Behavior** | amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, balance_change_orig, balance_change_dest | 7 |
| **Anomalies** | isFraud = 1 | 0.1% |

### Credit Card Fraud

**Source:** Kaggle Credit Card Fraud (European transactions)

| Type | Features | Count |
|------|----------|-------|
| **Context** | Time, Amount | 2 |
| **Behavior** | V1-V28 (PCA components) | 28 |
| **Anomalies** | Class = 1 | 0.17% |

**Note:** This dataset has weak context (Time, Amount) since original features were PCA-transformed for privacy.

---

## Preprocessing Pipeline

All datasets undergo:

1. **Categorical Encoding**
   - ≤20 unique values → One-hot encoding
   - >20 unique values → Frequency encoding

2. **Standardization**
   - z-score: `(x - μ) / σ`

3. **NaN Handling**
   - Fill with 0 (after standardization)

---

## Code Locations

| Dataset | Loader Function | File |
|---------|-----------------|------|
| Synthetic | `get_dataset()` | `src/data/synthetic.py` |
| Adult, Bank | `load_adult()`, `load_bank()` | `src/data/real_datasets.py` |
| SAML | `load_saml_account_level()` | `src/data/fraud_datasets.py` |
| SAML (injected) | `load_saml_with_injection()` | `src/data/fraud_datasets.py` |
| IEEE-CIS, PaySim, CreditCard, Thyroid | `load_*()` | `src/data/fraud_datasets.py` |
