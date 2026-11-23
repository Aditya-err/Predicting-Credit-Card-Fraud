# DATASET AUDIT & INTELLIGENCE REPORT
**Playground Series Season 6 Episode 11: Predicting Credit Card Fraud**
*Conducted by: Senior Kaggle Grandmaster & ML Research Engineer*
*Dataset Location: `E:\MCA projects\Credit card fraud detection\playground-series-s6e11`*

---

## 1. Dataset Dimensions & Storage Footprint

| File Name | Row Count | Column Count | File Size on Disk | In-Memory Footprint (pandas) |
| :--- | :--- | :--- | :--- | :--- |
| `train.csv` | 699,960 | 26 | 98.5 MB | ~138.8 MB |
| `test.csv` | 299,984 | 25 | 41.1 MB | ~57.2 MB |
| `sample_submission.csv` | 299,984 | 2 | 8.4 MB | ~4.6 MB |

- **Total Transactions Examined**: 999,944 transactions (nearly 1 million records).
- **Partition Ratio**: Exactly 70.0% Train (699,960) vs 30.0% Test (299,984).

---

## 2. Complete Schema & Datatype Inventory

All 26 columns in `train.csv` and 25 columns in `test.csv` were verified directly without assumptions:

| Column Name | Train Datatype | Test Datatype | Role | Null Count | Unique Count (Train) | Unique Count (Test) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `id` | int64 | int64 | Primary Key | 0 | 699,960 | 299,984 |
| `amount_usd` | float64 | float64 | Continuous Feature | 0 | 13,387 | 10,510 |
| `hours_since_last_txn` | float64 | float64 | Continuous Feature | 0 | 3,633 | 3,331 |
| `txn_count_last_24h` | int64 | int64 | Discrete Count | 0 | 12 | 11 |
| `distance_from_home_km` | float64 | float64 | Continuous Feature | 0 | 4,365 | 4,108 |
| `card_age_months` | int64 | int64 | Discrete Age | 0 | 49 | 49 |
| `customer_age` | int64 | int64 | Discrete Age | 0 | 64 | 64 |
| `account_balance_usd` | float64 | float64 | Continuous Feature | 0 | 43,317 | 31,803 |
| `cvv_retry_count` | int64 | int64 | Discrete Count | 0 | 4 | 4 |
| `velocity_score` | float64 | float64 | Continuous Feature | 0 | 666 | 644 |
| `time_of_day_hour` | int64 | int64 | Discrete Temporal | 0 | 24 | 24 |
| `day_of_week` | int64 | int64 | Discrete Temporal | 0 | 7 | 7 |
| `merchant_risk_score` | float64 | float64 | Continuous Feature | 0 | 950 | 917 |
| `prior_disputes` | int64 | int64 | Discrete Count | 0 | 5 | 5 |
| `merchant_category` | object | object | Categorical | 0 | 12 | 12 |
| `card_type` | object | object | Categorical | 0 | 5 | 5 |
| `auth_method` | object | object | Categorical | 0 | 5 | 5 |
| `channel` | object | object | Categorical | 0 | 5 | 5 |
| `device_type` | object | object | Categorical | 0 | 8 | 8 |
| `is_foreign_transaction` | bool | bool | Boolean Risk Flag | 0 | 2 | 2 |
| `is_new_merchant` | bool | bool | Boolean Risk Flag | 0 | 2 | 2 |
| `used_vpn` | bool | bool | Boolean Risk Flag | 0 | 2 | 2 |
| `ip_country_mismatch` | bool | bool | Boolean Risk Flag | 0 | 2 | 2 |
| `billing_shipping_mismatch` | bool | bool | Boolean Risk Flag | 0 | 2 | 2 |
| `is_ai_generated_scam_attempt` | bool | bool | Boolean Risk Flag | 0 | 2 | 2 |
| `is_fraud` | float64 | *Target* | Binary Target | 0 | 2 | — |

---

## 3. Target Distribution & Class Imbalance Analysis

The target variable is `is_fraud`:
- **Class 0.0 (Legitimate)**: 688,271 records (**98.33%**)
- **Class 1.0 (Fraudulent)**: 11,689 records (**1.67%**)
- **Exact Fraud Incidence Rate**: **0.0166995** (~1.67%)
- **Negative-to-Positive Imbalance Ratio**: **58.88 : 1**

### Strategic Implications of the 1.67% Fraud Rate:
1. Standard accuracy is an entirely deceptive metric (a trivial dummy model predicting all 0s achieves 98.33% accuracy with an ROC AUC of 0.500).
2. Resampling techniques like naive SMOTE or random undersampling should be avoided: they distort the natural posterior probability calibration without benefiting rank-based ROC AUC.
3. Gradient boosting algorithms natively handle this ratio via standard binary cross-entropy loss; tuning `scale_pos_weight` is unnecessary for ROC AUC since it produces a monotonic logit shift.

---

## 4. Missing Values, Duplicates & Data Integrity Checks

| Audit Metric | Train Value | Test Value | Status / Finding |
| :--- | :--- | :--- | :--- |
| **Total Missing Values** | 0 (0.00%) | 0 (0.00%) | Perfect data completeness across all columns |
| **Exact Duplicate Rows** | 0 | 0 | Zero redundant records |
| **Feature Duplicates (ignoring `id`)** | 0 | 0 | Every transaction profile is unique |
| **Train/Test Feature Set Overlap** | 0 | 0 | Zero exact record contamination across splits |
| **ID Column Monotonicity** | Strict step +1 | Strict step +1 | Train: `0` to `699,959`; Test: `699,960` to `999,943` |
| **Test ID Continuity** | Begins at `699,960` | Matches `test.csv` | Seamless contiguous primary key |
| **ID Target Leakage Check** | ROC AUC = **0.49855** | Deciles: 1.58%–1.73% | Zero temporal or target sorting leakage in `id` |

---

## 5. Train vs Test Distribution Stability (Covariate Shift Analysis)

To test whether the test set comes from the exact same data-generating distribution, a two-sample **Kolmogorov-Smirnov (KS) Test** was computed for every numerical feature:

| Numerical Feature | Train Mean (Std) | Test Mean (Std) | Train Median | Test Median | KS Statistic | KS p-value | Shift Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `amount_usd` | 79.89 (205.17) | 79.92 (201.85) | 27.25 | 27.28 | 0.00153 | 0.7058 | **Identical (No drift)** |
| `hours_since_last_txn` | 6.63 (6.29) | 6.64 (6.31) | 4.73 | 4.72 | 0.00138 | 0.8199 | **Identical (No drift)** |
| `txn_count_last_24h` | 2.98 (1.50) | 2.98 (1.50) | 3.00 | 3.00 | 0.00065 | 0.9999 | **Identical (No drift)** |
| `distance_from_home_km` | 13.59 (11.62) | 13.60 (11.61) | 10.22 | 10.27 | 0.00197 | 0.3890 | **Identical (No drift)** |
| `card_age_months` | 46.86 (5.58) | 46.84 (5.59) | 47.00 | 47.00 | 0.00266 | 0.1014 | **Identical (No drift)** |
| `customer_age` | 51.43 (17.90) | 51.42 (17.89) | 52.00 | 52.00 | 0.00080 | 0.9993 | **Identical (No drift)** |
| `account_balance_usd` | 2064.99 (3400.13) | 2071.47 (3430.01) | 1382.74 | 1388.54 | 0.00186 | 0.4629 | **Identical (No drift)** |
| `cvv_retry_count` | 0.0855 (0.2895) | 0.0854 (0.2891) | 0.00 | 0.00 | 0.00008 | 1.0000 | **Identical (No drift)** |
| `velocity_score` | 18.93 (10.86) | 18.94 (10.87) | 18.60 | 18.70 | 0.00139 | 0.8147 | **Identical (No drift)** |
| `time_of_day_hour` | 11.76 (6.87) | 11.75 (6.87) | 12.00 | 12.00 | 0.00157 | 0.6793 | **Identical (No drift)** |
| `day_of_week` | 2.98 (2.00) | 2.97 (2.00) | 3.00 | 3.00 | 0.00383 | 0.0043 | Minor uniform jitter |
| `merchant_risk_score` | 35.22 (12.63) | 35.22 (12.62) | 34.20 | 34.20 | 0.00215 | 0.2846 | **Identical (No drift)** |
| `prior_disputes` | 0.1528 (0.3811) | 0.1539 (0.3831) | 0.00 | 0.00 | 0.00082 | 0.9988 | **Identical (No drift)** |

### Categorical & Boolean Distribution Alignment
- **Unseen Categories in Test**: **0** across all 5 categorical features (`merchant_category`, `card_type`, `auth_method`, `channel`, `device_type`).
- **Boolean Risk Proportions**:
  - `is_foreign_transaction`: Train 1.73% vs Test 1.72%
  - `is_new_merchant`: Train 15.25% vs Test 15.24%
  - `used_vpn`: Train 3.38% vs Test 3.39%
  - `ip_country_mismatch`: Train 1.62% vs Test 1.61%
  - `billing_shipping_mismatch`: Train 1.29% vs Test 1.30%
  - `is_ai_generated_scam_attempt`: Train 0.33% vs Test 0.33%

### Covariate Shift Verdict:
Train and test were produced by the exact same synthetic generator and partitioned via random split. There is **no covariate shift or concept drift**. A properly configured Stratified 5-Fold Cross-Validation scheme will correlate with the Leaderboard.

---

## 6. Numerical Feature Distributions, Skewness & Outliers

| Feature | Min | 1st Pct | 25th Pct | 50th Pct | 75th Pct | 95th Pct | 99th Pct | 99.9th Pct | Max | Skewness | Kurtosis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `amount_usd` | $1.00 | $4.42 | $15.81 | $27.25 | $43.11 | $542.46 | $936.95 | $1,694.19 | $6,872.69 | +7.00 | 97.27 |
| `account_balance_usd` | $58.25 | $454.85 | $834.12 | $1,382.74 | $2,517.02 | $4,861.18 | $9,082.81 | $57,720.99 | $99,701.39 | +14.44 | 278.35 |
| `velocity_score` | 0.0 | 0.0 | 11.4 | 18.6 | 26.0 | 37.4 | 46.1 | 55.5 | 72.9 | +0.26 | -0.15 |
| `merchant_risk_score` | 0.0 | 11.2 | 26.4 | 34.2 | 42.7 | 57.3 | 72.0 | 89.4 | 100.0 | +0.62 | 0.99 |
| `hours_since_last_txn` | 0.01 | 0.11 | 2.11 | 4.73 | 9.19 | 19.40 | 29.38 | 41.01 | 51.98 | +1.82 | 4.40 |
| `distance_from_home_km`| 0.00 | 0.08 | 4.43 | 10.22 | 19.75 | 37.94 | 47.86 | 50.96 | 57.93 | +1.06 | 0.45 |
| `customer_age` | 18 | 19 | 38 | 52 | 66 | 79 | 81 | 81 | 81 | -0.14 | -1.05 |
| `card_age_months` | 25 | 34 | 43 | 47 | 51 | 56 | 60 | 65 | 73 | +0.13 | -0.02 |

### Key Distributional Discoveries:
1. **Extreme Right-Skew in Monetary Values**: Both `amount_usd` (skew = +7.00, kurtosis = 97.27) and `account_balance_usd` (skew = +14.44, kurtosis = 278.35) have long fat tails. 75% of all transactions are under $43.11, but the maximum reaches $6,872.69.
2. **Artificial Bounding / Truncation**:
   - `distance_from_home_km` is strictly truncated at ~58 km (max 57.93 in train, 60.47 in test).
   - `customer_age` is capped strictly at 81.
   - `merchant_risk_score` is scaled in $[0.0, 100.0]$.
3. **Discrete Count Zero-Inflation**:
   - `cvv_retry_count`: 91.73% are 0.
   - `prior_disputes`: 85.49% are 0.
   - `velocity_score`: 6.93% are 0.0.

---

## 7. Multicollinearity Assessment

Pairwise correlation matrix check across all numerical columns revealed:
- **`txn_count_last_24h` vs `velocity_score`**: Pearson correlation = **+0.6214** (moderate collinearity by domain definition).
- **All other feature pairs**: Pearson $|r| < 0.25$.
- **Finding**: The feature space is exceptionally orthogonal. Tree-based models will not suffer from severe split dilution caused by collinear redundancies.
