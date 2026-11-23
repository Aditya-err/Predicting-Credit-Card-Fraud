# BASELINE EXPERIMENTS REPORT: EXP-001 THROUGH EXP-004
**Playground Series S6E11: Predicting Credit Card Fraud**
*Evaluation Metric: ROC AUC | Validation: Stratified 5-Fold CV (Seed 42)*

---

## 1. Executive Summary & Comparative Leaderboard

| Exp ID | Model Architecture | Features | OOF ROC AUC | Fold Mean (Std) | Runtime (s) | Delta vs EXP-001 | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-001** | LightGBM | Raw Features | **0.94519** | 0.94534 (±0.00319) | 114.2s | 0.00000 (Base) | `COMPLETED` |
| **EXP-002** | CatBoost | Raw Features | **0.94614** | 0.94619 (±0.00298) | 591.1s | +0.00094 | `COMPLETED` |
| **EXP-003** | XGBoost | Raw Features | **0.94580** | 0.94591 (±0.00322) | 202.3s | +0.00060 | `COMPLETED` |
| **EXP-004** | Ridge_Logistic_Regression | Raw Features | **0.94185** | 0.94188 (±0.00345) | 36.8s | -0.00334 | `COMPLETED` |

---

## 2. Fold-by-Fold Performance Breakdown

| Exp ID | Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | OOF Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-001** | LightGBM | 0.94352 | 0.94731 | 0.94961 | 0.94593 | 0.94033 | **0.94519** |
| **EXP-002** | CatBoost | 0.94444 | 0.94752 | 0.95081 | 0.94624 | 0.94193 | **0.94614** |
| **EXP-003** | XGBoost | 0.94393 | 0.94775 | 0.95051 | 0.94627 | 0.94108 | **0.94580** |
| **EXP-004** | Ridge_Logistic_Regression | 0.94128 | 0.94340 | 0.94688 | 0.94158 | 0.93625 | **0.94185** |

---

## 3. Feature Importance Analysis across Baseline Models

- **EXP-001 (LightGBM) Top Predictors**: `merchant_risk_score, velocity_score, customer_age, hours_since_last_txn, amount_usd`
- **EXP-002 (CatBoost) Top Predictors**: `merchant_risk_score, velocity_score, auth_method, customer_age, cvv_retry_count`
- **EXP-003 (XGBoost) Top Predictors**: `cvv_retry_count, ip_country_mismatch, is_new_merchant, billing_shipping_mismatch, is_foreign_transaction`
- **EXP-004 (Ridge_Logistic_Regression) Top Predictors**: `is_ai_generated_scam_attempt, billing_shipping_mismatch, ip_country_mismatch, auth_method_Biometric, used_vpn`

---

## 4. First Valid Submission Verification (EXP-001 LightGBM)

- **File Location**: `outputs/submissions/submission_EXP-001_lightgbm.csv`
- **Verification Status**: `VALID = True`
- **Row Count**: Exactly 299,984 rows (Matches sample submission)
- **Columns**: `['id', 'is_fraud']`
- **Null Count**: 0 missing values
- **ID Alignment**: Exactly matching `test.csv` (contiguous 699,960 to 999,943)
- **Prediction Statistics**: Min = 0.000008, Max = 0.994763, Mean = 0.016536, Median = 0.001953

---

## 5. Phase 5 Feature Preparation Audit (Inactive Features)

The 5 engineered features prepared in `src/features.py` were audited for distribution integrity across train and test:
- **`feat_merchant_x_auth`**:
  - Total unique combinations: 60
  - Low-frequency categories (<50 occurrences in train): 0
  - Unseen categories in test: **0** (Zero drift risk)
- **`feat_channel_x_auth`**:
  - Total unique combinations: 25
  - Low-frequency categories (<50 occurrences in train): 0
  - Unseen categories in test: **0** (Zero drift risk)
