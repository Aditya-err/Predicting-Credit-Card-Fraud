# FEATURE ANALYSIS & DOMAIN INTELLIGENCE
**Playground Series S6E11: Predicting Credit Card Fraud**
*Conducted by: Senior Kaggle Grandmaster & ML Research Engineer*

---

## 1. Master Univariate Predictive Ranking

All 24 predictor features evaluated against the binary target `is_fraud` on the full training set (699,960 samples), ranked by standalone **Univariate ROC AUC**:

| Rank | Feature Name | Type | Cardinality | Univariate ROC AUC | Pearson Corr ($r$) | Spearman Corr ($\rho$) | Fraud Risk Spread (Min $\to$ Max) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | `merchant_risk_score` | Numerical | 950 | **0.74894** | +0.1254 | +0.1105 | Continuous monotonic |
| **2** | `velocity_score` | Numerical | 666 | **0.71079** | +0.1016 | +0.0936 | Zero: 0.8% $\to$ High: 6.2% |
| **3** | `cvv_retry_count` | Numerical (Count)| 4 | **0.70808** | +0.2123 | +0.1936 | 0: 1.1% $\to$ 3: 28.4% |
| **4** | `auth_method` | Categorical | 5 | **0.68657** | — | — | Biometric: 0.50% $\to$ No Auth: 6.45% |
| **5** | `is_new_merchant` | Boolean Flag | 2 | **0.64224** | +0.0817 | +0.0817 | False: 1.12% $\to$ True: 4.73% |
| **6** | `merchant_category` | Categorical | 12 | **0.63900** | — | — | Retail: 1.11% $\to$ Gift Cards: 7.30% |
| **7** | `customer_age` | Numerical | 64 | **0.62472** | +0.0561 | +0.0554 | 18: 0.8% $\to$ 81: 3.1% |
| **8** | `txn_count_last_24h` | Numerical (Count)| 12 | **0.62031** | +0.0580 | +0.0545 | 0: 0.6% $\to$ 8+: 4.8% |
| **9** | `ip_country_mismatch`| Boolean Flag | 2 | **0.57296** | +0.0768 | +0.0768 | False: 1.43% $\to$ True: 16.50% |
| **10** | `hours_since_last_txn`| Numerical | 3,633 | **0.56822** | -0.0205 | -0.0303 | Fast: 2.8% $\to$ Slow: 1.1% |
| **11** | `is_foreign_transaction`| Boolean Flag | 2 | **0.55738** | +0.0617 | +0.0617 | False: 1.48% $\to$ True: 12.57% |
| **12** | `used_vpn` | Boolean Flag | 2 | **0.55024** | +0.0537 | +0.0537 | False: 1.50% $\to$ True: 6.55% |
| **13** | `amount_usd` | Numerical | 13,387 | **0.54485** | +0.0194 | +0.0199 | Low: 1.2% $\to$ High: 3.8% |
| **14** | `billing_shipping_mismatch`| Boolean Flag | 2 | **0.54458** | +0.0494 | +0.0494 | False: 1.52% $\to$ True: 12.99% |
| **15** | `prior_disputes` | Numerical (Count)| 5 | **0.53218** | +0.0236 | +0.0234 | 0: 1.4% $\to$ 4: 5.1% |
| **16** | `is_ai_generated_scam_attempt`| Boolean Flag | 2 | **0.51356** | +0.0381 | +0.0381 | False: 1.63% $\to$ True: 15.22% |
| **17** | `card_type` | Categorical | 5 | **0.51186** | — | — | Discover: 1.46% $\to$ Amex: 1.90% |
| **18** | `channel` | Categorical | 5 | **0.51122** | — | — | Contactless: 1.55% $\to$ Online: 1.74% |
| **19** | `device_type` | Categorical | 8 | **0.50867** | — | — | Android: 1.61% $\to$ POS: 1.73% |
| **20** | `day_of_week` | Numerical (Cycle)| 7 | **0.50813** | +0.0037 | +0.0036 | Mon: 1.58% $\to$ Sun: 1.78% |
| **21** | `account_balance_usd`| Numerical | 43,317 | **0.50625** | +0.0011 | +0.0028 | Weak non-linear relationship |
| **22** | `distance_from_home_km`| Numerical | 4,365 | **0.50387** | +0.0018 | +0.0017 | Weak standalone effect |
| **23** | `card_age_months` | Numerical | 49 | **0.50191** | +0.0007 | +0.0008 | Near-zero standalone effect |
| **24** | `time_of_day_hour` | Numerical (Cycle)| 24 | **0.50020** | -0.0001 | -0.0001 | Hour 1: 1.44% $\to$ Hour 21: 1.83% |

---

## 2. In-Depth Domain Vector Investigations

### 1. Transaction Amount & Financial Capacity Patterns
- **`amount_usd` Distribution**:
  - Median: $27.25 | Mean: $79.89 | 95th Pct: $542.46 | 99th Pct: $936.95 | Max: $6,872.69
  - Standalone AUC is moderate (**0.54485**), but it operates as a heavy risk amplifier. Fraud transactions are 2.3x more prevalent when transaction amounts exceed $500.
- **Cent / Digit Forensics**:
  - Transactions ending in `.00` (e.g. $100.00): **0.94%** of rows, fraud rate = **2.089%** (+25% over baseline).
  - Transactions ending in `.99` (e.g. $49.99): **0.95%** of rows, fraud rate = **1.462%** (-12% below baseline).
  - All other cent endings: fraud rate = **1.668%**.
  - *Engineering take*: Binary indicator `is_round_amount = (amount_cents == 0)` carries a distinct positive signal.
- **Financial Balance Interaction (`amount_usd` vs `account_balance_usd`)**:
  - In 7,470 instances, `amount_usd > account_balance_usd`. In this subset, the fraud rate spikes to **2.48%** (compared to 1.66% when within balance).
  - Feature: `amount_to_balance_ratio = amount_usd / (account_balance_usd + 1)` has standalone AUC of **0.53444**.

### 2. Merchant Profile & Category Risk
- **`merchant_risk_score`**:
  - The single strongest continuous feature in the dataset (AUC = **0.74894**).
  - Range: 0.0 to 100.0. Transactions with score > 70.0 exhibit fraud rates exceeding 14.5%.
- **`merchant_category` Stratification**:
  - High-Risk Tier:
    - *Gift Cards*: 9,456 transactions (1.35% of total), fraud rate = **7.297%** (4.4x baseline).
    - *Crypto Exchange*: 5,285 transactions (0.76% of total), fraud rate = **6.528%** (3.9x baseline).
    - *Gaming*: 23,784 transactions (3.40% of total), fraud rate = **5.512%** (3.3x baseline).
    - *Electronics*: 48,693 transactions (6.96% of total), fraud rate = **3.083%** (1.8x baseline).
  - Moderate-Risk Tier:
    - *Streaming* (2.25%), *Travel* (1.86%).
  - Low-Risk Baseline Tier:
    - *Utilities* (1.38%), *Healthcare* (1.29%), *Fuel* (1.24%), *Restaurants* (1.14%), *Groceries* (1.11%), *Online Retail* (1.11%).
- **`is_new_merchant`**:
  - When buying from an unfamiliar merchant, fraud rate jumps from **1.12% to 4.73%** (4.2x multiplier, AUC = **0.64224**).

### 3. Authentication & Channel Dynamics
- **`auth_method` Disruption**:
  - *No Authentication*: 67,060 transactions (9.58% volume) $\to$ Fraud rate: **6.445%** (AUC = **0.68657**).
  - *PIN*: 154,020 transactions (22.0% volume) $\to$ Fraud rate: **0.868%**.
  - *Biometric*: 93,666 transactions (13.38% volume) $\to$ Fraud rate: **0.500%** (13x safer than No Auth!).
  - *OTP* (1.41%) and *3D Secure* (1.48%) hold near-baseline risk.
- **The "Lethal Intersection" (`merchant_category` × `auth_method`)**:
  When high-risk categories lack authentication, fraud rates reach critical levels:
  - `Gift Cards` + `No Authentication`: **23.01%** fraud rate (1,104 transactions).
  - `Crypto Exchange` + `No Authentication`: **22.32%** fraud rate (466 transactions).
  - `Gaming` + `No Authentication`: **19.09%** fraud rate (2,629 transactions).
  - `Electronics` + `No Authentication`: **11.34%** fraud rate (4,543 transactions).
  - *Takeaway*: A composite category interaction `merchant_cat_x_auth = merchant_category + "_" + auth_method` is an essential engineered feature.

### 4. Behavioral Velocity & Time Deltas
- **`velocity_score`** (AUC = **0.71079**): Directly measures card velocity spikes.
- **`txn_count_last_24h`** (AUC = **0.62031**): Count of transactions in 24 hours.
- **`hours_since_last_txn`** (AUC = **0.56822**): Time elapsed since previous transaction. Fraudsters act fast before accounts are flagged (smaller hours = higher fraud).
- **The Discovery Feature: `velocity_score * merchant_risk_score`**:
  - Multiplying these two features creates a unified behavioral-contextual risk index.
  - **Standalone Univariate ROC AUC = 0.79805 (~0.80)!**
  - This is by far the highest single-feature signal in the entire dataset.

### 5. Boolean Risk Flags: The Exponential Compounding Pattern
The dataset features 6 explicit boolean security flags. Each has high precision when activated:
1. `ip_country_mismatch`: True fraud rate = **16.50%** (vs 1.43% False)
2. `is_ai_generated_scam_attempt`: True fraud rate = **15.22%** (vs 1.63% False)
3. `billing_shipping_mismatch`: True fraud rate = **12.99%** (vs 1.52% False)
4. `is_foreign_transaction`: True fraud rate = **12.57%** (vs 1.48% False)
5. `used_vpn`: True fraud rate = **6.55%** (vs 1.50% False)
6. `is_new_merchant`: True fraud rate = **4.73%** (vs 1.12% False)

#### The Exponential Risk Cascade (`risk_flag_count`):
When computing the integer sum of active risk flags per row:
$$\text{risk\_flag\_count} = \sum \text{flag}_i$$

| Active Flags | Transaction Count | Pct of Dataset | Fraud Count | Fraud Rate | Risk Multiplier vs 0 Flags |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0 Flags** | 549,155 | 78.46% | 3,871 | **0.705%** | 1.0x (Ultra Safe) |
| **1 Flag** | 137,966 | 19.71% | 4,980 | **3.609%** | **5.1x** |
| **2 Flags** | 11,455 | 1.64% | 2,082 | **18.175%** | **25.8x** |
| **3 Flags** | 1,268 | 0.18% | 658 | **51.893%** | **73.6x (Majority Fraud)** |
| **4 Flags** | 112 | 0.016% | 94 | **83.929%** | **119.0x** |
| **5 Flags** | 4 | 0.0006% | 4 | **100.000%** | **141.8x (Certain Fraud)** |
| **6 Flags** | 0 | 0.00% | 0 | — | — |

This exponential progression is an explicit confirmation of additive/logistic risk generation. Incorporating `risk_flag_count` and its non-linear polynomials will directly feed gradient boosting split efficiency.

### 6. Synthetic Generation Artifacts & Integrity Findings
- **Geographic Decoupling**: In real-world data, `is_foreign_transaction = True` would mandate a `distance_from_home_km` of thousands of kilometers. In this synthetic dataset, the mean distance for foreign transactions is **13.64 km** (virtually identical to 13.58 km for domestic transactions), capped at 58 km. Tree algorithms must not attempt to enforce geographic consistency.
- **Cardholder Demographics**: `customer_age` is uniformly populated from 18 to 81 with higher fraud rates at older ages (age 70+ fraud rate ~3.1% vs age 20-30 ~1.0%).
- **Time Invariance**: `time_of_day_hour` and `day_of_week` show minimal standalone correlation ($r \approx 0.00$), but hour 21 has the peak fraud rate (1.83%) and hour 1 has the trough (1.44%).
