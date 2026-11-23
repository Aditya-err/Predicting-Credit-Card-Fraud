# Kaggle Playground Series S6E11: Predicting Credit Card Fraud

An end-to-end, Grandmaster-tier competitive machine learning pipeline and rigorous statistical audit for **Kaggle Playground Series – Season 6 Episode 11 (Predicting Credit Card Fraud)**.

[![Metric](https://img.shields.io/badge/Metric-ROC%20AUC-blue.svg)](https://www.kaggle.com/competitions/playground-series-s6e11)
[![Public LB](https://img.shields.io/badge/Kaggle%20Public%20LB-0.94801-brightgreen.svg)](https://www.kaggle.com/competitions/playground-series-s6e11)
[![Final OOF AUC](https://img.shields.io/badge/5--Fold%20CV%20OOF%20AUC-0.94703-success.svg)](reports/FINAL_RESULTS.md)
[![Validation](https://img.shields.io/badge/Validation-Stratified%205--Fold%20CV-orange.svg)](reports/WINNING_STRATEGY.md)

---

## Overview

- **Competition**: [Kaggle Playground Series S6E11](https://www.kaggle.com/competitions/playground-series-s6e11/overview)
- **Objective**: Predict the continuous probability of fraudulent credit card transactions (`is_fraud`).
- **Official Evaluation Metric**: **ROC AUC** (Area Under the Receiver Operating Characteristic Curve).
- **Dataset Scale**:
  - `train.csv`: 699,960 rows × 26 columns (~1.67% fraud rate, 58.88:1 class imbalance).
  - `test.csv`: 299,984 rows × 25 columns.
  - Zero missing values, zero exact duplicates.

---

## Key Breakthroughs & Methodological Discoveries

1. **The Shallow Architecture Breakthrough (`EXP-008` to `EXP-010`)**:
   - The synthetic dataset generator exhibits a strong additive logistic risk structure.
   - Default deep gradient boosting trees overfit continuous interaction permutations. Restricting LightGBM to `max_depth: 4`, `num_leaves: 10`, CatBoost to `depth: 5`, and XGBoost to `max_depth: 4` yielded a major breakthrough (+0.00156 OOF ROC AUC over baseline).
2. **Behavioral Velocity × Contextual Risk Interaction**:
   - Engineered feature `velocity_score * merchant_risk_score` delivers a standalone univariate ROC AUC of **0.79805**, acting as the core non-linear split driver across all tree models.
3. **Exponential Risk Flag Compounding**:
   - The integer sum of 6 security flags (`risk_flag_count`) captures an exponential monotonic fraud risk cascade:
     - 0 flags: **0.70%** fraud rate
     - 1 flag: **3.61%** fraud rate
     - 2 flags: **18.18%** fraud rate
     - 3 flags: **51.89%** fraud rate (majority fraud)
     - 4 flags: **83.93%** fraud rate
     - 5 flags: **100.0%** fraud rate
4. **Percentile Rank Fusion Ensemble (`EXP-017`)**:
   - Because ROC AUC evaluates pairwise rank ordering (Wilcoxon-Mann-Whitney U statistic), raw probability averaging suffers from inter-model calibration skew.
   - Normalizing predictions to uniform percentile ranks (`scipy.stats.rankdata(p) / N`) and optimizing ensemble weights with the Nelder-Mead simplex algorithm produced our top result: **0.94703 OOF ROC AUC**.
5. **Zero Optimism Bias in Nested Cross-Validation**:
   - A strict 5-fold nested cross-validation confirmed zero in-sample weight overfitting (nested holdout AUC: **0.947027**, matching full OOF AUC to 6 decimal places).

---

## Experimentation Progression Leaderboard

All 17 experiments evaluated using **Stratified 5-Fold Cross-Validation** (Seed 42) across all 699,960 training samples:

| Experiment ID | Model Architecture | Feature Strategy | Out-of-Fold ROC AUC | 5-Fold Mean | Sample Std ($s$) | Runtime | $\Delta$ vs Baseline | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `EXP-001` | LightGBM Baseline | Raw Features | 0.94519 | 0.94534 | 0.00319 | 114.2s | `0.00000` | Verified |
| `EXP-002` | CatBoost Baseline | Raw Features | 0.94614 | 0.94619 | 0.00298 | 591.1s | `+0.00094` | Verified |
| `EXP-003` | XGBoost Baseline | Raw Features | 0.94580 | 0.94591 | 0.00322 | 202.3s | `+0.00060` | Verified |
| `EXP-004` | Ridge Logistic Reg | Scaled Raw Features | 0.94185 | 0.94188 | 0.00345 | 36.8s | `-0.00334` | Verified |
| `EXP-005` | LightGBM | Feature Pack 1 | 0.94486 | 0.94492 | 0.00345 | 91.0s | `-0.00033` | Verified |
| `EXP-006` | CatBoost | Feature Pack 1 | 0.94565 | 0.94568 | 0.00302 | 495.5s | `+0.00046` | Verified |
| `EXP-007` | XGBoost | Feature Pack 1 | 0.94496 | 0.94503 | 0.00320 | 163.4s | `-0.00023` | Verified |
| `EXP-008` | **Shallow LightGBM** | **Pack 1 (`depth: 4`, `leaves: 10`)** | **0.94675** | **0.94683** | **0.00313** | **223.1s** | **`+0.00156`** | **Breakthrough** |
| `EXP-009` | Shallow CatBoost | Pack 1 (`depth: 5`, `l2: 5.0`) | 0.94603 | 0.94607 | 0.00328 | 652.7s | `+0.00084` | Verified |
| `EXP-010` | Shallow XGBoost | Pack 1 (`max_depth: 4`) | 0.94637 | 0.94642 | 0.00333 | 338.2s | `+0.00118` | Verified |
| `EXP-011` | LightGBM Target Enc | Pack 1 + Smoothed OOF TE | 0.94671 | 0.94682 | 0.00309 | 200.3s | `+0.00152` | Verified |
| `EXP-012` | LightGBM Pack 2 | Pack 2 (Cent & Log Forensics) | 0.94673 | 0.94678 | 0.00294 | 187.5s | `+0.00154` | Verified |
| `EXP-013` | External Data Audit | Schema Drift Analysis | 0.94673 | 0.94678 | 0.00294 | 0.0s | `+0.00154` | Incompatible |
| `EXP-014` | Ridge Logistic Reg | Scaled Pack 2 | 0.94180 | 0.94183 | 0.00384 | 60.8s | `-0.00339` | Diversity Anchor |
| `EXP-015` | Multi-Seed LightGBM | Pack 2 (3 Seeds: `42, 2026, 777`) | 0.94679 | 0.94683 | 0.00329 | 360.0s | `+0.00160` | Variance Reduced |
| `EXP-016` | Probability Blend | Linear Weighted Blend | 0.94700 | 0.94704 | 0.00329 | 15.0s | `+0.00181` | Verified |
| **`EXP-017`** | **Rank Fusion Blend** | **Percentile Rank Fusion** | **0.94703** | **0.94707** | **0.00327** | **15.0s** | **`+0.00184`** | **WINNER** |

---

## Project Structure

```text
├── configs/
│   └── config.yaml               # Model parameters, file paths, CV settings
├── src/
│   ├── data.py                   # Memory downcasting & safe CSV loaders
│   ├── features.py               # Feature engineering packs & in-fold target encoders
│   ├── models.py                 # LightGBM, CatBoost, XGBoost, and Ridge model factories
│   ├── validation.py             # Stratified 5-Fold CV harness (leakage-safe)
│   ├── evaluation.py             # ROC AUC, PR-AUC, Brier score calculations
│   ├── ensembling.py             # Percentile rank fusion & Nelder-Mead weight optimizer
│   └── submission.py             # Validation & generation of submission CSVs
├── reports/
│   ├── COMPETITION_ANALYSIS.md   # Domain understanding & metric math
│   ├── DATASET_AUDIT.md          # 999k rows statistical distributions & KS drift tests
│   ├── FEATURE_ANALYSIS.md       # Univariate rankings & risk multiplier forensic checks
│   ├── COMPETITION_RESEARCH.md   # Research into public baselines & shallow tree hypotheses
│   ├── WINNING_STRATEGY.md       # End-to-end architecture & measured performance
│   ├── BASELINE_RESULTS.md       # Baselines EXP-001 through EXP-004 verification
│   ├── FINAL_RESULTS.md          # Final complete results & submission diagnostics
│   └── FINAL_NESTED_VALIDATION_CHECK.md # Independent nested CV proof (zero optimism bias)
├── tests/                        # 13 automated unit tests (100% passing)
├── experiments/
│   └── experiment_log.csv        # Detailed sequential experiment tracking
├── outputs/
│   ├── oof/                      # Out-of-fold prediction matrices for all models
│   ├── predictions/              # Test prediction matrices for all models
│   └── submissions/              # Verified competition submission files
├── run_baselines.py              # Script to run EXP-001 to EXP-004
├── run_advanced_pipeline.py      # Script to run EXP-005 to EXP-017
└── submission.csv                # Final competition submission (ready for upload)
```

---

## Quickstart & Reproduction

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Aditya-err/<your-repo-name>.git
cd "Credit card fraud detection"

# Install dependencies
pip install lightgbm catboost xgboost scikit-learn pandas scipy pyyaml pytest psutil
```

### 2. Download Data
Place the Kaggle competition CSV files into `playground-series-s6e11/`:
- `train.csv`
- `test.csv`
- `sample_submission.csv`

### 3. Run Automated Tests
```bash
python -m pytest tests/
```

### 4. Run Baseline Experiments (EXP-001 to EXP-004)
```bash
python run_baselines.py
```

### 5. Run Full Advanced Pipeline & Ensembling (EXP-005 to EXP-017)
```bash
python run_advanced_pipeline.py
```

The final evaluated submission will be output to `submission.csv`.

---

## Verification & Compliance

The output file `submission.csv` has been checked against all official Kaggle submission criteria:
- **Row count**: Exactly **299,984** rows.
- **Columns**: `id`, `is_fraud`.
- **Integrity**: 0 null / NaN values; test IDs match `test.csv` sequence exactly (contiguous `699960` to `999943`).
- **Predictions**: Scaled continuously in $[0.0, 1.0]$.
