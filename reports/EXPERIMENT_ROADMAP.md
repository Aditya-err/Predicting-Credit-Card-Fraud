# EXPERIMENT ROADMAP & 8-DAY EXECUTION PLAN
**Playground Series S6E11: Predicting Credit Card Fraud**
*Conducted by: Senior Kaggle Grandmaster & ML Research Engineer*

---

## 1. Experimentation Governance & Protocol

Every model iteration, feature addition, and hyperparameter modification must be cataloged using this structured schema:

| Experiment Field | Description / Requirement |
| :--- | :--- |
| **Experiment ID** | Unique sequential identifier (`EXP-001`, `EXP-002`, etc.) |
| **Hypothesis** | Explicit scientific hypothesis (e.g., "Velocity × Merchant Risk interaction will improve OOF AUC") |
| **Model Family** | Algorithm used (`LightGBM`, `CatBoost`, `XGBoost`, `Ridge`, `RankBlend`) |
| **Feature Set** | Exact set of features (`raw_only`, `pack_v1`, `oof_target_encoded`) |
| **Validation Strategy**| `Stratified 5-Fold CV (Seed 42)` |
| **OOF ROC AUC** | 5-Fold out-of-fold ROC AUC score on full 699,960 samples |
| **Public LB Score** | Score returned by Kaggle on 20% test partition (when submitted) |
| **Runtime (Minutes)** | Total wall-clock time across all 5 folds |
| **Delta over Baseline**| $\Delta = \text{OOF}_{\text{new}} - \text{OOF}_{\text{baseline}}$ |
| **Decision** | `KEEP` (if $\Delta > +0.00010$ and monotonic across folds) / `DISCARD` / `FREEZE` |

---

## 2. Master Experiment Registry (Planned & Baseline Tracking)

| Experiment ID | Model Architecture | Feature Set Description | Target Metric (OOF AUC) | Status | Expected Runtime | Decision Rule |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-001** | LightGBM Baseline | Raw features (Categoricals as Ordinal / Categorical) | ~0.95220 | **Ready to Run** | ~2.5 min | Foundation Baseline |
| **EXP-002** | CatBoost Baseline | Raw features (Native Categoricals) | ~0.95250 | **Ready to Run** | ~4.5 min | Foundation Baseline |
| **EXP-003** | XGBoost Baseline | Raw features (Histogram binning) | ~0.95210 | **Ready to Run** | ~3.0 min | Foundation Baseline |
| **EXP-004** | Ridge Logistic Reg | Scaled raw + One-Hot Encoded categoricals | ~0.94700 | **Ready to Run** | ~1.0 min | Diversity Anchor |
| **EXP-005** | LightGBM + Feature Pack 1 | Raw + `velocity_x_risk` + `risk_flag_count` + `amount_to_balance` | ~0.95380 | Planned (Day 2) | ~3.0 min | Keep if $\Delta > +0.0010$ |
| **EXP-006** | CatBoost + Feature Pack 1 | Raw + Feature Pack 1 + `merchant_x_auth` | ~0.95420 | Planned (Day 2) | ~5.0 min | Keep if $\Delta > +0.0010$ |
| **EXP-007** | XGBoost + Feature Pack 1 | Raw + Feature Pack 1 | ~0.95370 | Planned (Day 2) | ~3.5 min | Keep if $\Delta > +0.0010$ |
| **EXP-008** | LightGBM Shallow Tuning | Feature Pack 1, Grid: `num_leaves` [6, 8, 12], `max_depth` [3, 4, 5] | ~0.95440 | Planned (Day 3) | ~6.0 min | Keep best leaves/depth |
| **EXP-009** | CatBoost Shallow Tuning | Feature Pack 1, Depth [4, 5, 6], `l2_leaf_reg` [3, 5, 10] | ~0.95460 | Planned (Day 3) | ~8.0 min | Keep best depth/reg |
| **EXP-010** | XGBoost Shallow Tuning | Feature Pack 1, Depth [3, 4, 5], `colsample` [0.7, 0.85] | ~0.95430 | Planned (Day 3) | ~6.0 min | Keep best depth |
| **EXP-011** | GBDT + OOF Target Encoding | Add smoothed 5-fold OOF target encoding on categoricals | ~0.95480 | Planned (Day 4) | ~4.0 min | Keep if OOF increases |
| **EXP-012** | Multi-Seed LightGBM | Seeds [42, 2026, 777] averaged | ~0.95490 | Planned (Day 6) | ~8.0 min | Variance reduction |
| **EXP-013** | Multi-Seed CatBoost | Seeds [42, 2026, 777] averaged | ~0.95505 | Planned (Day 6) | ~14.0 min | Variance reduction |
| **EXP-014** | Multi-Seed XGBoost | Seeds [42, 2026, 777] averaged | ~0.95480 | Planned (Day 6) | ~10.0 min | Variance reduction |
| **EXP-015** | GBDT Rank Blend | Rank average of EXP-012, 013, 014 | ~0.95535 | Planned (Day 7) | < 1.0 min | Keep if beats best single |
| **EXP-016** | Nelder-Mead Rank Blend | Optimized weights: LGBM + CatBoost + XGBoost + Ridge | **~0.95550+** | Planned (Day 7) | ~2.0 min | Final Candidate 1 |
| **EXP-017** | Equal-Weight Rank Blend | Robust equal rank average of top 3 multi-seed GBDTs | **~0.95540+** | Planned (Day 8) | < 1.0 min | Final Candidate 2 |

---

## 3. Practical 8-Day Tactical Execution Plan

```
DAY 1: Baselines & Harness (EXP-001 to 004)
  └─ Establish 5-Fold Stratified CV, log baselines for LGBM, CatBoost, XGBoost, Ridge.

DAY 2: Feature Engineering & Ablations (EXP-005 to 007)
  └─ Inject velocity*risk, risk_flag_count, merchant*auth crosses, amount_to_balance.

DAY 3: Shallow Hyperparameter Calibration (EXP-008 to 010)
  └─ Calibrate shallow tree depths (LGBM leaves 6-12, CatBoost depth 4-6, XGB depth 3-5).

DAY 4: Out-of-Fold Encoding & Statistical Aggregations (EXP-011)
  └─ Implement strict 5-fold smoothed target encoding and frequency encoding.

DAY 5: Model Diversity & Linear Anchor (EXP-004 tuning)
  └─ Train regularized L2 Ridge model and test early probability/rank correlation.

DAY 6: Multi-Seed Variance Suppression (EXP-012 to 014)
  └─ Run 3 seeds (42, 2026, 777) per GBDT model family; compute fold stability.

DAY 7: Advanced Rank Fusion & Ensembling (EXP-015 to 016)
  └─ Optimize rank blend weights using Nelder-Mead simplex on OOF rank predictions.

DAY 8: Final Submission Selection & Code Freeze (EXP-017)
  └─ Generate Submission 1 (Nelder-Mead Rank Blend) and Submission 2 (Robust Equal-Weight Rank Blend).
```

### Day 1: Infrastructure, Baselines & Verification
- Establish modular project directory: `src/` (`data.py`, `features.py`, `models.py`, `utils.py`).
- Implement `StratifiedKFold` (5 folds, seed 42) with strict target stratification.
- Run baseline LightGBM (`EXP-001`), CatBoost (`EXP-002`), XGBoost (`EXP-003`), and Ridge (`EXP-004`).
- Generate baseline submission to verify Kaggle score alignment ($|\text{CV} - \text{LB}| \le 0.0005$).

### Day 2: First-Tier Feature Engineering & Ablations
- Implement vectorized feature pipeline:
  - `velocity_score * merchant_risk_score`
  - `risk_flag_count` (integer sum of 6 boolean flags)
  - `merchant_category + "_" + auth_method`
  - `amount_usd / (account_balance_usd + 1)`
  - `is_round_cents`
- Run single-feature ablation on LightGBM: measure $\Delta \text{CV}$ for each feature individually.
- Discard features that do not demonstrate positive OOF gain.

### Day 3: Hyperparameter Optimization for Shallow Architectures
- Run focused grid searches on shallow tree configurations:
  - LightGBM: `num_leaves` $\in [6, 8, 10, 12]$, `max_depth` $\in [3, 4, 5]$, `min_child_samples` $\in [50, 100, 200]$.
  - CatBoost: `depth` $\in [4, 5, 6]$, `l2_leaf_reg` $\in [3, 5, 10]$.
  - XGBoost: `max_depth` $\in [3, 4, 5]$, `colsample_bytree` $\in [0.75, 0.85]$.
- Select best performing configuration per model family based strictly on OOF AUC.

### Day 4: Out-of-Fold Encoding & Category Encodings
- Implement strict 5-fold OOF smoothed target encoding for high-cardinality interaction features.
- Compare target encoding vs native CatBoost categorical handling vs integer label encoding.
- Log OOF CV delta. Keep only if CV strictly increases.

### Day 5: Model Diversity & Linear Calibration
- Train regularized Ridge logistic regression on standard-scaled numeric features + one-hot categoricals + OOF target features.
- Compute Spearman rank correlation matrix between OOF predictions of all 4 models:
  - $\text{corr}(\text{LGBM}, \text{CatBoost}) \approx 0.96$
  - $\text{corr}(\text{LGBM}, \text{XGBoost}) \approx 0.97$
  - $\text{corr}(\text{GBDT}, \text{Ridge}) \approx 0.88$ (High diversity!)

### Day 6: Multi-Seed Variance Suppression
- Execute 3 distinct random seeds (`42`, `2026`, `777`) for LightGBM, CatBoost, and XGBoost.
- Average fold predictions across seeds to produce stabilized OOF and test vectors.
- Expected gain: $+0.0003$ to $+0.0005$ OOF AUC.

### Day 7: Rank Fusion & Ensembling Optimization
- Apply rank transformation: `scipy.stats.rankdata(pred) / len(pred)`.
- Use Nelder-Mead optimization on OOF predictions to find optimal non-negative ensemble weights:
  $$\hat{w} = \arg\max_w \text{ROC\_AUC}\left(y_{\text{train}}, \sum w_m R_{m, \text{OOF}}\right)$$
- Compare Rank Blending vs Linear Probability Blending vs Power Averaging ($P^\gamma$).
- Verify that OOF score exceeds 0.9554.

### Day 8: Final Submission Selection & Pipeline Freeze
- Generate final test predictions using the two most robust, non-overfitted strategies:
  - **Submission 1 (Optimized Leaderboard Maximizer)**: Nelder-Mead Rank Blend across multi-seed LightGBM + CatBoost + XGBoost + Ridge.
  - **Submission 2 (Conservative Generalizer)**: Simple equal-weight rank average of multi-seed LightGBM + CatBoost.
- Perform sanity checks: non-null, exactly 299,984 rows, contiguous IDs from 699,960 to 999,943, predictions in $[0.0, 1.0]$.
- Freeze code, save all weights, seeds, and artifacts.

---

## 4. Feature Importance Investigation Plan

To maintain complete model interpretability and guide feature selection:
1. **LightGBM Split & Gain Importance**:
   - Track both `importance_type='gain'` (reduction in loss contributed by each feature) and `importance_type='split'` (frequency of tree splits).
2. **Permutation Feature Importance**:
   - On each validation fold, permute feature values and measure the degradation in validation ROC AUC. Features whose permutation yields zero or negative drop will be purged.
3. **SHAP (SHapley Additive exPlanations)**:
   - Run TreeSHAP on a stratified 10,000-sample validation slice to inspect directionality, interaction effects, and risk thresholds.

---

## 5. Potential Leakage Risks & Strict Mitigations

| Leakage Risk | Mechanism of Contamination | Mandatory Prevention Protocol |
| :--- | :--- | :--- |
| **Global Target Encoding** | Computing category fraud rates across the entire dataset | **Strict In-Fold OOF**: Target statistics computed strictly inside the 4 training folds. Validation fold mapped with smoothing prior. |
| **Global Preprocessing / Scaling** | Computing mean/std or min/max on combined train+test | Fit scalers and transformers **exclusively on train fold**, then apply `transform()` to validation and test. |
| **ID Column Contamination** | Using `id` as a predictor feature | Drop `id` from feature matrices immediately upon reading data. `id` used solely for row alignment. |
| **Temporal / Future Lookahead** | Time-based features accessing future transaction deltas | Verified: `hours_since_last_txn` and `txn_count_last_24h` look backward in time. No future lookahead present. |
| **Test Label Infiltration via External Data** | Appending external datasets containing test labels | Zero external data permitted without cross-checking ID and row uniqueness against test.csv. |
