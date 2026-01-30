# DWTS Fan-Vote Estimation — Model Summary

## Purpose
- Estimate fan votes per contestant/week from judge scores.
- Validate elimination reproduction, quantify uncertainty, compare rank vs percent combination methods, and analyze impact of contestant/pro-dancer features.

## Data
- Input CSV: `data/raw/2026_MCM_Problem_C_Data.csv` (columns: celebrity_name, ballroom_partner, celebrity_industry, celebrity_age_during_season, season, results, placement, weekX_judgeY_score...)

## High-level Pipeline (code sketch)
```python
# load
df = pd.read_csv('data/raw/2026_MCM_Problem_C_Data.csv')

# Preprocess -> weekly judge totals + elimination labels
estimator = DWTSFanVoteEstimator(data_path)
estimator.preprocess_data()

# Per-week estimation
estimator.estimate_all_weeks()

# Uncertainty (LP + Monte Carlo)
estimator.calculate_uncertainty(season, week, judge_scores, eliminated_idx, method)

# Validate: elimination reproduction, correlations
validation = estimator.validate_estimates()

# Time-series residual modeling (per-contestant)
estimator.arima_residual_pipeline(df_long)
# or
estimator.lstm_residual_pipeline(df_long)

# Feature analysis and reports
estimator.analyze_industry_age_partner_effects()
estimator.visualize_results()
```

## Core Methods
- `estimate_percentage_method(judge_scores, eliminated_idx)` — constrained optimization (SLSQP) that finds fan-vote vector whose percentages combined with judge percentages place the known eliminated contestant lowest.
- `estimate_rank_method(judge_scores, eliminated_idx)` — rank-based heuristic producing integer ranks then scaled to vote-like magnitudes.
- `heuristic_percentage_method(...)` — fallback when optimization fails.
- `calculate_uncertainty(...)` — linear-program style bounds + Monte-Carlo perturbation to estimate uncertainty intervals and a normalized uncertainty score.
 - `estimate_fan_votes_percentage(judge_scores, eliminated_idx)` — added: constrained/penalized solver (SLSQP) to find a normalized fan-vote vector consistent with the observed elimination by combined percent.
 - `estimate_fan_votes_rank(judge_scores, eliminated_idx)` — added: rank-based fan-vote heuristic producing normalized fan-vote estimates consistent with the observed elimination by ranks.
 - `monte_carlo_uncertainty(...)` — added: Monte Carlo perturbation of judge scores to estimate standard deviations/uncertainty of per-week fan-vote estimates.
- `validate_estimates()` — computes elimination reproduction accuracy for both combination-by-percent and by-rank, Spearman correlation between judge scores and estimated fan votes, and mean uncertainty.
- Time-series residual models: ARIMA or LSTM per-contestant on estimated fan_votes; residuals modeled with RandomForest (optionally XGBoost / LightGBM) for nonlinear effects.

## Outputs / Deliverables
- Per-week estimated `fan_votes` and uncertainty per `(season, week, contestant)` saved in `estimator.estimates` and `estimator.uncertainty`.
 - Per-week estimated `fan_votes` and Monte Carlo uncertainty per `(season, week, contestant)` are saved to `data/processed/model2_estimates.csv` when the CSV is available.
- Validation summary: elimination accuracy, mean uncertainty, method-switch impact rates.
- Industry / age / partner impact analyses and visualizations (plots saved by `visualize_results`).

## Mermaid Architecture Diagram
```mermaid
flowchart LR
  A[Raw CSV: judge scores & contestant info]
  A --> B[Preprocessing]
  B --> C[Weekly judge totals & elimination labels]
  C --> D{Choose method per season}
  D -->|percentage| E[Constrained optimization]
  D -->|rank| F[Rank-based heuristic]
  E --> G[Estimated fan_votes]
  F --> G
  G --> H[Uncertainty quantification (LP + MC)]
  G --> I[Long-form dataset with historical features]
  I --> J[Time-series models (ARIMA or LSTM)]
  J --> K[Residuals]
  K --> L[Tree learners (RF / XGBoost / LightGBM)]
  L --> M[Residual-corrected predictions]
  H --> N[Validation: elimination reproduction & correlations]
  M --> N
  N --> O[Reports: plots, method comparisons, policy recommendations]
```

## Where this lives in the repo
- Implementation: `script/test/model1.py` (reads `data/raw/2026_MCM_Problem_C_Data.csv`, performs estimation, uncertainty, validation, time-series + residual pipelines, and visualizations).
- Complementary framework: `script/test/model2.py` (comprehensive demo; main() currently runs a synthetic-data demo but framework classes can accept the real CSV).
 - Complementary framework: `script/test/model2.py` (now loads `data/raw/2026_MCM_Problem_C_Data.csv` when present, runs the full hybrid framework, computes per-week percentage- and rank-based fan-vote estimates with Monte Carlo uncertainty, and writes `data/processed/model2_estimates.csv`).
 - Complementary framework: `script/test/model2.py` (now loads `data/raw/2026_MCM_Problem_C_Data.csv` when present, runs the full hybrid framework, computes per-week percentage- and rank-based fan-vote estimates with Monte Carlo uncertainty, writes `data/processed/model2_estimates.csv`, and—when available—invokes `script/test/model1.py`'s `DWTSFanVoteEstimator` to produce its native estimates and visualizations and saves them to `data/processed/model1_estimates.csv`).



