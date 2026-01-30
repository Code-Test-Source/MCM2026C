from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = "../../../data/processed/2026_MCM_Problem_C_Data_popularity_features.csv"
PCA_INPUT_PATH = "../../../data/processed/2026_MCM_Problem_C_Data_popularity_pca.csv"
OUTPUT_PATH = (
    "../../../data/processed/2026_MCM_Problem_C_Data_popularity_features_with_attractiveness.csv"
)
WEIGHTS_PATH = "../../../data/processed/attractiveness_weights.csv"

ID_COLS = [
    "season",
    "week",
    "celebrity_name",
    "ballroom_partner",
]

TARGET_COL = "placement"
RANK_BASED_SEASONS = set([1, 2, 28, 29, 30, 31, 32, 33, 34])
JUDGE_METRICS_RANK = {
    "weekly_judge_rank_pct": -1.0,
    "week_total_judge": 1.0,
}
JUDGE_METRICS_PERCENT = {
    "judge_share": 1.0,
    "week_total_judge": 1.0,
}
TARGET_ALPHA = 0.5
EXCLUDE_COLS = [
    "results",
]

CANDIDATE_COLS = [
    "judge_pct",
    "rank_pct",
    "popularity_deviation",
    "survival_advantage",
    "relative_cumulative_weeks",
    "weekly_judge_rank_pct",
]

RANDOM_SEED = 42


def _spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    a_rank = pd.Series(a).rank().to_numpy(dtype=float)
    b_rank = pd.Series(b).rank().to_numpy(dtype=float)
    if np.std(a_rank) == 0 or np.std(b_rank) == 0:
        return 0.0
    return float(np.corrcoef(a_rank, b_rank)[0, 1])


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    means = df.mean(axis=0)
    stds = df.std(axis=0).replace(0, 1)
    return (df - means) / stds


def _normalize_weights(w: np.ndarray) -> np.ndarray:
    denom = np.sum(np.abs(w))
    if denom == 0:
        return w
    return w / denom


def _objective(weights: np.ndarray, x: np.ndarray, target: np.ndarray) -> float:
    score = x @ weights
    return _spearman_corr(score, target)


def _build_composite_target(df: pd.DataFrame) -> np.ndarray:
    placement = df[TARGET_COL].to_numpy(dtype=float)
    placement = np.nan_to_num(placement, nan=np.nanmedian(placement))
    placement_z = (placement - np.mean(placement)) / (np.std(placement) or 1)
    placement_score = -placement_z

    rank_cols = [c for c in JUDGE_METRICS_RANK if c in df.columns]
    percent_cols = [c for c in JUDGE_METRICS_PERCENT if c in df.columns]

    judge_score_rank = np.zeros(len(df), dtype=float)
    if rank_cols:
        judge_rank_df = df[rank_cols].copy()
        judge_rank_df = judge_rank_df.fillna(judge_rank_df.median(numeric_only=True))
        judge_rank_z = _standardize(judge_rank_df)
        rank_weights = np.array([JUDGE_METRICS_RANK[c] for c in rank_cols], dtype=float)
        judge_score_rank = judge_rank_z.to_numpy(dtype=float) @ rank_weights
        judge_score_rank = (
            judge_score_rank - np.mean(judge_score_rank)
        ) / (np.std(judge_score_rank) or 1)

    judge_score_percent = np.zeros(len(df), dtype=float)
    if percent_cols:
        judge_percent_df = df[percent_cols].copy()
        judge_percent_df = judge_percent_df.fillna(
            judge_percent_df.median(numeric_only=True)
        )
        judge_percent_z = _standardize(judge_percent_df)
        percent_weights = np.array(
            [JUDGE_METRICS_PERCENT[c] for c in percent_cols], dtype=float
        )
        judge_score_percent = judge_percent_z.to_numpy(dtype=float) @ percent_weights
        judge_score_percent = (
            judge_score_percent - np.mean(judge_score_percent)
        ) / (np.std(judge_score_percent) or 1)

    use_rank = df["season"].isin(RANK_BASED_SEASONS).to_numpy(dtype=bool)
    judge_score = np.where(use_rank, judge_score_rank, judge_score_percent)

    alpha = TARGET_ALPHA
    return alpha * placement_score + (1 - alpha) * judge_score


def simulated_annealing(
    x: np.ndarray,
    target: np.ndarray,
    n_iter: int = 5000,
    init_temp: float = 1.0,
    cooling: float = 0.995,
    step_scale: float = 0.15,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(RANDOM_SEED)
    w = _normalize_weights(rng.normal(0, 1, size=x.shape[1]))
    best_w = w.copy()
    best_score = _objective(w, x, target)

    score = best_score
    for i in range(n_iter):
        temp = init_temp * (cooling**i)
        proposal = _normalize_weights(w + rng.normal(0, step_scale, size=w.shape[0]))
        proposal_score = _objective(proposal, x, target)
        delta = proposal_score - score
        if delta >= 0 or rng.random() < np.exp(delta / max(temp, 1e-8)):
            w = proposal
            score = proposal_score
            if score > best_score:
                best_score = score
                best_w = w.copy()

    return best_w, best_score


def main() -> None:
    df = pd.read_csv(INPUT_PATH, na_values=["N/A", "NA", ""])
    pca_df = pd.read_csv(PCA_INPUT_PATH, na_values=["N/A", "NA", ""])

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    pca_feature_cols = [c for c in pca_df.columns if c.startswith("pca_")]
    if not pca_feature_cols:
        raise ValueError("No PCA components found in PCA input file.")

    feature_df = pca_df[pca_feature_cols].copy()
    feature_df = feature_df.fillna(feature_df.median(numeric_only=True))
    x = _standardize(feature_df).to_numpy(dtype=float)

    train_mask = df["season"].isin(RANK_BASED_SEASONS).to_numpy(dtype=bool)
    if not train_mask.any():
        train_mask = np.ones(len(df), dtype=bool)

    target = _build_composite_target(df[train_mask])
    x_train = x[train_mask]

    weights, best_score = simulated_annealing(x_train, target)
    raw_score = x @ weights

    min_v = float(np.min(raw_score))
    max_v = float(np.max(raw_score))
    if max_v - min_v == 0:
        attractiveness = np.zeros_like(raw_score)
    else:
        attractiveness = (raw_score - min_v) / (max_v - min_v)

    output_df = df.copy()
    output_df["composite_attractiveness"] = attractiveness

    output_df.to_csv(OUTPUT_PATH, index=False)

    weights_df = pd.DataFrame(
        {
            "feature": pca_feature_cols,
            "weight": weights,
        }
    ).sort_values("weight", key=lambda s: s.abs(), ascending=False)
    weights_df["best_spearman_corr"] = best_score
    weights_df.to_csv(WEIGHTS_PATH, index=False)

    print("优化完成，最佳Spearman相关系数:", best_score)
    print("权重结果已保存:", WEIGHTS_PATH)
    print(weights_df.head(10).to_string(index=False))
    print("带综合吸引力结果已保存:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
