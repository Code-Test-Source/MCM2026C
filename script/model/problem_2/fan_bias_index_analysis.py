from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler


INPUT_SCORE_PATH = (
    "../../../data/processed/problem2/2026_MCM_Problem_C_Attractiveness_Scores_All_Seasons_Both_Methods.csv"
)

OUTPUT_CONFLICT_PATH = (
    "../../../data/processed/problem2/2026_MCM_Problem_C_FBI_conflict_samples.csv"
)
OUTPUT_FBI_SUMMARY_PATH = (
    "../../../data/processed/problem2/2026_MCM_Problem_C_FBI_summary.csv"
)
OUTPUT_FBI_WEEKLY_PATH = (
    "../../../data/processed/problem2/2026_MCM_Problem_C_FBI_weekly.csv"
)
OUTPUT_FBI_WEEKLY_DIFF_PATH = (
    "../../../data/processed/problem2/2026_MCM_Problem_C_FBI_weekly_difference.csv"
)

# -------- 可调参数 --------
CONFLICT_THRESHOLD_RANK = 0
CONFLICT_THRESHOLD_PERCENT = 0

# 手动设置权重（FBI1, FBI2, FBI3），若为 None 则使用熵权法
MANUAL_FBI_WEIGHTS: Optional[Tuple[float, float, float]] = None

# 最小回归样本数（避免样本过少导致回归不稳定）
MIN_REG_SAMPLES = 3


@dataclass
class FBISummary:
    method: str
    fbi1: float
    fbi2: float
    fbi3: float
    composite: float


def _safe_rank(series: pd.Series, ascending: bool) -> pd.Series:
    ranked = series.rank(method="min", ascending=ascending)
    ranked = ranked.replace([np.inf, -np.inf], np.nan)
    ranked = ranked.where(ranked > 0)
    return ranked


def _final_percent_from_rank(rank: pd.Series) -> pd.Series:
    if rank.empty:
        return rank.astype(float)
    max_rank = float(rank.max())
    if max_rank <= 1:
        return pd.Series(1.0, index=rank.index)
    return 1.0 - (rank - 1.0) / (max_rank - 1.0)


def _zscore(values: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(values.values)
    return pd.DataFrame(scaled, index=values.index, columns=values.columns)


def _entropy_weights(values: pd.DataFrame) -> Tuple[float, float, float]:
    # values shape: (methods, indicators)
    if values.shape[0] < 2:
        return (1.0 / values.shape[1],) * values.shape[1]

    normalized = values.copy()
    for col in normalized.columns:
        min_v = normalized[col].min()
        max_v = normalized[col].max()
        if math.isclose(max_v - min_v, 0.0):
            normalized[col] = 0.0
        else:
            normalized[col] = (normalized[col] - min_v) / (max_v - min_v)

    eps = 1e-12
    proportions = normalized + eps
    proportions = proportions.div(proportions.sum(axis=0), axis=1)

    k = 1.0 / math.log(len(normalized))
    entropy = -k * (proportions * np.log(proportions)).sum(axis=0)
    divergence = 1.0 - entropy

    if math.isclose(float(divergence.sum()), 0.0):
        weights = np.array([1.0 / len(divergence)] * len(divergence))
    else:
        weights = divergence / divergence.sum()

    return tuple(float(x) for x in weights.values)


def _compute_fbi1(conflict_df: pd.DataFrame) -> float:
    a_group = conflict_df[conflict_df["conflict_type"] == "A"]
    b_group = conflict_df[conflict_df["conflict_type"] == "B"]

    if len(a_group) == 0 or len(b_group) == 0:
        return 0.0

    a_success = (a_group["final_rank"] < a_group["judge_rank"]).mean()
    b_success = (b_group["final_rank"] > b_group["judge_rank"]).mean()
    return float((a_success + b_success) / 2.0)


def _compute_fbi2_rank(conflict_df: pd.DataFrame) -> float:
    usable = conflict_df.dropna(subset=["final_rank", "judge_rank", "fan_rank"])
    if len(usable) < MIN_REG_SAMPLES:
        return 0.0

    X = usable[["judge_rank", "fan_rank"]]
    X_std = _zscore(X)
    y = usable["final_rank"].astype(float)

    X_std = sm.add_constant(X_std)
    model = sm.OLS(y, X_std).fit()

    beta = model.params.get("judge_rank", np.nan)
    gamma = model.params.get("fan_rank", np.nan)

    if pd.isna(beta) or math.isclose(float(beta), 0.0):
        return 0.0
    return float(gamma / beta)


def _compute_fbi2_percent(conflict_df: pd.DataFrame) -> float:
    usable = conflict_df.dropna(
        subset=["final_percent", "judge_percent", "fan_percent"]
    )
    if len(usable) < MIN_REG_SAMPLES:
        return 0.0

    X = usable[["judge_percent", "fan_percent"]]
    X_std = _zscore(X)
    y = usable["final_percent"].astype(float)

    X_std = sm.add_constant(X_std)
    model = sm.OLS(y, X_std).fit()

    beta = model.params.get("judge_percent", np.nan)
    gamma = model.params.get("fan_percent", np.nan)

    if pd.isna(beta) or math.isclose(float(beta), 0.0):
        return 0.0
    return float(gamma / beta)


def _compute_fbi3(conflict_df: pd.DataFrame) -> float:
    usable = conflict_df.dropna(subset=["final_rank", "judge_rank"])
    if len(usable) == 0:
        return 0.0
    delta = (usable["final_rank"] - usable["judge_rank"]).abs()
    return float(delta.mean())


def _compute_season_fbi(conflict_df: pd.DataFrame, method: str) -> pd.DataFrame:
    results: List[Dict[str, object]] = []

    for season, group in conflict_df.groupby("season", sort=True):
        if method == "rank":
            fbi1 = _compute_fbi1(group)
            fbi2 = _compute_fbi2_rank(group)
        else:
            fbi1 = _compute_fbi1(group)
            fbi2 = _compute_fbi2_percent(group)
        fbi3 = _compute_fbi3(group)

        results.append(
            {
                "season": int(season),
                "method": method,
                "fbi1": fbi1,
                "fbi2": fbi2,
                "fbi3": fbi3,
                "sample_count": len(group),
            }
        )

    return pd.DataFrame(results)


def _composite_fbi(
    fbi_values: pd.DataFrame, weights: Tuple[float, float, float]
) -> pd.Series:
    return (
        fbi_values["fbi1"] * weights[0]
        + fbi_values["fbi2"] * weights[1]
        + fbi_values["fbi3"] * weights[2]
    )


def _clean_percent(series: pd.Series) -> pd.Series:
    return series.clip(lower=0.0, upper=1.0)


def main() -> None:
    df = pd.read_csv(INPUT_SCORE_PATH, na_values=["N/A", "NA", ""])

    required_cols = [
        "season",
        "week",
        "celebrity_name",
        "week_total_judge",
        "judge_rank",
        "fan_rank",
        "rank_sum",
        "judge_percent",
        "fan_percent",
        "combined_percent",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["judge_percent"] = _clean_percent(df["judge_percent"])
    df["fan_percent"] = _clean_percent(df["fan_percent"])
    df["combined_percent"] = _clean_percent(df["combined_percent"])

    # 计算每周最终名次（1最好）
    df["final_rank_rank"] = (
        df.groupby(["season", "week"])["rank_sum"]
        .transform(lambda s: _safe_rank(s, ascending=True))
    )
    df["final_rank_percent"] = (
        df.groupby(["season", "week"])["combined_percent"]
        .transform(lambda s: _safe_rank(s, ascending=False))
    )
    df["final_percent"] = df.groupby(["season", "week"])["final_rank_percent"].transform(
        _final_percent_from_rank
    )

    # 计算冲突度与类型（用于冲突样本输出）
    rank_conflict = df.copy()
    rank_conflict["conflict_degree"] = (
        (rank_conflict["judge_rank"] - rank_conflict["fan_rank"]).abs()
        / rank_conflict[["judge_rank", "fan_rank"]].max(axis=1)
    )
    rank_conflict = rank_conflict[rank_conflict["conflict_degree"] >= CONFLICT_THRESHOLD_RANK]
    rank_conflict["conflict_type"] = np.where(
        rank_conflict["fan_rank"] < rank_conflict["judge_rank"], "A", "B"
    )
    rank_conflict["final_rank"] = rank_conflict["final_rank_rank"]
    rank_conflict["method"] = "rank"

    percent_conflict = df.copy()
    percent_conflict["conflict_degree"] = (
        percent_conflict["judge_percent"] - percent_conflict["fan_percent"]
    ).abs()
    percent_conflict = percent_conflict[
        percent_conflict["conflict_degree"] >= CONFLICT_THRESHOLD_PERCENT
    ]
    percent_conflict["conflict_type"] = np.where(
        percent_conflict["fan_percent"] > percent_conflict["judge_percent"], "A", "B"
    )
    percent_conflict["final_rank"] = percent_conflict["final_rank_percent"]
    percent_conflict["method"] = "percent"

    # 全量样本（用于周度输出，保证每周都有结果）
    rank_all = df.copy()
    rank_all["conflict_type"] = np.where(
        rank_all["fan_rank"] < rank_all["judge_rank"], "A", "B"
    )
    rank_all["final_rank"] = rank_all["final_rank_rank"]
    rank_all["method"] = "rank"

    percent_all = df.copy()
    percent_all["conflict_type"] = np.where(
        percent_all["fan_percent"] > percent_all["judge_percent"], "A", "B"
    )
    percent_all["final_rank"] = percent_all["final_rank_percent"]
    percent_all["method"] = "percent"

    conflict_df = pd.concat([rank_conflict, percent_conflict], ignore_index=True)

    conflict_df = conflict_df[
        [
            "season",
            "week",
            "celebrity_name",
            "ballroom_partner",
            "method",
            "conflict_degree",
            "conflict_type",
            "judge_rank",
            "fan_rank",
            "judge_percent",
            "fan_percent",
            "final_rank",
            "final_percent",
        ]
    ]
    conflict_df.to_csv(OUTPUT_CONFLICT_PATH, index=False)

    # 计算 FBI 指数（整体）
    rank_conflict_short = conflict_df[conflict_df["method"] == "rank"]
    percent_conflict_short = conflict_df[conflict_df["method"] == "percent"]

    fbi1_rank = _compute_fbi1(rank_conflict_short)
    fbi2_rank = _compute_fbi2_rank(rank_conflict_short)
    fbi3_rank = _compute_fbi3(rank_conflict_short)

    fbi1_percent = _compute_fbi1(percent_conflict_short)
    fbi2_percent = _compute_fbi2_percent(percent_conflict_short)
    fbi3_percent = _compute_fbi3(percent_conflict_short)

    fbi_table = pd.DataFrame(
        {
            "method": ["rank", "percent"],
            "fbi1": [fbi1_rank, fbi1_percent],
            "fbi2": [fbi2_rank, fbi2_percent],
            "fbi3": [fbi3_rank, fbi3_percent],
        }
    )

    # 赛季 FBI
    season_rank = _compute_season_fbi(rank_all, "rank")
    season_percent = _compute_season_fbi(percent_all, "percent")
    season_fbi = pd.concat([season_rank, season_percent], ignore_index=True)
    season_fbi["fbi1"] = season_fbi["fbi1"].fillna(0.0)
    season_fbi["fbi2"] = season_fbi["fbi2"].fillna(0.0)
    season_fbi["fbi3"] = season_fbi["fbi3"].fillna(0.0)

    if MANUAL_FBI_WEIGHTS is None:
        weights = _entropy_weights(season_fbi[["fbi1", "fbi2", "fbi3"]])
    else:
        weights = MANUAL_FBI_WEIGHTS

    fbi_table["composite_fbi"] = _composite_fbi(fbi_table, weights)
    season_fbi["composite_fbi"] = _composite_fbi(season_fbi, weights)
    season_fbi["composite_fbi"] = season_fbi["composite_fbi"].fillna(0.0)

    # 仅保留两种方法都存在的赛季
    pivot = season_fbi.pivot_table(
        index=["season"],
        columns="method",
        values="composite_fbi",
        aggfunc="mean",
    )

    has_rank = "rank" in pivot.columns
    has_percent = "percent" in pivot.columns

    season_fbi.to_csv(OUTPUT_FBI_WEEKLY_PATH, index=False)

    if has_rank and has_percent and not pivot.empty:
        weekly_diff = pivot.copy()
        weekly_diff["diff_rank_minus_percent"] = (
            weekly_diff["rank"] - weekly_diff["percent"]
        )
        weekly_diff = weekly_diff.reset_index()
    else:
        weekly_diff = pd.DataFrame(
            columns=["season", "rank", "percent", "diff_rank_minus_percent"]
        )
    weekly_diff.to_csv(OUTPUT_FBI_WEEKLY_DIFF_PATH, index=False)

    fbi_table["weight_fbi1"] = weights[0]
    fbi_table["weight_fbi2"] = weights[1]
    fbi_table["weight_fbi3"] = weights[2]
    fbi_table.to_csv(OUTPUT_FBI_SUMMARY_PATH, index=False)


if __name__ == "__main__":
    main()
