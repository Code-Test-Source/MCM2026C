from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

INPUT_SCORE_PATH = (
    "../../../data/processed/problem2/2026_MCM_Problem_C_Attractiveness_Scores_All_Seasons_Both_Methods.csv"
)

OUTPUT_CORR_SUMMARY_PATH = (
    "../../../data/processed/problem2/2026_MCM_Problem_C_FBI_corr_summary.csv"
)
OUTPUT_CORR_WEEKLY_PATH = (
    "../../../data/processed/problem2/2026_MCM_Problem_C_FBI_corr_weekly.csv"
)
OUTPUT_CORR_CONCLUSION_PATH = (
    "../../../data/processed/problem2/2026_MCM_Problem_C_FBI_corr_conclusions.md"
)


def _safe_corr(x: pd.Series, y: pd.Series) -> Tuple[float, float, int]:
    data = pd.concat([x, y], axis=1).dropna()
    n = len(data)
    if n < 3:
        return float("nan"), float("nan"), n
    pearson = float(stats.pearsonr(data.iloc[:, 0], data.iloc[:, 1]).statistic)
    spearman = float(stats.spearmanr(data.iloc[:, 0], data.iloc[:, 1]).correlation)
    return pearson, spearman, n


def _build_rank_method(df: pd.DataFrame) -> pd.DataFrame:
    rank_df = df.copy()
    rank_df["final_signal"] = -rank_df["rank_sum"].astype(float)
    rank_df["fan_signal"] = -rank_df["fan_rank"].astype(float)
    rank_df["judge_signal"] = -rank_df["judge_rank"].astype(float)
    rank_df["method"] = "rank"
    return rank_df


def _build_percent_method(df: pd.DataFrame) -> pd.DataFrame:
    percent_df = df.copy()
    percent_df["final_signal"] = percent_df["combined_percent"].astype(float)
    percent_df["fan_signal"] = percent_df["fan_percent"].astype(float)
    percent_df["judge_signal"] = percent_df["judge_percent"].astype(float)
    percent_df["method"] = "percent"
    return percent_df


def _summary_for_method(df: pd.DataFrame, method: str) -> Dict[str, object]:
    subset = df[df["method"] == method]
    fan_p, fan_s, fan_n = _safe_corr(subset["fan_signal"], subset["final_signal"])
    judge_p, judge_s, judge_n = _safe_corr(
        subset["judge_signal"], subset["final_signal"]
    )
    sample_n = int(min(fan_n, judge_n))
    return {
        "method": method,
        "sample_count": sample_n,
        "fan_final_pearson": fan_p,
        "fan_final_spearman": fan_s,
        "judge_final_pearson": judge_p,
        "judge_final_spearman": judge_s,
    }


def _weekly_for_method(df: pd.DataFrame, method: str) -> List[Dict[str, object]]:
    subset = df[df["method"] == method]
    rows: List[Dict[str, object]] = []
    for (season, week), group in subset.groupby(["season", "week"], sort=True):
        fan_p, fan_s, fan_n = _safe_corr(group["fan_signal"], group["final_signal"])
        judge_p, judge_s, judge_n = _safe_corr(
            group["judge_signal"], group["final_signal"]
        )
        rows.append(
            {
                "season": int(season),
                "week": int(week),
                "method": method,
                "sample_count": int(min(fan_n, judge_n)),
                "fan_final_pearson": fan_p,
                "fan_final_spearman": fan_s,
                "judge_final_pearson": judge_p,
                "judge_final_spearman": judge_s,
            }
        )
    return rows


def _compare_simple(summary_df: pd.DataFrame, col: str) -> str:
    rank_val = summary_df.loc[summary_df["method"] == "rank", col].iloc[0]
    percent_val = summary_df.loc[summary_df["method"] == "percent", col].iloc[0]
    if pd.isna(rank_val) or pd.isna(percent_val):
        return "不足以比较"
    if abs(rank_val) > abs(percent_val):
        return "排名法更强"
    if abs(rank_val) < abs(percent_val):
        return "百分比法更强"
    return "两者相当"


def main() -> None:
    df = pd.read_csv(INPUT_SCORE_PATH, na_values=["N/A", "NA", ""])

    required_cols = [
        "season",
        "week",
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

    rank_df = _build_rank_method(df)
    percent_df = _build_percent_method(df)
    combined = pd.concat([rank_df, percent_df], ignore_index=True)

    summary_rows = [
        _summary_for_method(combined, "rank"),
        _summary_for_method(combined, "percent"),
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_CORR_SUMMARY_PATH, index=False)

    weekly_rows = _weekly_for_method(combined, "rank") + _weekly_for_method(
        combined, "percent"
    )
    weekly_df = pd.DataFrame(weekly_rows)
    weekly_df.to_csv(OUTPUT_CORR_WEEKLY_PATH, index=False)

    conclusion_lines = [
        "# 简化相关系数比较结论",
        f"- 粉丝影响（Spearman）对比：{_compare_simple(summary_df, 'fan_final_spearman')}",
        f"- 粉丝影响（Pearson）对比：{_compare_simple(summary_df, 'fan_final_pearson')}",
        f"- 评委影响（Spearman）对比：{_compare_simple(summary_df, 'judge_final_spearman')}",
        f"- 评委影响（Pearson）对比：{_compare_simple(summary_df, 'judge_final_pearson')}",
    ]

    with open(OUTPUT_CORR_CONCLUSION_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(conclusion_lines))


if __name__ == "__main__":
    main()
