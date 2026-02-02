# 依赖安装命令：
# pip install pandas numpy

from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd

# ===== 输入/输出路径（相对路径，放在显眼位置） =====
INPUT_PATH = "../../../data/result/judge_score_calibrated_by_week.csv"
OUTPUT_PATH = "../../../data/result/dynamic_vote_weights_by_week.csv"

# ===== 评委/粉丝权重的合理范围 =====
# 每个 season 第 1 周 fan_weight 从 0.55 开始
FAN_WEIGHT_START = 0.55
JUDGE_WEIGHT_MIN = 1.0 - FAN_WEIGHT_START
JUDGE_WEIGHT_MAX = 0.75

# ===== 粉丝可靠性设置 =====
RELIABILITY_ALPHA = 1.5  # z 分差异的惩罚强度
RELIABILITY_ADJ_STRENGTH = 0.6  # 可靠性对权重的调整强度（0-1）

# ===== 指定使用的列 =====
JUDGE_PERCENT_COL = "judge_score_final_percent"
FAN_PERCENT_COL = "fan_votes_relative_percent"


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    df_pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df_pair) < 3:
        return 0.0
    xr = df_pair["x"].rank(method="average")
    yr = df_pair["y"].rank(method="average")
    if xr.std(ddof=0) == 0 or yr.std(ddof=0) == 0:
        return 0.0
    corr = np.corrcoef(xr, yr)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(np.clip(corr, -1.0, 1.0))


def _mean_abs_zdiff(x: pd.Series, y: pd.Series) -> float:
    df_pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if df_pair.empty:
        return 0.0
    x_mean = df_pair["x"].mean()
    y_mean = df_pair["y"].mean()
    x_std = df_pair["x"].std(ddof=0)
    y_std = df_pair["y"].std(ddof=0)
    if x_std == 0 or y_std == 0:
        return 0.0
    zx = (df_pair["x"] - x_mean) / x_std
    zy = (df_pair["y"] - y_mean) / y_std
    return float((zx - zy).abs().mean())


def _calc_base_judge_weight(week: int, season_start_week: int, season_total_weeks: int) -> float:
    span = season_total_weeks - season_start_week
    if span <= 0:
        return JUDGE_WEIGHT_MIN
    progress = (week - season_start_week) / span
    weight = JUDGE_WEIGHT_MIN + (JUDGE_WEIGHT_MAX - JUDGE_WEIGHT_MIN) * progress
    return float(np.clip(weight, JUDGE_WEIGHT_MIN, JUDGE_WEIGHT_MAX))


def _calc_reliability(week_df: pd.DataFrame) -> float:
    judge = pd.to_numeric(week_df[JUDGE_PERCENT_COL], errors="coerce")
    fan = pd.to_numeric(week_df[FAN_PERCENT_COL], errors="coerce")
    z_diff = _mean_abs_zdiff(judge, fan)
    reliability = np.exp(-RELIABILITY_ALPHA * z_diff)
    return float(np.clip(reliability, 0.0, 1.0))


def main() -> None:
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"输入文件不存在: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    required_cols = ["season", "week", "celebrity_name"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}")

    if JUDGE_PERCENT_COL not in df.columns or FAN_PERCENT_COL not in df.columns:
        missing_cols = [c for c in [JUDGE_PERCENT_COL, FAN_PERCENT_COL] if c not in df.columns]
        raise ValueError(f"缺少指定列: {missing_cols}")

    # 周度可靠性（Spearman + z 分差异）
    reliability_week = (
        df.groupby(["season", "week"], as_index=False)
        .apply(lambda g: pd.Series({"week_reliability": _calc_reliability(g)}))
    )

    # 累计可靠性（仅使用本 season 开始至今的数据）
    reliability_week = reliability_week.sort_values(["season", "week"]).reset_index(drop=True)
    reliability_week["cum_reliability"] = (
        reliability_week.groupby("season")["week_reliability"]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )

    # 取每个 season 的起始周与总周数
    season_start_week = df.groupby("season")["week"].min().astype(int).to_dict()
    if "season_total_weeks" in df.columns:
        season_weeks = (
            df.groupby("season")["season_total_weeks"]
            .max()
            .fillna(df.groupby("season")["week"].max())
            .astype(int)
            .to_dict()
        )
    else:
        season_weeks = df.groupby("season")["week"].max().astype(int).to_dict()

    # 计算权重
    def _calc_row(row: pd.Series) -> pd.Series:
        season = int(row["season"])
        week = int(row["week"])
        total_weeks = int(season_weeks.get(season, week))
        start_week = int(season_start_week.get(season, week))
        base_judge = _calc_base_judge_weight(week, start_week, total_weeks)
        base_fan = 1.0 - base_judge

        reliability = float(row["cum_reliability"])
        reliability = float(np.clip(reliability, 0.0, 1.0))

        # 可靠性越低，评委权重在 base_judge 与 JUDGE_WEIGHT_MAX 之间上调
        judge_weight = base_judge + (JUDGE_WEIGHT_MAX - base_judge) * (1.0 - reliability) * RELIABILITY_ADJ_STRENGTH
        judge_weight = float(np.clip(judge_weight, JUDGE_WEIGHT_MIN, JUDGE_WEIGHT_MAX))
        fan_weight = 1.0 - judge_weight
        return pd.Series({"judge_weight": judge_weight, "fan_weight": fan_weight})

    weights = reliability_week.apply(_calc_row, axis=1)
    weights_by_week = pd.concat([reliability_week[["season", "week"]], weights], axis=1)

    # 合并权重到原始数据，计算选手总得分
    df_scores = df[["season", "week", "celebrity_name", JUDGE_PERCENT_COL, FAN_PERCENT_COL]].copy()
    df_scores[JUDGE_PERCENT_COL] = pd.to_numeric(df_scores[JUDGE_PERCENT_COL], errors="coerce")
    df_scores[FAN_PERCENT_COL] = pd.to_numeric(df_scores[FAN_PERCENT_COL], errors="coerce")


    merged = df_scores.merge(weights_by_week, on=["season", "week"], how="left")
    merged["total_score"] = (
        merged[JUDGE_PERCENT_COL] * merged["judge_weight"]
        + merged[FAN_PERCENT_COL] * merged["fan_weight"]
    )

    # 输出结果（包含选手总得分与权重）
    output = merged.sort_values(["season", "week", "celebrity_name"]).reset_index(drop=True)
    output.to_csv(OUTPUT_PATH, index=False)

    print(output)


if __name__ == "__main__":
    main()
