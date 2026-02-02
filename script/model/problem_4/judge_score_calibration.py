# 依赖安装命令：
# pip install pandas numpy

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ===== 输入/输出路径（相对路径，放在显眼位置） =====
INPUT_PATH = (
    "../../../data/processed/pr4/"
    "2026_MCM_Problem_C_Attractiveness_Scores_All_Seasons_Both_Methods_Keep_Features.csv"
)
AGE_SUMMARY_PATH = "../../../data/result/age_group_judge_impact_summary.csv"
REGION_SUMMARY_PATH = "../../../data/result/region_judge_impact_summary.csv"
OUTPUT_PATH = "../../../data/result/judge_score_calibrated_by_week.csv"

# ===== 调整开关与幅度约束 =====
APPLY_ADJUSTMENT = True  # 是否执行校准（False 时系数全为1）
COEF_MIN = 0.9  # 调整系数下限
COEF_MAX = 1.1  # 调整系数上限
MAX_ABS_DELTA = 0.1  # 单因素最大调整幅度（绝对值）

# ===== 评委权重（按周）设置 =====
WEIGHT_ADJ_STRENGTH = 0.2  # 权重调整强度，越小越接近 1
WEIGHT_EPS = 1e-6

# ===== 评委打分候选列 =====
JUDGE_COL_CANDIDATES = [
    "week_total_judge",
    "week_total_judge_percent",
]

JUDGE_SCORE_COL_PATTERN = re.compile(r"week(\d+)_judge(\d+)_score(?:_(rank|percent))?$")

# ===== 与分析脚本保持一致的标签构造配置 =====
AGE_GROUP_COL = "age_group_rank"
COUNTRY_COL = "celebrity_homecountry/region_rank"
STATE_COL = "celebrity_homestate_rank"

AGE_MIN_FREQ_RATIO = 0.03
AGE_MERGE_RARE = False

REGION_MIN_FREQ_RATIO = 0.03
REGION_MERGE_RARE = True
US_NAMES = ["United States", "USA", "US", "U.S."]


def _pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"未找到评委打分列: {candidates}")


def _validate_columns(df: pd.DataFrame, required_cols: List[str], label: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label}缺少所需列: {missing}")


def _build_age_group_labels(
    df: pd.DataFrame,
    age_col: str,
    min_freq_ratio: float = 0.03,
    merge_rare: bool = True,
) -> pd.Series:
    series = df[age_col].fillna("Unknown").astype(str)
    if merge_rare:
        total = len(series)
        freq = series.value_counts(dropna=False)
        rare = freq[freq / total < min_freq_ratio].index.tolist()
        series = series.where(~series.isin(rare), other="others")
    return series


def _build_region_labels(
    df: pd.DataFrame,
    country_col: str,
    state_col: str,
    min_freq_ratio: float = 0.03,
    merge_rare: bool = True,
    us_names: Optional[List[str]] = None,
) -> pd.Series:
    if us_names is None:
        us_names = US_NAMES
    country = df[country_col].fillna("Unknown").astype(str)
    state = df[state_col].fillna("Unknown").astype(str)
    is_us = country.isin(us_names)
    combined = np.where(is_us, "United States - " + state, country)
    combined = pd.Series(combined, index=df.index).fillna("Unknown")
    if merge_rare:
        total = len(combined)
        freq = combined.value_counts(dropna=False)
        rare = freq[freq / total < min_freq_ratio].index.tolist()
        combined = combined.where(~combined.isin(rare), other="others")
    return combined


def _load_significant_map(
    summary_path: str,
    label_col: str,
) -> Dict[str, Dict[str, float]]:
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"影响分析文件不存在: {summary_path}")

    summary = pd.read_csv(summary_path)
    required = [label_col, "significant_and_unidirectional", "direction", "coef_ctrl", "mean_diff"]
    _validate_columns(summary, required, "影响分析结果")

    selected = summary[summary["significant_and_unidirectional"] == True].copy()
    if selected.empty:
        return {}

    # 优先使用控制回归系数，若缺失则退回均值差
    selected["impact_raw"] = selected["coef_ctrl"].abs()
    missing_mask = selected["impact_raw"].isna()
    selected.loc[missing_mask, "impact_raw"] = selected.loc[missing_mask, "mean_diff"].abs()

    # 归一化为 [0,1]
    max_impact = float(selected["impact_raw"].max()) if not selected["impact_raw"].isna().all() else 0.0
    if max_impact <= 0:
        selected["impact_norm"] = 0.0
    else:
        selected["impact_norm"] = selected["impact_raw"] / max_impact

    impact_map: Dict[str, Dict[str, float]] = {}
    for _, row in selected.iterrows():
        label = str(row[label_col])
        direction = str(row["direction"])
        impact_norm = float(row["impact_norm"])
        impact_map[label] = {"direction": direction, "impact_norm": impact_norm}
    return impact_map


def _direction_to_sign(direction: str) -> int:
    if direction == "正向":
        return 1
    if direction == "负向":
        return -1
    return 0


def _calc_factor(direction: str, impact_norm: float) -> float:
    sign = _direction_to_sign(direction)
    if sign == 0:
        return 1.0
    delta = min(MAX_ABS_DELTA, max(0.0, impact_norm * MAX_ABS_DELTA))
    return 1.0 - sign * delta


def _collect_week_judge_cols(df: pd.DataFrame) -> Dict[int, Dict[int, str]]:
    mapping: Dict[int, Dict[int, Tuple[int, str]]] = {}
    for col in df.columns:
        match = JUDGE_SCORE_COL_PATTERN.fullmatch(col)
        if not match:
            continue
        week = int(match.group(1))
        judge = int(match.group(2))
        suffix = match.group(3)
        if suffix is None:
            priority = 0
        elif suffix == "rank":
            priority = 1
        else:
            priority = 2

        current = mapping.setdefault(week, {}).get(judge)
        if current is None or priority < current[0]:
            mapping[week][judge] = (priority, col)

    cleaned: Dict[int, Dict[int, str]] = {}
    for week, judges in mapping.items():
        cleaned[week] = {judge: col for judge, (priority, col) in judges.items()}
    return cleaned


def _compute_weekly_judge_weights(
    group: pd.DataFrame,
    judge_cols: Dict[int, str],
) -> Dict[int, float]:
    cols = [judge_cols[j] for j in sorted(judge_cols)]
    scores = group[cols].apply(pd.to_numeric, errors="coerce")
    row_means = scores.mean(axis=1, skipna=True)
    abs_dev = scores.sub(row_means, axis=0).abs()
    dev_by_col = abs_dev.mean(axis=0, skipna=True)
    mean_dev = float(dev_by_col.mean(skipna=True)) if not dev_by_col.empty else 0.0

    weights: Dict[int, float] = {}
    if mean_dev <= 0:
        for judge in judge_cols:
            weights[judge] = 1.0
        return weights

    dev_by_col = dev_by_col.fillna(mean_dev)
    for judge in judge_cols:
        col = judge_cols[judge]
        dev = float(dev_by_col[col])
        weight = 1.0 + WEIGHT_ADJ_STRENGTH * (mean_dev - dev) / (mean_dev + WEIGHT_EPS)
        weights[judge] = weight
    total_weight = sum(weights.values())
    if total_weight > 0:
        scale = len(weights) / total_weight
        for judge in weights:
            weights[judge] = weights[judge] * scale
    return weights


def main() -> None:
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"输入文件不存在: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, na_values=["N/A", "NA", ""])

    judge_col = _pick_first_existing(df, JUDGE_COL_CANDIDATES)
    _validate_columns(df, ["season", "week", judge_col], "主数据")

    week_judge_cols = _collect_week_judge_cols(df)

    if AGE_GROUP_COL in df.columns:
        df["age_group_label"] = _build_age_group_labels(
            df,
            age_col=AGE_GROUP_COL,
            min_freq_ratio=AGE_MIN_FREQ_RATIO,
            merge_rare=AGE_MERGE_RARE,
        )
    else:
        df["age_group_label"] = "Unknown"

    if COUNTRY_COL in df.columns and STATE_COL in df.columns:
        df["region_label"] = _build_region_labels(
            df,
            country_col=COUNTRY_COL,
            state_col=STATE_COL,
            min_freq_ratio=REGION_MIN_FREQ_RATIO,
            merge_rare=REGION_MERGE_RARE,
        )
    else:
        df["region_label"] = "Unknown"

    age_impact_map = _load_significant_map(AGE_SUMMARY_PATH, "age_group")
    region_impact_map = _load_significant_map(REGION_SUMMARY_PATH, "region")

    def _row_factor(row: pd.Series) -> Tuple[float, str]:
        if not APPLY_ADJUSTMENT:
            return 1.0, "关闭调整"

        age_label = str(row["age_group_label"])
        region_label = str(row["region_label"])

        age_info = age_impact_map.get(age_label)
        region_info = region_impact_map.get(region_label)

        age_factor = 1.0
        region_factor = 1.0
        notes: List[str] = []

        if age_info:
            age_factor = _calc_factor(age_info["direction"], age_info["impact_norm"])
            notes.append(f"age:{age_label}")
        if region_info:
            region_factor = _calc_factor(region_info["direction"], region_info["impact_norm"])
            notes.append(f"region:{region_label}")

        combined = age_factor * region_factor
        combined = min(COEF_MAX, max(COEF_MIN, combined))
        note = "+".join(notes) if notes else "无显著单向影响"
        return combined, note

    factors: List[float] = []
    notes: List[str] = []
    for _, row in df.iterrows():
        factor, note = _row_factor(row)
        factors.append(factor)
        notes.append(note)

    df["judge_adjustment_factor"] = factors
    df["judge_adjustment_note"] = notes

    judge_values = pd.to_numeric(df[judge_col], errors="coerce")
    valid_mask = judge_values.notna() & (judge_values > 0)

    adjusted = judge_values.copy()
    adjusted.loc[valid_mask] = judge_values.loc[valid_mask] * df.loc[valid_mask, "judge_adjustment_factor"]
    df["judge_score_original"] = judge_values
    df["judge_score_adjusted"] = adjusted

    # ===== 按周评委权重与加权总分 =====
    weight_columns: Dict[int, str] = {}
    if week_judge_cols:
        for judge_id in sorted({j for cols in week_judge_cols.values() for j in cols}):
            col_name = f"week_judge{judge_id}_weight"
            weight_columns[judge_id] = col_name
            df[col_name] = np.nan

    df["week_total_judge_weighted"] = np.nan
    df["judge_score_weighted_adjusted"] = np.nan

    if week_judge_cols:
        for (season, week), group in df.groupby(["season", "week"], sort=True):
            week_num = int(week)
            judge_cols = week_judge_cols.get(week_num)
            if not judge_cols:
                continue

            weights = _compute_weekly_judge_weights(group, judge_cols)
            for judge_id, weight in weights.items():
                df.loc[group.index, weight_columns[judge_id]] = weight

            cols = [judge_cols[j] for j in sorted(judge_cols)]
            scores = group[cols].apply(pd.to_numeric, errors="coerce")
            weight_series = pd.Series(
                {judge_cols[j]: weights[j] for j in judge_cols}, index=cols
            )
            weighted_total = scores.mul(weight_series, axis=1).sum(axis=1, skipna=True)
            df.loc[group.index, "week_total_judge_weighted"] = weighted_total

        weighted_values = pd.to_numeric(df["week_total_judge_weighted"], errors="coerce")
        weighted_valid = weighted_values.notna() & (weighted_values > 0)
        weighted_adjusted = weighted_values.copy()
        weighted_adjusted.loc[weighted_valid] = (
            weighted_values.loc[weighted_valid]
            * df.loc[weighted_valid, "judge_adjustment_factor"]
        )
        df["judge_score_weighted_adjusted"] = weighted_adjusted

    # ===== 计算评委最终分数占比（percent法） =====
    if df["judge_score_weighted_adjusted"].notna().any():
        final_score = df["judge_score_weighted_adjusted"]
    else:
        final_score = df["judge_score_adjusted"]

    df["judge_score_final_adjusted"] = final_score
    valid_final = pd.to_numeric(final_score, errors="coerce")
    valid_final_mask = valid_final.notna() & (valid_final > 0)
    week_sum = (
        valid_final.where(valid_final_mask)
        .groupby([df["season"], df["week"]])
        .transform("sum")
    )
    df["judge_score_final_percent"] = valid_final / week_sum

    # ===== 计算粉丝 relative 的 percent =====
    if "fan_votes_relative" in df.columns:
        fan_values = pd.to_numeric(df["fan_votes_relative"], errors="coerce")
        fan_valid_mask = fan_values.notna() & (fan_values > 0)
        fan_week_sum = (
            fan_values.where(fan_valid_mask)
            .groupby([df["season"], df["week"]])
            .transform("sum")
        )
        df["fan_votes_relative_percent"] = fan_values / fan_week_sum

    # 统计信息
    applied_mask = valid_mask & (df["judge_adjustment_factor"] != 1.0)
    adjusted_count = int(applied_mask.sum())
    avg_abs_delta = float(
        (df.loc[valid_mask, "judge_adjustment_factor"] - 1.0).abs().mean()
    ) if valid_mask.any() else 0.0
    max_abs_delta = float(
        (df.loc[valid_mask, "judge_adjustment_factor"] - 1.0).abs().max()
    ) if valid_mask.any() else 0.0

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("=== 评委评分校准完成 ===")
    print(f"评委打分列: {judge_col}")
    print(f"是否调整: {APPLY_ADJUSTMENT}")
    print(f"输出文件: {OUTPUT_PATH}")
    print(f"调整样本数: {adjusted_count}")
    print(f"平均绝对调整幅度: {avg_abs_delta:.4f}")
    print(f"最大绝对调整幅度: {max_abs_delta:.4f}")


if __name__ == "__main__":
    main()
