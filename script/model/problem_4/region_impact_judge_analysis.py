# 依赖安装命令：
# pip install pandas numpy scipy statsmodels

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.api as sm
except ImportError as exc:
    raise SystemExit("缺少statsmodels依赖，请先执行: pip install statsmodels") from exc

# ===== 输入/输出路径（相对路径，放在显眼位置） =====
INPUT_PATH = (
    "../../../data/processed/problem3/"
    "2026_MCM_Problem_C_Attractiveness_Scores_All_Seasons_Both_Methods_Keep_Features.csv"
)
OUTPUT_SUMMARY_PATH = "../../../data/result/region_judge_impact_summary.csv"

# ===== 配置区（需替换为实际列名） =====
COUNTRY_COL = "celebrity_homecountry/region_rank"  # 需替换为实际country列名
STATE_COL = "celebrity_homestate_rank"  # 需替换为实际state列名
JUDGE_COL_CANDIDATES = [
    "week_total_judge",
    "week_total_judge_rank",
    "week_total_judge_percent",
]

# ===== 地域降维设置 =====
MIN_FREQ_RATIO = 0.03
MERGE_RARE_CATEGORIES = True
US_NAMES = ["United States", "USA", "US", "U.S."]

# ===== 数值标准化设置 =====
STANDARDIZE_NUMERIC = True

# ===== 控制变量设置（排除其他因素影响） =====
CONTROL_INDUSTRY_COL = "celebrity_industry_rank"
CONTROL_AGE_GROUP_COL = "age_group_rank"
CONTROL_PARTNER_PREFIX = "partner_"
CONTROL_SEASON_COL = "season"
CONTROL_WEEK_COL = "week"

# ===== 显著性与单向性判定设置 =====
ALPHA = 0.05
MIN_GROUP_SIZE = 30
MIN_SEASON_GROUP_SIZE = 10
DIRECTION_CONSISTENCY_THRESHOLD = 0.8


def _pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"未找到评委打分列: {candidates}")


def _validate_columns(df: pd.DataFrame, required_cols: List[str], label: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label}缺少所需列: {missing}")


def _build_region_labels(
    df: pd.DataFrame,
    country_col: str,
    state_col: str,
    min_freq_ratio: float = 0.03,
    merge_rare: bool = True,
    us_names: Optional[List[str]] = None,
) -> Tuple[pd.Series, Dict[str, int]]:
    """
    与 feature_engineering.py/age_group_impact_judge_analysis.py 保持一致：美国细分到州，其他国家单独列出。
    """
    if us_names is None:
        us_names = US_NAMES

    country = df[country_col].fillna("Unknown").astype(str)
    state = df[state_col].fillna("Unknown").astype(str)
    is_us = country.isin(us_names)
    combined = np.where(is_us, "United States - " + state, country)
    combined = pd.Series(combined, index=df.index).fillna("Unknown")
    # 合并低频
    if merge_rare:
        total = len(combined)
        freq = combined.value_counts(dropna=False)
        rare = freq[freq / total < min_freq_ratio].index.tolist()
        combined = combined.where(~combined.isin(rare), other="others")
    freq_map = combined.value_counts(dropna=False).to_dict()
    return combined, freq_map


def _build_control_matrix(df: pd.DataFrame) -> pd.DataFrame:
    control_cols: List[str] = []
    for col in [CONTROL_INDUSTRY_COL, CONTROL_AGE_GROUP_COL, CONTROL_SEASON_COL, CONTROL_WEEK_COL]:
        if col in df.columns:
            control_cols.append(col)

    partner_cols = [c for c in df.columns if c.startswith(CONTROL_PARTNER_PREFIX)]
    control_cols += partner_cols

    if not control_cols:
        raise ValueError("未找到可用的控制变量列，请检查输入文件或配置区。")

    control_df = df[control_cols].copy()

    # 数值列处理
    numeric_cols = control_df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        control_df[col] = pd.to_numeric(control_df[col], errors="coerce")
        control_df[col] = control_df[col].fillna(control_df[col].median())

    if STANDARDIZE_NUMERIC and numeric_cols:
        for col in numeric_cols:
            values = control_df[col].astype(float)
            std = float(values.std(ddof=0))
            if std == 0 or np.isnan(std):
                control_df[col] = 0.0
            else:
                control_df[col] = (values - float(values.mean())) / std

    # 类别列独热（含合并低频）
    categorical_cols = [c for c in control_df.columns if c not in numeric_cols]
    for col in categorical_cols:
        series = control_df[col].fillna("Unknown").astype(str)
        if MERGE_RARE_CATEGORIES:
            total = len(series)
            freq = series.value_counts(dropna=False)
            rare = freq[freq / total < MIN_FREQ_RATIO].index.tolist()
            series = series.where(~series.isin(rare), other="others")
        dummies = pd.get_dummies(series, prefix=col, drop_first=True)
        control_df = control_df.drop(columns=[col])
        control_df = pd.concat([control_df, dummies], axis=1)

    return control_df


def _cohen_d(mean1: float, mean2: float, std1: float, std2: float, n1: int, n2: int) -> float:
    if n1 + n2 < 3:
        return 0.0
    pooled_var = ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / max(n1 + n2 - 2, 1)
    pooled_std = np.sqrt(pooled_var) if pooled_var > 0 else 0.0
    if pooled_std == 0:
        return 0.0
    return (mean1 - mean2) / pooled_std


def _point_biserial_r(mean1: float, mean2: float, std_all: float, n1: int, n2: int) -> float:
    n = n1 + n2
    if n == 0 or std_all == 0:
        return 0.0
    return (mean1 - mean2) / std_all * np.sqrt((n1 * n2) / (n**2))


def _direction_consistency(
    df: pd.DataFrame,
    region_col: str,
    judge_col: str,
    region: str,
) -> Tuple[str, float]:
    signs: List[int] = []
    for season, group in df.groupby("season"):
        in_region = group[group[region_col] == region][judge_col]
        out_region = group[group[region_col] != region][judge_col]
        if len(in_region) < MIN_SEASON_GROUP_SIZE or len(out_region) < MIN_SEASON_GROUP_SIZE:
            continue
        diff = in_region.mean() - out_region.mean()
        if diff > 0:
            signs.append(1)
        elif diff < 0:
            signs.append(-1)
    if not signs:
        return "不稳定", 0.0

    pos_ratio = signs.count(1) / len(signs)
    neg_ratio = signs.count(-1) / len(signs)
    best_ratio = max(pos_ratio, neg_ratio)
    if best_ratio >= DIRECTION_CONSISTENCY_THRESHOLD:
        return ("正向" if pos_ratio >= neg_ratio else "负向"), best_ratio
    return "不稳定", best_ratio


def _run_control_regression(
    y: pd.Series,
    X: pd.DataFrame,
    flag_name: str,
) -> Tuple[float, float, str]:
    X = X.copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    y = pd.to_numeric(y, errors="coerce").replace([np.inf, -np.inf], np.nan)

    data = pd.concat([y.rename("y"), X], axis=1).dropna()
    if data.empty:
        return np.nan, np.nan, "回归数据为空"

    y_clean = data["y"].astype(float)
    X_clean = sm.add_constant(data.drop(columns=["y"]), has_constant="add")
    X_clean = X_clean.astype(float)

    try:
        model = sm.OLS(y_clean, X_clean).fit()
        coef = model.params.get(flag_name, np.nan)
        p_value = model.pvalues.get(flag_name, np.nan)
        return coef, p_value, ""
    except Exception as exc:
        return np.nan, np.nan, f"回归失败: {type(exc).__name__}"


def main() -> None:
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"输入文件不存在: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, na_values=["N/A", "NA", ""])
    _validate_columns(df, [COUNTRY_COL, STATE_COL], "地域列")

    judge_col = _pick_first_existing(df, JUDGE_COL_CANDIDATES)
    _validate_columns(df, [judge_col], "评委打分列")

    # 仅保留评委打分有效的周度记录
    df = df[df[judge_col].notna()].copy()
    df = df[df[judge_col] > 0].copy()

    df["region_label"], freq_map = _build_region_labels(
        df,
        country_col=COUNTRY_COL,
        state_col=STATE_COL,
        min_freq_ratio=MIN_FREQ_RATIO,
        merge_rare=MERGE_RARE_CATEGORIES,
    )

    control_df = _build_control_matrix(df)

    rows: List[Dict[str, object]] = []
    regions = df["region_label"].value_counts().index.tolist()
    for region in regions:
        group = df[df["region_label"] == region][judge_col]
        others = df[df["region_label"] != region][judge_col]

        n1, n2 = len(group), len(others)
        mean1, mean2 = group.mean(), others.mean()
        std1, std2 = group.std(ddof=1), others.std(ddof=1)
        std_all = df[judge_col].std(ddof=1)
        mean_diff = mean1 - mean2

        if n1 < MIN_GROUP_SIZE or n2 < MIN_GROUP_SIZE:
            t_stat, p_value = np.nan, np.nan
        else:
            t_stat, p_value = stats.ttest_ind(group, others, equal_var=False)

        cohen_d = _cohen_d(mean1, mean2, std1, std2, n1, n2)
        r_pb = _point_biserial_r(mean1, mean2, std_all, n1, n2)

        # 控制其他因素后的回归检验（地域指示变量）
        region_flag = (df["region_label"] == region).astype(int)
        X = pd.concat([control_df, region_flag.rename("region_flag")], axis=1)
        X = sm.add_constant(X, has_constant="add")
        y = df[judge_col].astype(float)

        coef, p_value_ctrl, regression_note = _run_control_regression(
            y,
            X,
            "region_flag",
        )

        direction, consistency = _direction_consistency(df, "region_label", judge_col, region)
        significant = bool(p_value_ctrl < ALPHA) if pd.notna(p_value_ctrl) else False
        significant_and_unidirectional = bool(significant and direction in ["正向", "负向"] and consistency >= DIRECTION_CONSISTENCY_THRESHOLD)

        note = ""
        if n1 < MIN_GROUP_SIZE:
            note = "样本量偏小"
        elif direction == "不稳定":
            note = "方向不稳定"
        if regression_note:
            note = f"{note}; {regression_note}" if note else regression_note

        rows.append(
            {
                "region": region,
                "count": n1,
                "mean_judge": round(float(mean1), 4),
                "mean_others": round(float(mean2), 4),
                "mean_diff": round(float(mean_diff), 4),
                "t_stat": None if pd.isna(t_stat) else round(float(t_stat), 4),
                "p_value_raw": None if pd.isna(p_value) else round(float(p_value), 6),
                "coef_ctrl": None if pd.isna(coef) else round(float(coef), 6),
                "p_value_ctrl": None if pd.isna(p_value_ctrl) else round(float(p_value_ctrl), 6),
                "cohen_d": round(float(cohen_d), 4),
                "point_biserial_r": round(float(r_pb), 4),
                "direction": direction,
                "direction_consistency": round(float(consistency), 3),
                "significant": significant,
                "significant_and_unidirectional": significant_and_unidirectional,
                "note": note,
            }
        )

    result = pd.DataFrame(rows)
    result = result.sort_values(["significant_and_unidirectional", "significant", "count"], ascending=[False, False, False])

    os.makedirs(os.path.dirname(OUTPUT_SUMMARY_PATH), exist_ok=True)
    result.to_csv(OUTPUT_SUMMARY_PATH, index=False, encoding="utf-8-sig")

    print("=== 地域对评委打分影响：量化分析摘要 ===")
    print(f"评委打分列: {judge_col}")
    print(f"地域合并阈值: {MIN_FREQ_RATIO} | 合并低频: {MERGE_RARE_CATEGORIES}")
    print(f"结果已保存: {OUTPUT_SUMMARY_PATH}")

    # 面向大众的简要输出
    for _, row in result.iterrows():
        region = row["region"]
        if region == "others":
            continue
        verdict = "显著且单向" if row["significant_and_unidirectional"] else ("显著但不稳定" if row["significant"] else "不显著")
        direction = row["direction"]
        mean_diff = row["mean_diff"]
        p_value = row["p_value_ctrl"]
        print(
            f"地域={region} | 结论={verdict} | 方向={direction} | 平均差={mean_diff} | p值(控制后)={p_value}"
        )


if __name__ == "__main__":
    main()
