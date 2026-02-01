# 依赖安装命令：
# pip install pandas numpy

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ===== 输入/输出路径（相对路径，放在显眼位置） =====
INPUT_PATH = "../../../data/processed/problem3/2026_MCM_Problem_C_Attractiveness_Scores_All_Seasons_Both_Methods_Keep_Features.csv"
OUTPUT_PATH = (
    "../../../data/processed/problem3/2026_MCM_Problem_C_Attractiveness_Scores_All_Seasons_Both_Methods_Keep_Features_processed.csv"
)

# ===== 配置区（需替换为实际列名） =====
PARTNER_FEATURE_COLS = [
    "partner_total_seasons_rank",  # 需替换为实际partner特征列名
    "partner_avg_placement_rank",  # 需替换为实际partner特征列名
    "partner_final_rate_rank",  # 需替换为实际partner特征列名
]

INDUSTRY_COL = "celebrity_industry_rank"  # 需替换为实际celebrity_industry列名
COUNTRY_COL = "celebrity_homecountry/region_rank"  # 需替换为实际country列名
STATE_COL = "celebrity_homestate_rank"  # 需替换为实际state列名
AGE_GROUP_COL = "age_group_rank"  # 需替换为实际age_group列名

AGE_GROUP_METHOD = "ordinal"  # "ordinal" 或 "onehot"
AGE_GROUP_ORDER: Optional[List[str]] = None  # 例如: ["18-24", "25-34", "35-44", "45-54", "55+"]

MIN_FREQ_RATIO = 0.05
MERGE_RARE_CATEGORIES = False
US_NAMES = ["United States", "USA", "US", "U.S."]

# 评委相关数据列（需替换为实际列名）
JUDGE_COLS: List[str] = [
    "week_total_judge_rank",  # 评委总分（原始总分）
]

# 粉丝相关数据列（需替换为实际列名）
FAN_COLS: List[str] = [
    "fan_votes_relative",  # 粉丝票相对预测结果
]

# 额外保留列（可选）
EXTRA_KEEP_COLS: List[str] = []

# ===== 输出列命名规范 =====
# 以下列为“真正的排名列”，保留 _rank 后缀；其余以 _rank 结尾的列将去掉后缀
RANK_SUFFIX_KEEP_COLS: List[str] = [
    "judge_rank",
    "fan_rank",
    "rank_sum",
    "predicted_eliminated_rank",
]


def _validate_columns(df: pd.DataFrame, required_cols: List[str], label: str) -> None:
    """检查所需列是否存在。"""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label}缺少所需列: {missing}")


def one_hot_with_others(
    df: pd.DataFrame,
    col: str,
    min_freq_ratio: float = 0.05,
    others_label: str = "others",
    merge_rare: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """将低频类别合并为others后进行独热编码。"""
    _validate_columns(df, [col], "独热编码")

    series = df[col].fillna("Unknown").astype(str)
    rare: List[str] = []
    if merge_rare:
        total = len(series)
        freq = series.value_counts(dropna=False)
        rare = freq[freq / total < min_freq_ratio].index.tolist()
        series = series.where(~series.isin(rare), other=others_label)

    dummies = pd.get_dummies(series, prefix=col)

    df = df.drop(columns=[col])
    df = pd.concat([df, dummies], axis=1)
    return df, rare


def hierarchical_country_state_encoding(
    df: pd.DataFrame,
    country_col: str,
    state_col: str,
    min_freq_ratio: float = 0.05,
    us_names: Optional[List[str]] = None,
    others_label: str = "others",
    merge_rare: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """国家+美国州分层独热编码。"""
    _validate_columns(df, [country_col, state_col], "国家/州")

    if us_names is None:
        us_names = US_NAMES

    country = df[country_col].fillna("Unknown").astype(str)
    state = df[state_col].fillna("Unknown").astype(str)
    is_us = country.isin(us_names)

    rare_countries: List[str] = []
    country_merged = country
    if merge_rare:
        total = len(country)
        freq = country.value_counts(dropna=False)
        rare_countries = freq[freq / total < min_freq_ratio].index.tolist()
        country_merged = country.where(
            ~country.isin(rare_countries), other=others_label
        )

    combined = np.where(is_us, "United States - " + state, country_merged)
    combined = pd.Series(combined, index=df.index).fillna(others_label)

    dummies = pd.get_dummies(combined, prefix="country_state")
    df = df.drop(columns=[country_col, state_col])
    df = pd.concat([df, dummies], axis=1)
    return df, rare_countries


def encode_age_group(
    df: pd.DataFrame,
    col: str,
    method: str = "ordinal",
    order: Optional[List[str]] = None,
) -> pd.DataFrame:
    """对age_group进行有序编码或独热编码，默认有序编码。"""
    _validate_columns(df, [col], "age_group")

    series = df[col].fillna("Unknown").astype(str)

    if method == "onehot":
        dummies = pd.get_dummies(series, prefix=col)
        df = df.drop(columns=[col])
        df = pd.concat([df, dummies], axis=1)
        return df

    if order is None:
        order = sorted(series.unique())
    order_map = {k: i for i, k in enumerate(order)}
    df[col + "_ordinal"] = series.map(order_map).fillna(-1).astype(int)
    df = df.drop(columns=[col])
    return df


def filter_columns_whitelist(df: pd.DataFrame, keep_cols: List[str]) -> pd.DataFrame:
    """白名单保留列，避免误删。"""
    existing = [c for c in keep_cols if c in df.columns]
    if not existing:
        raise ValueError("白名单列在数据中不存在，请检查配置。")
    return df[existing].copy()


def main() -> None:
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"输入文件不存在: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, na_values=["N/A", "NA", ""])

    # 1) celebrity_industry 独热
    df, rare_industries = one_hot_with_others(
        df,
        INDUSTRY_COL,
        min_freq_ratio=MIN_FREQ_RATIO,
        merge_rare=MERGE_RARE_CATEGORIES,
    )
    if MERGE_RARE_CATEGORIES:
        print(f"冷门industry合并为others: {rare_industries}")

    # 2) 国家/州 分层独热
    df, rare_countries = hierarchical_country_state_encoding(
        df,
        country_col=COUNTRY_COL,
        state_col=STATE_COL,
        min_freq_ratio=MIN_FREQ_RATIO,
        us_names=US_NAMES,
        merge_rare=MERGE_RARE_CATEGORIES,
    )
    if MERGE_RARE_CATEGORIES:
        print(f"冷门国家合并为others: {rare_countries}")

    # 3) age_group 编码
    df = encode_age_group(
        df,
        AGE_GROUP_COL,
        method=AGE_GROUP_METHOD,
        order=AGE_GROUP_ORDER,
    )


    # ===== placement顺序编码综合排名 =====
    if "placement" in df.columns:
        # placement越小排名越高，按每个season分组编码
        df["placement_ordered_rank"] = df.groupby("season")['placement'].rank(method="min", ascending=True)
    elif "placement_rank" in df.columns:
        # 若只有placement_rank，直接用placement_rank顺序编码
        df["placement_ordered_rank"] = df.groupby("season")['placement_rank'].rank(method="min", ascending=True)
    else:
        print("未找到placement或placement_rank列，无法生成placement_ordered_rank")

    # ===== 白名单保留列 =====
    generated_cols = [c for c in df.columns if c.startswith(INDUSTRY_COL + "_")]
    generated_cols += [c for c in df.columns if c.startswith("country_state_")]
    generated_cols += [c for c in PARTNER_FEATURE_COLS if c in df.columns]
    if AGE_GROUP_METHOD == "ordinal":
        generated_cols += [AGE_GROUP_COL + "_ordinal"]
    else:
        generated_cols += [c for c in df.columns if c.startswith(AGE_GROUP_COL + "_")]
    generated_cols += ["placement_ordered_rank"]

    # # 保留season特征
    # if "season" in df.columns:
    #     generated_cols += ["season"]

    keep_cols = generated_cols + JUDGE_COLS + FAN_COLS + EXTRA_KEEP_COLS
    df_final = filter_columns_whitelist(df, keep_cols)

    # ===== 列名规范化：非排名列去掉 _rank 后缀 =====
    rename_map = {}
    for col in df_final.columns:
        if col.endswith("_rank") and col not in RANK_SUFFIX_KEEP_COLS and col != "placement_ordered_rank":
            rename_map[col] = col[: -len("_rank")]
    if rename_map:
        df_final = df_final.rename(columns=rename_map)

    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"已输出文件: {OUTPUT_PATH}")
    print(f"输出行列: {df_final.shape}")


if __name__ == "__main__":
    main()
