from __future__ import annotations

import re

import numpy as np
import pandas as pd

INPUT_PATH = (
    "../../../data/processed/2026_MCM_Problem_C_Data_popularity_features_with_attractiveness_xgboost.csv"
)
OUTPUT_PATH = (
    "../../../data/processed/2026_MCM_Problem_C_Data_fan_votes_with_certainty.csv"
)

ELIM_PATTERN = re.compile(r"(eliminated|withdrew)\s+week\s+(\d+)", re.IGNORECASE)


def _extract_elimination_week(results: pd.Series) -> pd.Series:
    extracted = results.fillna("").str.extract(ELIM_PATTERN)
    return pd.to_numeric(extracted[1], errors="coerce")


def _infer_missing_elimination_week(df: pd.DataFrame) -> pd.Series:
    last_active_week = (
        df[df["week_total_judge"].notna() & (df["week_total_judge"] > 0)]
        .groupby(["season", "celebrity_name"])["week"]
        .max()
    )

    results_lower = df["results"].fillna("").str.lower()
    withdrew_mask = results_lower.str.contains("withdrew", regex=False)
    missing_week_mask = df["elimination_week"].isna() & withdrew_mask

    inferred = df["elimination_week"].copy()
    if missing_week_mask.any():
        inferred.loc[missing_week_mask] = (
            df.loc[missing_week_mask, ["season", "celebrity_name"]]
            .merge(
                last_active_week.rename("last_active_week"),
                left_on=["season", "celebrity_name"],
                right_index=True,
                how="left",
            )["last_active_week"]
            .to_numpy()
        )

    final_mask = df["elimination_week"].isna() & results_lower.str.contains(
        r"final|winner|champion|1st|2nd|3rd|4th|5th|place",
        regex=True,
    )
    if final_mask.any():
        inferred.loc[final_mask] = (
            df.loc[final_mask, ["season", "celebrity_name"]]
            .merge(
                last_active_week.rename("last_active_week"),
                left_on=["season", "celebrity_name"],
                right_index=True,
                how="left",
            )["last_active_week"]
            .add(1)
            .to_numpy()
        )

    remaining_mask = inferred.isna()
    if remaining_mask.any():
        inferred.loc[remaining_mask] = (
            df.loc[remaining_mask, ["season", "celebrity_name"]]
            .merge(
                last_active_week.rename("last_active_week"),
                left_on=["season", "celebrity_name"],
                right_index=True,
                how="left",
            )["last_active_week"]
            .add(1)
            .to_numpy()
        )

    return inferred


def _fan_share_from_attractiveness(attractiveness: pd.Series) -> pd.Series:
    values = attractiveness.astype(float)
    min_v = float(values.min())
    max_v = float(values.max())
    if max_v - min_v == 0:
        return pd.Series(0.0, index=attractiveness.index)
    return (values - min_v) / (max_v - min_v)


def _certainty_from_distribution(
    attractiveness: pd.Series, fan_share: pd.Series
) -> pd.Series:
    values = attractiveness.astype(float).to_numpy()
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std == 0:
        z = np.zeros_like(values)
    else:
        z = (values - mean) / std
    base = 1.0 / (1.0 + np.exp(-np.abs(z)))

    share = fan_share.astype(float).to_numpy()
    share = share / max(share.sum(), 1e-12)
    if len(share) <= 1:
        concentration = 0.0
    else:
        entropy = -np.sum(share * np.log(share + 1e-12))
        concentration = 1.0 - float(entropy / np.log(len(share)))
    concentration = float(np.clip(concentration, 0.0, 1.0))

    certainty = base * (0.5 + 0.5 * concentration)
    certainty = np.clip(certainty, 0.0, 1.0)
    return pd.Series(certainty, index=attractiveness.index)


def main() -> None:
    df = pd.read_csv(INPUT_PATH, na_values=["N/A", "NA", ""])

    required_cols = [
        "season",
        "week",
        "celebrity_name",
        "ballroom_partner",
        "results",
        "composite_attractiveness",
        "week_total_judge",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["elimination_week"] = _extract_elimination_week(df["results"])
    df["elimination_week"] = _infer_missing_elimination_week(df)
    df["is_eliminated_this_week"] = (
        df["elimination_week"].notna() & (df["elimination_week"] == df["week"])
    )

    df["fan_votes_relative"] = 0.0
    df["fan_votes_rank"] = pd.NA
    df["certainty"] = 0.0

    for (season, week), group in df.groupby(["season", "week"], sort=True):
        if group.empty:
            continue

        active_mask = (
            (group["week_total_judge"].notna() & (group["week_total_judge"] > 0))
            | group["is_eliminated_this_week"]
        )
        active = group[active_mask].copy()
        if active.empty:
            continue

        attractiveness = active["composite_attractiveness"].astype(float)
        attractiveness = attractiveness.fillna(attractiveness.median())

        fan_share = _fan_share_from_attractiveness(attractiveness)
        certainty = _certainty_from_distribution(attractiveness, fan_share)
        fan_rank = fan_share.rank(method="min", ascending=False).astype(int)

        df.loc[active.index, "fan_votes_relative"] = fan_share
        df.loc[active.index, "fan_votes_rank"] = fan_rank
        df.loc[active.index, "certainty"] = certainty

    df.to_csv(OUTPUT_PATH, index=False)
    print("已输出 fan_votes_relative 与 certainty:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
