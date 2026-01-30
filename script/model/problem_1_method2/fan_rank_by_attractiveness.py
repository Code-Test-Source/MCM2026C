from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


INPUT_PATH = (
    "../../../data/processed/2026_MCM_Problem_C_Data_popularity_features_with_attractiveness.csv"
)
OUTPUT_PATH = "../../../data/processed/2026_MCM_Problem_C_fan_rank_by_attractiveness.csv"

TARGET_SEASONS = set([1, 2, 28, 29, 30, 31, 32, 33, 34])
ELIM_PATTERN = re.compile(r"(eliminated|withdrew)\s+week\s+(\d+)", re.IGNORECASE)


def _extract_elimination_week(results: pd.Series) -> pd.Series:
    extracted = results.fillna("").str.extract(ELIM_PATTERN)
    return pd.to_numeric(extracted[1], errors="coerce")


def main() -> None:
    df = pd.read_csv(INPUT_PATH, na_values=["N/A", "NA", ""])

    required_cols = ["season", "week", "results", "composite_attractiveness"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[df["season"].isin(TARGET_SEASONS)].copy()
    df["elimination_week"] = _extract_elimination_week(df["results"])
    df["is_eliminated_this_week"] = (
        df["elimination_week"].notna() & (df["elimination_week"] == df["week"])
    )

    df["fan_rank"] = pd.NA

    for (season, week), group in df.groupby(["season", "week"], sort=True):
        if group.empty:
            continue

        n = int(len(group))
        eliminated_mask = group["is_eliminated_this_week"].to_numpy(dtype=bool)
        non_elim = group[~group["is_eliminated_this_week"]]

        if not non_elim.empty:
            non_elim_rank = (
                non_elim["composite_attractiveness"]
                .rank(method="min", ascending=False)
                .astype(int)
            )
            df.loc[non_elim.index, "fan_rank"] = non_elim_rank

        if eliminated_mask.any():
            df.loc[group.index[eliminated_mask], "fan_rank"] = n

    df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
