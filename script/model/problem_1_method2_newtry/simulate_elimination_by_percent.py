from __future__ import annotations

import re

import pandas as pd


INPUT_PATH = (
    "../../../data/processed/2026_MCM_Problem_C_Data_popularity_features_with_attractiveness_xgboost_percent.csv"
)
OUTPUT_PATH = (
    "../../../data/processed/2026_MCM_Problem_C_Attractiveness_Elimination_Sim_Percent.csv"
)
SUMMARY_PATH = (
    "../../../data/processed/2026_MCM_Problem_C_Attractiveness_Elimination_Accuracy_Percent.csv"
)

PERCENT_BASED_SEASONS = set(range(3, 28))
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

    return inferred


def _minmax(series: pd.Series) -> pd.Series:
    min_v = float(series.min())
    max_v = float(series.max())
    if max_v - min_v == 0:
        return pd.Series(0.0, index=series.index)
    return (series - min_v) / (max_v - min_v)


def _to_percent(series: pd.Series) -> pd.Series:
    total = float(series.sum())
    if total == 0:
        return pd.Series(1.0 / len(series), index=series.index)
    return series / total


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

    df = df[df["season"].isin(PERCENT_BASED_SEASONS)].copy()
    df["elimination_week"] = _extract_elimination_week(df["results"])
    df["elimination_week"] = _infer_missing_elimination_week(df)
    df["is_eliminated_this_week"] = (
        df["elimination_week"].notna() & (df["elimination_week"] == df["week"])
    )

    records = []
    accuracy_percent = []
    exact_match_percent = []

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

        actual_elim = active[active["is_eliminated_this_week"]]
        if actual_elim.empty:
            continue

        attractiveness = active["composite_attractiveness"].astype(float)
        attractiveness = attractiveness.fillna(attractiveness.median())
        fan_base = _minmax(attractiveness)

        judge_scores = active["week_total_judge"].astype(float)
        judge_scores = judge_scores.fillna(judge_scores.median())
        judge_percent = _to_percent(judge_scores)

        actual_names = set(actual_elim["celebrity_name"].tolist())
        actual_count = len(actual_names)

        fan_proxy = fan_base.clip(lower=0)
        fan_percent = _to_percent(fan_proxy)
        combined = judge_percent + fan_percent

        ranked = active.copy()
        ranked["judge_percent"] = judge_percent
        ranked["fan_percent"] = fan_percent
        ranked["combined_percent"] = combined
        ranked = ranked.sort_values(
            ["combined_percent", "fan_percent", "judge_percent"],
            ascending=[True, True, True],
        )

        predicted_names = ranked["celebrity_name"].head(actual_count).tolist()
        predicted_set = set(predicted_names)
        week_best_correct = len(predicted_set & actual_names) / max(actual_count, 1)
        week_best_exact = int(predicted_set == actual_names)

        records.append(
            {
                "season": season,
                "week": week,
                "predicted_eliminated_percent": ", ".join(predicted_names),
                "predicted_partner_percent": ", ".join(
                    ranked["ballroom_partner"].head(actual_count).tolist()
                ),
                "actual_eliminated": ", ".join(actual_names),
                "actual_count": len(actual_names),
                "correct_percent": week_best_correct,
                "exact_match_percent": week_best_exact,
            }
        )

        accuracy_percent.append(float(week_best_correct))
        exact_match_percent.append(week_best_exact)

    result_df = pd.DataFrame(records)
    result_df.to_csv(OUTPUT_PATH, index=False)

    accuracy_value = (
        float(sum(accuracy_percent) / len(accuracy_percent))
        if accuracy_percent
        else 0.0
    )
    summary_df = pd.DataFrame(
        {
            "total_weeks": [len(accuracy_percent)],
            "avg_recall_percent": [accuracy_value],
            "exact_match_weeks_percent": [sum(exact_match_percent)],
            "exact_match_rate_percent": [
                float(sum(exact_match_percent) / len(exact_match_percent))
                if exact_match_percent
                else 0.0
            ],
        }
    )
    summary_df.to_csv(SUMMARY_PATH, index=False)


if __name__ == "__main__":
    main()
