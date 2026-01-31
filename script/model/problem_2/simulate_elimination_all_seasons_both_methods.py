from __future__ import annotations

import re
from typing import Dict, List, Tuple

import pandas as pd


INPUT_RANK_PATH = (
    "../../../data/processed/2026_MCM_Problem_C_Data_popularity_features_with_attractiveness_xgboost.csv"
)
INPUT_PERCENT_PATH = (
    "../../../data/processed/2026_MCM_Problem_C_Data_popularity_features_with_attractiveness_xgboost_percent.csv"
)

OUTPUT_WEEKLY_PATH = (
    "../../../data/processed/problem2/2026_MCM_Problem_C_Attractiveness_Elimination_Sim_All_Seasons_Both_Methods.csv"
)
OUTPUT_SCORE_PATH = (
    "../../../data/processed/problem2/2026_MCM_Problem_C_Attractiveness_Scores_All_Seasons_Both_Methods.csv"
)
OUTPUT_SUMMARY_PATH = (
    "../../../data/processed/problem2/2026_MCM_Problem_C_Attractiveness_Elimination_Summary_All_Seasons_Both_Methods.csv"
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


def _validate_columns(df: pd.DataFrame, required_cols: List[str], label: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["elimination_week"] = _extract_elimination_week(df["results"])
    df["elimination_week"] = _infer_missing_elimination_week(df)
    df["is_eliminated_this_week"] = (
        df["elimination_week"].notna() & (df["elimination_week"] == df["week"])
    )
    return df


def _init_week_record(season: int, week: int) -> Dict[str, object]:
    return {
        "season": season,
        "week": week,
        "actual_eliminated": "",
        "actual_count": 0,
        "predicted_eliminated_rank": "",
        "predicted_partner_rank": "",
        "correct_rank": None,
        "exact_match_rank": None,
        "predicted_eliminated_percent": "",
        "predicted_partner_percent": "",
        "correct_percent": None,
        "exact_match_percent": None,
    }


def _compute_rank_method(
    df: pd.DataFrame,
    week_records: Dict[Tuple[int, int], Dict[str, object]],
) -> Tuple[pd.DataFrame, List[float], List[int]]:
    score_rows: List[Dict[str, object]] = []
    accuracy_rank: List[float] = []
    exact_match_rank: List[int] = []

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

        judge_rank = (
            active["week_total_judge"].astype(float).rank(method="min", ascending=False)
        ).astype(int)

        actual_names = set(actual_elim["celebrity_name"].tolist())
        actual_count = len(actual_names)

        fan_rank = fan_base.rank(method="min", ascending=False).astype(int)
        rank_sum = judge_rank + fan_rank

        ranked = active.copy()
        ranked["judge_rank"] = judge_rank
        ranked["fan_rank"] = fan_rank
        ranked["rank_sum"] = rank_sum
        ranked = ranked.sort_values(
            ["rank_sum", "fan_rank", "judge_rank"],
            ascending=[False, False, False],
        )

        predicted_rank_names = ranked["celebrity_name"].head(actual_count).tolist()
        predicted_rank_set = set(predicted_rank_names)
        week_best_correct = len(predicted_rank_set & actual_names) / max(actual_count, 1)
        week_best_exact = int(predicted_rank_set == actual_names)

        key = (int(season), int(week))
        week_record = week_records.setdefault(key, _init_week_record(int(season), int(week)))
        week_record["actual_eliminated"] = ", ".join(actual_names)
        week_record["actual_count"] = actual_count
        week_record["predicted_eliminated_rank"] = ", ".join(predicted_rank_names)
        week_record["predicted_partner_rank"] = ", ".join(
            ranked["ballroom_partner"].head(actual_count).tolist()
        )
        week_record["correct_rank"] = week_best_correct
        week_record["exact_match_rank"] = week_best_exact

        accuracy_rank.append(float(week_best_correct))
        exact_match_rank.append(week_best_exact)

        predicted_mask = ranked["celebrity_name"].isin(predicted_rank_names)
        ranked["predicted_eliminated_rank"] = predicted_mask

        for _, row in ranked.iterrows():
            score_rows.append(
                {
                    "season": int(season),
                    "week": int(week),
                    "celebrity_name": row["celebrity_name"],
                    "ballroom_partner": row["ballroom_partner"],
                    "results": row["results"],
                    "elimination_week": row["elimination_week"],
                    "week_total_judge": row["week_total_judge"],
                    "composite_attractiveness": row["composite_attractiveness"],
                    "judge_rank": row["judge_rank"],
                    "fan_rank": row["fan_rank"],
                    "rank_sum": row["rank_sum"],
                    "predicted_eliminated_rank": bool(row["predicted_eliminated_rank"]),
                    "is_eliminated_this_week": bool(row["is_eliminated_this_week"]),
                }
            )

    return pd.DataFrame(score_rows), accuracy_rank, exact_match_rank


def _compute_percent_method(
    df: pd.DataFrame,
    week_records: Dict[Tuple[int, int], Dict[str, object]],
) -> Tuple[pd.DataFrame, List[float], List[int]]:
    score_rows: List[Dict[str, object]] = []
    accuracy_percent: List[float] = []
    exact_match_percent: List[int] = []

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

        key = (int(season), int(week))
        week_record = week_records.setdefault(key, _init_week_record(int(season), int(week)))
        week_record["actual_eliminated"] = ", ".join(actual_names)
        week_record["actual_count"] = actual_count
        week_record["predicted_eliminated_percent"] = ", ".join(predicted_names)
        week_record["predicted_partner_percent"] = ", ".join(
            ranked["ballroom_partner"].head(actual_count).tolist()
        )
        week_record["correct_percent"] = week_best_correct
        week_record["exact_match_percent"] = week_best_exact

        accuracy_percent.append(float(week_best_correct))
        exact_match_percent.append(week_best_exact)

        predicted_mask = ranked["celebrity_name"].isin(predicted_names)
        ranked["predicted_eliminated_percent"] = predicted_mask

        for _, row in ranked.iterrows():
            score_rows.append(
                {
                    "season": int(season),
                    "week": int(week),
                    "celebrity_name": row["celebrity_name"],
                    "ballroom_partner": row["ballroom_partner"],
                    "judge_percent": row["judge_percent"],
                    "fan_percent": row["fan_percent"],
                    "combined_percent": row["combined_percent"],
                    "predicted_eliminated_percent": bool(row["predicted_eliminated_percent"]),
                }
            )

    return pd.DataFrame(score_rows), accuracy_percent, exact_match_percent


def main() -> None:
    required_cols = [
        "season",
        "week",
        "celebrity_name",
        "ballroom_partner",
        "results",
        "composite_attractiveness",
        "week_total_judge",
    ]

    rank_df = pd.read_csv(INPUT_RANK_PATH, na_values=["N/A", "NA", ""])
    percent_df = pd.read_csv(INPUT_PERCENT_PATH, na_values=["N/A", "NA", ""])

    _validate_columns(rank_df, required_cols, "rank input")
    _validate_columns(percent_df, required_cols, "percent input")

    rank_df = _prepare_df(rank_df)
    percent_df = _prepare_df(percent_df)

    week_records: Dict[Tuple[int, int], Dict[str, object]] = {}

    rank_scores, accuracy_rank, exact_match_rank = _compute_rank_method(
        rank_df, week_records
    )
    percent_scores, accuracy_percent, exact_match_percent = _compute_percent_method(
        percent_df, week_records
    )

    week_df = pd.DataFrame(week_records.values()).sort_values(["season", "week"])
    week_df.to_csv(OUTPUT_WEEKLY_PATH, index=False)

    score_df = pd.merge(
        rank_scores,
        percent_scores,
        on=["season", "week", "celebrity_name", "ballroom_partner"],
        how="outer",
    ).sort_values(["season", "week", "celebrity_name"])
    score_df.to_csv(OUTPUT_SCORE_PATH, index=False)

    summary_df = pd.DataFrame(
        {
            "method": ["rank", "percent"],
            "total_weeks": [len(accuracy_rank), len(accuracy_percent)],
            "avg_recall": [
                float(sum(accuracy_rank) / len(accuracy_rank)) if accuracy_rank else 0.0,
                float(sum(accuracy_percent) / len(accuracy_percent))
                if accuracy_percent
                else 0.0,
            ],
            "exact_match_weeks": [sum(exact_match_rank), sum(exact_match_percent)],
            "exact_match_rate": [
                float(sum(exact_match_rank) / len(exact_match_rank))
                if exact_match_rank
                else 0.0,
                float(sum(exact_match_percent) / len(exact_match_percent))
                if exact_match_percent
                else 0.0,
            ],
        }
    )
    summary_df.to_csv(OUTPUT_SUMMARY_PATH, index=False)


if __name__ == "__main__":
    main()
