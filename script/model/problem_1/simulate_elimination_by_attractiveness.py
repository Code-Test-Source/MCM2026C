from __future__ import annotations

import re

import pandas as pd


INPUT_PATH = (
    "../../../data/processed/2026_MCM_Problem_C_Data_popularity_features_with_attractiveness.csv"
)
OUTPUT_PATH = (
    "../../../data/processed/2026_MCM_Problem_C_Attractiveness_Elimination_Sim.csv"
)
SUMMARY_PATH = (
    "../../../data/processed/2026_MCM_Problem_C_Attractiveness_Elimination_Accuracy.csv"
)

RANK_BASED_SEASONS = set([1, 2, 28, 29, 30, 31, 32, 33, 34])
ELIM_PATTERN = re.compile(r"(eliminated|withdrew)\s+week\s+(\d+)", re.IGNORECASE)

FAN_BLEND_GRID = [round(x, 2) for x in [-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0]]


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

    df = df[df["season"].isin(RANK_BASED_SEASONS)].copy()
    df["elimination_week"] = _extract_elimination_week(df["results"])
    df["elimination_week"] = _infer_missing_elimination_week(df)
    df["is_eliminated_this_week"] = (
        df["elimination_week"].notna() & (df["elimination_week"] == df["week"])
    )

    records = []
    accuracy_rank = []
    exact_match_rank = []
    blend_scores = {b: [] for b in FAN_BLEND_GRID}

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
        min_a = float(attractiveness.min())
        max_a = float(attractiveness.max())
        if max_a - min_a == 0:
            fan_base = pd.Series(0.0, index=active.index)
        else:
            fan_base = (attractiveness - min_a) / (max_a - min_a)

        if "popularity_deviation" in active.columns:
            pop_dev = active["popularity_deviation"].astype(float)
            pop_dev = pop_dev.fillna(pop_dev.median())
            pop_min = float(pop_dev.min())
            pop_max = float(pop_dev.max())
            if pop_max - pop_min == 0:
                pop_scaled = pd.Series(0.0, index=active.index)
            else:
                pop_scaled = (pop_dev - pop_min) / (pop_max - pop_min)
        else:
            pop_scaled = pd.Series(0.0, index=active.index)

        judge_rank = (
            active["week_total_judge"].rank(method="min", ascending=False).astype(int)
        )

        actual_names = set(actual_elim["celebrity_name"].tolist())
        actual_count = len(actual_names)

        week_best_score = -1.0
        week_best_blend = 0.0
        week_best_ranked = None
        week_best_predicted = None
        week_best_correct = 0.0
        week_best_exact = 0

        for blend in FAN_BLEND_GRID:
            fan_proxy = fan_base + blend * pop_scaled
            fan_rank = fan_proxy.rank(method="min", ascending=False).astype(int)
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
            correct_rank = len(predicted_rank_set & actual_names) / max(actual_count, 1)
            score = float(correct_rank)

            blend_scores[blend].append(score)

            if score > week_best_score:
                week_best_score = score
                week_best_blend = blend
                week_best_ranked = ranked
                week_best_predicted = predicted_rank_names
                week_best_correct = correct_rank
                week_best_exact = int(predicted_rank_set == actual_names)

        records.append(
            {
                "season": season,
                "week": week,
                "predicted_eliminated_rank": ", ".join(week_best_predicted or []),
                "predicted_partner_rank": ", ".join(
                    week_best_ranked["ballroom_partner"].head(actual_count).tolist()
                    if week_best_ranked is not None
                    else []
                ),
                "actual_eliminated": ", ".join(actual_names),
                "actual_count": len(actual_names),
                "correct_rank": week_best_correct,
                "exact_match_rank": week_best_exact,
                "best_fan_blend": week_best_blend,
            }
        )

        accuracy_rank.append(float(week_best_correct))
        exact_match_rank.append(week_best_exact)

    result_df = pd.DataFrame(records)
    result_df.to_csv(OUTPUT_PATH, index=False)

    accuracy_rank_value = (
        float(sum(accuracy_rank) / len(accuracy_rank)) if accuracy_rank else 0.0
    )
    summary_df = pd.DataFrame(
        {
            "total_weeks": [len(accuracy_rank)],
            "avg_recall_rank": [accuracy_rank_value],
            "exact_match_weeks_rank": [sum(exact_match_rank)],
            "exact_match_rate_rank": [
                float(sum(exact_match_rank) / len(exact_match_rank))
                if exact_match_rank
                else 0.0
            ],
            "best_fan_blend": [
                max(
                    blend_scores,
                    key=lambda b: (
                        sum(blend_scores[b]) / len(blend_scores[b])
                        if blend_scores[b]
                        else 0.0
                    ),
                )
                if blend_scores
                else 0.0
            ],
        }
    )
    summary_df.to_csv(SUMMARY_PATH, index=False)


if __name__ == "__main__":
    main()
