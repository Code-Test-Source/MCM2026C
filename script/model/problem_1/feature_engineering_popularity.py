import re
from pathlib import Path

import pandas as pd



DATA_PATH = "../../../data/processed/2026_MCM_Problem_C_Data_processed.csv"
OUTPUT_PATH = "../../../data/processed/2026_MCM_Problem_C_Data_popularity_features.csv"


def main() -> None:
    df = pd.read_csv(
        DATA_PATH,
        na_values=["N/A", "NA", ""],
        dtype={
            "celebrity_name": "string",
            "ballroom_partner": "string",
            "celebrity_industry": "string",
            "celebrity_homestate": "string",
            "celebrity_homecountry/region": "string",
            "results": "string",
        },
    )
    print('打印results列所有取值', df['results'].unique())
    print('打印placement列所有取值', df['placement'].unique())
    
    base_cols = [
        "celebrity_name",
        "ballroom_partner",
        "celebrity_industry",
        "celebrity_homestate",
        "celebrity_homecountry/region",
        "celebrity_age_during_season",
        "season",
        "results",
        "placement",
    ]
    df.fillna({"celebrity_homestate": "Unknown"}, inplace=True)
    df.fillna({"celebrity_industry": "Unknown"}, inplace=True)
    df.fillna({"celebrity_homecountry/region": "Unknown"}, inplace=True)

    df["placement"] = pd.to_numeric(df["placement"], errors="coerce")
    results_lower_base = df["results"].fillna("").str.lower()
    is_finalist = (df["placement"] <= 3) | results_lower_base.str.contains(
        "final", regex=False
    )
    partner_stats = (
        df.groupby("ballroom_partner")
        .agg(
            partner_total_seasons=("season", "nunique"),
            partner_avg_placement=("placement", "mean"),
            partner_final_rate=("placement", lambda s: is_finalist.loc[s.index].mean()),
        )
        .reset_index()
    )

    judge_score_cols = [
        c for c in df.columns if re.match(r"^week\d+_judge[1-4]_score$", c)
    ]
    for c in judge_score_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    week_nums = sorted(
        {
            int(re.match(r"^week(\d+)_judge[1-4]_score$", c).group(1))
            for c in judge_score_cols
        }
    )

    long_frames = []
    for w in week_nums:
        week_cols = [
            c for c in judge_score_cols if re.match(rf"^week{w}_judge[1-4]_score$", c)
        ]
        tmp = df[base_cols + week_cols].copy()
        tmp["week"] = w
        tmp["week_total_judge"] = tmp[week_cols].sum(axis=1, skipna=True, min_count=1)
        long_frames.append(tmp[base_cols + ["week", "week_total_judge"]])

    long_df = pd.concat(long_frames, ignore_index=True)
    long_df = long_df.merge(partner_stats, on="ballroom_partner", how="left")

    results_lower = long_df["results"].fillna("").str.lower()
    eliminated_or_withdrew = results_lower.str.contains("eliminated|withdrew", regex=True)

    long_df["is_active"] = 1
    long_df.loc[long_df["week_total_judge"].isna(), "is_active"] = 0
    long_df.loc[
        (long_df["week_total_judge"] == 0) & eliminated_or_withdrew, "is_active"
    ] = 0

    active_df = long_df[long_df["is_active"] == 1].copy()

    active_df["age_group"] = pd.cut(
        active_df["celebrity_age_during_season"],
        bins=[0, 24, 34, 44, 54, 64, 200],
        labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"],
        right=True,
        include_lowest=True,
    )

    season_total_weeks = (
        active_df.groupby("season")["week"].max().rename("season_total_weeks")
    )
    active_df = active_df.merge(
        season_total_weeks, left_on="season", right_index=True, how="left"
    )

    weekly_counts = (
        active_df.groupby(["season", "week"]).size().rename("weekly_contestant_count")
    )
    active_df = active_df.merge(
        weekly_counts, left_on=["season", "week"], right_index=True, how="left"
    )

    weekly_max = (
        active_df.groupby(["season", "week"])["week_total_judge"]
        .max()
        .rename("weekly_max_judge")
    )
    active_df = active_df.merge(
        weekly_max, left_on=["season", "week"], right_index=True, how="left"
    )

    active_df["judge_pct"] = active_df["week_total_judge"] / active_df[
        "weekly_max_judge"
    ]

    weekly_rank = (
        active_df.groupby(["season", "week"])["week_total_judge"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    active_df["rank_pct"] = 1 - (
        (weekly_rank - 1) / active_df["weekly_contestant_count"]
    )

    active_df = active_df.sort_values(["season", "celebrity_name", "week"])
    active_df["cumulative_weeks"] = (
        active_df.groupby(["season", "celebrity_name"])
        .cumcount()
        .add(1)
        .astype(int)
    )
    active_df["relative_cumulative_weeks"] = (
        active_df["cumulative_weeks"] / active_df["season_total_weeks"]
    )

    weekly_cum_mean = (
        active_df.groupby(["season", "week"])["cumulative_weeks"]
        .mean()
        .rename("weekly_cum_mean")
    )
    active_df = active_df.merge(
        weekly_cum_mean, left_on=["season", "week"], right_index=True, how="left"
    )
    active_df["survival_advantage"] = (
        active_df["cumulative_weeks"] - active_df["weekly_cum_mean"]
    )

    active_df["weekly_judge_rank"] = (
        active_df.groupby(["season", "week"])["week_total_judge"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    active_df["weekly_judge_rank_pct"] = (
        active_df["weekly_judge_rank"] / active_df["weekly_contestant_count"]
    )

    active_df["popularity_deviation"] = (
        active_df["rank_pct"] - active_df["judge_pct"]
    )

    output_cols = [
        "season",
        "week",
        "celebrity_name",
        "ballroom_partner",
        "partner_total_seasons",
        "partner_avg_placement",
        "partner_final_rate",
        "celebrity_industry",
        "celebrity_homestate",
        "celebrity_homecountry/region",
        "celebrity_age_during_season",
        "age_group",
        "results",
        "placement",
        "week_total_judge",
        "season_total_weeks",
        "weekly_contestant_count",
        "judge_pct",
        "rank_pct",
        "popularity_deviation",
        "cumulative_weeks",
        "relative_cumulative_weeks",
        "survival_advantage",
        "weekly_judge_rank",
        "weekly_judge_rank_pct",
    ]

    active_df[output_cols].to_csv(OUTPUT_PATH, index=False)




if __name__ == "__main__":
    main()
