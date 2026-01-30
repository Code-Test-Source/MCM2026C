import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
JUDGE_COL_PATTERN = re.compile(r"^week(\d+)_judge(\d+)_score$")
ELIM_PATTERN = re.compile(r"Eliminated\s*Week\s*(\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process 2026 MCM Problem C data with judge totals and ranks."
    )
    parser.add_argument(
        "-i",
        "--input",
        default=None,
        help=(
            "Path to 2026_MCM_Problem_C_Data.csv. Defaults to "
            "./2026_MCM_Problem_C_Data.csv"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output CSV path. Defaults to ../processed/<stem>_processed.csv",
    )
    return parser.parse_args()


def find_judge_columns(columns: list[str]) -> dict[int, list[str]]:
    week_to_cols: dict[int, list[str]] = {}
    for col in columns:
        match = JUDGE_COL_PATTERN.match(col)
        if match:
            week = int(match.group(1))
            week_to_cols.setdefault(week, []).append(col)
    return week_to_cols


def extract_elim_week(results_value: str | float) -> int | None:
    if pd.isna(results_value):
        return None
    match = ELIM_PATTERN.search(str(results_value))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def compute_week_totals(df: pd.DataFrame, week_to_cols: dict[int, list[str]]) -> pd.DataFrame:
    df = df.copy()
    df["_elim_week"] = df["results"].apply(extract_elim_week)

    for week, cols in sorted(week_to_cols.items()):
        total_col = f"week{week}_total_judge_score"
        score_block = df[cols]
        non_nan_counts = score_block.notna().sum(axis=1)
        totals = score_block.sum(axis=1, skipna=True)
        totals = totals.where(non_nan_counts > 0, np.nan)

        elim_mask = df["_elim_week"].notna() & (df["_elim_week"] <= week)
        totals = totals.mask(elim_mask & totals.notna(), 0)
        df[total_col] = totals

    return df


def validate_judge_scores(df: pd.DataFrame, judge_cols: list[str]) -> list[dict]:
    anomalies: list[dict] = []
    for col in judge_cols:
        series = df[col]
        numeric = pd.to_numeric(series, errors="coerce")
        non_numeric_mask = series.notna() & numeric.isna()
        out_of_range_mask = numeric.notna() & ((numeric < 0) | (numeric > 15))

        for idx in df.index[non_numeric_mask]:
            anomalies.append(
                {
                    "type": "non-numeric",
                    "column": col,
                    "row_index": int(idx),
                    "value": series.loc[idx],
                }
            )
        for idx in df.index[out_of_range_mask]:
            anomalies.append(
                {
                    "type": "out-of-range",
                    "column": col,
                    "row_index": int(idx),
                    "value": series.loc[idx],
                }
            )
    return anomalies


def validate_elimination_rule(
    df: pd.DataFrame, week_to_cols: dict[int, list[str]]
) -> list[dict]:
    violations: list[dict] = []
    weeks = sorted(week_to_cols.keys())
    total_cols = [f"week{week}_total_judge_score" for week in weeks]

    for idx, row in df.iterrows():
        totals = row[total_cols]
        first_zero_week = None
        for week, total in zip(weeks, totals):
            if pd.isna(total):
                continue
            if total == 0:
                first_zero_week = week
                break
        if first_zero_week is None:
            continue
        for week, total in zip(weeks, totals):
            if week <= first_zero_week:
                continue
            if pd.isna(total):
                continue
            if total != 0:
                violations.append(
                    {
                        "celebrity_name": row.get("celebrity_name", ""),
                        "season": row.get("season", ""),
                        "week": week,
                        "value": total,
                    }
                )
                break
    return violations


def compute_week_ranks(df: pd.DataFrame, week_to_cols: dict[int, list[str]]) -> pd.DataFrame:
    df = df.copy()
    for week in sorted(week_to_cols.keys()):
        total_col = f"week{week}_total_judge_score"
        rank_col = f"week{week}_judge_rank"
        df[rank_col] = np.nan

        for season, group in df.groupby("season"):
            totals = group[total_col]
            valid_mask = totals.notna() & (totals > 0)
            if valid_mask.any():
                ranks = totals[valid_mask].rank(ascending=False, method="min")
                df.loc[group.index[valid_mask], rank_col] = ranks

            eliminated_mask = totals.notna() & (totals == 0)
            df.loc[group.index[eliminated_mask], rank_col] = "Eliminated"
    return df


def build_report(
    df: pd.DataFrame,
    anomalies: list[dict],
    violations: list[dict],
    week_to_cols: dict[int, list[str]],
) -> str:
    lines: list[str] = []
    lines.append("验证报告")
    lines.append("=" * 40)

    if not anomalies and not violations:
        lines.append("数据验证通过，无异常。")
    else:
        if anomalies:
            lines.append(f"评委得分异常记录数: {len(anomalies)}")
            sample = anomalies[:10]
            for item in sample:
                lines.append(
                    f"- {item['type']} @ row {item['row_index']}, "
                    f"{item['column']} = {item['value']}"
                )
            if len(anomalies) > 10:
                lines.append("...（仅展示前10条）")
        else:
            lines.append("评委得分有效性检查：无异常。")

        if violations:
            lines.append(f"淘汰后得分规则违规数: {len(violations)}")
            sample = violations[:10]
            for item in sample:
                lines.append(
                    f"- {item['celebrity_name']} (season {item['season']}), "
                    f"week {item['week']} total = {item['value']}"
                )
            if len(violations) > 10:
                lines.append("...（仅展示前10条）")
        else:
            lines.append("淘汰后得分规则检查：无异常。")

    lines.append("")
    lines.append("数据概况")
    lines.append("=" * 40)
    lines.append(f"筛选后行数: {len(df)}")
    lines.append(f"涉及赛季数: {df['season'].nunique()}")
    lines.append(f"选手数: {df['celebrity_name'].nunique()}")

    anomaly_names = set()
    for item in violations:
        if item.get("celebrity_name") is not None:
            anomaly_names.add(item.get("celebrity_name"))
    for item in anomalies:
        row_index = item.get("row_index")
        if row_index in df.index:
            anomaly_names.add(df.loc[row_index, "celebrity_name"])
    total_names = df["celebrity_name"].nunique()
    no_anomaly_ratio = (
        1 - len(anomaly_names) / total_names if total_names else 1.0
    )
    lines.append(f"无异常选手占比: {no_anomaly_ratio:.2%}")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent  # 获取脚本所在目录
    input_path = Path(args.input) if args.input else script_dir.parent.parent / "data" / "raw" / "2026_MCM_Problem_C_Data.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    week_to_cols = find_judge_columns(df.columns.tolist())
    if not week_to_cols:
        raise ValueError("No judge score columns found.")

    judge_cols = [col for cols in week_to_cols.values() for col in cols]
    df = compute_week_totals(df, week_to_cols)

    anomalies = validate_judge_scores(df, judge_cols)
    violations = validate_elimination_rule(df, week_to_cols)

    df = compute_week_ranks(df, week_to_cols)
    df = df.drop(columns=["_elim_week"])

    output_path = (
        Path(args.output)
        if args.output
        else script_dir.parent.parent / "data" / "processed" / f"{input_path.stem}_processed.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    report = build_report(df, anomalies, violations, week_to_cols)
    print(report)
    print(f"\n已保存: {output_path}")


if __name__ == "__main__":
    main()
