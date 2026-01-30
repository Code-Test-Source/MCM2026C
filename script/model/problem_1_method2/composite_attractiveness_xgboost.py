from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import ElasticNet
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "缺少依赖，请先安装：pip install scikit-learn xgboost"
    ) from exc


INPUT_PATH = "../../../data/processed/2026_MCM_Problem_C_Data_popularity_features.csv"
PCA_INPUT_PATH = "../../../data/processed/2026_MCM_Problem_C_Data_popularity_pca.csv"
OUTPUT_PATH = (
    "../../../data/processed/2026_MCM_Problem_C_Data_popularity_features_with_attractiveness_xgboost.csv"
)
WEIGHTS_PATH = "../../../data/processed/attractiveness_weights_xgboost.csv"

ID_COLS = [
    "season",
    "week",
    "celebrity_name",
    "ballroom_partner",
]

TARGET_COL = "placement"
RANK_BASED_SEASONS = set([1, 2, 28, 29, 30, 31, 32, 33, 34])
JUDGE_METRICS_RANK = {
    "weekly_judge_rank_pct": -1.0,
    "week_total_judge": 1.0,
}
JUDGE_METRICS_PERCENT = {
    "judge_share": 1.0,
    "week_total_judge": 1.0,
}
TARGET_ALPHA = 0.5
EXCLUDE_COLS = [
    "results",
]

RANDOM_SEED = 42


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    means = df.mean(axis=0)
    stds = df.std(axis=0).replace(0, 1)
    return (df - means) / stds


def _build_composite_target(df: pd.DataFrame) -> np.ndarray:
    placement = df[TARGET_COL].to_numpy(dtype=float)
    placement = np.nan_to_num(placement, nan=np.nanmedian(placement))
    placement_z = (placement - np.mean(placement)) / (np.std(placement) or 1)
    placement_score = -placement_z

    rank_cols = [c for c in JUDGE_METRICS_RANK if c in df.columns]
    percent_cols = [c for c in JUDGE_METRICS_PERCENT if c in df.columns]

    judge_score_rank = np.zeros(len(df), dtype=float)
    if rank_cols:
        judge_rank_df = df[rank_cols].copy()
        judge_rank_df = judge_rank_df.fillna(judge_rank_df.median(numeric_only=True))
        judge_rank_z = _standardize(judge_rank_df)
        rank_weights = np.array([JUDGE_METRICS_RANK[c] for c in rank_cols], dtype=float)
        judge_score_rank = judge_rank_z.to_numpy(dtype=float) @ rank_weights
        judge_score_rank = (
            judge_score_rank - np.mean(judge_score_rank)
        ) / (np.std(judge_score_rank) or 1)

    judge_score_percent = np.zeros(len(df), dtype=float)
    if percent_cols:
        judge_percent_df = df[percent_cols].copy()
        judge_percent_df = judge_percent_df.fillna(
            judge_percent_df.median(numeric_only=True)
        )
        judge_percent_z = _standardize(judge_percent_df)
        percent_weights = np.array(
            [JUDGE_METRICS_PERCENT[c] for c in percent_cols], dtype=float
        )
        judge_score_percent = judge_percent_z.to_numpy(dtype=float) @ percent_weights
        judge_score_percent = (
            judge_score_percent - np.mean(judge_score_percent)
        ) / (np.std(judge_score_percent) or 1)

    use_rank = df["season"].isin(RANK_BASED_SEASONS).to_numpy(dtype=bool)
    judge_score = np.where(use_rank, judge_score_rank, judge_score_percent)

    alpha = TARGET_ALPHA
    return alpha * placement_score + (1 - alpha) * judge_score


def _minmax(x: np.ndarray) -> np.ndarray:
    min_v = float(np.min(x))
    max_v = float(np.max(x))
    if max_v - min_v == 0:
        return np.zeros_like(x)
    return (x - min_v) / (max_v - min_v)


def _stacking_blend(
    x: np.ndarray, y: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    oof_en = np.zeros(len(y), dtype=float)
    oof_xgb = np.zeros(len(y), dtype=float)

    for train_idx, val_idx in kf.split(x):
        x_tr, x_val = x[train_idx], x[val_idx]
        y_tr = y[train_idx]

        en = ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=seed)
        en.fit(x_tr, y_tr)
        oof_en[val_idx] = en.predict(x_val)

        xgb = XGBRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=seed,
            objective="reg:squarederror",
            n_jobs=-1,
        )
        xgb.fit(x_tr, y_tr)
        oof_xgb[val_idx] = xgb.predict(x_val)

    oof_en_mm = _minmax(oof_en)
    oof_xgb_mm = _minmax(oof_xgb)

    blend = 0.35 * oof_en_mm + 0.65 * oof_xgb_mm

    metrics = {
        "mse": mean_squared_error(y, blend),
        "mae": mean_absolute_error(y, blend),
        "r2": r2_score(y, blend),
    }

    return oof_en, oof_xgb, metrics


def main() -> None:
    df = pd.read_csv(INPUT_PATH, na_values=["N/A", "NA", ""])
    pca_df = pd.read_csv(PCA_INPUT_PATH, na_values=["N/A", "NA", ""])

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    pca_feature_cols = [c for c in pca_df.columns if c.startswith("pca_")]
    if not pca_feature_cols:
        raise ValueError("No PCA components found in PCA input file.")

    feature_df = pca_df[pca_feature_cols].copy()
    feature_df = feature_df.fillna(feature_df.median(numeric_only=True))

    scaler = StandardScaler()
    x = scaler.fit_transform(feature_df.to_numpy(dtype=float))

    train_mask = df["season"].isin(RANK_BASED_SEASONS).to_numpy(dtype=bool)
    if not train_mask.any():
        train_mask = np.ones(len(df), dtype=bool)

    target = _build_composite_target(df[train_mask])
    x_train = x[train_mask]

    oof_en, oof_xgb, metrics = _stacking_blend(x_train, target, RANDOM_SEED)

    en_full = ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=RANDOM_SEED)
    en_full.fit(x_train, target)

    xgb_full = XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        objective="reg:squarederror",
        n_jobs=-1,
    )
    xgb_full.fit(x_train, target)

    pred_en = en_full.predict(x)
    pred_xgb = xgb_full.predict(x)

    blended_pred = 0.35 * _minmax(pred_en) + 0.65 * _minmax(pred_xgb)
    attractiveness = _minmax(blended_pred)

    output_df = df.copy()
    output_df["composite_attractiveness"] = attractiveness
    output_df.to_csv(OUTPUT_PATH, index=False)

    weights_df = pd.DataFrame(
        {
            "feature": pca_feature_cols,
            "elastic_net_coef": en_full.coef_,
            "xgb_importance": xgb_full.feature_importances_,
        }
    ).sort_values("xgb_importance", ascending=False)
    weights_df["cv_mse"] = metrics["mse"]
    weights_df["cv_mae"] = metrics["mae"]
    weights_df["cv_r2"] = metrics["r2"]
    weights_df.to_csv(WEIGHTS_PATH, index=False)

    print("训练完成(非相关系数)，CV-MSE:", metrics["mse"])
    print("训练完成(非相关系数)，CV-MAE:", metrics["mae"])
    print("训练完成(非相关系数)，CV-R2:", metrics["r2"])
    print("权重结果已保存:", WEIGHTS_PATH)
    print("带综合吸引力结果已保存:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
