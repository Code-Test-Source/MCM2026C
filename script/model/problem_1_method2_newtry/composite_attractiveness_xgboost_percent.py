from __future__ import annotations

import re

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import ElasticNet
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    )
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
    "../../../data/processed/2026_MCM_Problem_C_Data_popularity_features_with_attractiveness_xgboost_percent.csv"
)
WEIGHTS_PATH = "../../../data/processed/attractiveness_weights_xgboost_percent.csv"

ID_COLS = [
    "season",
    "week",
    "celebrity_name",
    "ballroom_partner",
]

TARGET_COL = "placement"
PERCENT_BASED_SEASONS = set(range(3, 28))
JUDGE_METRICS_PERCENT = {
    "judge_share": 1.0,
    "week_total_judge": 1.0,
}
TARGET_ALPHA = 0.5
EXCLUDE_COLS = [
    "results",
]

ELIM_PATTERN = re.compile(r"(eliminated|withdrew)\s+week\s+(\d+)", re.IGNORECASE)

EXTRA_FEATURE_COLS: list[str] = []

RANDOM_SEED = 42


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    means = df.mean(axis=0)
    stds = df.std(axis=0).replace(0, 1)
    return (df - means) / stds


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


def _build_elimination_target(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    working = df.copy()
    working["elimination_week"] = _extract_elimination_week(working["results"])
    working["elimination_week"] = _infer_missing_elimination_week(working)
    known_mask = working["elimination_week"].notna()
    target = (
        known_mask & (working["elimination_week"] == working["week"])
    ).astype(float)
    return target.to_numpy(dtype=float), known_mask.to_numpy(dtype=bool)


def _minmax(x: np.ndarray) -> np.ndarray:
    min_v = float(np.min(x))
    max_v = float(np.max(x))
    if max_v - min_v == 0:
        return np.zeros_like(x)
    return (x - min_v) / (max_v - min_v)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _stacking_blend(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, dict[str, float], np.ndarray]:
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    oof_en = np.zeros(len(y), dtype=float)
    oof_xgb = np.zeros(len(y), dtype=float)

    for train_idx, val_idx in kf.split(x):
        x_tr, x_val = x[train_idx], x[val_idx]
        y_tr = y[train_idx]
        w_tr = weights[train_idx]

        en = ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=seed)
        en.fit(x_tr, y_tr, sample_weight=w_tr)
        oof_en[val_idx] = en.predict(x_val)

        neg_count = float((y == 0).sum())
        pos_count = float((y == 1).sum())
        xgb = XGBRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=seed,
            objective="binary:logistic",
            scale_pos_weight=neg_count / (pos_count or 1.0),
            n_jobs=-1,
        )
        xgb.fit(x_tr, y_tr, sample_weight=w_tr)
        oof_xgb[val_idx] = xgb.predict(x_val)

    oof_en_mm = _minmax(oof_en)
    oof_xgb_mm = _minmax(oof_xgb)

    preds = np.vstack([oof_en_mm, oof_xgb_mm]).T
    weights_raw, *_ = np.linalg.lstsq(preds, y, rcond=None)
    weights_raw = np.clip(weights_raw, 0, None)
    if weights_raw.sum() == 0:
        weights_raw = np.array([0.5, 0.5], dtype=float)
    blend_weights = weights_raw / weights_raw.sum()

    blend = blend_weights[0] * oof_en_mm + blend_weights[1] * oof_xgb_mm

    metrics = {
        "mse": mean_squared_error(y, blend, sample_weight=weights),
        "mae": mean_absolute_error(y, blend, sample_weight=weights),
        "r2": r2_score(y, blend, sample_weight=weights),
        "auc": _safe_auc(y, blend),
    }

    return oof_en, oof_xgb, metrics, blend_weights


def main() -> None:
    df = pd.read_csv(INPUT_PATH, na_values=["N/A", "NA", ""])
    pca_df = pd.read_csv(PCA_INPUT_PATH, na_values=["N/A", "NA", ""])

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    pca_feature_cols = [c for c in pca_df.columns if c.startswith("pca_")]
    if not pca_feature_cols:
        raise ValueError("No PCA components found in PCA input file.")

    if set(ID_COLS).issubset(df.columns) and set(ID_COLS).issubset(pca_df.columns):
        merged = df[ID_COLS].merge(
            pca_df[ID_COLS + pca_feature_cols],
            on=ID_COLS,
            how="left",
            validate="one_to_one",
        )
        feature_df = merged[pca_feature_cols].copy()
        if feature_df.isna().all(axis=1).any():
            raise ValueError("PCA 特征与原始数据无法对齐，请检查 ID 列是否一致。")
    else:
        feature_df = pca_df[pca_feature_cols].copy()
    extra_cols = [c for c in EXTRA_FEATURE_COLS if c in df.columns]
    extra_df = df[extra_cols].apply(pd.to_numeric, errors="coerce")
    feature_df = pd.concat(
        [feature_df.reset_index(drop=True), extra_df.reset_index(drop=True)], axis=1
    )
    feature_df = feature_df.fillna(feature_df.median(numeric_only=True))

    target, known_mask = _build_elimination_target(df)
    train_mask = (
        df["season"].isin(PERCENT_BASED_SEASONS).to_numpy(dtype=bool) & known_mask
    )
    if not train_mask.any():
        train_mask = np.ones(len(df), dtype=bool)

    scaler = StandardScaler()
    x_full = feature_df.to_numpy(dtype=float)
    x_train_raw = x_full[train_mask]
    x = scaler.fit(x_train_raw).transform(x_full)

    x_train = x[train_mask]
    y_train = target[train_mask]
    pos_count = float((y_train == 1).sum())
    neg_count = float((y_train == 0).sum())
    pos_weight = neg_count / (pos_count or 1.0)
    sample_weights = np.where(y_train == 1, pos_weight, 1.0)

    oof_en, oof_xgb, metrics, blend_weights = _stacking_blend(
        x_train, y_train, sample_weights, RANDOM_SEED
    )

    en_full = ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=RANDOM_SEED)
    en_full.fit(x_train, y_train, sample_weight=sample_weights)

    xgb_full = XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        objective="binary:logistic",
        scale_pos_weight=pos_weight,
        n_jobs=-1,
    )
    xgb_full.fit(x_train, y_train, sample_weight=sample_weights)

    pred_en = en_full.predict(x)
    pred_xgb = xgb_full.predict(x)

    blended_pred = (
        blend_weights[0] * _minmax(pred_en) + blend_weights[1] * _minmax(pred_xgb)
    )
    attractiveness = 1 - _minmax(blended_pred)

    output_df = df.copy()
    output_df["composite_attractiveness"] = attractiveness
    output_df.to_csv(OUTPUT_PATH, index=False)

    all_feature_cols = pca_feature_cols + extra_cols
    weights_df = pd.DataFrame(
        {
            "feature": all_feature_cols,
            "elastic_net_coef": en_full.coef_,
            "xgb_importance": xgb_full.feature_importances_,
        }
    ).sort_values("xgb_importance", ascending=False)
    weights_df["cv_mse"] = metrics["mse"]
    weights_df["cv_mae"] = metrics["mae"]
    weights_df["cv_r2"] = metrics["r2"]
    weights_df["cv_auc"] = metrics["auc"]
    weights_df["blend_elasticnet_weight"] = blend_weights[0]
    weights_df["blend_xgb_weight"] = blend_weights[1]
    weights_df.to_csv(WEIGHTS_PATH, index=False)

    print("训练完成(非相关系数)，CV-MSE:", metrics["mse"])
    print("训练完成(非相关系数)，CV-MAE:", metrics["mae"])
    print("训练完成(非相关系数)，CV-R2:", metrics["r2"])
    print("训练完成(非相关系数)，CV-AUC:", metrics["auc"])
    print("权重结果已保存:", WEIGHTS_PATH)
    print("带综合吸引力结果已保存:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
