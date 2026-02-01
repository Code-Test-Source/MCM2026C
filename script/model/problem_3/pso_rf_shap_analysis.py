# 依赖安装命令：
# pip install pandas numpy scikit-learn shap matplotlib pyswarms
#
# 说明：
# 1) 读取当前目录下 full.md 的第三题内容（关键段落）。
# 2) 读取当前目录下与任务相关的文件（feature_engineering.py、CSV）。
# 3) 使用 PSO + RF + SHAP 评估四大特征（舞伴、年龄、行业、家乡）
#    对三类目标变量的影响。
# 4) 输出结果并给出结论（基于实际运行结果）。

from __future__ import annotations

import os
import re
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

try:
    import shap
except ImportError as exc:
    raise SystemExit(
        "缺少shap依赖，请先执行: pip install shap"
    ) from exc

try:
    from pyswarms.discrete import BinaryPSO
except ImportError as exc:
    raise SystemExit(
        "缺少pyswarms依赖，请先执行: pip install pyswarms"
    ) from exc


# ===== 文件路径（相对路径） =====
FULL_MD_PATH = "full.md"
FEATURE_ENGINEERING_PATH = "feature_engineering.py"
PROCESSED_INPUT_CSV = (
    "../../../data/processed/problem3/"
    "2026_MCM_Problem_C_Attractiveness_Scores_All_Seasons_Both_Methods_Keep_Features_processed.csv"
)
# 输出结果CSV路径（与输入路径放在一起，显眼位置）
RESULT_CSV_PATH = "../../../data/result/pso_rf_shap_results.csv"

# ===== PSO配置 =====
PSO_PARTICLES = 20
PSO_ITERS = 30
CV_FOLDS = 3
ALPHA_PENALTY = 0.01  # 特征数量惩罚权重

# ===== 可选可视化 =====
PLOT = False


def read_third_task_from_full_md(md_path: str) -> str:
    if not os.path.exists(md_path):
        return "未找到full.md"
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
    pattern = re.compile(
        r"Use the data including your fan vote estimates to develop a model[\s\S]*?\n\n",
        re.IGNORECASE,
    )
    match = pattern.search(content)
    if match:
        return match.group(0).strip()
    return "未能从full.md中匹配到第三题段落，请手动确认。"


def load_input_csv() -> pd.DataFrame:
    if os.path.exists(PROCESSED_INPUT_CSV):
        return pd.read_csv(PROCESSED_INPUT_CSV, na_values=["N/A", "NA", ""])
    raise FileNotFoundError("未找到输入CSV文件（仅支持processed文件）。")


def pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"未找到候选目标列: {candidates}")


def infer_feature_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    cols = df.columns

    partner_cols = [c for c in cols if c.startswith("partner_")]
    industry_cols = [c for c in cols if c.startswith("celebrity_industry")]
    hometown_cols = [c for c in cols if c.startswith("country_state_")]
    hometown_cols += [c for c in cols if c.startswith("celebrity_homecountry")]
    hometown_cols += [c for c in cols if c.startswith("celebrity_homestate")]
    age_cols = [c for c in cols if c.startswith("age_group_")]
    if "age_group_ordinal" in cols:
        age_cols.append("age_group_ordinal")
    if "celebrity_age_during_season" in cols:
        age_cols.append("celebrity_age_during_season")

    def _unique_keep(items: List[str]) -> List[str]:
        seen = set()
        out = []
        for item in items:
            if item in cols and item not in seen:
                out.append(item)
                seen.add(item)
        return out

    return {
        "partner": _unique_keep(partner_cols),
        "industry": _unique_keep(industry_cols),
        "hometown": _unique_keep(hometown_cols),
        "age": _unique_keep(age_cols),
    }


def build_dataset(df: pd.DataFrame, target_col: str, feature_groups: Dict[str, List[str]]) -> Tuple[pd.DataFrame, pd.Series]:
    feature_cols = [c for group in feature_groups.values() for c in group]
    feature_cols = [c for c in feature_cols if c in df.columns]
    if not feature_cols:
        raise ValueError("未找到四大特征对应的列，请检查输入文件。")

    data = df[feature_cols + [target_col]].copy()
    data = data.dropna(subset=[target_col])

    for col in feature_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        if data[col].isna().all():
            data[col] = 0.0
        else:
            data[col] = data[col].fillna(data[col].median())

    X = data[feature_cols]
    y = data[target_col]
    return X, y


def pso_select_features(
    X: pd.DataFrame, y: pd.Series, feature_groups: Dict[str, List[str]]
) -> List[str]:
    feature_cols = X.columns.tolist()
    group_map = {}
    for group, cols in feature_groups.items():
        group_map[group] = [feature_cols.index(c) for c in cols if c in feature_cols]

    # PSO适应度函数（越小越好）
    def _objective(mask_matrix: np.ndarray) -> np.ndarray:
        costs = []
        for mask in mask_matrix:
            selected_idx = np.where(mask > 0.5)[0]
            if len(selected_idx) == 0:
                costs.append(1e6)
                continue

            # 保证四大特征每组至少选一个
            missing_group = False
            for group, idxs in group_map.items():
                if idxs and len(set(selected_idx) & set(idxs)) == 0:
                    missing_group = True
                    break
            if missing_group:
                costs.append(1e5)
                continue

            X_sel = X.iloc[:, selected_idx]
            model = RandomForestRegressor(
                n_estimators=150, random_state=42, n_jobs=-1
            )
            # 负MSE转换为RMSE
            scores = cross_val_score(
                model, X_sel, y, cv=CV_FOLDS, scoring="neg_mean_squared_error"
            )
            rmse = np.sqrt(-scores.mean())
            penalty = ALPHA_PENALTY * (len(selected_idx) / len(feature_cols))
            costs.append(rmse + penalty)
        return np.array(costs)

    optimizer = BinaryPSO(
        n_particles=PSO_PARTICLES,
        dimensions=len(feature_cols),
        options={"c1": 1.5, "c2": 1.5, "w": 0.6, "k": 5, "p": 2},
    )

    _, best_pos = optimizer.optimize(_objective, iters=PSO_ITERS)
    selected_idx = np.where(best_pos > 0.5)[0]
    if len(selected_idx) == 0:
        return feature_cols
    return [feature_cols[i] for i in selected_idx]


def rf_shap_analysis(X: pd.DataFrame, y: pd.Series, feature_groups: Dict[str, List[str]]) -> Dict[str, object]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300, random_state=42, n_jobs=-1
    )
    start = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start

    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap_abs_mean = np.abs(shap_values).mean(axis=0)

    feature_importance = pd.Series(shap_abs_mean, index=X_train.columns).sort_values(ascending=False)

    group_importance = {}
    for group, cols in feature_groups.items():
        cols_in = [c for c in cols if c in X_train.columns]
        if cols_in:
            group_importance[group] = feature_importance[cols_in].sum()
        else:
            group_importance[group] = 0.0

    group_importance = pd.Series(group_importance).sort_values(ascending=False)

    if PLOT:
        shap.summary_plot(shap_values, X_train, show=True)

    return {
        "model": model,
        "r2": r2,
        "mae": mae,
        "fit_time": fit_time,
        "feature_importance": feature_importance,
        "group_importance": group_importance,
    }


def main() -> None:
    print("=== 读取 full.md 第三题段落 ===")
    third_task = read_third_task_from_full_md(FULL_MD_PATH)
    print(third_task)

    print("\n=== 读取相关文件 ===")
    for path in [FULL_MD_PATH, FEATURE_ENGINEERING_PATH, PROCESSED_INPUT_CSV]:
        print(f"存在: {os.path.exists(path)} -> {path}")

    df = load_input_csv()
    print(f"\n输入数据形状: {df.shape}")

    target_fan = pick_first_existing(
        df, ["fan_votes_relative", "fan_rank", "fan_percent", "fan_percent_percent"]
    )
    target_judge = pick_first_existing(df, ["week_total_judge", "week_total_judge_rank", "week_total_judge_percent"])
    target_rank = pick_first_existing(df, ["placement_ordered_rank", "placement", "placement_rank"])

    feature_groups = infer_feature_groups(df)
    print("\n四大特征组列数：")
    for k, v in feature_groups.items():
        print(f"{k}: {len(v)}")

    results = {}
    csv_rows = []
    group_names = list(feature_groups.keys())
    for target_name, target_col in [
        ("观众投票数预计", target_fan),
        ("裁判投票总分", target_judge),
        ("处理后的总排名", target_rank),
    ]:
        print(f"\n=== 目标变量: {target_name} ({target_col}) ===")
        X, y = build_dataset(df, target_col, feature_groups)

        print("开始PSO特征筛选...")
        start = time.time()
        selected_cols = pso_select_features(X, y, feature_groups)
        pso_time = time.time() - start
        print(f"PSO选中特征数: {len(selected_cols)} | 耗时: {pso_time:.2f}s")

        X_selected = X[selected_cols]
        out = rf_shap_analysis(X_selected, y, feature_groups)
        results[target_name] = out

        print(f"R2: {out['r2']:.4f}")
        print(f"MAE: {out['mae']:.4f}")
        print(f"训练耗时: {out['fit_time']:.2f}s")
        print("四大特征重要性(聚合):")
        print(out["group_importance"])

        # 构建一行结果
        row = {
            "target": target_name,
            "R2": out["r2"],
            "MAE": out["mae"],
            "fit_time": out["fit_time"],
        }
        for group in group_names:
            row[f"importance_{group}"] = out["group_importance"].get(group, 0.0)
        csv_rows.append(row)

    # 保存为CSV
    result_df = pd.DataFrame(csv_rows)
    result_df.to_csv(RESULT_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存到: {RESULT_CSV_PATH}")

    print("\n=== 结论（基于本次运行结果） ===")
    print("方法：PSO + RF + SHAP。请与RF+SHAP脚本结果对比R2/MAE/耗时与重要性排序一致性。")


if __name__ == "__main__":
    main()
