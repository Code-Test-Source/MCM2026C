# 第三题模型说明与使用指南

本目录包含第三题“名人特征对表现/投票影响”的实现与结果生成脚本。整体流程：
1) 先用淘汰模拟脚本生成“带特征的周度评分/投票数据”；
2) 再做特征工程（行业/家乡/年龄分组等编码）；
3) 最后用 PSO + RF + SHAP 评估四类特征（舞伴、年龄、行业、家乡）对三类目标变量的影响，并输出结果表。

---

## 1. 模型逻辑（面向论文描述）

### 1.1 研究问题
第三题要求分析：
- 舞伴（职业舞者）、名人年龄、行业、家乡等特征是否影响表现；
- 这些特征对“裁判评分”和“粉丝投票”的影响是否一致；
- 给出可解释的影响排序/权重。

### 1.2 数据与目标变量
基于已处理的周度数据，构造三类目标变量用于对比：
- 粉丝投票相关：`fan_votes_relative` （优先选择，选不到则顺延）/ `fan_rank` / `fan_percent`（脚本自动选存在的列）
- 裁判打分相关：`week_total_judge` （优先）/ `week_total_judge_rank` / `week_total_judge_percent`
- 赛季名次相关：`placement_ordered_rank` （优先）/ `placement` / `placement_rank`

### 1.3 特征分组（四大类）
脚本会自动从列名中识别并分组：
- 舞伴特征：以 `partner_` 前缀开头的列
- 行业特征：以 `celebrity_industry` 前缀开头的列
- 家乡特征：以 `country_state_` 或 `celebrity_homecountry` / `celebrity_homestate` 前缀开头的列
- 年龄特征：`age_group_` 或 `age_group_ordinal`，以及 `celebrity_age_during_season`

### 1.4 方法：PSO + 随机森林 + SHAP
1. **PSO 特征筛选**：
   - 将所有候选特征作为粒子维度；
   - 适应度=交叉验证 RMSE + 特征数惩罚；
   - 约束：四大特征组都至少选中一个列；
2. **随机森林回归**：
   - 拟合目标变量，输出 R2、MAE；
3. **SHAP**：
   - 计算每个特征的平均绝对 SHAP 值；
   - 组内求和得到四大特征组的重要性（可解释性）。

### 1.5 输出结论（用于论文）
输出表格包含：
- 每个目标变量对应的 R2、MAE、训练耗时
- 四大特征组的重要性（舞伴/年龄/行业/家乡）

论文可据此比较：
- 影响粉丝投票 vs 影响裁判评分的主导特征是否不同；
- “名次”是否更依赖舞伴/年龄等客观因素；
- 用于“更公平”方案的依据（例如：若粉丝投票更受行业/家乡影响，说明存在偏好）。

---

## 2. 文件说明

- [full.md](full.md)
  - 题面与问题描述；`pso_rf_shap_analysis.py` 会从中读取第三题描述做日志输出。

- [simulate_elimination_all_seasons_both_methods_keep_features.py](simulate_elimination_all_seasons_both_methods_keep_features.py)
  - 作用：按“排名法/百分比法”模拟每周淘汰，生成用于第三题的周度评分/投票数据（含特征）。
  - 输入：
    - 2026_MCM_Problem_C_Data_popularity_features_with_attractiveness_xgboost.csv
    - 2026_MCM_Problem_C_Data_popularity_features_with_attractiveness_xgboost_percent.csv
  - 输出（写入 data/processed/problem3/）：
    - 2026_MCM_Problem_C_Attractiveness_Scores_All_Seasons_Both_Methods_Keep_Features.csv
    - 2026_MCM_Problem_C_Attractiveness_Elimination_Sim_All_Seasons_Both_Methods_Keep_Features.csv
    - 2026_MCM_Problem_C_Attractiveness_Elimination_Summary_All_Seasons_Both_Methods_Keep_Features.csv

- [feature_engineering.py](feature_engineering.py)
  - 作用：对第三题数据做特征编码，生成最终建模输入表。
  - 核心处理：
    - 行业独热（可合并低频为 others）
    - 国家/州分层独热（美国细分到州，其他国家保留国家）
    - 年龄组有序编码或独热
    - 生成 `placement_ordered_rank`
    - 保留白名单列、规范列名（非排名列去掉 `_rank` 后缀）
  - 输出：
    - 2026_MCM_Problem_C_Attractiveness_Scores_All_Seasons_Both_Methods_Keep_Features_processed.csv

- [pso_rf_shap_analysis.py](pso_rf_shap_analysis.py)
  - 作用：PSO + RF + SHAP 训练与解释，输出四大特征组重要性。
  - 输出：
    - data/result/pso_rf_shap_results.csv

---

## 3. 运行顺序（推荐）

1) 生成第三题数据（若尚未生成）：
- 运行 [simulate_elimination_all_seasons_both_methods_keep_features.py](simulate_elimination_all_seasons_both_methods_keep_features.py)

2) 特征工程：
- 运行 [feature_engineering.py](feature_engineering.py)

3) PSO + RF + SHAP 分析：
- 运行 [pso_rf_shap_analysis.py](pso_rf_shap_analysis.py)

---

## 4. 依赖说明

- pandas, numpy
- scikit-learn
- shap
- pyswarms
- matplotlib（仅在需要可视化时）

---

## 5. 常见问题（简短）

- **找不到输入文件**：检查 data/processed 与 data/result 路径是否存在，或修改脚本顶部路径变量。
- **特征列不匹配**：在 [feature_engineering.py](feature_engineering.py) 顶部配置区替换实际列名（`PARTNER_FEATURE_COLS` 等）。
- **SHAP 过慢**：可减少随机森林树数或关闭 `PLOT`。

---

## 6. 结果用于论文的建议表达
- “舞伴特征”重要性显著高于其他组，说明职业舞者带来的训练与编舞水平对表现影响最大。
- “行业/家乡”对粉丝投票的影响更大，表明存在群体偏好与地域效应。
- 若“年龄”对裁判评分重要但对粉丝投票弱，可据此提出更公平的加权方案。

（以上为模板表述，最终应以实际输出结果为准。）
