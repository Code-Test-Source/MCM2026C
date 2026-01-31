# Problem 1 模型脚本说明

## 概览
本文件夹包含“吸引力（attractiveness）+评委评分”驱动的淘汰模拟、特征工程与结果汇总相关脚本。整体流程以“评委排名 + 观众排名（由吸引力构成）”的排名和方法为核心，输出逐周淘汰预测与准确率统计。

## 建议流程
1. 生成特征：先构建热度/人气相关特征。
2. 构建吸引力指标：进行PCA并合成吸引力。
3. 运行淘汰模拟：按周预测淘汰并汇总准确率。

## 主要文件
- [feature_engineering_popularity.py](feature_engineering_popularity.py)：热度/人气特征工程。
- [pca.py](pca.py)：PCA降维与特征合成。
- [composite_attractiveness_sa.py](composite_attractiveness_sa.py)：吸引力指标的权重寻优与生成。
- [fan_rank_by_attractiveness.py](fan_rank_by_attractiveness.py)：基于吸引力构造观众排名。
- [simulate_elimination_by_attractiveness.py](simulate_elimination_by_attractiveness.py)：淘汰模拟与准确率统计。
- [full.md](full.md)：完整说明（若已有更详细文档，可作为主文档）。

## 输出文件
- [2026_MCM_Problem_C_Attractiveness_Elimination_Sim.csv](2026_MCM_Problem_C_Attractiveness_Elimination_Sim.csv)：逐周淘汰预测结果。
- [2026_MCM_Problem_C_Attractiveness_Elimination_Accuracy.csv](2026_MCM_Problem_C_Attractiveness_Elimination_Accuracy.csv)：整体准确率统计。
- [fan_attraction_rf_results.csv](fan_attraction_rf_results.csv)：吸引力/人气相关结果缓存或中间输出。
- [rank_sum_sa_results.csv](rank_sum_sa_results.csv)：排名和方案的优化或记录。

## 说明与约定
- 排名逻辑为“评委排名 + 观众排名”的等权排名和。
- 观众排名由吸引力指标（及其相关特征）构成。
- 若需更改数据路径或列名，请优先检查相关脚本顶部的常量配置。
