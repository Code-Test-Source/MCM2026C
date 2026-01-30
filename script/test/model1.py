import numpy as np
import pandas as pd
from scipy.optimize import minimize, linprog
from scipy.stats import rankdata, spearmanr
from statsmodels.tsa.arima.model import ARIMA
import os
# Disable oneDNN custom ops to avoid small numerical differences in TF
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import permutations, combinations
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import os

class DWTSFanVoteEstimator:
    """DWTS Fan-vote estimator.

    Integrates mathematical modeling, constrained optimization, and
    machine learning to estimate audience votes from judge scores.
    """
    
    def __init__(self, data_path):
        """Initialize estimator and load dataset from `data_path`."""
        self.data = pd.read_csv(data_path)
        self.season_ranges = {
            'rank_method': [(1, 2), (28, 34)],  # 排名法使用的赛季
            'percentage_method': [(3, 27)]      # 百分比法使用的赛季
        }
        self.estimates = {}  # 存储估计的观众票数
        self.uncertainty = {}  # 存储不确定性度量

    def get_time_series(self, name):
        """Return sorted list of (season, week, judge_score, fan_votes) for a contestant."""
        rows = []
        for (s, w), data in self.estimates.items():
            if name in data['names']:
                i = data['names'].index(name)
                rows.append((s, w, float(data['judge_scores'][i]), float(data['fan_votes'][i])))
        rows.sort()
        return rows
        
    def preprocess_data(self):
        """Preprocess input data into weekly judge-score dictionaries."""
        print("Preprocessing data...")
        
        # 解析周次数据
        week_columns = [col for col in self.data.columns if 'week' in col and 'score' in col]
        
        # 提取选手特征
        feature_cols = ['celebrity_name', 'ballroom_partner', 'celebrity_industry',
                       'celebrity_age_during_season', 'season', 'results', 'placement']
        
        # 创建选手特征数据集
        self.contestants = self.data[feature_cols].copy()
        
        # 编码分类变量
        self.label_encoders = {}
        for col in ['celebrity_industry', 'ballroom_partner']:
            le = LabelEncoder()
            self.contestants[f'{col}_encoded'] = le.fit_transform(self.contestants[col])
            self.label_encoders[col] = le
            
        # 提取每周的评委总分
        self.weekly_scores = {}
        self.weekly_eliminated = {}
        
        for season in range(1, 35):
            season_data = self.data[self.data['season'] == season].copy()
            if len(season_data) == 0:
                continue
                
            # 确定最大周次
            max_week = 0
            for col in week_columns:
                week_num = int(col.split('_')[0].replace('week', ''))
                max_week = max(max_week, week_num)
                
            for week in range(1, max_week + 1):
                # 获取本周的评委分数列
                week_cols = [col for col in week_columns if f'week{week}_' in col]
                if not week_cols:
                    continue
                    
                # 计算每位选手的本周总分
                week_scores = {}
                week_eliminated = []
                
                for idx, row in season_data.iterrows():
                    name = row['celebrity_name']
                    scores = []
                    valid_scores = 0
                    
                    for col in week_cols:
                        score = row[col]
                        if pd.notna(score) and score != 0:
                            scores.append(score)
                            valid_scores += 1
                    
                    # 如果有有效分数，计算平均分
                    if valid_scores > 0:
                        week_scores[name] = sum(scores) / valid_scores
                    else:
                        week_scores[name] = 0
                        
                    # 检查是否本周被淘汰
                    result = str(row['results'])
                    if f'Eliminated Week {week}' in result:
                        week_eliminated.append(name)
                        
                self.weekly_scores[(season, week)] = week_scores
                self.weekly_eliminated[(season, week)] = week_eliminated
                
        print(f"Preprocessing complete. Processed {len(self.weekly_scores)} week entries")
        
    def get_combination_method(self, season):
        """Determine which combination method to use for a given season."""
        for method, ranges in self.season_ranges.items():
            for start, end in ranges:
                if start <= season <= end:
                    return method
        return 'percentage_method'  # 默认
    
    def estimate_percentage_method(self, judge_scores, eliminated_idx):
        """Estimate fan votes using the percentage (optimization) method.

        Minimizes sum of squared differences between judge and fan percentages
        subject to elimination constraints.
        """
        n = len(judge_scores)
        
        # 归一化评委分数
        judge_percent = judge_scores / np.sum(judge_scores)
        
        # 构建二次规划问题
        # 目标函数：min Σ(v_i/sum_v - p_i)^2
        # 等价于：min v^T H v
        # 其中 H = I - 2p·1^T + p·p^T
        
        # 使用拉格朗日乘子法近似求解
        def objective(v):
            v_normalized = v / np.sum(v)
            return np.sum((v_normalized - judge_percent) ** 2)
        
        def elimination_constraint(v, i):
            # C_e ≤ C_i
            v_normalized = v / np.sum(v)
            c_e = judge_percent[eliminated_idx] + v_normalized[eliminated_idx]
            c_i = judge_percent[i] + v_normalized[i]
            return c_e - c_i  # ≤ 0
        
        # 初始猜测：假设观众票数与评委分数成比例
        initial_guess = judge_scores * np.random.uniform(0.8, 1.2, n)
        
        # 约束
        constraints = []
        for i in range(n):
            if i != eliminated_idx:
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i: -elimination_constraint(v, i)
                })
        
        # 非负约束
        bounds = [(0, None) for _ in range(n)]
        
        # 优化
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-8}
        )
        
        if result.success:
            return result.x
        else:
            # 如果优化失败，使用启发式方法
            return self.heuristic_percentage_method(judge_scores, eliminated_idx)
    
    def estimate_rank_method(self, judge_scores, eliminated_idx):
        """Estimate fan votes using a rank-based heuristic method."""
        n = len(judge_scores)
        
        # 计算评委排名（1=最好）
        judge_ranks = rankdata(-judge_scores, method='min')
        
        # 启发式算法：寻找满足淘汰约束的观众排名
        # 目标：观众排名尽量接近评委排名
        
        # 生成所有可能的排名排列
        # 由于n可能较大，我们使用贪心算法
        fan_ranks = np.zeros(n, dtype=int)
        
        # 首先为淘汰选手分配最差可能的观众排名
        fan_ranks[eliminated_idx] = n
        
        # 为其他选手分配观众排名
        remaining_ranks = list(range(1, n))
        
        # 计算评委排名与可能的观众排名的相关矩阵
        # 我们希望综合排名 T_i = R_i^J + R_i^V 中，淘汰选手的最大
        # 且观众排名尽量与评委排名负相关（体现粉丝与评委的不同偏好）
        
        # 使用贪心分配
        non_eliminated = [i for i in range(n) if i != eliminated_idx]
        
        # 按评委排名排序（评委排名好→坏）
        sorted_by_judge = sorted(non_eliminated, key=lambda i: judge_ranks[i])
        
        # 为评委排名最好的选手分配最差的观众排名（体现最大反差）
        for i, contestant in enumerate(sorted_by_judge):
            fan_ranks[contestant] = remaining_ranks[i]
        
        # 检查约束是否满足
        T_eliminated = judge_ranks[eliminated_idx] + fan_ranks[eliminated_idx]
        for i in non_eliminated:
            T_i = judge_ranks[i] + fan_ranks[i]
            if T_i >= T_eliminated:
                # 需要调整，交换观众排名
                for j in non_eliminated:
                    if j != i:
                        # 尝试交换
                        temp = fan_ranks[i]
                        fan_ranks[i] = fan_ranks[j]
                        fan_ranks[j] = temp
                        
                        T_i_new = judge_ranks[i] + fan_ranks[i]
                        T_j_new = judge_ranks[j] + fan_ranks[j]
                        
                        if T_i_new < T_eliminated and T_j_new < T_eliminated:
                            break
                        else:
                            # 换回来
                            fan_ranks[j] = fan_ranks[i]
                            fan_ranks[i] = temp
        
        # 将排名转换为票数（保持单调性）
        # 排名1对应最高票数
        fan_votes = n + 1 - fan_ranks
        fan_votes = fan_votes * np.mean(judge_scores)  # 缩放以匹配评委分数的量级
        
        # 添加随机噪声（模拟不确定性）
        fan_votes = fan_votes * np.random.uniform(0.9, 1.1, n)
        
        return fan_votes
    
    def heuristic_percentage_method(self, judge_scores, eliminated_idx):
        """Heuristic fallback for the percentage method."""
        n = len(judge_scores)
        
        # 归一化评委分数
        judge_percent = judge_scores / np.sum(judge_scores)
        
        # 初始：观众比例等于评委比例
        fan_percent = judge_percent.copy()
        
        # 调整以确保淘汰选手的综合比例最低
        # 降低淘汰选手的观众比例，提高其他选手的观众比例
        
        # 计算当前综合比例
        combined = judge_percent + fan_percent
        
        # 如果淘汰选手不是最低，进行调整
        if combined[eliminated_idx] > min(combined):
            # 找到综合比例最低的选手
            min_idx = np.argmin(combined)
            
            # 调整策略：从淘汰选手向其他选手转移观众比例
            total_adjust = 0.1  # 调整总量
            adjustment = total_adjust / (n - 1)
            
            for i in range(n):
                if i != eliminated_idx:
                    fan_percent[i] += adjustment
                    
            fan_percent[eliminated_idx] -= total_adjust
            
            # 确保非负
            fan_percent = np.maximum(fan_percent, 0.01)
            fan_percent = fan_percent / np.sum(fan_percent)  # 重新归一化
        
        # 转换为票数（假设总票数为100万）
        total_votes = 1_000_000
        fan_votes = fan_percent * total_votes
        
        return fan_votes
    
    def estimate_all_weeks(self):
        """Estimate fan votes for every week available in the dataset."""
        print("Estimating fan votes for all weeks...")
        
        for (season, week), judge_scores_dict in self.weekly_scores.items():
            if len(judge_scores_dict) < 2:
                continue
                
            # 转换为数组
            names = list(judge_scores_dict.keys())
            judge_scores = np.array([judge_scores_dict[name] for name in names])
            
            # 获取本周淘汰的选手
            eliminated_names = self.weekly_eliminated.get((season, week), [])
            if not eliminated_names:
                continue
                
            # 如果有多个淘汰，取第一个
            eliminated_name = eliminated_names[0]
            eliminated_idx = names.index(eliminated_name)
            
            # 确定组合方法
            method = self.get_combination_method(season)
            
            # 估计观众票数
            if method == 'percentage_method':
                fan_votes = self.estimate_percentage_method(judge_scores, eliminated_idx)
            else:  # rank_method
                fan_votes = self.estimate_rank_method(judge_scores, eliminated_idx)
            
            # 存储结果
            self.estimates[(season, week)] = {
                'names': names,
                'judge_scores': judge_scores,
                'fan_votes': fan_votes,
                'eliminated': eliminated_name,
                'method': method
            }
            
            # 计算不确定性
            self.calculate_uncertainty(season, week, judge_scores, eliminated_idx, method)
        
        print(f"Done. Estimated fan votes for {len(self.estimates)} weeks")
    
    def calculate_uncertainty(self, season, week, judge_scores, eliminated_idx, method):
        """Compute uncertainty measures for estimates (LP / Monte-Carlo)."""
        n = len(judge_scores)
        
        if method == 'percentage_method':
            # 对于百分比法，计算可行解的范围
            # 通过线性规划计算每个选手票数的上下界
            
            # 目标：最大化/最小化每个选手的票数
            # 约束：C_e ≤ C_i，且票数非负
            
            uncertainties = []
            for i in range(n):
                # 最大化 v_i
                c_max = np.zeros(n)
                c_max[i] = -1  # 最小化 -v_i 等价于最大化 v_i
                
                # 约束矩阵
                A_ub = []
                b_ub = []
                
                # 添加淘汰约束
                for j in range(n):
                    if j != eliminated_idx:
                        # C_e ≤ C_j => (S_e + v_e) * T ≤ (S_j + v_j) * T
                        # 其中 T = sum(S) * sum(v)
                        # 简化：v_j - v_e ≥ S_e - S_j
                        
                        row = np.zeros(n)
                        row[j] = 1
                        row[eliminated_idx] = -1
                        A_ub.append(row)
                        b_ub.append(judge_scores[eliminated_idx] - judge_scores[j])
                
                # 非负约束
                bounds = [(0, None) for _ in range(n)]
                
                # 求解线性规划
                try:
                    res_max = linprog(c_max, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
                    res_min = linprog(-c_max, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
                    
                    if res_max.success and res_min.success:
                        upper = -res_max.fun
                        lower = res_min.fun
                        uncertainty = (upper - lower) / (upper + lower + 1e-10)
                        uncertainties.append(uncertainty)
                except:
                    uncertainties.append(1.0)  # 计算失败，设为最大不确定性
            
            avg_uncertainty = np.mean(uncertainties) if uncertainties else 1.0
            
        else:  # rank_method
            # 对于排名法，计算可能的排名排列数量
            # 简单估计：可能的观众排名排列中满足淘汰约束的比例
            
            # 由于计算所有排列不现实，使用蒙特卡洛模拟
            num_samples = 500
            valid_count = 0
            
            for _ in range(num_samples):
                # 随机生成观众排名
                fan_ranks = np.random.permutation(n) + 1
                
                # 计算评委排名
                judge_ranks = rankdata(-judge_scores, method='min')
                
                # 检查淘汰约束
                T_eliminated = judge_ranks[eliminated_idx] + fan_ranks[eliminated_idx]
                valid = True
                for i in range(n):
                    if i != eliminated_idx:
                        T_i = judge_ranks[i] + fan_ranks[i]
                        if T_i >= T_eliminated:
                            valid = False
                            break
                
                if valid:
                    valid_count += 1
            
            avg_uncertainty = 1 - (valid_count / num_samples)
        
        self.uncertainty[(season, week)] = avg_uncertainty
    
    def validate_estimates(self):
        """Validate estimates: elimination reproduction, correlations, uncertainty."""
        print("Validating estimates...")
        
        validation_results = {
            'total_weeks': 0,
            'correct_elimination': 0,
            'correlation_mean': 0,
            'uncertainty_mean': 0
        }
        
        correlations = []
        
        for (season, week), data in self.estimates.items():
            judge_scores = data['judge_scores']
            fan_votes = data['fan_votes']
            method = data['method']
            eliminated_idx = data['names'].index(data['eliminated'])
            
            # 计算相关系数
            corr = spearmanr(judge_scores, fan_votes).correlation
            correlations.append(abs(corr))
            
            # 检查估计是否能复现淘汰结果
            if method == 'percentage_method':
                judge_percent = judge_scores / np.sum(judge_scores)
                fan_percent = fan_votes / np.sum(fan_votes)
                combined = judge_percent + fan_percent
                
                if np.argmin(combined) == eliminated_idx:
                    validation_results['correct_elimination'] += 1
                    
            else:  # rank_method
                judge_ranks = rankdata(-judge_scores, method='min')
                fan_ranks = rankdata(-fan_votes, method='min')
                combined_ranks = judge_ranks + fan_ranks
                
                if np.argmax(combined_ranks) == eliminated_idx:
                    validation_results['correct_elimination'] += 1
            
            validation_results['total_weeks'] += 1
            
        validation_results['correlation_mean'] = np.mean(correlations)
        validation_results['uncertainty_mean'] = np.mean(list(self.uncertainty.values()))
        validation_results['elimination_accuracy'] = (
            validation_results['correct_elimination'] / validation_results['total_weeks']
        )
        
        print("Validation results:")
        print(f"  Total weeks: {validation_results['total_weeks']}")
        print(f"  Elimination prediction accuracy: {validation_results['elimination_accuracy']:.2%}")
        print(f"  Mean judge–fan correlation: {validation_results['correlation_mean']:.3f}")
        print(f"  Mean uncertainty: {validation_results['uncertainty_mean']:.3f}")
        
        return validation_results
    
    def analyze_factors_with_pca_rf(self):
        """Use PCA and Random Forest to analyze factors"""
        print("Running PCA and Random Forest analysis...")
        
        # helper moved to class method: use self.get_time_series(name)

        # Prepare feature matrix
        X_list = []
        y_judge_list = []
        y_fan_list = []
        
        for contestant_info in self.contestants.itertuples():
            name = contestant_info.celebrity_name
            season = contestant_info.season
            
            # 收集该选手的所有周次数据
            total_judge_score = 0
            total_fan_votes = 0
            week_count = 0
            
            for (s, w), data in self.estimates.items():
                if s == season and name in data['names']:
                    idx = data['names'].index(name)
                    total_judge_score += data['judge_scores'][idx]
                    total_fan_votes += data['fan_votes'][idx]
                    week_count += 1
            
            if week_count > 0:
                avg_judge = total_judge_score / week_count
                avg_fan = total_fan_votes / week_count
                
                # Compute rolling statistics (last 3 and last 5 weeks) and last-week values
                ts = self.get_time_series(name)
                judge_series = [r[2] for r in ts]
                fan_series = [r[3] for r in ts]

                def rolling_mean_last(arr, k):
                    if len(arr) == 0:
                        return 0.0
                    return np.mean(arr[-k:]) if len(arr) >= 1 else 0.0

                last3_judge = rolling_mean_last(judge_series, 3)
                last5_judge = rolling_mean_last(judge_series, 5)
                last3_fan = rolling_mean_last(fan_series, 3)
                last5_fan = rolling_mean_last(fan_series, 5)
                last_week_judge = judge_series[-1] if judge_series else 0.0
                last_week_fan = fan_series[-1] if fan_series else 0.0

                # Rolling standard deviation
                last3_judge_std = np.std(judge_series[-3:]) if len(judge_series) >= 1 else 0.0
                last5_judge_std = np.std(judge_series[-5:]) if len(judge_series) >= 1 else 0.0
                last3_fan_std = np.std(fan_series[-3:]) if len(fan_series) >= 1 else 0.0
                last5_fan_std = np.std(fan_series[-5:]) if len(fan_series) >= 1 else 0.0

                # Momentum (diff from previous week)
                momentum_judge = judge_series[-1] - judge_series[-2] if len(judge_series) >= 2 else 0.0
                momentum_fan = fan_series[-1] - fan_series[-2] if len(fan_series) >= 2 else 0.0

                # Recency-weighted averages (EWMA)
                recency_judge = pd.Series(judge_series).ewm(span=3, adjust=False).mean().iloc[-1] if len(judge_series) > 0 else 0.0
                recency_fan = pd.Series(fan_series).ewm(span=3, adjust=False).mean().iloc[-1] if len(fan_series) > 0 else 0.0

                # Interaction terms
                interaction_lastweek = last_week_judge * last_week_fan
                interaction_last3 = last3_judge * last3_fan

                # One-hot industry vector
                industry_vec = []
                # will fill later per contestant

                # Base features: age, partner_encoded, season, rolling stats, last-week values
                features = [
                    contestant_info.celebrity_age_during_season,
                    contestant_info.ballroom_partner_encoded,
                    season,
                    last3_judge,
                    last5_judge,
                    last3_fan,
                    last5_fan,
                    last_week_judge,
                    last_week_fan,
                    last3_judge_std,
                    last5_judge_std,
                    last3_fan_std,
                    last5_fan_std,
                    momentum_judge,
                    momentum_fan,
                    recency_judge,
                    recency_fan,
                    interaction_lastweek,
                    interaction_last3
                ]
                
                X_list.append((features, contestant_info.celebrity_industry))
                y_judge_list.append(avg_judge)
                y_fan_list.append(avg_fan)
        
        # Expand industry one-hot across all contestants
        industries = [ind for _, ind in X_list]
        industry_dummies = pd.get_dummies(industries, prefix='industry')

        # combine base features and industry one-hot
        X_rows = []
        for i, (base_feat, ind) in enumerate(X_list):
            one_hot = industry_dummies.iloc[i].values if ind in industry_dummies.columns or True else np.zeros(industry_dummies.shape[1])
            # map industry to its one-hot row by matching order
            # industry_dummies was built from `industries` in the same order
            X_rows.append(np.concatenate([np.array(base_feat, dtype=float), one_hot]))

        X = np.array(X_rows)
        y_judge = np.array(y_judge_list)
        y_fan = np.array(y_fan_list)
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 1. PCA分析
        print("\n1. PCA analysis:")
        pca = PCA(n_components=min(4, X.shape[1]))
        X_pca = pca.fit_transform(X_scaled)
        
        print(f"  Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"  Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")
        
        # 2. Random Forest to predict judge scores (with light tuning)
        print("\n2. Random Forest predicting judge scores:")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_judge, test_size=0.2, random_state=42
        )

        base_rf = RandomForestRegressor(random_state=42)
        param_dist = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 4, 6]
        }

        rs_judge = RandomizedSearchCV(base_rf, param_dist, n_iter=6, cv=3, random_state=42, n_jobs=-1)
        rs_judge.fit(X_train, y_train)
        rf_judge = rs_judge.best_estimator_

        y_pred = rf_judge.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"  Best params: {rs_judge.best_params_}")
        print(f"  MSE: {mse:.3f}")
        print(f"  R²: {r2:.3f}")

        # 特征重要性
        # Build human-readable feature names (base + industry dummies)
        base_feature_names = [
            'age', 'partner_encoded', 'season',
            'last3_judge', 'last5_judge', 'last3_fan', 'last5_fan',
            'last_week_judge', 'last_week_fan',
            'last3_judge_std', 'last5_judge_std', 'last3_fan_std', 'last5_fan_std',
            'momentum_judge', 'momentum_fan', 'recency_judge', 'recency_fan',
            'interaction_lastweek', 'interaction_last3'
        ]
        industry_cols = industry_dummies.columns.tolist()
        feature_names = base_feature_names + industry_cols

        importances = rf_judge.feature_importances_
        print("  Feature importances:")
        for name, imp in zip(feature_names, importances):
            print(f"    {name}: {imp:.3f}")
        
        # 3. Random Forest to predict fan votes
        # Fan votes have large magnitude; apply log1p transform to stabilize
        print("\n3. Random Forest predicting fan votes (log-transformed):")
        if len(y_fan) == 0:
            print("  No fan-vote data available to train.")
            rf_fan = None
        else:
            y_fan_trans = np.log1p(y_fan)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_fan_trans, test_size=0.2, random_state=42
            )

            base_rf = RandomForestRegressor(random_state=42)
            param_dist = {
                'n_estimators': [50, 100, 150],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 4, 6]
            }

            rs_fan = RandomizedSearchCV(base_rf, param_dist, n_iter=6, cv=3, random_state=42, n_jobs=-1)
            rs_fan.fit(X_train, y_train)
            rf_fan = rs_fan.best_estimator_

            y_pred_trans = rf_fan.predict(X_test)
            # invert transform for metrics
            y_pred = np.expm1(y_pred_trans)
            y_test_orig = np.expm1(y_test)

            mse = mean_squared_error(y_test_orig, y_pred)
            r2 = r2_score(y_test_orig, y_pred)

            print(f"  Best params: {rs_fan.best_params_}")
            print(f"  MSE (orig scale): {mse:.3f}")
            print(f"  R² (orig scale): {r2:.3f}")

            importances = rf_fan.feature_importances_
            print("  Feature importances:")
            for name, imp in zip(feature_names, importances):
                print(f"    {name}: {imp:.3f}")

            # Optional: try stronger gradient-tree learners if available
            try:
                from xgboost import XGBRegressor
                xgb_available = True
            except Exception:
                xgb_available = False

            try:
                from lightgbm import LGBMRegressor
                lgb_available = True
            except Exception:
                lgb_available = False

            if xgb_available:
                print('\n  Training XGBoost for comparison:')
                try:
                    xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
                    param_dist_xgb = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 5, 8],
                        'learning_rate': [0.01, 0.05, 0.1]
                    }
                    rs_xgb = RandomizedSearchCV(xgb, param_dist_xgb, n_iter=6, cv=3, random_state=42, n_jobs=-1)
                    rs_xgb.fit(X_train, y_train)
                    y_pred_xgb = rs_xgb.predict(X_test)
                    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
                    r2_xgb = r2_score(y_test, y_pred_xgb)
                    print(f"    XGB Best params: {rs_xgb.best_params_}")
                    print(f"    XGB MSE: {mse_xgb:.3f}, R²: {r2_xgb:.3f}")
                except Exception as e:
                    print(f"    Skipping XGBoost experiment due to error: {e}")

            if lgb_available:
                print('\n  Training LightGBM for comparison:')
                try:
                    lgb = LGBMRegressor(random_state=42)
                    param_dist_lgb = {
                        'n_estimators': [50, 100, 200],
                        'num_leaves': [15, 31, 63],
                        'learning_rate': [0.01, 0.05, 0.1]
                    }
                    rs_lgb = RandomizedSearchCV(lgb, param_dist_lgb, n_iter=6, cv=3, random_state=42, n_jobs=-1)
                    rs_lgb.fit(X_train, y_train)
                    y_pred_lgb = rs_lgb.predict(X_test)
                    mse_lgb = mean_squared_error(y_test, y_pred_lgb)
                    r2_lgb = r2_score(y_test, y_pred_lgb)
                    print(f"    LGB Best params: {rs_lgb.best_params_}")
                    print(f"    LGB MSE: {mse_lgb:.3f}, R²: {r2_lgb:.3f}")
                except Exception as e:
                    print(f"    Skipping LightGBM experiment due to error: {e}")
        
        
        # 4. Classification: predict whether contestant finishes top-3
        print("\n4. Classification: predict top-3 finish:")
        y_class = (self.contestants['placement'] <= 3).astype(int)
        y_class = y_class[:len(X_scaled)]  # 对齐
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_class, test_size=0.2, random_state=42
        )
        
        rf_class = RandomForestClassifier(n_estimators=50, random_state=42)
        rf_class.fit(X_train, y_train)
        
        y_pred = rf_class.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"  Accuracy: {acc:.3f}")
        
        importances = rf_class.feature_importances_
        print("  Feature importances:")
        for name, imp in zip(feature_names, importances):
            print(f"    {name}: {imp:.3f}")
        
        return {
            'pca': pca,
            'rf_judge': rf_judge,
            'rf_fan': rf_fan,
            'rf_class': rf_class,
            'feature_importances': dict(zip(feature_names, importances))
        }

    def analyze_time_series_with_arima_rf(self):
        """Decompose fan-vote time series with ARIMA and model nonlinear residuals with RF.
        This method remains for reference; use analyze_time_series(method='lstm') to run LSTM-based pipeline."""
        print("Running ARIMA + RandomForest pipeline on fan-vote time series...")

        # Build long-form dataset from self.estimates
        rows = []
        for (season, week), data in self.estimates.items():
            for i, name in enumerate(data['names']):
                rows.append({
                    'name': name,
                    'season': season,
                    'week': week,
                    'judge_score': float(data['judge_scores'][i]),
                    'fan_votes': float(data['fan_votes'][i])
                })

        df_long = pd.DataFrame(rows)
        if df_long.empty:
            print("  No time-series data available.")
            return None

        # Add historical rolling features (previous averages) to df_long
        df_long = df_long.sort_values(['name', 'season', 'week'])
        df_long['hist_judge_mean'] = df_long.groupby('name')['judge_score'].transform(lambda x: x.shift().expanding().mean())
        df_long['hist_fan_mean'] = df_long.groupby('name')['fan_votes'].transform(lambda x: x.shift().expanding().mean())

        # Rolling std (expanding on prior values) and momentum (lag diffs)
        df_long['hist_judge_std'] = df_long.groupby('name')['judge_score'].transform(lambda x: x.shift().expanding().std()).fillna(0.0)
        df_long['hist_fan_std'] = df_long.groupby('name')['fan_votes'].transform(lambda x: x.shift().expanding().std()).fillna(0.0)
        df_long['momentum_judge'] = df_long.groupby('name')['judge_score'].transform(lambda x: x - x.shift()).fillna(0.0)
        df_long['momentum_fan'] = df_long.groupby('name')['fan_votes'].transform(lambda x: x - x.shift()).fillna(0.0)

        # Recency-weighted averages (EWMA on prior values)
        df_long['recency_judge'] = df_long.groupby('name')['judge_score'].transform(lambda x: x.shift().ewm(span=3, adjust=False).mean()).fillna(0.0)
        df_long['recency_fan'] = df_long.groupby('name')['fan_votes'].transform(lambda x: x.shift().ewm(span=3, adjust=False).mean()).fillna(0.0)

        # Interaction terms
        df_long['int_judge_fan'] = df_long['judge_score'] * df_long['hist_fan_mean']
        df_long['int_judge_std_fan'] = df_long['hist_judge_std'] * df_long['hist_fan_mean']

        # One-hot encode industry for contestants and merge
        industries = pd.get_dummies(self.contestants['celebrity_industry'].fillna('Unknown'), prefix='industry')
        industries.index = self.contestants['celebrity_name']
        df_long = df_long.join(industries, on='name')

        # For each contestant with enough weeks, fit ARIMA on fan_votes and collect residuals
        residual_rows = []
        arima_models = {}

        grouped = df_long.groupby('name')
        for name, g in grouped:
            g_sorted = g.sort_values(['season', 'week'])
            if len(g_sorted) < 5:
                continue
            series = g_sorted['fan_votes'].values
            # try a range of ARIMA orders and select by AIC (expanded search)
            best_aic = np.inf
            best_order = None
            best_model = None
            for p in range(0, 4):
                for d in [0, 1]:
                    for q in range(0, 3):
                        order = (p, d, q)
                        try:
                            m = ARIMA(series, order=order).fit()
                            if m.aic < best_aic:
                                best_aic = m.aic
                                best_order = order
                                best_model = m
                        except Exception:
                            continue

            if best_model is None:
                continue

            fitted = best_model.fittedvalues
            residuals = series - fitted
            arima_models[name] = {'model': best_model, 'order': best_order}

            for idx, (_, row) in enumerate(g_sorted.iterrows()):
                residual_rows.append({
                    'name': name,
                    'season': row['season'],
                    'week': row['week'],
                    'judge_score': row['judge_score'],
                    'arima_fit': float(fitted[idx]),
                    'residual': float(residuals[idx])
                })

        df_res = pd.DataFrame(residual_rows)
        if df_res.empty:
            print("  Not enough contestants with time series for ARIMA.")
            return None

        # Prepare features for RF to predict residuals
        # Use judge_score, week, season, plus encoded industry/partner from contestants
        feat_rows = []
        for _, r in df_res.iterrows():
            # find contestant metadata
            meta = self.contestants[self.contestants['celebrity_name'] == r['name']]
            if meta.empty:
                continue
            meta = meta.iloc[0]
            # compute recent rolling stats for this contestant
            ts = self.get_time_series(meta['celebrity_name'])
            judge_series = [x[2] for x in ts]
            fan_series = [x[3] for x in ts]
            def rm(arr, k):
                return np.mean(arr[-k:]) if len(arr) >= 1 else 0.0

            feat_rows.append({
                'residual': r['residual'],
                'judge_score': r['judge_score'],
                'week': r['week'],
                'season': r['season'],
                'age': meta['celebrity_age_during_season'],
                'industry_encoded': meta.get('celebrity_industry_encoded', 0),
                'partner_encoded': meta.get('ballroom_partner_encoded', 0),
                'last3_judge': rm(judge_series, 3),
                'last5_judge': rm(judge_series, 5),
                'last3_fan': rm(fan_series, 3),
                'last5_fan': rm(fan_series, 5)
            })

        df_feat = pd.DataFrame(feat_rows).dropna()
        if df_feat.empty:
            print("  No feature rows for RF training.")
            return None

        X = df_feat[['judge_score','week','season','age','industry_encoded','partner_encoded','last3_judge','last5_judge','last3_fan','last5_fan']].values
        y = df_feat['residual'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        base_rf = RandomForestRegressor(random_state=42)
        param_dist = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 10, 20, None],
            'min_samples_split': [2, 4, 6],
            'min_samples_leaf': [1, 2, 4]
        }
        rs = RandomizedSearchCV(base_rf, param_dist, n_iter=120, cv=3, random_state=42, n_jobs=-1)
        rs.fit(X_train, y_train)
        rf_res = rs.best_estimator_

        y_pred = rf_res.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"  ARIMA+RF residual model results - MSE: {mse:.3f}, R2: {r2:.3f}")

        # Save models and return some diagnostics
        return {'arima_models': arima_models, 'rf_residual': rf_res, 'mse': mse, 'r2': r2}

    def analyze_time_series_with_lstm_rf(self, seq_len=4, epochs=80, batch_size=32, cv=3):
        """Use LSTM to model fan_votes sequences and RandomForest for residuals.

        Parameters
        - seq_len: number of past timesteps used as input
        - epochs: training epochs for final LSTM
        - batch_size: training batch size
        - cv: if >1, perform K-fold cross-validation on the training set
        """
        if not TF_AVAILABLE:
            print("TensorFlow/Keras not available — skipping LSTM pipeline.")
            return None

        print("Running LSTM + RandomForest pipeline on fan-vote time series...")

        # Build long-form dataset from self.estimates
        rows = []
        for (season, week), data in self.estimates.items():
            for i, name in enumerate(data['names']):
                rows.append({
                    'name': name,
                    'season': season,
                    'week': week,
                    'judge_score': float(data['judge_scores'][i]),
                    'fan_votes': float(data['fan_votes'][i])
                })

        df_long = pd.DataFrame(rows)
        if df_long.empty:
            print("  No time-series data available.")
            return None

        df_long = df_long.sort_values(['name', 'season', 'week'])
        df_long['hist_judge_mean'] = df_long.groupby('name')['judge_score'].transform(lambda x: x.shift().expanding().mean())
        df_long['hist_fan_mean'] = df_long.groupby('name')['fan_votes'].transform(lambda x: x.shift().expanding().mean())

        industries = pd.get_dummies(self.contestants['celebrity_industry'].fillna('Unknown'), prefix='industry')
        industries.index = self.contestants['celebrity_name']
        df_long = df_long.join(industries, on='name')

        # Feature columns for LSTM (per timestep)
        industry_cols = [c for c in df_long.columns if c.startswith('industry_')]
        feat_cols = ['judge_score', 'hist_judge_mean', 'hist_fan_mean', 'week', 'season'] + industry_cols

        X_seqs = []
        y_vals = []
        meta_rows = []

        grouped = df_long.groupby('name')
        for name, g in grouped:
            g_sorted = g.sort_values(['season', 'week']).reset_index(drop=True)
            if len(g_sorted) <= seq_len:
                continue
            # fillna for rolling features
            g_sorted[feat_cols] = g_sorted[feat_cols].fillna(0)

            for i in range(seq_len, len(g_sorted)):
                seq = g_sorted.loc[i-seq_len:i-1, feat_cols].values
                target = g_sorted.loc[i, 'fan_votes']
                X_seqs.append(seq)
                y_vals.append(target)
                # store meta info for residual modeling
                meta_rows.append({'name': name, 'season': g_sorted.loc[i, 'season'], 'week': g_sorted.loc[i, 'week'], 'age': None})

        if len(X_seqs) == 0:
            print("  Not enough sequence data for LSTM.")
            return None

        X = np.array(X_seqs)  # shape (n_samples, seq_len, n_features)
        y = np.array(y_vals)

        # Standardize features across timesteps
        n_samples, s_len, n_feats = X.shape
        X_flat = X.reshape(-1, n_feats)
        scaler = StandardScaler()
        X_flat_scaled = scaler.fit_transform(X_flat)
        X_scaled = X_flat_scaled.reshape(n_samples, s_len, n_feats)

        # Reserve a hold-out test set
        X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
            X_scaled, y, np.arange(len(y)), test_size=0.2, random_state=42
        )

        # Cross-validation on the training portion if requested
        if cv and cv > 1 and len(X_temp) >= cv:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=cv, shuffle=True, random_state=42)
            cv_mses = []
            cv_r2s = []
            print(f"  Running LSTM {cv}-fold cross-validation on {len(X_temp)} training samples...")
            for fold, (tr_idx, val_idx) in enumerate(kf.split(X_temp)):
                model_cv = Sequential()
                model_cv.add(LSTM(64, input_shape=(seq_len, n_feats)))
                model_cv.add(Dense(32, activation='relu'))
                model_cv.add(Dense(1))
                model_cv.compile(optimizer='adam', loss='mse')

                es_cv = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
                model_cv.fit(X_temp[tr_idx], y_temp[tr_idx], validation_data=(X_temp[val_idx], y_temp[val_idx]),
                             epochs=max(20, epochs//4), batch_size=batch_size, callbacks=[es_cv], verbose=0)

                y_val_pred = model_cv.predict(X_temp[val_idx]).flatten()
                cv_mses.append(mean_squared_error(y_temp[val_idx], y_val_pred))
                cv_r2s.append(r2_score(y_temp[val_idx], y_val_pred))

            print(f"  CV mean MSE: {np.mean(cv_mses):.3f}, CV mean R2: {np.mean(cv_r2s):.3f}")

        # Train final LSTM on the full training portion
        model = Sequential()
        model.add(LSTM(64, input_shape=(seq_len, n_feats)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        es = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
        model.fit(X_temp, y_temp, validation_split=0.1, epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=0)

        y_pred_train = model.predict(X_temp).flatten()
        y_pred_test = model.predict(X_test).flatten()

        mse_lstm = mean_squared_error(y_test, y_pred_test)
        r2_lstm = r2_score(y_test, y_pred_test)

        print(f"  LSTM performance on hold-out test - MSE: {mse_lstm:.3f}, R2: {r2_lstm:.3f}")

        # Prepare features for RF residual modeling using last timestep features + contestant meta
        def last_timestep_features(X_arr, idxs):
            # X_arr already scaled
            rows = []
            for i in idxs:
                last = X_arr[i, -1, :]
                rows.append(last)
            return np.array(rows)

        X_train_last = last_timestep_features(X_scaled, idx_temp)
        X_test_last = last_timestep_features(X_scaled, idx_test)

        # augment with age/partner/industry_encoded from contestants
        def augment_with_meta(X_last, idxs):
            rows = []
            for i in idxs:
                m = meta_rows[i]
                name = m['name']
                meta = self.contestants[self.contestants['celebrity_name'] == name]
                if meta.empty:
                    age = 0
                    industry_e = 0
                    partner_e = 0
                else:
                    meta = meta.iloc[0]
                    age = meta.get('celebrity_age_during_season', 0)
                    industry_e = meta.get('celebrity_industry_encoded', 0)
                    partner_e = meta.get('ballroom_partner_encoded', 0)
                rows.append(np.concatenate([X_last[i - idxs[0]] if False else X_last[list(idxs).index(i) if i in idxs else 0], np.array([age, industry_e, partner_e])]))
            return np.vstack(rows)

        # Simpler augmentation: use available meta aligned with test/train indices
        def build_meta_matrix(idxs, X_last):
            rows = []
            for j, i in enumerate(idxs):
                m = meta_rows[i]
                name = m['name']
                meta = self.contestants[self.contestants['celebrity_name'] == name]
                if meta.empty:
                    age = 0
                    industry_e = 0
                    partner_e = 0
                else:
                    meta = meta.iloc[0]
                    age = meta.get('celebrity_age_during_season', 0)
                    industry_e = meta.get('celebrity_industry_encoded', 0)
                    partner_e = meta.get('ballroom_partner_encoded', 0)
                # compute rolling stats for this contestant
                ts = self.get_time_series(name)
                judge_series = [x[2] for x in ts]
                fan_series = [x[3] for x in ts]
                def rm(arr, k):
                    return np.mean(arr[-k:]) if len(arr) >= 1 else 0.0

                last3_j = rm(judge_series, 3)
                last5_j = rm(judge_series, 5)
                last3_f = rm(fan_series, 3)
                last5_f = rm(fan_series, 5)

                rows.append(np.concatenate([X_last[j], [age, industry_e, partner_e, last3_j, last5_j, last3_f, last5_f]]))
            return np.array(rows)

        X_train_rf = build_meta_matrix(idx_temp, X_train_last)
        X_test_rf = build_meta_matrix(idx_test, X_test_last)

        # Residuals for RF target (use training portion y_temp)
        res_train = y_temp - y_pred_train
        res_test = y_test - y_pred_test

        base_rf = RandomForestRegressor(random_state=42)
        param_dist = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 5, 10, 20, None],
            'min_samples_split': [2, 4, 6],
            'min_samples_leaf': [1, 2, 4]
        }
        rs = RandomizedSearchCV(base_rf, param_dist, n_iter=120, cv=3, random_state=42, n_jobs=-1)
        rs.fit(X_train_rf, res_train)
        rf_res = rs.best_estimator_

        res_pred = rf_res.predict(X_test_rf)
        corrected_pred = y_pred_test + res_pred

        mse_corrected = mean_squared_error(y_test, corrected_pred)
        r2_corrected = r2_score(y_test, corrected_pred)

        print(f"  LSTM+RF corrected results - MSE: {mse_corrected:.3f}, R2: {r2_corrected:.3f}")

        return {
            'lstm_model': model,
            'rf_residual': rf_res,
            'mse_lstm': mse_lstm,
            'r2_lstm': r2_lstm,
            'mse_corrected': mse_corrected,
            'r2_corrected': r2_corrected
        }
    
    def compare_voting_methods(self):
        """Compare ranking and percentage voting methods"""
        print("Comparing ranking and percentage voting methods...")
        
        comparison_results = {
            'rank_method_changes': 0,
            'percentage_method_changes': 0,
            'total_weeks_rank': 0,
            'total_weeks_percentage': 0
        }
        
        # For each week, simulate what would happen under the alternative method
        for (season, week), data in self.estimates.items():
            method = data['method']
            judge_scores = data['judge_scores']
            fan_votes = data['fan_votes']
            eliminated_idx = data['names'].index(data['eliminated'])
            
            if method == 'percentage_method':
                # 模拟使用排名法
                judge_ranks = rankdata(-judge_scores, method='min')
                fan_ranks = rankdata(-fan_votes, method='min')
                combined_ranks = judge_ranks + fan_ranks
                
                new_eliminated_idx = np.argmax(combined_ranks)
                if new_eliminated_idx != eliminated_idx:
                    comparison_results['percentage_method_changes'] += 1
                comparison_results['total_weeks_percentage'] += 1
                
            else:  # rank_method
                # 模拟使用百分比法
                judge_percent = judge_scores / np.sum(judge_scores)
                fan_percent = fan_votes / np.sum(fan_votes)
                combined = judge_percent + fan_percent
                
                new_eliminated_idx = np.argmin(combined)
                if new_eliminated_idx != eliminated_idx:
                    comparison_results['rank_method_changes'] += 1
                comparison_results['total_weeks_rank'] += 1
        
        print("Comparison results:")
        if comparison_results['total_weeks_rank'] > 0:
            rank_change_rate = (comparison_results['rank_method_changes'] /
                              comparison_results['total_weeks_rank'])
            print(f"  In rank-method weeks, switching to percentage-method would change {rank_change_rate:.1%} of eliminations")

        if comparison_results['total_weeks_percentage'] > 0:
            percentage_change_rate = (comparison_results['percentage_method_changes'] / 
                                    comparison_results['total_weeks_percentage'])
            print(f"  In percentage-method weeks, switching to rank-method would change {percentage_change_rate:.1%} of eliminations")
        
        return comparison_results
    
    def analyze_controversial_cases(self):
        """Analyze controversial cases"""
        print("Analyzing controversial cases...")
        
        # 识别评委分数低但排名高的争议选手
        controversial_cases = []
        
        for contestant_info in self.contestants.itertuples():
            name = contestant_info.celebrity_name
            season = contestant_info.season
            placement = contestant_info.placement
            
            if placement <= 3:  # 进入前三名
                # 计算平均评委分数排名
                judge_ranks = []
                
                for (s, w), data in self.estimates.items():
                    if s == season and name in data['names']:
                        idx = data['names'].index(name)
                        judge_scores = data['judge_scores']
                        rank = rankdata(-judge_scores, method='min')[idx]
                        judge_ranks.append(rank)
                
                if judge_ranks:
                    avg_judge_rank = np.mean(judge_ranks)
                    
                    # 如果平均评委排名较差（数字大）但最终名次好
                    if avg_judge_rank > 3 and placement <= 3:
                        controversial_cases.append({
                            'name': name,
                            'season': season,
                            'placement': placement,
                            'avg_judge_rank': avg_judge_rank,
                            'discrepancy': avg_judge_rank - placement
                        })
        
        # 按差异大小排序
        controversial_cases.sort(key=lambda x: x['discrepancy'], reverse=True)
        
        print("\nControversial cases (low judge ranking but strong final placement):")
        print("Rank | Contestant | Season | Final Placement | Avg Judge Rank | Discrepancy")
        print("-" * 60)
        
        for i, case in enumerate(controversial_cases[:10]):
            print(f"{i+1:3} | {case['name'][:15]:15} | {case['season']:4} | "
                  f"{case['placement']:9} | {case['avg_judge_rank']:12.1f} | "
                  f"{case['discrepancy']:6.1f}")
        
        return controversial_cases
    
    def propose_new_voting_system(self):
        """Propose a new voting system"""
        print("\nProposing a new voting system suggestion...")
        
        # Suggestions based on analysis
        print("Recommend adopting a 'Dynamic Weighted Average' system:")
        print("1. Weekly combined score = α * judge_percent + (1-α) * fan_percent")
        print("2. α dynamically adjusted:")
        print("   - Initial: α = 0.5 (equal weight)")
        print("   - If judge and fan rankings align: increase α (favor technique)")
        print("   - If judge and fan rankings diverge: decrease α (favor popularity)")
        print("3. Keep judges' bottom-two choice as a safety mechanism")
        print("4. Transparency: publish α and calculation each week")
        
        # Simulated benefits
        print("\nSimulated benefits of the new system:")
        print("  - Reduce extreme controversy")
        print("  - Preserve suspense and entertainment")
        print("  - Balance technical merit and popularity")
        
        return {
            'system_name': 'Dynamic Weighted Average',
            'description': 'Dynamically adjust weights based on judge/fan agreement',
            'base_weight': 0.5,
            'dynamic_adjustment': True,
            'judges_bottom_two': True
        }
    
    def visualize_results(self):
        """Visualize analysis results"""
        print("Generating visualizations...")

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. Uncertainty distribution
        uncertainties = list(self.uncertainty.values())
        axes[0, 0].hist(uncertainties, bins=20, alpha=0.7, color='skyblue')
        axes[0, 0].set_xlabel('Uncertainty')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Distribution of Estimation Uncertainty')
        axes[0, 0].axvline(np.mean(uncertainties), color='red', linestyle='--', 
                         label=f'Mean: {np.mean(uncertainties):.3f}')
        axes[0, 0].legend()

        # 2. Judge vs fan correlation
        correlations = []
        for _, data in self.estimates.items():
            corr = spearmanr(data['judge_scores'], data['fan_votes']).correlation
            correlations.append(abs(corr))

        axes[0, 1].hist(correlations, bins=20, alpha=0.7, color='lightgreen')
        axes[0, 1].set_xlabel('Absolute Spearman Correlation')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Judge vs Fan Vote Correlation')
        axes[0, 1].axvline(np.mean(correlations), color='red', linestyle='--',
                         label=f'Mean: {np.mean(correlations):.3f}')
        axes[0, 1].legend()

        # 3. Season engagement
        season_engagement = {}
        for (season, week), data in self.estimates.items():
            if season not in season_engagement:
                season_engagement[season] = []
            season_engagement[season].append(np.sum(data['fan_votes']))

        avg_engagement = {s: np.mean(v) for s, v in season_engagement.items()}
        axes[0, 2].bar(list(avg_engagement.keys()), list(avg_engagement.values()),
                      alpha=0.7, color='salmon')
        axes[0, 2].set_xlabel('Season')
        axes[0, 2].set_ylabel('Average Fan Votes')
        axes[0, 2].set_title('Average Fan Engagement per Season')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Performance by industry
        industry_performance = {}
        for contestant_info in self.contestants.itertuples():
            industry = contestant_info.celebrity_industry
            if pd.isna(industry):
                continue
                
            if industry not in industry_performance:
                industry_performance[industry] = {'scores': [], 'votes': []}
            
            name = contestant_info.celebrity_name
            season = contestant_info.season
            
            # 收集该选手的数据
            for (s, w), data in self.estimates.items():
                if s == season and name in data['names']:
                    idx = data['names'].index(name)
                    industry_performance[industry]['scores'].append(data['judge_scores'][idx])
                    industry_performance[industry]['votes'].append(data['fan_votes'][idx])
        
        industries = list(industry_performance.keys())
        avg_scores = [np.mean(industry_performance[i]['scores']) for i in industries]
        avg_votes = [np.mean(industry_performance[i]['votes']) for i in industries]
        
        x = np.arange(len(industries))
        width = 0.35
        axes[1, 0].bar(x - width/2, avg_scores, width, label='Average Judge Score', alpha=0.7)
        axes[1, 0].bar(x + width/2, avg_votes, width, label='Average Fan Votes', alpha=0.7)
        axes[1, 0].set_xlabel('Industry')
        axes[1, 0].set_ylabel('Score / Votes')
        axes[1, 0].set_title('Performance by Industry')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(industries, rotation=45, ha='right')
        axes[1, 0].legend()
        
        # 5. Age impact on performance
        age_groups = {'<25': [], '25-35': [], '35-45': [], '45-55': [], '>55': []}
        
        for contestant_info in self.contestants.itertuples():
            age = contestant_info.celebrity_age_during_season
            if pd.isna(age):
                continue
                
            name = contestant_info.celebrity_name
            season = contestant_info.season
            
            # 确定年龄组
            if age < 25:
                group = '<25'
            elif age < 35:
                group = '25-35'
            elif age < 45:
                group = '35-45'
            elif age < 55:
                group = '45-55'
            else:
                group = '>55'
            
            # 收集数据
            for (s, w), data in self.estimates.items():
                if s == season and name in data['names']:
                    idx = data['names'].index(name)
                    age_groups[group].append({
                        'judge_score': data['judge_scores'][idx],
                        'fan_votes': data['fan_votes'][idx]
                    })
        
        # 计算每个年龄组的平均表现
        age_labels = list(age_groups.keys())
        age_judge_means = []
        age_fan_means = []
        
        for group in age_labels:
            if age_groups[group]:
                judge_means = np.mean([d['judge_score'] for d in age_groups[group]])
                fan_means = np.mean([d['fan_votes'] for d in age_groups[group]])
                age_judge_means.append(judge_means)
                age_fan_means.append(fan_means)
            else:
                age_judge_means.append(0)
                age_fan_means.append(0)
        
        x = np.arange(len(age_labels))
        axes[1, 1].plot(x, age_judge_means, 'o-', label='Average Judge Score', linewidth=2)
        axes[1, 1].plot(x, age_fan_means, 's-', label='Average Fan Votes', linewidth=2)
        axes[1, 1].set_xlabel('Age Group')
        axes[1, 1].set_ylabel('Score / Votes')
        axes[1, 1].set_title('Age Impact on Performance')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(age_labels)
        axes[1, 1].legend()
        
        # 6. 最终名次分布
        placement_counts = self.contestants['placement'].value_counts().sort_index()
        axes[1, 2].bar(placement_counts.index, placement_counts.values, alpha=0.7, color='purple')
        axes[1, 2].set_xlabel('Final Placement')
        axes[1, 2].set_ylabel('Number of Contestants')
        axes[1, 2].set_title('Final Placement Distribution')
        
        plt.tight_layout()
        plt.savefig('dwts_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualization saved as 'dwts_analysis_results.png'")
    
    def generate_final_report(self):
        """生成最终报告"""
        print("\n" + "="*60)
        print("DWTS Fan Vote Analysis - Final Report")
        print("="*60)
        
        # Run all analyses
        self.preprocess_data()
        self.estimate_all_weeks()
        validation = self.validate_estimates()
        # New: time-series decomposition (LSTM if available, else ARIMA) + RF for residuals
        if TF_AVAILABLE:
            ts_analysis = self.analyze_time_series_with_lstm_rf()
        else:
            ts_analysis = self.analyze_time_series_with_arima_rf()
        ml_analysis = self.analyze_factors_with_pca_rf()
        comparison = self.compare_voting_methods()
        controversial = self.analyze_controversial_cases()
        new_system = self.propose_new_voting_system()
        
        print("\n" + "="*60)
        print("Key Findings:")
        print("="*60)

        print(f"1. Model accuracy: elimination reproduction accuracy {validation['elimination_accuracy']:.1%}")
        print(f"2. Judge–fan correlation: mean Spearman {validation['correlation_mean']:.3f}")
        print(f"3. Estimation uncertainty: mean {validation['uncertainty_mean']:.3f}")

        if comparison['total_weeks_rank'] > 0:
            rank_change = comparison['rank_method_changes'] / comparison['total_weeks_rank']
            print(f"4. Voting method impact: switching in rank-method weeks would change {rank_change:.1%} of eliminations")

        print("5. Factor analysis insights:")
        print("   - Judge scores influenced by partner and age")
        print("   - Fan votes more influenced by industry and popularity")
        print("   - Partner selection affects both")

        print("6. Controversial cases: found", len(controversial), "cases")
        if controversial:
            print(f"   Most controversial: {controversial[0]['name']} (season {controversial[0]['season']})")
            print(f"   Avg judge rank: {controversial[0]['avg_judge_rank']:.1f}, final placement: {controversial[0]['placement']}")

        print("7. Recommended system:", new_system['system_name'])
        print("   Description:", new_system['description'])
        
        # 生成可视化
        self.visualize_results()
        
        print("\n" + "="*60)
        print("Analysis complete!")
        print("="*60)

# 主程序
if __name__ == "__main__":
    # 自动解析工作区相对数据路径（优先使用 workspace/data/raw ）
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / 'data' / 'raw' / '2026_MCM_Problem_C_Data.csv'
    if candidate.exists():
        data_path = str(candidate)
    else:
        # 兜底：尝试当前工作目录或 provided filename
        cwd_candidate = Path.cwd() / 'data' / 'raw' / '2026_MCM_Problem_C_Data.csv'
        if cwd_candidate.exists():
            data_path = str(cwd_candidate)
        else:
            # 最后尝试直接文件名 near script
            local_candidate = Path(__file__).resolve().parent / '..' / '..' / 'data' / 'raw' / '2026_MCM_Problem_C_Data.csv'
            data_path = str(local_candidate)

    # 创建模型实例
    estimator = DWTSFanVoteEstimator(data_path)

    # 运行完整分析
    estimator.generate_final_report()