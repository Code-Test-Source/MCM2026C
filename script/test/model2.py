"""
Comprehensive DWTS Analysis Framework with Advanced Models
Feature Engineering, Predictive Modeling, and Clustering Analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize, linalg
from scipy.signal import find_peaks
from scipy.spatial.distance import pdist, squareform, cdist
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

# Statistical Analysis
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.gam.api import GLMGam, BSplines
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
try:
    from statsmodels.discrete.discrete_model import LogisticRegression as StatsLogistic
except Exception:
    from statsmodels.discrete.discrete_model import Logit as StatsLogistic
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

# Machine Learning
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression, LogisticRegression, ElasticNet, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    BaggingRegressor, BaggingClassifier,
    AdaBoostRegressor, AdaBoostClassifier
)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, precision_score, recall_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
from sklearn.covariance import EllipticEnvelope

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, LSTM, GRU, Dropout, BatchNormalization,
    Input, Conv1D, MaxPooling1D, Flatten,
    Bidirectional, Attention, Multiply
)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
tf.random.set_seed(42)

# Try to import DWTSFanVoteEstimator from model1.py so model2 can run the same end-to-end pipeline
try:
    from script.test.model1 import DWTSFanVoteEstimator
except Exception:
    try:
        import importlib.util, sys, os
        spec = importlib.util.spec_from_file_location("model1", os.path.join(os.path.dirname(__file__), "model1.py"))
        model1 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model1)
        DWTSFanVoteEstimator = getattr(model1, 'DWTSFanVoteEstimator', None)
    except Exception:
        DWTSFanVoteEstimator = None

# Bayesian Methods
try:
    import pymc3 as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    print("PyMC3 not available. Bayesian methods will use alternative implementations.")

class AdvancedFeatureEngineering:
    """
    Comprehensive feature engineering for DWTS data analysis
    """
    
    def __init__(self, data):
        self.data = data
        self.features = None
        self.feature_names = []
        
    def extract_all_features(self):
        """
        Extract comprehensive feature set
        """
        features_dict = {}
        
        # 1. Basic demographic features
        features_dict.update(self._extract_demographic_features())
        
        # 2. Performance time-series features
        features_dict.update(self._extract_performance_features())
        
        # 3. Statistical moment features
        features_dict.update(self._extract_statistical_features())
        
        # 4. Temporal pattern features
        features_dict.update(self._extract_temporal_features())
        
        # 5. Season and context features
        features_dict.update(self._extract_contextual_features())
        
        # 6. Partner and pairing features
        features_dict.update(self._extract_pairing_features())
        
        # 7. Competition dynamics features
        features_dict.update(self._extract_competition_features())
        
        # Combine all features
        self.features = pd.DataFrame(features_dict)
        self.feature_names = list(features_dict.keys())
        
        return self.features
    
    def _extract_demographic_features(self):
        """Extract demographic and background features"""
        features = {}
        
        # Age features
        if 'age' in self.data.columns:
            features['age'] = self.data['age']
            features['age_group'] = pd.cut(self.data['age'], 
                                          bins=[0, 25, 35, 45, 100],
                                          labels=['young', 'young_adult', 'adult', 'senior'])
        
        # Industry encoding
        if 'industry' in self.data.columns:
            le = LabelEncoder()
            features['industry_encoded'] = le.fit_transform(self.data['industry'].fillna('Unknown'))
        
        # Gender features (if available)
        if 'gender' in self.data.columns:
            features['is_male'] = (self.data['gender'] == 'male').astype(int)
            features['is_female'] = (self.data['gender'] == 'female').astype(int)
        
        # Experience features
        if 'prior_experience' in self.data.columns:
            features['has_dance_experience'] = (self.data['prior_experience'] > 0).astype(int)
            features['experience_level'] = pd.cut(self.data['prior_experience'],
                                                 bins=[-1, 0, 2, 5, 100],
                                                 labels=['none', 'beginner', 'intermediate', 'expert'])
        
        return features
    
    def _extract_performance_features(self):
        """Extract performance-related features"""
        features = {}
        
        # Find score columns
        score_columns = [col for col in self.data.columns if 'score' in col.lower()]
        
        if score_columns:
            all_scores = []
            for idx, row in self.data.iterrows():
                scores = []
                for col in sorted(score_columns):
                    score = row[col]
                    if pd.notna(score) and score > 0:
                        scores.append(score)
                all_scores.append(scores)
            
            # Basic statistics
            features['mean_score'] = [np.mean(s) if s else 0 for s in all_scores]
            features['median_score'] = [np.median(s) if s else 0 for s in all_scores]
            features['std_score'] = [np.std(s) if len(s) > 1 else 0 for s in all_scores]
            features['max_score'] = [np.max(s) if s else 0 for s in all_scores]
            features['min_score'] = [np.min(s) if s else 0 for s in all_scores]
            features['score_range'] = [np.ptp(s) if s else 0 for s in all_scores]
            
            # Higher order statistics
            features['skewness'] = [stats.skew(s) if len(s) > 2 else 0 for s in all_scores]
            features['kurtosis'] = [stats.kurtosis(s) if len(s) > 3 else 0 for s in all_scores]
            features['coefficient_variation'] = [
                np.std(s)/np.mean(s) if np.mean(s) > 0 else 0 for s in all_scores
            ]
            
            # Trend features
            for i, scores in enumerate(all_scores):
                if len(scores) >= 3:
                    # Linear trend
                    x = np.arange(len(scores))
                    slope, intercept = np.polyfit(x, scores, 1)
                    features.setdefault('trend_slope', []).append(slope)
                    features.setdefault('trend_intercept', []).append(intercept)
                    
                    # Quadratic trend
                    if len(scores) >= 4:
                        coeffs = np.polyfit(x, scores, 2)
                        features.setdefault('quadratic_a', []).append(coeffs[0])
                        features.setdefault('quadratic_b', []).append(coeffs[1])
                    else:
                        features.setdefault('quadratic_a', []).append(0)
                        features.setdefault('quadratic_b', []).append(0)
                        
                    # Improvement rate
                    first_half = scores[:len(scores)//2]
                    second_half = scores[len(scores)//2:]
                    if first_half and second_half:
                        features.setdefault('improvement_rate', []).append(
                            np.mean(second_half) - np.mean(first_half)
                        )
                    else:
                        features.setdefault('improvement_rate', []).append(0)
                else:
                    features.setdefault('trend_slope', []).append(0)
                    features.setdefault('trend_intercept', []).append(0)
                    features.setdefault('quadratic_a', []).append(0)
                    features.setdefault('quadratic_b', []).append(0)
                    features.setdefault('improvement_rate', []).append(0)
        
        return features
    
    def _extract_statistical_features(self):
        """Extract statistical moment and distribution features"""
        features = {}
        
        score_columns = [col for col in self.data.columns if 'score' in col.lower()]
        
        if score_columns:
            for idx, row in self.data.iterrows():
                scores = []
                for col in score_columns:
                    score = row[col]
                    if pd.notna(score) and score > 0:
                        scores.append(score)
                
                if scores:
                    # Percentiles
                    for p in [25, 50, 75, 90]:
                        features.setdefault(f'percentile_{p}', []).append(np.percentile(scores, p))
                    
                    # Robust statistics
                    features.setdefault('iqr', []).append(stats.iqr(scores))
                    features.setdefault('mad', []).append(stats.median_abs_deviation(scores))
                    
                    # Normality tests
                    if len(scores) >= 8:
                        _, p_value = stats.shapiro(scores)
                        features.setdefault('shapiro_p', []).append(p_value)
                        features.setdefault('is_normal', []).append(int(p_value > 0.05))
                    else:
                        features.setdefault('shapiro_p', []).append(np.nan)
                        features.setdefault('is_normal', []).append(np.nan)
                else:
                    for p in [25, 50, 75, 90]:
                        features.setdefault(f'percentile_{p}', []).append(0)
                    features.setdefault('iqr', []).append(0)
                    features.setdefault('mad', []).append(0)
                    features.setdefault('shapiro_p', []).append(np.nan)
                    features.setdefault('is_normal', []).append(np.nan)
        
        return features
    
    def _extract_temporal_features(self):
        """Extract time-series and temporal pattern features"""
        features = {}
        
        score_columns = [col for col in self.data.columns if 'score' in col.lower()]
        
        if score_columns:
            for idx, row in self.data.iterrows():
                scores = []
                for col in sorted(score_columns):
                    score = row[col]
                    if pd.notna(score) and score > 0:
                        scores.append(score)
                
                if len(scores) >= 3:
                    # Autocorrelation features
                    lag1_autocorr = np.corrcoef(scores[:-1], scores[1:])[0, 1] if len(scores) > 1 else 0
                    features.setdefault('autocorr_lag1', []).append(lag1_autocorr)
                    
                    # Peak detection
                    peaks, properties = find_peaks(scores, prominence=1)
                    features.setdefault('num_peaks', []).append(len(peaks))
                    features.setdefault('peak_prominence_mean', []).append(
                        np.mean(properties['prominences']) if len(peaks) > 0 else 0
                    )
                    
                    # Volatility
                    returns = np.diff(scores) / scores[:-1]
                    features.setdefault('volatility', []).append(np.std(returns) if len(returns) > 0 else 0)
                    
                    # Seasonality (if enough data)
                    if len(scores) >= 6:
                        try:
                            # Simple seasonal decomposition
                            seasonal_period = min(3, len(scores)//2)
                            seasonal_component = []
                            for i in range(len(scores)):
                                idx_range = list(range(max(0, i-seasonal_period), 
                                                      min(len(scores), i+seasonal_period+1)))
                                if idx_range:
                                    seasonal_component.append(scores[i] - np.mean([scores[j] for j in idx_range]))
                            features.setdefault('seasonality_strength', []).append(np.std(seasonal_component))
                        except:
                            features.setdefault('seasonality_strength', []).append(0)
                    else:
                        features.setdefault('seasonality_strength', []).append(0)
                else:
                    features.setdefault('autocorr_lag1', []).append(0)
                    features.setdefault('num_peaks', []).append(0)
                    features.setdefault('peak_prominence_mean', []).append(0)
                    features.setdefault('volatility', []).append(0)
                    features.setdefault('seasonality_strength', []).append(0)
        
        return features
    
    def _extract_contextual_features(self):
        """Extract season and competition context features"""
        features = {}
        
        # Season features
        if 'season' in self.data.columns:
            features['season_number'] = self.data['season']
            features['season_quarter'] = self.data['season'] % 4  # Assuming 4 seasons per year
        
        # Week features
        if 'week' in self.data.columns:
            features['week_number'] = self.data['week']
            features['is_final_week'] = (self.data['week'] >= 10).astype(int)
        
        # Result features
        if 'placement' in self.data.columns:
            features['placement'] = self.data['placement']
            features['is_winner'] = (self.data['placement'] == 1).astype(int)
            features['is_finalist'] = (self.data['placement'] <= 3).astype(int)
            features['was_eliminated_early'] = (self.data['placement'] > 8).astype(int)
        
        return features
    
    def _extract_pairing_features(self):
        """Extract partner and pairing features"""
        features = {}
        
        # Partner encoding
        if 'partner' in self.data.columns:
            le = LabelEncoder()
            features['partner_id'] = le.fit_transform(self.data['partner'].fillna('Unknown'))
            
            # Partner experience (if available)
            partner_counts = self.data['partner'].value_counts()
            features['partner_experience'] = self.data['partner'].map(partner_counts)
        
        # Gender pairing (if both gender and partner gender available)
        if 'gender' in self.data.columns and 'partner_gender' in self.data.columns:
            features['same_gender_pair'] = (
                self.data['gender'] == self.data['partner_gender']
            ).astype(int)
        
        return features
    
    def _extract_competition_features(self):
        """Extract competition dynamics features"""
        features = {}
        
        # Competition intensity (based on scores of other contestants)
        score_columns = [col for col in self.data.columns if 'score' in col.lower()]
        
        if score_columns and 'season' in self.data.columns and 'week' in self.data.columns:
            competition_intensity = []
            for idx, row in self.data.iterrows():
                season = row['season']
                week = row['week']
                
                # Get all contestants in same season and week
                same_week = self.data[
                    (self.data['season'] == season) & 
                    (self.data['week'] == week)
                ]
                
                if len(same_week) > 1:
                    # Calculate average score of competitors
                    competitor_scores = []
                    for _, comp_row in same_week.iterrows():
                        if comp_row.name != idx:  # Exclude self
                            scores = []
                            for col in score_columns:
                                score = comp_row[col]
                                if pd.notna(score) and score > 0:
                                    scores.append(score)
                            if scores:
                                competitor_scores.append(np.mean(scores))
                    
                    if competitor_scores:
                        competition_intensity.append(np.mean(competitor_scores))
                    else:
                        competition_intensity.append(0)
                else:
                    competition_intensity.append(0)
            
            features['competition_intensity'] = competition_intensity
        
        return features

class StatisticalAnalysisModels:
    """
    Statistical analysis models for DWTS data
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def generalized_linear_regression(self, X, y, family='gaussian', link='identity'):
        """
        Generalized Linear Regression
        """
        try:
            if family == 'gaussian' and link == 'identity':
                # Use OLS for Gaussian family with identity link
                X_with_const = sm.add_constant(X)
                model = sm.GLM(y, X_with_const, family=sm.families.Gaussian())
                result = model.fit()
                
                self.models['glm'] = model
                self.results['glm'] = {
                    'params': result.params,
                    'pvalues': result.pvalues,
                    'aic': result.aic,
                    'bic': result.bic,
                    'llf': result.llf,
                    'resid': result.resid_response,
                    'fitted': result.fittedvalues,
                    'summary': result.summary()
                }
                
                return self.results['glm']
            else:
                print(f"GLM family {family} with link {link} not implemented")
                return None
        except Exception as e:
            print(f"GLM error: {e}")
            return None
    
    def principal_component_analysis(self, X, n_components=None, kernel=None):
        """
        Principal Component Analysis with kernel options
        """
        if n_components is None:
            n_components = min(X.shape[0], X.shape[1])
        
        if kernel is None:
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X)
            
            self.models['pca'] = pca
            self.results['pca'] = {
                'components': pca.components_,
                'explained_variance': pca.explained_variance_,
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'transformed_data': X_pca,
                'singular_values': pca.singular_values_
            }
        else:
            # Kernel PCA
            kpca = KernelPCA(n_components=n_components, kernel=kernel)
            X_pca = kpca.fit_transform(X)
            
            self.models['kpca'] = kpca
            self.results['kpca'] = {
                'transformed_data': X_pca,
                'kernel': kernel,
                'n_components': n_components
            }
        
        return self.results['pca' if kernel is None else 'kpca']
    
    def arima_model(self, time_series, order=(1, 0, 1), seasonal_order=(0, 0, 0, 0)):
        """
        ARIMA model for time series analysis
        """
        try:
            model = ARIMA(time_series, order=order, seasonal_order=seasonal_order)
            model_fit = model.fit()
            
            # Forecast
            forecast_steps = min(5, len(time_series))
            forecast = model_fit.forecast(steps=forecast_steps)
            
            # Diagnostics
            residuals = model_fit.resid
            
            self.models['arima'] = model_fit
            self.results['arima'] = {
                'model': model_fit,
                'forecast': forecast,
                'residuals': residuals,
                'aic': model_fit.aic,
                'bic': model_fit.bic,
                'params': model_fit.params,
                'conf_int': model_fit.conf_int()
            }
            
            return self.results['arima']
        except Exception as e:
            print(f"ARIMA error: {e}")
            return None
    
    def bayesian_linear_regression(self, X, y, n_samples=2000, tune=1000):
        """
        Bayesian Linear Regression using PyMC3
        """
        if not PYMC_AVAILABLE:
            print("PyMC3 not available for Bayesian methods")
            return None
        
        try:
            n_features = X.shape[1]
            
            with pm.Model() as model:
                # Priors
                alpha = pm.Normal('alpha', mu=0, sigma=10)
                betas = pm.Normal('betas', mu=0, sigma=10, shape=n_features)
                sigma = pm.HalfNormal('sigma', sigma=1)
                
                # Expected value
                mu = alpha + pm.math.dot(X, betas)
                
                # Likelihood
                likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y)
                
                # Sample from posterior
                trace = pm.sample(n_samples, tune=tune, chains=2, cores=1, progressbar=False)
                
                # Posterior predictive
                posterior_predictive = pm.sample_posterior_predictive(trace, samples=500, model=model)
            
            self.models['bayesian_lr'] = model
            self.results['bayesian_lr'] = {
                'trace': trace,
                'posterior_predictive': posterior_predictive,
                'summary': az.summary(trace)
            }
            
            return self.results['bayesian_lr']
        except Exception as e:
            print(f"Bayesian regression error: {e}")
            return None
    
    def breakpoint_regression(self, X, y, max_breakpoints=3):
        """
        Breakpoint (Segmented) Regression
        """
        try:
            # Sort by X for breakpoint detection
            sort_idx = np.argsort(X.flatten())
            X_sorted = X[sort_idx]
            y_sorted = y[sort_idx]
            
            # Find optimal breakpoints using grid search
            n_points = len(X_sorted)
            best_r2 = -np.inf
            best_breakpoints = []
            best_models = []
            
            # Simple grid search for breakpoints
            if max_breakpoints > 0:
                for bp1 in range(1, n_points-2):
                    if max_breakpoints >= 1:
                        # Single breakpoint
                        X1 = X_sorted[:bp1]
                        y1 = y_sorted[:bp1]
                        X2 = X_sorted[bp1:]
                        y2 = y_sorted[bp1:]
                        
                        model1 = LinearRegression().fit(X1, y1)
                        model2 = LinearRegression().fit(X2, y2)
                        
                        y_pred = np.concatenate([model1.predict(X1), model2.predict(X2)])
                        r2 = r2_score(y_sorted, y_pred)
                        
                        if r2 > best_r2:
                            best_r2 = r2
                            best_breakpoints = [bp1]
                            best_models = [model1, model2]
            
            self.models['breakpoint'] = best_models
            self.results['breakpoint'] = {
                'breakpoints': best_breakpoints,
                'r2': best_r2,
                'models': best_models
            }
            
            return self.results['breakpoint']
        except Exception as e:
            print(f"Breakpoint regression error: {e}")
            return None

class MachineLearningModels:
    """
    Machine learning models for DWTS analysis
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def bp_neural_network(self, X_train, X_test, y_train, y_test, 
                          hidden_layers=(64, 32, 16), epochs=100):
        """
        Backpropagation Neural Network
        """
        # Build model
        model = Sequential()
        model.add(Dense(hidden_layers[0], activation='relu', input_shape=(X_train.shape[1],)))
        
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(0.3))
        
        model.add(Dense(1))
        
        # Compile
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='mse',
                     metrics=['mae'])
        
        # Train
        history = model.fit(X_train, y_train,
                          epochs=epochs,
                          batch_size=32,
                          validation_split=0.2,
                          verbose=0,
                          callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])
        
        # Predict
        y_pred = model.predict(X_test).flatten()
        
        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.models['bp_nn'] = model
        self.results['bp_nn'] = {
            'model': model,
            'history': history.history,
            'y_pred': y_pred,
            'mse': mse,
            'r2': r2,
            'y_test': y_test
        }
        
        return self.results['bp_nn']
    
    def dbscan_clustering(self, X, eps=None, min_samples=5):
        """
        DBSCAN clustering with automatic parameter tuning
        """
        from sklearn.neighbors import NearestNeighbors
        
        # Estimate eps if not provided
        if eps is None:
            # Compute k-distance graph
            neighbors = NearestNeighbors(n_neighbors=min_samples)
            neighbors_fit = neighbors.fit(X)
            distances, _ = neighbors_fit.kneighbors(X)
            
            # Sort k-distances
            k_distances = np.sort(distances[:, min_samples-1])
            
            # Find elbow point
            elbow_idx = self._find_elbow_point(k_distances)
            eps = k_distances[elbow_idx]
        
        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        
        # Calculate metrics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # Silhouette score (excluding noise)
        if n_clusters > 1:
            mask = labels != -1
            if np.sum(mask) > 1:
                silhouette = silhouette_score(X[mask], labels[mask])
            else:
                silhouette = -1
        else:
            silhouette = -1
        
        self.models['dbscan'] = dbscan
        self.results['dbscan'] = {
            'model': dbscan,
            'labels': labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette_score': silhouette,
            'eps': eps,
            'min_samples': min_samples
        }
        
        return self.results['dbscan']
    
    def _find_elbow_point(self, values):
        """Find elbow point in curve"""
        # Normalize
        values_norm = (values - values.min()) / (values.max() - values.min())
        
        # Create line from first to last point
        line = np.linspace(0, 1, len(values_norm))
        
        # Calculate distances
        distances = np.abs(values_norm - line)
        
        return np.argmax(distances)
    
    def decision_tree_ensemble(self, X_train, X_test, y_train, y_test, 
                               model_type='regression', n_estimators=100):
        """
        Decision tree ensemble (Random Forest, Gradient Boosting, etc.)
        """
        if model_type == 'regression':
            # Random Forest
            rf_model = RandomForestRegressor(n_estimators=n_estimators,
                                            random_state=42,
                                            n_jobs=-1)
            rf_model.fit(X_train, y_train)
            rf_pred = rf_model.predict(X_test)
            rf_mse = mean_squared_error(y_test, rf_pred)
            rf_r2 = r2_score(y_test, rf_pred)
            
            # Gradient Boosting
            gb_model = GradientBoostingRegressor(n_estimators=n_estimators,
                                                random_state=42)
            gb_model.fit(X_train, y_train)
            gb_pred = gb_model.predict(X_test)
            gb_mse = mean_squared_error(y_test, gb_pred)
            gb_r2 = r2_score(y_test, gb_pred)
            
            # Bagging
            bag_model = BaggingRegressor(estimator=DecisionTreeRegressor(),
                                        n_estimators=n_estimators,
                                        random_state=42)
            bag_model.fit(X_train, y_train)
            bag_pred = bag_model.predict(X_test)
            bag_mse = mean_squared_error(y_test, bag_pred)
            bag_r2 = r2_score(y_test, bag_pred)
            
            self.models['tree_ensemble'] = {
                'random_forest': rf_model,
                'gradient_boosting': gb_model,
                'bagging': bag_model
            }
            
            self.results['tree_ensemble'] = {
                'rf': {'mse': rf_mse, 'r2': rf_r2, 'y_pred': rf_pred},
                'gb': {'mse': gb_mse, 'r2': gb_r2, 'y_pred': gb_pred},
                'bag': {'mse': bag_mse, 'r2': bag_r2, 'y_pred': bag_pred}
            }
            
        else:  # classification
            # Random Forest
            rf_model = RandomForestClassifier(n_estimators=n_estimators,
                                             random_state=42,
                                             n_jobs=-1)
            rf_model.fit(X_train, y_train)
            rf_pred = rf_model.predict(X_test)
            rf_acc = accuracy_score(y_test, rf_pred)
            
            # Gradient Boosting
            gb_model = GradientBoostingClassifier(n_estimators=n_estimators,
                                                 random_state=42)
            gb_model.fit(X_train, y_train)
            gb_pred = gb_model.predict(X_test)
            gb_acc = accuracy_score(y_test, gb_pred)
            
            self.models['tree_ensemble'] = {
                'random_forest': rf_model,
                'gradient_boosting': gb_model
            }
            
            self.results['tree_ensemble'] = {
                'rf': {'accuracy': rf_acc, 'y_pred': rf_pred},
                'gb': {'accuracy': gb_acc, 'y_pred': gb_pred}
            }
        
        return self.results['tree_ensemble']
    
    def lstm_model(self, X_train, X_test, y_train, y_test, 
                   sequence_length=5, lstm_units=50):
        """
        LSTM model for sequential data
        """
        # Reshape data for LSTM
        n_features = X_train.shape[1]
        
        # Ensure sequence length divides features
        if n_features % sequence_length != 0:
            # Pad features
            pad_length = sequence_length - (n_features % sequence_length)
            X_train_padded = np.pad(X_train, ((0, 0), (0, pad_length)), 'constant')
            X_test_padded = np.pad(X_test, ((0, 0), (0, pad_length)), 'constant')
            n_features = X_train_padded.shape[1]
        else:
            X_train_padded = X_train
            X_test_padded = X_test
        
        # Reshape
        X_train_lstm = X_train_padded.reshape(-1, sequence_length, n_features // sequence_length)
        X_test_lstm = X_test_padded.reshape(-1, sequence_length, n_features // sequence_length)
        
        # Build LSTM model
        model = Sequential([
            LSTM(lstm_units, return_sequences=True, 
                 input_shape=(sequence_length, n_features // sequence_length)),
            Dropout(0.3),
            LSTM(lstm_units),
            Dropout(0.3),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        # Compile
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='mse',
                     metrics=['mae'])
        
        # Train
        history = model.fit(X_train_lstm, y_train,
                          epochs=50,
                          batch_size=16,
                          validation_split=0.2,
                          verbose=0,
                          callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])
        
        # Predict
        y_pred = model.predict(X_test_lstm).flatten()
        
        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.models['lstm'] = model
        self.results['lstm'] = {
            'model': model,
            'history': history.history,
            'y_pred': y_pred,
            'mse': mse,
            'r2': r2
        }
        
        return self.results['lstm']
    
    def gaussian_mixture_model(self, X, n_components_range=range(2, 10)):
        """
        Gaussian Mixture Model clustering
        """
        best_gmm = None
        best_bic = np.inf
        best_n = 2
        
        # Find optimal number of components
        bic_scores = []
        aic_scores = []
        
        for n_components in n_components_range:
            gmm = GaussianMixture(n_components=n_components,
                                 covariance_type='full',
                                 random_state=42)
            gmm.fit(X)
            
            bic = gmm.bic(X)
            aic = gmm.aic(X)
            
            bic_scores.append(bic)
            aic_scores.append(aic)
            
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
                best_n = n_components
        
        # Fit with optimal components
        labels = best_gmm.predict(X)
        probabilities = best_gmm.predict_proba(X)
        
        # Calculate metrics
        if best_n > 1:
            silhouette = silhouette_score(X, labels)
        else:
            silhouette = -1
        
        self.models['gmm'] = best_gmm
        self.results['gmm'] = {
            'model': best_gmm,
            'labels': labels,
            'probabilities': probabilities,
            'n_components': best_n,
            'bic_scores': bic_scores,
            'aic_scores': aic_scores,
            'silhouette_score': silhouette
        }
        
        return self.results['gmm']
    
    def multiple_linear_regression(self, X_train, X_test, y_train, y_test, 
                                   regularization=None, alpha=1.0):
        """
        Multiple Linear Regression with optional regularization
        """
        if regularization is None:
            model = LinearRegression()
        elif regularization == 'ridge':
            model = Ridge(alpha=alpha)
        elif regularization == 'lasso':
            model = Lasso(alpha=alpha)
        elif regularization == 'elastic':
            model = ElasticNet(alpha=alpha, l1_ratio=0.5)
        else:
            model = LinearRegression()
        
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Feature importance
        if hasattr(model, 'coef_'):
            feature_importance = np.abs(model.coef_)
        else:
            feature_importance = None
        
        self.models['mlr'] = model
        self.results['mlr'] = {
            'model': model,
            'y_pred': y_pred,
            'mse': mse,
            'r2': r2,
            'feature_importance': feature_importance,
            'coefficients': model.coef_ if hasattr(model, 'coef_') else None
        }
        
        return self.results['mlr']
    
    def feedforward_neural_network(self, X_train, X_test, y_train, y_test,
                                   hidden_layers=(128, 64, 32), dropout_rate=0.3):
        """
        Advanced Feedforward Neural Network
        """
        # Build model with batch normalization
        model = Sequential()
        model.add(Dense(hidden_layers[0], activation='relu', 
                       input_shape=(X_train.shape[1],),
                       kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation='relu',
                           kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
        
        model.add(Dense(1))
        
        # Compile
        model.compile(optimizer=Adam(learning_rate=0.001, decay=1e-6),
                     loss='mse',
                     metrics=['mae', 'mse'])
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        ]
        
        # Train
        history = model.fit(X_train, y_train,
                          epochs=200,
                          batch_size=32,
                          validation_split=0.2,
                          verbose=0,
                          callbacks=callbacks)
        
        # Predict
        y_pred = model.predict(X_test).flatten()
        
        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.models['ffnn'] = model
        self.results['ffnn'] = {
            'model': model,
            'history': history.history,
            'y_pred': y_pred,
            'mse': mse,
            'r2': r2
        }
        
        return self.results['ffnn']
    
    def bootstrap_ensemble(self, X_train, X_test, y_train, y_test, 
                           n_bootstrap=100, base_model='decision_tree'):
        """
        Bootstrap Ensemble Model
        """
        n_samples = len(X_train)
        predictions = np.zeros((len(X_test), n_bootstrap))
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X_train[indices]
            y_boot = y_train[indices]
            
            # Create base model
            if base_model == 'decision_tree':
                model = DecisionTreeRegressor(random_state=i)
            elif base_model == 'linear':
                model = LinearRegression()
            elif base_model == 'ridge':
                model = Ridge(alpha=1.0)
            else:
                model = DecisionTreeRegressor(random_state=i)
            
            # Train and predict
            model.fit(X_boot, y_boot)
            predictions[:, i] = model.predict(X_test)
        
        # Ensemble prediction (mean)
        y_pred = np.mean(predictions, axis=1)
        
        # Calculate prediction intervals
        pred_std = np.std(predictions, axis=1)
        ci_lower = np.percentile(predictions, 2.5, axis=1)
        ci_upper = np.percentile(predictions, 97.5, axis=1)
        
        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Coverage probability
        coverage = np.mean((y_test >= ci_lower) & (y_test <= ci_upper))
        
        self.results['bootstrap'] = {
            'y_pred': y_pred,
            'predictions': predictions,
            'pred_std': pred_std,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'mse': mse,
            'r2': r2,
            'coverage': coverage,
            'n_bootstrap': n_bootstrap
        }
        
        return self.results['bootstrap']
    
    def logistic_regression(self, X_train, X_test, y_train, y_test,
                            regularization='l2', C=1.0):
        """
        Logistic Regression for classification
        """
        # Ensure binary labels
        if len(np.unique(y_train)) > 2:
            print("Multi-class logistic regression not implemented")
            return None
        
        model = LogisticRegression(penalty=regularization, C=C, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        self.models['logistic'] = model
        self.results['logistic'] = {
            'model': model,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'coefficients': model.coef_
        }
        
        return self.results['logistic']
    
    def monte_carlo_simulation(self, X, y, n_simulations=1000):
        """
        Monte Carlo Simulation for uncertainty analysis
        """
        n_samples = len(X)
        n_features = X.shape[1]
        
        # Store results
        coefficient_samples = np.zeros((n_simulations, n_features))
        intercept_samples = np.zeros(n_simulations)
        r2_samples = np.zeros(n_simulations)
        
        for i in range(n_simulations):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Fit linear regression
            model = LinearRegression()
            model.fit(X_boot, y_boot)
            
            # Store parameters
            coefficient_samples[i, :] = model.coef_
            intercept_samples[i] = model.intercept_
            
            # Calculate R²
            y_pred = model.predict(X_boot)
            r2_samples[i] = r2_score(y_boot, y_pred)
        
        # Calculate statistics
        coef_mean = np.mean(coefficient_samples, axis=0)
        coef_std = np.std(coefficient_samples, axis=0)
        coef_ci_lower = np.percentile(coefficient_samples, 2.5, axis=0)
        coef_ci_upper = np.percentile(coefficient_samples, 97.5, axis=0)
        
        self.results['monte_carlo'] = {
            'coefficient_samples': coefficient_samples,
            'intercept_samples': intercept_samples,
            'r2_samples': r2_samples,
            'coef_mean': coef_mean,
            'coef_std': coef_std,
            'coef_ci_lower': coef_ci_lower,
            'coef_ci_upper': coef_ci_upper,
            'n_simulations': n_simulations
        }
        
        return self.results['monte_carlo']

class HybridModelFramework:
    """
    Hybrid modeling framework combining multiple approaches
    """
    
    def __init__(self):
        self.statistical_models = StatisticalAnalysisModels()
        self.ml_models = MachineLearningModels()
        self.results = {}
    
    def comprehensive_analysis(self, X, y, test_size=0.2):
        """
        Run comprehensive analysis with all models
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print("Starting comprehensive analysis...")
        print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
        print(f"Features: {X_train.shape[1]}")
        print()
        
        # 1. Statistical Models
        print("1. Running Statistical Models...")
        
        # GLM
        print("   - Generalized Linear Regression...")
        glm_results = self.statistical_models.generalized_linear_regression(
            X_train_scaled, y_train
        )
        
        # PCA
        print("   - Principal Component Analysis...")
        pca_results = self.statistical_models.principal_component_analysis(
            X_train_scaled, n_components=5
        )
        
        # Bayesian Regression
        print("   - Bayesian Linear Regression...")
        bayesian_results = self.statistical_models.bayesian_linear_regression(
            X_train_scaled, y_train
        )
        
        # Breakpoint Regression
        print("   - Breakpoint Regression...")
        breakpoint_results = self.statistical_models.breakpoint_regression(
            X_train_scaled, y_train, max_breakpoints=2
        )
        
        # 2. Machine Learning Models
        print("\n2. Running Machine Learning Models...")
        
        # Multiple Linear Regression
        print("   - Multiple Linear Regression...")
        mlr_results = self.ml_models.multiple_linear_regression(
            X_train_scaled, X_test_scaled, y_train, y_test
        )
        
        # Random Forest
        print("   - Random Forest...")
        rf_results = self.ml_models.decision_tree_ensemble(
            X_train_scaled, X_test_scaled, y_train, y_test, 
            model_type='regression', n_estimators=100
        )
        
        # Gradient Boosting
        print("   - Gradient Boosting...")
        # Already included in decision_tree_ensemble
        
        # Neural Networks
        print("   - Neural Networks (BP and FFNN)...")
        bp_results = self.ml_models.bp_neural_network(
            X_train_scaled, X_test_scaled, y_train, y_test
        )
        ffnn_results = self.ml_models.feedforward_neural_network(
            X_train_scaled, X_test_scaled, y_train, y_test
        )
        
        # LSTM
        print("   - LSTM Network...")
        lstm_results = self.ml_models.lstm_model(
            X_train_scaled, X_test_scaled, y_train, y_test
        )
        
        # Bootstrap Ensemble
        print("   - Bootstrap Ensemble...")
        bootstrap_results = self.ml_models.bootstrap_ensemble(
            X_train_scaled, X_test_scaled, y_train, y_test
        )
        
        # Logistic Regression (if binary classification)
        if len(np.unique(y)) == 2:
            print("   - Logistic Regression...")
            logistic_results = self.ml_models.logistic_regression(
                X_train_scaled, X_test_scaled, y_train, y_test
            )
        
        # Monte Carlo Simulation
        print("   - Monte Carlo Simulation...")
        monte_carlo_results = self.ml_models.monte_carlo_simulation(
            X_train_scaled, y_train
        )
        
        # 3. Clustering Models
        print("\n3. Running Clustering Models...")
        
        # DBSCAN
        print("   - DBSCAN Clustering...")
        dbscan_results = self.ml_models.dbscan_clustering(X_train_scaled)
        
        # Gaussian Mixture Model
        print("   - Gaussian Mixture Model...")
        gmm_results = self.ml_models.gaussian_mixture_model(X_train_scaled)
        
        # Compile results
        self.results = {
            'statistical': {
                'glm': glm_results,
                'pca': pca_results,
                'bayesian': bayesian_results,
                'breakpoint': breakpoint_results
            },
            'machine_learning': {
                'multiple_linear_regression': mlr_results,
                'random_forest': rf_results.get('rf') if rf_results else None,
                'gradient_boosting': rf_results.get('gb') if rf_results else None,
                'bp_neural_network': bp_results,
                'feedforward_nn': ffnn_results,
                'lstm': lstm_results,
                'bootstrap': bootstrap_results,
                'logistic': logistic_results if 'logistic_results' in locals() else None,
                'monte_carlo': monte_carlo_results
            },
            'clustering': {
                'dbscan': dbscan_results,
                'gmm': gmm_results
            },
            'data_info': {
                'X_train': X_train_scaled,
                'X_test': X_test_scaled,
                'y_train': y_train,
                'y_test': y_test,
                'scaler': scaler
            }
        }
        
        print("\nComprehensive analysis complete!")
        
        return self.results
    
    def compare_model_performance(self):
        """
        Compare performance of all models
        """
        if not self.results:
            print("No results available. Run comprehensive_analysis first.")
            return None
        
        comparison = []
        
        # Collect regression model performances
        ml_results = self.results['machine_learning']
        
        # Multiple Linear Regression
        if ml_results['multiple_linear_regression']:
            comparison.append({
                'Model': 'Multiple Linear Regression',
                'MSE': ml_results['multiple_linear_regression']['mse'],
                'R²': ml_results['multiple_linear_regression']['r2']
            })
        
        # Random Forest
        if ml_results['random_forest']:
            comparison.append({
                'Model': 'Random Forest',
                'MSE': ml_results['random_forest']['mse'],
                'R²': ml_results['random_forest']['r2']
            })
        
        # Gradient Boosting
        if ml_results['gradient_boosting']:
            comparison.append({
                'Model': 'Gradient Boosting',
                'MSE': ml_results['gradient_boosting']['mse'],
                'R²': ml_results['gradient_boosting']['r2']
            })
        
        # BP Neural Network
        if ml_results['bp_neural_network']:
            comparison.append({
                'Model': 'BP Neural Network',
                'MSE': ml_results['bp_neural_network']['mse'],
                'R²': ml_results['bp_neural_network']['r2']
            })
        
        # Feedforward NN
        if ml_results['feedforward_nn']:
            comparison.append({
                'Model': 'Feedforward NN',
                'MSE': ml_results['feedforward_nn']['mse'],
                'R²': ml_results['feedforward_nn']['r2']
            })
        
        # LSTM
        if ml_results['lstm']:
            comparison.append({
                'Model': 'LSTM',
                'MSE': ml_results['lstm']['mse'],
                'R²': ml_results['lstm']['r2']
            })
        
        # Bootstrap
        if ml_results['bootstrap']:
            comparison.append({
                'Model': 'Bootstrap Ensemble',
                'MSE': ml_results['bootstrap']['mse'],
                'R²': ml_results['bootstrap']['r2']
            })
        
        # Create DataFrame
        comparison_df = pd.DataFrame(comparison)
        
        # Sort by R² (descending)
        comparison_df = comparison_df.sort_values('R²', ascending=False).reset_index(drop=True)
        
        return comparison_df
    
    def visualize_comprehensive_results(self):
        """
        Create comprehensive visualizations
        """
        if not self.results:
            print("No results available. Run comprehensive_analysis first.")
            return None
        
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Model Performance Comparison
        ax1 = plt.subplot(3, 3, 1)
        comparison_df = self.compare_model_performance()
        if comparison_df is not None:
            x = np.arange(len(comparison_df))
            width = 0.35
            
            ax1.bar(x - width/2, comparison_df['MSE'], width, label='MSE', alpha=0.7)
            ax1.bar(x + width/2, comparison_df['R²'], width, label='R²', alpha=0.7)
            
            ax1.set_xlabel('Models')
            ax1.set_title('Model Performance Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. PCA Visualization
        ax2 = plt.subplot(3, 3, 2)
        if self.results['statistical']['pca']:
            pca_data = self.results['statistical']['pca']['transformed_data']
            if pca_data.shape[1] >= 2:
                ax2.scatter(pca_data[:, 0], pca_data[:, 1], alpha=0.6)
                ax2.set_xlabel('PC1')
                ax2.set_ylabel('PC2')
                ax2.set_title('PCA: First Two Components')
                ax2.grid(True, alpha=0.3)
        
        # 3. DBSCAN Clustering
        ax3 = plt.subplot(3, 3, 3)
        if self.results['clustering']['dbscan']:
            X_train = self.results['data_info']['X_train']
            dbscan_labels = self.results['clustering']['dbscan']['labels']
            
            if X_train.shape[1] >= 2:
                scatter = ax3.scatter(X_train[:, 0], X_train[:, 1], 
                                     c=dbscan_labels, cmap='tab20', alpha=0.7)
                ax3.set_xlabel('Feature 1')
                ax3.set_ylabel('Feature 2')
                ax3.set_title('DBSCAN Clustering')
                ax3.grid(True, alpha=0.3)
                
                n_clusters = self.results['clustering']['dbscan']['n_clusters']
                n_noise = self.results['clustering']['dbscan']['n_noise']
                ax3.text(0.05, 0.95, f'Clusters: {n_clusters}\nNoise: {n_noise}',
                         transform=ax3.transAxes, fontsize=9,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 4. GMM Clustering
        ax4 = plt.subplot(3, 3, 4)
        if self.results['clustering']['gmm']:
            X_train = self.results['data_info']['X_train']
            gmm_labels = self.results['clustering']['gmm']['labels']
            
            if X_train.shape[1] >= 2:
                scatter = ax4.scatter(X_train[:, 0], X_train[:, 1], 
                                     c=gmm_labels, cmap='tab20c', alpha=0.7)
                ax4.set_xlabel('Feature 1')
                ax4.set_ylabel('Feature 2')
                ax4.set_title('Gaussian Mixture Model')
                ax4.grid(True, alpha=0.3)
                
                n_components = self.results['clustering']['gmm']['n_components']
                silhouette = self.results['clustering']['gmm']['silhouette_score']
                ax4.text(0.05, 0.95, f'Components: {n_components}\nSilhouette: {silhouette:.3f}',
                         transform=ax4.transAxes, fontsize=9,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 5. Monte Carlo Coefficients
        ax5 = plt.subplot(3, 3, 5)
        if self.results['machine_learning']['monte_carlo']:
            mc_results = self.results['machine_learning']['monte_carlo']
            n_features = min(10, len(mc_results['coef_mean']))
            
            x_pos = np.arange(n_features)
            ax5.bar(x_pos, mc_results['coef_mean'][:n_features],
                   yerr=mc_results['coef_std'][:n_features],
                   capsize=5, alpha=0.7, error_kw={'elinewidth': 2, 'capsize': 3})
            
            ax5.set_xlabel('Feature Index')
            ax5.set_ylabel('Coefficient Value')
            ax5.set_title('Monte Carlo: Coefficient Uncertainty')
            ax5.set_xticks(x_pos)
            ax5.set_xticklabels([f'F{i}' for i in range(n_features)])
            ax5.grid(True, alpha=0.3)
        
        # 6. Bootstrap Prediction Intervals
        ax6 = plt.subplot(3, 3, 6)
        if self.results['machine_learning']['bootstrap']:
            bootstrap_results = self.results['machine_learning']['bootstrap']
            y_test = self.results['data_info']['y_test']
            
            n_display = min(30, len(y_test))
            indices = np.arange(n_display)
            
            ax6.errorbar(indices, bootstrap_results['y_pred'][:n_display],
                        yerr=[bootstrap_results['y_pred'][:n_display] - bootstrap_results['ci_lower'][:n_display],
                              bootstrap_results['ci_upper'][:n_display] - bootstrap_results['y_pred'][:n_display]],
                        fmt='o', capsize=5, alpha=0.7, label='Predictions with 95% CI')
            
            ax6.scatter(indices, y_test[:n_display], color='red', alpha=0.7, label='True Values')
            
            ax6.set_xlabel('Sample Index')
            ax6.set_ylabel('Value')
            ax6.set_title('Bootstrap Prediction Intervals')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # 7. Neural Network Training History
        ax7 = plt.subplot(3, 3, 7)
        if self.results['machine_learning']['bp_neural_network']:
            history = self.results['machine_learning']['bp_neural_network']['history']
            
            epochs = range(1, len(history['loss']) + 1)
            ax7.plot(epochs, history['loss'], 'b-', label='Training Loss', alpha=0.7)
            ax7.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', alpha=0.7)
            
            ax7.set_xlabel('Epochs')
            ax7.set_ylabel('Loss')
            ax7.set_title('Neural Network Training History')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        
        # 8. Feature Importance (Random Forest)
        ax8 = plt.subplot(3, 3, 8)
        if self.results['machine_learning']['multiple_linear_regression']:
            mlr_results = self.results['machine_learning']['multiple_linear_regression']
            if mlr_results['feature_importance'] is not None:
                importance = mlr_results['feature_importance']
                n_features = min(15, len(importance))
                
                top_indices = np.argsort(importance)[-n_features:]
                top_importance = importance[top_indices]
                
                y_pos = np.arange(n_features)
                ax8.barh(y_pos, top_importance, alpha=0.7)
                ax8.set_yticks(y_pos)
                ax8.set_yticklabels([f'Feature {i}' for i in top_indices])
                ax8.set_xlabel('Importance')
                ax8.set_title('Top Feature Importance (Linear Regression)')
                ax8.grid(True, alpha=0.3)
        
        # 9. Residual Plot
        ax9 = plt.subplot(3, 3, 9)
        if self.results['machine_learning']['multiple_linear_regression']:
            mlr_results = self.results['machine_learning']['multiple_linear_regression']
            y_test = self.results['data_info']['y_test']
            
            residuals = y_test - mlr_results['y_pred']
            ax9.scatter(mlr_results['y_pred'], residuals, alpha=0.6)
            ax9.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            ax9.set_xlabel('Predicted Values')
            ax9.set_ylabel('Residuals')
            ax9.set_title('Residual Plot (Linear Regression)')
            ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig


def estimate_fan_votes_percentage(judge_scores, eliminated_idx, eps=1e-6, penalty=1000.0):
    """Estimate fan vote percentages via a penalized optimization.

    judge_scores: 1d array of judge totals for contestants in the week
    eliminated_idx: index of the eliminated contestant in the arrays
    Returns normalized fan vote vector summing to 1.
    """
    judge_scores = np.asarray(judge_scores, dtype=float)
    n = len(judge_scores)
    if n < 2:
        return np.array([1.0])

    # For large contestant counts the SLSQP optimizer can be slow; fallback to rank heuristic
    if n > 12:
        return estimate_fan_votes_rank(judge_scores, eliminated_idx)

    judge_pct = judge_scores / (judge_scores.sum() + eps)

    # initial guess uniform
    x0 = np.ones(n) / n

    def objective(v):
        v = np.clip(v, 1e-8, 1.0)
        v = v / v.sum()
        # regularizer to keep near-uniform
        reg = np.sum((v - x0) ** 2)
        combined = judge_pct + v
        # penalty if eliminated combined is NOT <= others
        diff = combined[eliminated_idx] - combined
        diff[eliminated_idx] = -1.0  # ignore self
        viol = np.maximum(0.0, diff + 1e-6)
        pen = penalty * np.sum(viol ** 2)
        return reg + pen

    # constraints: v_i >= 0, sum(v)=1
    cons = ({'type': 'eq', 'fun': lambda v: np.sum(v) - 1.0},)
    bounds = [(1e-8, 1.0) for _ in range(n)]

    res = optimize.minimize(objective, x0, bounds=bounds, constraints=cons, method='SLSQP')
    if not res.success:
        v = x0
    else:
        v = res.x
        v = np.clip(v, 0, None)
        v = v / (v.sum() + eps)
    return v


def estimate_fan_votes_rank(judge_scores, eliminated_idx):
    """Simple rank-based fan vote estimate: assign fan votes proportional to reversed judge ranks,
    ensuring eliminated contestant has the lowest fan estimate.
    """
    judge_scores = np.asarray(judge_scores, dtype=float)
    n = len(judge_scores)
    ranks = stats.rankdata(-judge_scores, method='average')  # 1 = best
    # produce fan ranks that favor lower judge ranks (so eliminated gets lowest fan rank)
    fan_rank_score = (ranks.max() - ranks) + 1.0
    # force eliminated to have smallest fan score
    fan_rank_score[eliminated_idx] = 1e-6
    v = fan_rank_score / fan_rank_score.sum()
    return v


def monte_carlo_uncertainty(judge_scores, eliminated_idx, method='percent', n_sim=200, noise_scale=0.02):
    """Monte Carlo perturbation of judge scores to estimate variability in fan vote estimates."""
    judge_scores = np.asarray(judge_scores, dtype=float)
    n = len(judge_scores)
    samples = np.zeros((n_sim, n))
    for i in range(n_sim):
        noise = np.random.normal(scale=noise_scale * (np.std(judge_scores) + 1e-8), size=n)
        perturbed = np.maximum(0.0, judge_scores + noise)
        try:
            if method == 'percent':
                v = estimate_fan_votes_percentage(perturbed, eliminated_idx)
            else:
                v = estimate_fan_votes_rank(perturbed, eliminated_idx)
        except Exception:
            # fallback
            if method == 'percent':
                v = np.ones(n) / n
            else:
                v = estimate_fan_votes_rank(perturbed, eliminated_idx)
        samples[i, :] = v
    return samples

# Main execution
def main():
    """
    Main execution function
    """
    print("=== Comprehensive DWTS Analysis Framework ===")
    print("Loading and preparing data...")

    import os
    csv_path = os.path.join('data', 'raw', '2026_MCM_Problem_C_Data.csv')
    use_synthetic = True

    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            print(f"Loaded data from {csv_path}, shape={df.shape}")

            # Try to detect an explicit target column
            target_candidates = [
                'fan_votes', 'fan_vote_percent', 'percent_fan_vote',
                'votes', 'total_fan_votes', 'votes_percent', 'fan_vote'
            ]
            target_col = None
            for c in target_candidates:
                if c in df.columns:
                    target_col = c
                    break

            # If no explicit target, try to build a proxy from score/placement
            score_cols = [col for col in df.columns if 'score' in col.lower()]
            if target_col is None:
                if score_cols:
                    df['proxy_target'] = df[score_cols].sum(axis=1)
                    target_col = 'proxy_target'
                    print('No explicit target found; using sum of score columns as proxy target.')
                elif 'placement' in df.columns:
                    df['proxy_target'] = -df['placement'].astype(float)
                    target_col = 'proxy_target'
                    print('No explicit target found; using inverted placement as proxy target.')
                else:
                    print('No suitable target found in CSV; falling back to synthetic data.')

            if target_col is not None:
                # Extract features using the AdvancedFeatureEngineering helper
                fe = AdvancedFeatureEngineering(df)
                features_df = fe.extract_all_features()

                # Align X and y
                X = features_df.fillna(0).values
                y = df[target_col].values
                use_synthetic = False
        except Exception as e:
            print('Failed to load or process CSV:', e)

    if use_synthetic:
        # Fallback to synthetic data (original demo behavior)
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] * 2.5 +
             X[:, 1] * 1.8 +
             X[:, 2] * (-1.2) +
             X[:, 3] * 0.8 * X[:, 4] +
             np.sin(X[:, 5]) * 1.5 +
             np.random.randn(n_samples) * 0.5)

    print(f"Data shape: X={X.shape}, y={getattr(y, 'shape', (len(y),))}")
    print()

    # Initialize hybrid framework
    framework = HybridModelFramework()

    # Run comprehensive analysis
    results = framework.comprehensive_analysis(X, y, test_size=0.3)
    
    # Compare model performance
    comparison_df = framework.compare_model_performance()
    print("\n=== Model Performance Comparison ===")
    print(comparison_df.to_string(index=False))
    print()
    
    # Identify best model
    best_model = comparison_df.loc[comparison_df['R²'].idxmax()]
    print(f"Best Model: {best_model['Model']}")
    print(f"Best R²: {best_model['R²']:.4f}")
    print(f"Best MSE: {best_model['MSE']:.4f}")
    print()
    
    # Generate visualizations
    print("Generating comprehensive visualizations...")
    framework.visualize_comprehensive_results()
    
    # Print clustering results
    print("\n=== Clustering Results ===")
    dbscan_info = results['clustering']['dbscan']
    gmm_info = results['clustering']['gmm']
    
    print(f"DBSCAN: Found {dbscan_info['n_clusters']} clusters with {dbscan_info['n_noise']} noise points")
    print(f"DBSCAN Silhouette Score: {dbscan_info['silhouette_score']:.3f}")
    print(f"GMM: Found {gmm_info['n_components']} components")
    print(f"GMM Silhouette Score: {gmm_info['silhouette_score']:.3f}")
    print()
    
    # Print uncertainty analysis
    print("=== Uncertainty Analysis ===")
    bootstrap_results = results['machine_learning']['bootstrap']
    mc_results = results['machine_learning']['monte_carlo']
    
    if bootstrap_results:
        print(f"Bootstrap Coverage Probability: {bootstrap_results['coverage']:.1%}")
    
    if mc_results:
        avg_coef_std = np.mean(mc_results['coef_std'])
        print(f"Average Coefficient Uncertainty: {avg_coef_std:.4f}")
    
    print("\nAnalysis complete!")

    # --- Per-week estimation (percentage & rank) + uncertainty ---
    try:
        # Build long-form per-contestant-week frame if original CSV was used
        if not use_synthetic and 'df' in locals():
            df_long = df.copy()
            # Attempt to find score columns and group by season/week/contestant (wide-format dataset)
            score_cols = [c for c in df_long.columns if 'week' in c.lower() and 'judge' in c.lower()]
            # accept either 'celebrity_name' or 'celebrity name' column names
            name_col = 'celebrity_name' if 'celebrity_name' in df_long.columns else ('celebrity name' if 'celebrity name' in df_long.columns else None)
            if score_cols and 'season' in df_long.columns and name_col is not None:
                out_rows = []
                seasons = df_long['season'].unique()
                # determine available week numbers from column names like 'week3_judge2_score'
                week_numbers = set()
                for c in score_cols:
                    try:
                        prefix = c.lower().split('_')[0]
                        if prefix.startswith('week'):
                            wk = int(prefix.replace('week', ''))
                            week_numbers.add(wk)
                    except Exception:
                        continue

                for season in seasons:
                    df_s = df_long[df_long['season'] == season]
                    for wk in sorted(week_numbers):
                        # gather judge columns for this week
                        judge_cols = [col for col in df_s.columns if col.lower().startswith(f'week{wk}_judge')]
                        if not judge_cols:
                            continue

                        # select contestants who have at least one non-null score this week
                        df_w = df_s[df_s[judge_cols].notna().any(axis=1)]
                        if df_w.shape[0] < 2:
                            continue

                        # compute judge totals per contestant for this week
                        judge_totals = df_w[judge_cols].sum(axis=1).astype(float).values
                        names = df_w[name_col].values

                        # find eliminated contestant in that week if possible
                        eliminated_idx = None
                        if 'results' in df_w.columns:
                            for i, r in enumerate(df_w['results'].values):
                                if isinstance(r, str) and f'Eliminated Week {wk}' in r:
                                    eliminated_idx = i
                                    break

                        # fallback: lowest judge total
                        if eliminated_idx is None:
                            eliminated_idx = int(np.argmin(judge_totals))

                        # estimate fan votes (percent method)
                        v_pct = estimate_fan_votes_percentage(judge_totals, eliminated_idx)
                        # estimate fan votes (rank method)
                        v_rank = estimate_fan_votes_rank(judge_totals, eliminated_idx)

                        # uncertainty via Monte Carlo perturbation
                        pct_samples = monte_carlo_uncertainty(judge_totals, eliminated_idx, method='percent', n_sim=50)
                        rank_samples = monte_carlo_uncertainty(judge_totals, eliminated_idx, method='rank', n_sim=50)

                        for i, name in enumerate(names):
                            out_rows.append({
                                'season': int(season),
                                'week': int(wk),
                                'celebrity_name': name,
                                'judge_total': float(judge_totals[i]),
                                'fan_est_percent': float(v_pct[i]),
                                'fan_est_rank': float(v_rank[i]),
                                'fan_est_percent_std': float(pct_samples[:, i].std()),
                                'fan_est_rank_std': float(rank_samples[:, i].std()),
                                'eliminated_flag': int(i == eliminated_idx)
                            })

                if out_rows:
                    out_df = pd.DataFrame(out_rows)
                    out_dir = os.path.join('data', 'processed')
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, 'model2_estimates.csv')
                    out_df.to_csv(out_path, index=False)
                    print(f"Per-week estimates saved to {out_path}")
                    # If available, also run the DWTSFanVoteEstimator from model1.py to produce its report
                    if DWTSFanVoteEstimator is not None:
                        try:
                            print('Running DWTSFanVoteEstimator (model1) to produce additional outputs...')
                            data_path = os.path.join('data', 'raw', '2026_MCM_Problem_C_Data.csv')
                            estimator = DWTSFanVoteEstimator(data_path)
                            estimator.preprocess_data()
                            estimator.estimate_all_weeks()
                            val = estimator.validate_estimates()
                            print('Model1 validation:', val)
                            # attempt to visualize and save model1 results
                            try:
                                estimator.visualize_results()
                            except Exception as viz_e:
                                print('model1 visualize_results failed:', viz_e)
                            # Save model1 estimates if available
                            try:
                                if hasattr(estimator, 'estimates') and estimator.estimates:
                                    rows = []
                                    for (s,w), d in estimator.estimates.items():
                                        for rec in d:
                                            r = dict(rec)
                                            r.update({'season': s, 'week': w})
                                            rows.append(r)
                                    m1_df = pd.DataFrame(rows)
                                    m1_path = os.path.join(out_dir, 'model1_estimates.csv')
                                    m1_df.to_csv(m1_path, index=False)
                                    print(f'model1 estimates saved to {m1_path}')
                            except Exception as save_e:
                                print('saving model1 estimates failed:', save_e)
                        except Exception as e:
                            print('Running DWTSFanVoteEstimator failed:', e)
    except Exception as e:
        print('Per-week estimation/save failed:', e)

if __name__ == "__main__":
    main()