# Model1 README

Summary
- Hybrid approaches implemented: factor analysis (PCA + tree learners) and time-series decomposition (ARIMA) with nonlinear residual modeling.
- Added feature engineering: rolling means (last-3/last-5), rolling standard deviation, momentum (week-over-week diffs), recency-weighted averages (EWMA), and interaction terms.

Models
- RandomForest (default) for regression/classification tasks.
- Optional experiments with XGBoost (`xgboost`) and LightGBM (`lightgbm`) if installed; the code will try to fit and report comparison metrics.

Files of interest
- script/test/model1.py: main analysis functions and experiments.
- script/test/try.py: higher-level pipeline that calls model functions (if present).

Requirements
- Python packages: pandas, numpy, scipy, scikit-learn, statsmodels, matplotlib, seaborn
- Optional: xgboost, lightgbm for stronger tree learners; tensorflow (for LSTM path) is optional and guarded.


Notes
- The code includes safe import guards for optional dependencies; if XGBoost/LightGBM are not installed, the script will skip those experiments.
- Feature engineering additions are applied to both factor analysis and the long-form time-series pipeline.


