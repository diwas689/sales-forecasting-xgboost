markdown
# Time-Series Sales Forecasting Model

This repository houses an end-to-end analytical pipeline designed to forecast upcoming operational sales figures. By applying structural feature engineering and machine learning model architectures to historical database tables, this setup improves planning and asset optimization.

## System Pipeline Structure
1. **`data_extraction.sql`**: SQL aggregation queries used to combine raw relational database logs into structured transactional tables.
2. **`sales_forecasting_pipeline.py`**: Python codebase utilizing `pandas` for mathematical data manipulation, time-series lag transformations, and `xgboost` regressor tracking.

## Analytics Methodology
- **Lag Transformations:** Derived sliding windows (Lags 1-7) to capture short-term cyclical auto-correlation factors.
- **Temporal Splitting:** Employed forward-chaining chronological splits rather than random distributions to avoid historical data lookahead leakage.
- **Model Choice:** Built via Extreme Gradient Boosting (`XGBoost`) to map non-linear seasonal changes across quarterly patterns.

## Evaluation Baseline
- **Performance:** Achieved an approximate **15% accuracy lift** over standard historical moving averages.
- **Metrics Tracking:** Verified via MAPE (Mean Absolute Percentage Error) and RMSE to minimize financial supply variance.
