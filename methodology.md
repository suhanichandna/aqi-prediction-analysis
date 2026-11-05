# Methodology (simple language)

## 1. Question
Can today's air readings predict tomorrow's AQI for Jaipur better than just saying "tomorrow = today"?

## 2. Data
Kaggle city_day.csv (CPCB daily city averages). Jaipur rows only. Decision rule: if Jaipur has roughly
700+ usable daily rows with short gaps, use it; otherwise switch to a CPCB CCR export aggregated to daily means.

## 3. Cleaning (why each step)
| Step | Why |
|---|---|
| Parse dates, drop repeated dates (keep first) | one row per day |
| Re-index to a full daily calendar | so "previous day" is exactly 1 day earlier |
| Negative concentrations -> blank | impossible values |
| Drop pollutant if >30% missing | too little real information |
| Forward-fill, max 3 days | uses only past data; mean/interpolation over the whole series would leak the future |
| IQR outliers flagged, kept | extreme pollution days are real events |
| Year, Month, Season | for trend analysis (Winter Dec-Feb, Summer Mar-May, Monsoon Jun-Sep, Post-monsoon Oct-Nov) |
| Target rows: AQI actually observed + complete previous-day predictors | we never train/test on filled targets |

## 4. No-leakage rule
Target is AQI(t). Every predictor is from day t-1. Same-day pollutant readings are never used.

## 5. Split
Chronological: first 80% train, last 20% test. No shuffling (shuffling would let the model "see the future").
All fitting uses train rows only.

## 6. Models
1. Simple linear: AQI(t) = b0 + b1*AQI(t-1)
2. Multiple linear: AQI(t-1) + PM2.5, PM10, NO2, SO2, CO, O3 at t-1 (those kept after cleaning). Caveat: multicollinearity.
3. Polynomial deg 2: AQI(t-1), AQI(t-1)^2, PM2.5(t-1), PM2.5(t-1)^2. Can over-fit / extrapolate badly.
Baseline: persistence. A model is useful only if it beats it.
Fitting method: ordinary least squares (numpy least squares).

## 7. Evaluation (test set only decides)
R2 (share of variation explained), MAE (average absolute error, in AQI points), RMSE (like MAE but punishes big misses).
Best model = lowest test RMSE. Train RMSE vs test RMSE shows over-fitting. MAPE not used (AQI can be near zero).

## 8. EDA
Mean, median, mode, min, max, range, variance, std, quartiles, skewness; correlation heatmap; CPCB AQI bands
(Good 0-50, Satisfactory 51-100, Moderate 101-200, Poor 201-300, Very Poor 301-400, Severe 401+);
monthly/yearly/seasonal trends; pollutant histograms; PM2.5 and PM10 vs AQI scatter with trendline; actual vs predicted chart.

## 9. Honest limits
Data ends mid-2020; daily averages; lag-1 only; no weather; if the best model does not beat persistence, say so.
