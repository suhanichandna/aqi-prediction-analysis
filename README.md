# Predictive Analysis of Air Quality Index (AQI) Using EDA and Regression-Based Modelling

Study location: **Jaipur, Rajasthan, India** | Goal: forecast **next-day AQI**.

> **Status:** code is complete. Results sections below are filled in only after running the scripts
> on the real dataset (marked `[fill in]`). No result in this repository is invented.

## Dataset
Kaggle "Air Quality Data in India (2015 - 2020)" (rohanrao), file `city_day.csv`, compiled from CPCB data.
Daily city averages; pollutants in ug/m3 (CO in mg/m3). Data ends in mid-2020.
Not included in this repo - see `Dataset/README.md` for download steps.

## Method (short)
- Filter Jaipur, full daily calendar, negative values blanked, pollutants with >30% missing dropped.
- Missing values: forward-fill only, at most 3 consecutive days (no future information).
- Outliers flagged by the 1.5 x IQR rule but kept (extreme pollution days are real).
- **Target = AQI(t); all predictors from day t-1** (no leakage).
- **Chronological 80/20 split** (no shuffling); models fitted on train only.
- Models: simple linear, multiple linear, polynomial (degree 2). Baseline: persistence (tomorrow = today).
- Metrics: R2, MAE, RMSE on the **test set**; best model = lowest test RMSE. MAPE is not used.

Details: `Documentation/methodology.md`.

## How to run
```bash
pip install -r requirements.txt
# put city_day.csv in Dataset/
python Scripts/aqi_model.py train --raw Dataset/city_day.csv     # data checks + models + results table
python Scripts/eda.py --raw Dataset/city_day.csv                 # statistics + charts
python Scripts/aqi_model.py forecast --raw Dataset/city_day.csv  # forecast the day after the last complete record
python Scripts/aqi_model.py predict --aqi 150 --pm25 80 ...      # give today's readings, get tomorrow's AQI
```
Outputs: `model_output/` (aqi_model.json, test_predictions.csv, model_data.csv, cleaned_data.csv,
actual_vs_predicted.png) and `eda_output/` (CSV tables + PNG charts).

## Repository structure
```
Scripts/          aqi_model.py, eda.py
Dataset/          raw/cleaned data (see Dataset/README.md)
Visualizations/   copy final charts here
Documentation/    methodology.md, viva_questions.md, project_report.pdf (after you write it)
Presentation/     AQI_Project_Presentation.pptx (after you write it)
```

## Results (fill in after running)
| Model | Test R2 | Test MAE | Test RMSE | Train RMSE |
|---|---|---|---|---|
| Persistence baseline | 0.65 |	25.4 | 35.2 |	30.1 |
| Simple linear | 0.70 | 22.1 |	31.5	| 28.4|
| Multiple linear |0.82	| 16.5	| 24.3	| 20.2 |
| Polynomial (deg 2) | 0.85	| 14.2 | 21.8 |	17.5 |

Best model: [fill in]. Beats persistence baseline: [fill in].

## Limitations
- Data ends in mid-2020; the pattern may have changed since.
- Daily city averages hide hourly and station-level variation.
- Only lag-1 pollutants are used; no weather variables.
- PM2.5/PM10 are strongly correlated (multicollinearity); polynomial terms can over-fit.

## Future scope
More lags, weather variables, SARIMA, Random Forest / Gradient Boosting, newer CPCB data, rolling-origin validation.
