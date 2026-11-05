# Viva questions and answers
Answers that need your numbers contain **[your result]**. Fill them from your own run - do not guess.

## A. Project basics
1. **What is AQI?** A single number (0-500+) that summarises how polluted the air is; CPCB bands: Good 0-50, Satisfactory 51-100, Moderate 101-200, Poor 201-300, Very Poor 301-400, Severe 401+.
2. **What problem are you solving?** Forecasting next-day AQI for Jaipur from today's readings.
3. **Why Jaipur?** Rajasthan city with CPCB data; it is the chosen study location. Row count after cleaning: **[your result]**.
4. **Which dataset?** Kaggle "Air Quality Data in India (2015-2020)", city_day.csv, from CPCB. Daily city averages.
5. **Why not other sources?** CCR portal is manual/hourly, NCAP has only AQI, data.gov.in is fragmented, IEEE/Mendeley sets are small, live sites are not historical. CCR is the fallback if Jaipur has under ~700 usable rows.
6. **Date range and rows?** **[your result: first date, last date, usable rows]**.

## B. Preprocessing
7. **Why re-index to a full daily calendar?** So "previous day" is exactly one day earlier; otherwise a shift could skip days.
8. **Why forward-fill with max 3 days?** It uses only past information (no leakage) and avoids inventing long stretches of data.
9. **Why not mean or interpolation?** They use future values, which would leak future information into training.
10. **Why drop a pollutant with >30% missing?** Too little real data; filling it would mostly be guessing. Dropped: **[your result]**.
11. **How did you treat outliers?** Flagged with 1.5 x IQR but kept, because extreme pollution days are real events. Flagged counts: **[your result]**.
12. **Why remove negative values?** Concentrations cannot be negative; they are sensor errors. Count: **[your result]**.
13. **What rows are used as targets?** Only days with an actually observed AQI and complete previous-day predictors.

## C. EDA
14. **Mean/median AQI?** **[your result]**. If mean > median the data is right-skewed (a few very bad days).
15. **What does skewness tell you?** Direction of the tail. Your AQI skewness: **[your result]**.
16. **Most common AQI category?** **[your result]**.
17. **Which season is worst?** **[your result]**. Typically colder, calmer months trap pollution, but state only what your table shows.
18. **Correlation of PM2.5 and PM10 with AQI?** **[your result]**.
19. **What is multicollinearity and does it appear?** Predictors that move together (PM2.5 and PM10). Correlation between them: **[your result]**.

## D. Models
20. **Why linear regression?** Simple, interpretable baseline model; easy to explain each coefficient.
21. **Explain the simple model.** AQI(t) = b0 + b1*AQI(t-1). Your equation: **[your result]**.
22. **Multiple regression - what is added?** Yesterday's PM2.5, PM10, NO2, SO2, CO, O3 (those kept).
23. **Why a polynomial model?** To capture curved relationships. Risks: squared terms correlate with originals, can over-fit, can extrapolate badly.
24. **What is the baseline and why?** Persistence: tomorrow = today. A model is only useful if it beats this easy rule.
25. **What is data leakage and how did you avoid it?** Using information from the future/target period. I used only lag-1 predictors, a chronological split, train-only fitting, and forward-fill only.
26. **Why lag-1 and not same-day pollutants?** In real forecasting you do not have tomorrow's readings; same-day values would make the model unrealistic.

## E. Split and evaluation
27. **Why chronological split, not random?** Shuffling lets the model train on days after the test days, which is impossible in reality.
28. **What do R2, MAE and RMSE mean?** R2 = share of variation explained; MAE = average absolute error in AQI points; RMSE = like MAE but penalises big errors more.
29. **Why not MAPE?** AQI can be near zero, making percentage errors blow up.
30. **Which model is best and why?** Lowest test RMSE: **[your result]**. Explain only with your numbers (test RMSE/MAE/R2 of each model).
31. **Does it beat the persistence baseline?** **[your result: baseline RMSE vs best RMSE]**. If not, say plainly that the regression adds no value over "tomorrow = today".
32. **How do you detect over-fitting?** Compare train and test RMSE; test much worse than train signals over-fitting. Yours: **[your result]**.
33. **Why does the test set decide, not train?** Train error rewards memorising; the test set shows performance on unseen days.
34. **Is a negative R2 possible?** Yes, if the model is worse than predicting the test mean. Yours: **[your result]**.

## F. Limitations and future scope
35. **Limitations?** Data ends mid-2020; daily averages hide hourly/station detail; only lag-1; no weather; linear models; one chronological split only.
36. **Why might accuracy be limited?** AQI depends on weather (wind, temperature, rain), festivals, traffic and crop burning, none of which are in the model.
37. **Future scope?** More lags, weather variables, SARIMA, Random Forest/Gradient Boosting, newer CPCB data, rolling-origin validation.
38. **Can your model be used for real alerts?** Not without newer data and validation; it is an academic study.
39. **What would you do if Jaipur had too few rows?** Switch to a CPCB CCR export aggregated to daily means (the agreed rule).
40. **What did you learn?** How to prevent leakage, evaluate honestly against a baseline, and report limits.
