# AirSense Jaipur - web app guide

## 1. How the website connects to your model (explain this in the viva)
1. `Scripts/aqi_model.py train` fits the three regression models ONCE and saves their coefficients
   in `model_output/aqi_model.json` (plus cleaned data and test predictions).
2. `Scripts/prepare_app_data.py` copies those files into `app_data/`.
3. `core.py` reads `app_data/aqi_model.json`. For one prediction it builds the features in the exact
   order and form saved from training (e.g. AQI(t-1), AQI(t-1)^2, PM2.5(t-1), PM2.5(t-1)^2) and
   computes `intercept + sum(coefficient x feature)`. Nothing is retrained when you click Predict.
4. `app.py` only draws the pages; all numbers shown come from `app_data/`.
5. Verification already done: `core.predict` reproduces the training script's `test_predictions.csv`
   to ~1e-14 for all three models (checked on synthetic data; repeat the check on your real data if you wish).

## 2. Things you must know (honest notes)
- **Previous-day AQI is an input.** Your simple and polynomial models are built on AQI(t-1), so the
  prediction form has a "Previous-day AQI" field in addition to the six pollutants. Without it the
  polynomial model cannot run. All inputs are YESTERDAY's values; the output is TOMORROW's AQI.
- **Polynomial is the default selection**, as you requested, but the app tells you which model had the
  best test RMSE in YOUR run. If that is a different model, the app says so - it never hides it.
- **No scikit-learn.** Your models were fitted with NumPy least squares (not scikit-learn). Using
  scikit-learn in the app would mean a different model object, so it is not used. Plotly replaces
  Matplotlib/Seaborn for charts.
- **Not live data.** The app labels all data as historical (date range shown on every relevant page).
- **Testing status.** I could test the model/prediction logic and run every page's data code with stubbed
  UI libraries. I could NOT launch Streamlit or render Plotly charts here (no internet to install them),
  and I have no real Jaipur results. Do the checklist in section 6 on your Mac; send me any error text.

## 3. Run locally (MacBook, VS Code)
```bash
cd aqi-prediction-analysis
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-training.txt        # for training scripts
pip install -r requirements.txt                 # for the web app
# put city_day.csv into Dataset/
python Scripts/aqi_model.py train --raw Dataset/city_day.csv
python Scripts/prepare_app_data.py
streamlit run app.py                            # opens http://localhost:8501
```
If `app_data/` is empty the app shows exactly which files are missing and the commands above.

## 4. Deploy on Streamlit Community Cloud
1. Make sure `app_data/` contains the 4 real files (they are small; they must be committed).
   Do NOT commit `Dataset/city_day.csv` or `.venv` (the `.gitignore` excludes them).
2. Upload to GitHub (you do this):
   ```bash
   git init && git add . && git commit -m "AirSense Jaipur app"
   git branch -M main
   git remote add origin https://github.com/suhanichandna/aqi-prediction-analysis.git
   git push -u origin main
   ```
   (If the repo already exists with files, use `git pull --rebase origin main` first.)
3. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**, choose the repository,
   branch `main`, main file `app.py`, then **Deploy**.
4. Share the generated `*.streamlit.app` link.

## 5. Troubleshooting
| Problem | Fix |
|---|---|
| "Model/data files are missing" | Run the train + prepare_app_data commands, commit `app_data/`, redeploy |
| `ModuleNotFoundError` on Cloud | Check the package is in `requirements.txt`; reboot the app |
| App works locally, not on Cloud | Confirm `app_data/*.json/*.csv` were pushed (`git ls-files app_data`) |
| Port busy locally | `streamlit run app.py --server.port 8502` |
| Old data shown after retraining | Re-run prepare_app_data, then in the app press `C` -> Clear cache, or restart |
| Predicted value shown is clipped | Raw model output outside 0-500; the page tells you the raw value |

## 6. Test checklist (tick each on your Mac)
- [ ] App starts with no error; sidebar shows the data date range
- [ ] Home: observations, mean/min/max AQI match `eda.py` output; 5 navigation buttons switch pages
- [ ] Prediction: click Predict with defaults -> number, category badge, gauge, contribution table appear
- [ ] Prediction: change inputs -> value changes; switch model -> value and equation terms change
- [ ] Prediction: value outside training range -> warning appears; fields cannot go negative or above max
- [ ] Prediction check: run `python Scripts/aqi_model.py predict --aqi .. --pm25 ..` with same inputs (best model) -> same number
- [ ] Data Analysis: all 9 charts render, hover works, each has a caption
- [ ] Model Performance: table equals the table printed by the train script; baseline verdict is correct
- [ ] Historical Explorer: year/month filters change charts; an empty selection shows a warning
- [ ] About: dataset rows/dates agree with the validation printout
- [ ] Temporarily rename `app_data/aqi_model.json`: the app shows the missing-file message (then rename back)
- [ ] Resize the browser to phone width: layout stacks, nothing unreadable
