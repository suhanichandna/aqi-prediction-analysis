"""
aqi_model.py - Next-day AQI forecasting for Jaipur (pandas + numpy + matplotlib only).

Usage (run from the repository root):
    python Scripts/aqi_model.py train    --raw Dataset/city_day.csv
    python Scripts/aqi_model.py forecast --raw Dataset/city_day.csv
    python Scripts/aqi_model.py predict  --aqi 150 --pm25 80 --pm10 160 --no2 30 --so2 10 --co 1.0 --o3 40

Method (see Documentation/methodology.md):
  * target = AQI on day t; ALL predictors come from day t-1 (lag-1) -> no leakage
  * forward-fill only, max 3 consecutive days; no interpolation / mean fill
  * chronological 80/20 split, models fitted on TRAIN only
  * best model = lowest TEST RMSE; compared against persistence (tomorrow = today)
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CITY = "Jaipur"
POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
MAX_FFILL_DAYS = 3
MAX_MISSING_FRACTION = 0.30
TRAIN_FRACTION = 0.80
OUT_DIR = "model_output"
ARG_NAMES = {"AQI": "aqi", "PM2.5": "pm25", "PM10": "pm10", "NO2": "no2",
             "SO2": "so2", "CO": "co", "O3": "o3"}


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def season_of(month):
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Summer"
    if month in (6, 7, 8, 9):
        return "Monsoon"
    return "Post-monsoon"


def longest_gap(series):
    """Longest run of consecutive missing values."""
    best = run = 0
    for isna in series.isna().values:
        run = run + 1 if isna else 0
        best = max(best, run)
    return best


def load_city(path, city=CITY):
    """Load raw CSV -> Jaipur only, full daily calendar, negatives blanked.
    Returns (dataframe, info_dict). Nothing is filled here."""
    raw = pd.read_csv(path)
    raw["City"] = raw["City"].astype(str).str.strip()
    df = raw[raw["City"].str.lower() == city.lower()].copy()
    if df.empty:
        sys.exit(f"No rows found for city '{city}'. Cities present: "
                 f"{sorted(raw['City'].unique())[:30]}")
    info = {"city": city, "raw_rows_for_city": int(len(df))}
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    info["unparseable_dates"] = int(df["Date"].isna().sum())
    df = df.dropna(subset=["Date"]).sort_values("Date")
    info["duplicate_dates_dropped"] = int(df["Date"].duplicated().sum())
    df = df.drop_duplicates("Date", keep="first").set_index("Date")
    info["first_date"] = str(df.index.min().date())
    info["last_date"] = str(df.index.max().date())
    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    info["calendar_days"] = int(len(full))
    info["days_missing_from_file"] = int(len(full) - len(df))
    df = df.reindex(full)
    df.index.name = "Date"

    cols = ["AQI"] + [c for c in POLLUTANTS if c in df.columns]
    neg = {}
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        n = int((df[c] < 0).sum())
        if n:
            neg[c] = n
        df.loc[df[c] < 0, c] = np.nan
    info["negative_values_blanked"] = neg
    info["missing_pct"] = {c: round(float(df[c].isna().mean() * 100), 1) for c in cols}
    info["longest_gap_days"] = {c: longest_gap(df[c]) for c in cols}
    info["observed_aqi_days"] = int(df["AQI"].notna().sum())

    # outliers flagged with 1.5*IQR (KEPT, only counted)
    out = {}
    for c in cols:
        s = df[c].dropna()
        if len(s):
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            out[c] = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
    info["iqr_outliers_flagged_kept"] = out
    return df, info


def build_dataset(df, info):
    """Drop sparse pollutants, forward-fill (limit 3), make lag-1 features."""
    kept = [c for c in POLLUTANTS if c in df.columns
            and info["missing_pct"][c] <= MAX_MISSING_FRACTION * 100]
    dropped = [c for c in POLLUTANTS if c in df.columns and c not in kept]
    info["pollutants_kept"] = kept
    info["pollutants_dropped_over_30pct_missing"] = dropped

    cols = ["AQI"] + kept
    filled = df[cols].ffill(limit=MAX_FFILL_DAYS)      # causal: past values only
    lag = filled.shift(1).add_suffix("_lag1")           # day t-1 info
    data = pd.concat([df["AQI"].rename("AQI"), lag], axis=1)
    data["Year"] = data.index.year
    data["Month"] = data.index.month
    data["Season"] = data.index.month.map(season_of)
    # target rows: AQI actually observed AND all lag-1 predictors complete
    model_df = data.dropna(subset=["AQI"] + [f"{c}_lag1" for c in cols]).copy()
    info["model_rows"] = int(len(model_df))
    return model_df, kept, filled


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def model_specs(kept):
    """Each model is a list of (variable, power) terms, all measured at t-1."""
    specs = {"Simple linear": [("AQI", 1)],
             "Multiple linear": [("AQI", 1)] + [(c, 1) for c in kept]}
    if "PM2.5" in kept:
        specs["Polynomial (deg 2)"] = [("AQI", 1), ("AQI", 2), ("PM2.5", 1), ("PM2.5", 2)]
    else:
        specs["Polynomial (deg 2)"] = [("AQI", 1), ("AQI", 2)]
    return specs


def design(df, terms):
    cols = [(df[f"{v}_lag1"].values ** p) for v, p in terms]
    return np.column_stack(cols)


def fit_ols(X, y):
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


def predict_ols(beta, X):
    return beta[0] + X @ beta[1:]


def metrics(y, p):
    err = y - p
    sse = float(np.sum(err ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {"R2": 1 - sse / sst if sst > 0 else float("nan"),
            "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2)))}


def equation(name, terms, beta):
    s = f"AQI(t) = {beta[0]:.4f}"
    for (v, p), b in zip(terms, beta[1:]):
        label = f"{v}(t-1)" + ("^2" if p == 2 else "")
        s += f" {'+' if b >= 0 else '-'} {abs(b):.4f}*{label}"
    return s


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
def cmd_train(args):
    os.makedirs(args.out, exist_ok=True)
    df, info = load_city(args.raw)
    model_df, kept, _ = build_dataset(df, info)

    print("\n=== DATA VALIDATION (paste this into the chat) ===")
    for k in ["city", "raw_rows_for_city", "unparseable_dates", "duplicate_dates_dropped",
              "first_date", "last_date", "calendar_days", "days_missing_from_file",
              "observed_aqi_days", "negative_values_blanked", "missing_pct",
              "longest_gap_days", "iqr_outliers_flagged_kept", "pollutants_kept",
              "pollutants_dropped_over_30pct_missing", "model_rows"]:
        print(f"{k}: {info[k]}")
    if info["model_rows"] < 700:
        print("\nWARNING: fewer than ~700 usable rows. Agreed rule: consider the "
              "CPCB CCR fallback and tell your guide/chat.")
    if len(model_df) < 50:
        sys.exit("Too few usable rows to train. Check the data.")

    n_train = int(len(model_df) * TRAIN_FRACTION)
    train, test = model_df.iloc[:n_train], model_df.iloc[n_train:]
    info["train_rows"], info["test_rows"] = int(len(train)), int(len(test))
    info["train_period"] = f"{train.index.min().date()} to {train.index.max().date()}"
    info["test_period"] = f"{test.index.min().date()} to {test.index.max().date()}"
    print(f"train_rows: {len(train)} ({info['train_period']})")
    print(f"test_rows: {len(test)} ({info['test_period']})")

    ytr, yte = train["AQI"].values, test["AQI"].values
    results, models, preds = {}, {}, {}

    base_te = test["AQI_lag1"].values
    base_tr = train["AQI_lag1"].values
    results["Persistence baseline"] = {
        "terms": None, **{f"Test_{k}": v for k, v in metrics(yte, base_te).items()},
        "Train_RMSE": metrics(ytr, base_tr)["RMSE"]}
    preds["Persistence baseline"] = base_te

    for name, terms in model_specs(kept).items():
        beta = fit_ols(design(train, terms), ytr)           # TRAIN only
        p_te = predict_ols(beta, design(test, terms))
        p_tr = predict_ols(beta, design(train, terms))
        m = metrics(yte, p_te)
        results[name] = {"terms": terms, "Test_R2": m["R2"], "Test_MAE": m["MAE"],
                         "Test_RMSE": m["RMSE"], "Train_RMSE": metrics(ytr, p_tr)["RMSE"]}
        models[name] = {"terms": terms, "beta": [float(b) for b in beta]}
        preds[name] = p_te

    best = min(models, key=lambda n: results[n]["Test_RMSE"])
    beats = results[best]["Test_RMSE"] < results["Persistence baseline"]["Test_RMSE"]

    print("\n=== RESULTS (TEST SET decides the best model) ===")
    print(f"{'Model':<24}{'Test R2':>10}{'Test MAE':>11}{'Test RMSE':>12}{'Train RMSE':>12}")
    for n, r in results.items():
        print(f"{n:<24}{r['Test_R2']:>10.4f}{r['Test_MAE']:>11.3f}"
              f"{r['Test_RMSE']:>12.3f}{r['Train_RMSE']:>12.3f}")
    print(f"\nBest regression model (lowest test RMSE): {best}")
    print("Does it beat the 'tomorrow = today' baseline on test RMSE? "
          + ("YES" if beats else "NO - report this honestly"))
    print("\n=== FITTED EQUATIONS (trained on train rows only) ===")
    for n, m in models.items():
        print(f"{n}: {equation(n, m['terms'], m['beta'])}")

    with open(os.path.join(args.out, "aqi_model.json"), "w") as f:
        json.dump({"best_model": best, "beats_baseline": bool(beats), "info": info,
                   "results": {k: {kk: vv for kk, vv in v.items() if kk != "terms"}
                               for k, v in results.items()},
                   "models": models}, f, indent=2, default=str)
    out = test[["AQI"]].rename(columns={"AQI": "Actual"}).copy()
    for n, p in preds.items():
        out[n] = p
    out.to_csv(os.path.join(args.out, "test_predictions.csv"))
    model_df.to_csv(os.path.join(args.out, "model_data.csv"))
    df.to_csv(os.path.join(args.out, "cleaned_data.csv"))

    plt.figure(figsize=(12, 5))
    plt.plot(test.index, yte, label="Actual", color="black", lw=1.2)
    plt.plot(test.index, preds[best], label=f"Predicted ({best})", color="tab:red", lw=1)
    plt.plot(test.index, base_te, label="Persistence", color="tab:blue", lw=0.8, alpha=0.5)
    plt.title(f"Jaipur next-day AQI: actual vs predicted (test set)")
    plt.xlabel("Date"); plt.ylabel("AQI"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out, "actual_vs_predicted.png"), dpi=150)
    plt.close()
    print(f"\nSaved files in ./{args.out}/ : aqi_model.json, test_predictions.csv, "
          "model_data.csv, cleaned_data.csv, actual_vs_predicted.png")


def _load_model(out):
    p = os.path.join(out, "aqi_model.json")
    if not os.path.exists(p):
        sys.exit("Run the 'train' command first.")
    with open(p) as f:
        return json.load(f)


def _predict_from_values(saved, vals):
    best = saved["best_model"]
    m = saved["models"][best]
    X = np.array([[vals[v] ** p for v, p in m["terms"]]], dtype=float)
    return best, float(predict_ols(np.array(m["beta"]), X)[0])


def cmd_forecast(args):
    saved = _load_model(args.out)
    df, info = load_city(args.raw)
    _, kept, filled = build_dataset(df, info)
    cols = ["AQI"] + kept
    ok = filled[cols].dropna()
    if ok.empty:
        sys.exit("No complete record available.")
    last = ok.index[-1]
    vals = {c: float(ok.loc[last, c]) for c in cols}
    best, p = _predict_from_values(saved, vals)
    print(f"Last complete record: {last.date()}  values: {vals}")
    print(f"Forecast AQI for {(last + pd.Timedelta(days=1)).date()} ({best}): {p:.1f}")


def cmd_predict(args):
    saved = _load_model(args.out)
    need = {v for v, _ in saved["models"][saved["best_model"]]["terms"]}
    vals = {}
    for v in need:
        a = getattr(args, ARG_NAMES[v])
        if a is None:
            sys.exit(f"Best model needs today's {v}: pass --{ARG_NAMES[v]} VALUE")
        vals[v] = a
    best, p = _predict_from_values(saved, vals)
    print(f"Tomorrow's predicted AQI ({best}): {p:.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    for name, fn in [("train", cmd_train), ("forecast", cmd_forecast)]:
        sp = sub.add_parser(name)
        sp.add_argument("--raw", default="Dataset/city_day.csv")
        sp.add_argument("--out", default=OUT_DIR)
        sp.set_defaults(fn=fn)
    sp = sub.add_parser("predict")
    sp.add_argument("--out", default=OUT_DIR)
    for v, a in ARG_NAMES.items():
        sp.add_argument(f"--{a}", type=float, help=f"today's {v}")
    sp.set_defaults(fn=cmd_predict)
    args = ap.parse_args()
    if not args.cmd:
        args = ap.parse_args(["train"])
    args.fn(args)


if __name__ == "__main__":
    main()
