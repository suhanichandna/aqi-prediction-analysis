"""
eda.py - Exploratory Data Analysis for Jaipur AQI (pandas + numpy + matplotlib only).

Usage (from repository root):
    python Scripts/eda.py --raw Dataset/city_day.csv

Creates eda_output/ with summary statistics (CSV + printed) and PNG charts.
Uses the same cleaning (no filling) as aqi_model.py so numbers agree.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aqi_model import load_city, season_of, POLLUTANTS  # noqa: E402

BANDS = [(0, 50, "Good"), (51, 100, "Satisfactory"), (101, 200, "Moderate"),
         (201, 300, "Poor"), (301, 400, "Very Poor"), (401, 10_000, "Severe")]


def band(v):
    if pd.isna(v):
        return np.nan
    for lo, hi, name in BANDS:
        if lo <= v <= hi:
            return name
    return np.nan


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="Dataset/city_day.csv")
    ap.add_argument("--out", default="eda_output")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df, info = load_city(a.raw)          # negatives blanked, NOT filled
    cols = ["AQI"] + [c for c in POLLUTANTS if c in df.columns]
    df["Year"], df["Month"] = df.index.year, df.index.month
    df["Season"] = df["Month"].map(season_of)
    df["Band"] = df["AQI"].map(band)

    # ---- descriptive statistics --------------------------------------------
    rows = {}
    for c in cols:
        s = df[c].dropna()
        mode = s.round(0).mode()
        rows[c] = {"count": len(s), "mean": s.mean(), "median": s.median(),
                   "mode(rounded)": mode.iloc[0] if len(mode) else np.nan,
                   "min": s.min(), "max": s.max(), "range": s.max() - s.min(),
                   "variance": s.var(), "std": s.std(), "Q1": s.quantile(.25),
                   "Q3": s.quantile(.75), "skewness": s.skew()}
    stats = pd.DataFrame(rows).T.round(3)
    stats.to_csv(os.path.join(a.out, "descriptive_statistics.csv"))
    print("\n=== DESCRIPTIVE STATISTICS ===")
    print(stats.to_string())

    corr = df[cols].corr().round(3)
    corr.to_csv(os.path.join(a.out, "correlation_matrix.csv"))
    print("\n=== CORRELATION MATRIX (same-day) ===")
    print(corr.to_string())

    band_order = [b[2] for b in BANDS]
    bc = df["Band"].value_counts().reindex(band_order).fillna(0).astype(int)
    print("\n=== AQI CATEGORY COUNTS (observed days) ===")
    print(bc.to_string())

    yearly = df.groupby("Year")["AQI"].agg(["count", "mean", "median", "max"]).round(2)
    monthly = df.groupby("Month")["AQI"].agg(["count", "mean", "median", "max"]).round(2)
    seasonal = df.groupby("Season")["AQI"].agg(["count", "mean", "median", "max"]).round(2)
    for nm, t in [("yearly", yearly), ("monthly", monthly), ("seasonal", seasonal)]:
        t.to_csv(os.path.join(a.out, f"aqi_{nm}.csv"))
        print(f"\n=== AQI {nm.upper()} ===")
        print(t.to_string())

    # ---- charts --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df["AQI"], lw=0.8)
    ax.set(title="Jaipur daily AQI over time", xlabel="Date", ylabel="AQI")
    save(fig, a.out, "aqi_time_series.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(bc.index, bc.values, color="tab:orange")
    ax.set(title="AQI category distribution (CPCB bands)", ylabel="Days")
    plt.setp(ax.get_xticklabels(), rotation=25)
    save(fig, a.out, "aqi_category_distribution.png")

    for nm, t, lab in [("monthly", monthly, "Month"), ("yearly", yearly, "Year")]:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t.index, t["mean"], marker="o")
        ax.set(title=f"Mean AQI by {lab.lower()}", xlabel=lab, ylabel="Mean AQI")
        save(fig, a.out, f"aqi_{nm}_trend.png")
    order = ["Winter", "Summer", "Monsoon", "Post-monsoon"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(order, seasonal["mean"].reindex(order).values, color="tab:green")
    ax.set(title="Mean AQI by season", ylabel="Mean AQI")
    save(fig, a.out, "aqi_seasonal_trend.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45)
    ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im)
    ax.set_title("Correlation heatmap")
    save(fig, a.out, "correlation_heatmap.png")

    n = len(cols)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(14, 7))
    for ax, c in zip(axes.ravel(), cols):
        ax.hist(df[c].dropna(), bins=40, color="tab:blue")
        ax.set_title(c)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    save(fig, a.out, "pollutant_distributions.png")

    for p in ("PM2.5", "PM10"):
        if p in df.columns:
            d = df[[p, "AQI"]].dropna()
            if len(d) > 2:
                m, b = np.polyfit(d[p], d["AQI"], 1)
                r = d[p].corr(d["AQI"])
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.scatter(d[p], d["AQI"], s=8, alpha=0.5)
                xs = np.array([d[p].min(), d[p].max()])
                ax.plot(xs, m * xs + b, color="red", label=f"trend (r={r:.2f})")
                ax.set(title=f"{p} vs AQI", xlabel=p, ylabel="AQI")
                ax.legend()
                save(fig, a.out, f"scatter_{p.replace('.', '')}_vs_aqi.png")
    print(f"\nSaved tables and charts in ./{a.out}/")


if __name__ == "__main__":
    main()
