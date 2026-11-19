"""Rolling-origin backtest at 3 / 6 / 9 months -> outputs/backtest/.

Every model forecasts log(y[t+h] / y[t]) with P10/P50/P90. Data for month t is assumed known at origin t.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import data as D

warnings.filterwarnings("ignore")
OUT = D.OUT / "backtest"
OUT.mkdir(parents=True, exist_ok=True)

HORIZONS = (3, 6, 9)
QS = (0.1, 0.5, 0.9)
FIRST_ORIGIN = pd.Timestamp("2020-01-01")
BOM_IDS = ["RM6_NE_IQI_433520", "RMAirSeT_IIQ_24kV"]
TARGET_IDS = list(D.TARGETS) + BOM_IDS
DRIVER_IDS = ["ED01", "ED02", "E09", "S04", "S03", "S01", "S06"]
FAMILY = {k: v[1] for k, v in D.TARGETS.items()} | {b: "BOM" for b in BOM_IDS}
GBM_REFIT_EVERY = 3  # months


def build_features(p):
    """(month, series) frame of features known at month t."""
    L = np.log(p)
    drv = pd.DataFrame({f"drv_{d}_r{k}": L[d].diff(k) for d in DRIVER_IDS for k in (1, 3)})
    fam_code = {f: i for i, f in enumerate(sorted(set(FAMILY.values())))}
    frames = []
    for s in TARGET_IDS:
        x = pd.DataFrame({f"r{k}": L[s].diff(k) for k in (1, 3, 6, 12)})
        x["dev12"] = L[s] - L[s].rolling(12).mean()
        x["vol12"] = L[s].diff().rolling(12).std()
        x = x.join(drv)
        x["family"] = fam_code[FAMILY[s]]
        x["series"] = s
        frames.append(x)
    return pd.concat(frames).set_index("series", append=True).sort_index()


def target_returns(p, h):
    L = np.log(p[TARGET_IDS])
    return L.shift(-h) - L


def empirical_spread(hist_ret):
    """P10 / P90 of past h-month log-returns, centred on their median."""
    r = hist_ret.dropna()
    if len(r) < 12:
        return -0.1, 0.1
    return r.quantile(0.1) - r.median(), r.quantile(0.9) - r.median()


def run():
    t0 = time.time()
    p = D.panel()
    X = build_features(p)
    cols = list(X.columns)
    rows = []
    for h in HORIZONS:
        Y = target_returns(p, h)
        origins = [m for m in p.index if m >= FIRST_ORIGIN and m + pd.DateOffset(months=h) <= p.index[-1]]
        gbm, last_fit = None, None
        for t in origins:
            train_end = t - pd.DateOffset(months=h)  # last origin whose outcome is known at t
            if gbm is None or (t - last_fit).days > 28 * GBM_REFIT_EVERY:
                Xtr = X.loc[(slice(None, train_end), slice(None)), :]
                ytr = Y.stack().rename("y").reindex(Xtr.index)
                ok = ytr.notna() & Xtr.notna().all(axis=1)
                gbm = {q: HistGradientBoostingRegressor(
                    loss="quantile", quantile=q, max_iter=150, learning_rate=0.05, max_depth=3,
                    min_samples_leaf=20, categorical_features=[cols.index("family")], random_state=0,
                ).fit(Xtr[ok].values, ytr[ok].values) for q in QS}
                last_fit = t
            Xt = X.loc[t]
            gbm_pred = {q: pd.Series(gbm[q].predict(Xt[cols].values), index=Xt.index) for q in QS}

            for s in TARGET_IDS:
                y_t, actual = p.at[t, s], p.at[t + pd.DateOffset(months=h), s]
                lo, hi = empirical_spread(Y[s].loc[:train_end])
                preds = {"naive": (lo, 0.0, hi)}
                drift = np.log(p[s]).diff().loc[:t].tail(36).mean() * h
                preds["drift"] = (drift + lo, drift, drift + hi)
                try:
                    ets = ExponentialSmoothing(np.log(p[s].loc[:t].values), trend="add", damped_trend=True,
                                               initialization_method="estimated").fit()
                    e = float(ets.forecast(h)[-1] - np.log(y_t))
                except Exception:
                    e = 0.0
                preds["ets_local"] = (e + lo, e, e + hi)
                xs = X.xs(s, level="series").loc[:train_end].drop(columns=["family"])
                ys = Y[s].reindex(xs.index)
                ok = ys.notna() & xs.notna().all(axis=1)
                if ok.sum() >= 24:
                    rg = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(xs[ok].values, ys[ok].values)
                    res = ys[ok].values - rg.predict(xs[ok].values)
                    mu = float(rg.predict(X.loc[(t, s)].drop("family").values.reshape(1, -1).astype(float))[0])
                    preds["ridge_local"] = (mu + np.quantile(res, 0.1), mu, mu + np.quantile(res, 0.9))
                preds["gbm_global"] = tuple(gbm_pred[q].get(s, np.nan) for q in QS)

                for name, q in preds.items():
                    q10, q50, q90 = np.sort(q)
                    rows.append({"model": name, "series": s, "h": h, "origin": t, "y_t": y_t, "actual": actual,
                                 "p10": y_t * np.exp(q10), "p50": y_t * np.exp(q50), "p90": y_t * np.exp(q90)})
        print(f"h={h}: {len(origins)} origins ({time.time() - t0:.0f}s)")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "backtest_predictions.csv", index=False)
    return res


def score(res, save=False):
    """Per-series and summary scores; only the full run saves (so sub-period checks never overwrite)."""
    r = res.copy()
    r["ae"] = (r["p50"] - r["actual"]).abs()
    r["ape"] = r["ae"] / r["actual"]
    r["covered"] = (r["actual"] >= r["p10"]) & (r["actual"] <= r["p90"])
    r["pinball"] = sum(np.maximum(q * (r["actual"] - r[f"p{int(q * 100)}"]), (q - 1) * (r["actual"] - r[f"p{int(q * 100)}"]))
                       for q in QS) / 3 / r["actual"]
    naive_ae = r[r.model == "naive"].set_index(["series", "h", "origin"])["ae"]
    r["naive_ae"] = naive_ae.reindex(pd.MultiIndex.from_frame(r[["series", "h", "origin"]])).values
    per = r.groupby(["model", "series", "h"]).agg(
        mape=("ape", "mean"), mae=("ae", "mean"), naive_mae=("naive_ae", "mean"),
        coverage80=("covered", "mean"), pinball=("pinball", "mean"), n=("ape", "size")).reset_index()
    per["rel_mae"] = per["mae"] / per["naive_mae"]
    summ = per.groupby(["model", "h"]).agg(
        mean_mape=("mape", "mean"), median_rel_mae=("rel_mae", "median"),
        series_beating_naive=("rel_mae", lambda x: int((x < 0.97).sum())),
        coverage80=("coverage80", "mean"), pinball=("pinball", "mean")).reset_index()
    if save:
        per.to_csv(OUT / "scores_per_series.csv", index=False, float_format="%.4f")
        summ.to_csv(OUT / "scores_summary.csv", index=False, float_format="%.4f")
    return per, summ


if __name__ == "__main__":
    per, summ = score(run(), save=True)
    print(summ.pivot(index="model", columns="h", values=["median_rel_mae", "coverage80"]).round(3).to_string())
