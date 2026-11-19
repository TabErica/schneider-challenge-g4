"""Rolling-origin backtest of baseline forecasters at 3 / 6 / 9 months.

Question it answers: is a naive forecast hard to beat, and is ONE global model (pooled across all
materials) better than one model PER material?

All models forecast the h-month log-return  log(y[t+h] / y[t])  and output p10 / p50 / p90.
Assumption (same as the challenge CSV): every series observed for month t is known at origin t.

Run:  python src/backtest.py
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


# ------------------------------------------------------------------------------------------ features
def build_features(p: pd.DataFrame) -> pd.DataFrame:
    """Long frame indexed by (month, series) with features known at month t."""
    L = np.log(p)
    drv = {}
    for d in DRIVER_IDS:
        drv[f"drv_{d}_r1"] = L[d].diff(1)
        drv[f"drv_{d}_r3"] = L[d].diff(3)
    drv = pd.DataFrame(drv)
    fam_code = {f: i for i, f in enumerate(sorted(set(FAMILY.values())))}
    frames = []
    for s in TARGETS_ALL:
        x = pd.DataFrame(index=p.index)
        for k in (1, 3, 6, 12):
            x[f"r{k}"] = L[s].diff(k)
        x["dev12"] = L[s] - L[s].rolling(12).mean()
        x["vol12"] = L[s].diff().rolling(12).std()
        x = x.join(drv)
        x["family"] = fam_code[FAMILY[s]]
        x["series"] = s
        frames.append(x)
    return pd.concat(frames).set_index("series", append=True).sort_index()


TARGETS_ALL = TARGET_IDS


def target_returns(p: pd.DataFrame, h: int) -> pd.DataFrame:
    L = np.log(p[TARGET_IDS])
    return (L.shift(-h) - L)  # indexed by origin month


# ------------------------------------------------------------------------------------------ models
def empirical_spread(hist_ret: pd.Series) -> tuple[float, float]:
    """10th / 90th pct of historical h-month log-returns, centred on their median."""
    r = hist_ret.dropna()
    if len(r) < 12:
        return -0.1, 0.1
    med = r.median()
    return r.quantile(0.1) - med, r.quantile(0.9) - med


def run():
    t0 = time.time()
    p = D.panel()
    X = build_features(p)
    feat_cols = [c for c in X.columns]
    months = p.index
    rows = []

    for h in HORIZONS:
        Y = target_returns(p, h)
        origins = [m for m in months if m >= FIRST_ORIGIN and m + pd.DateOffset(months=h) <= months[-1]]
        gbm_models, last_fit = None, None
        for t in origins:
            train_end = t - pd.DateOffset(months=h)  # last origin whose outcome is known at t
            # ---------- global GBM (one model for all series), refit every few months
            if gbm_models is None or (t - last_fit).days > 28 * GBM_REFIT_EVERY:
                Xtr = X.loc[(slice(None, train_end), slice(None)), :]
                ytr = Y.stack().rename("y").reindex(Xtr.index)
                ok = ytr.notna() & Xtr.notna().all(axis=1)
                gbm_models = {}
                for q in QS:
                    m = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=150,
                                                      learning_rate=0.05, max_depth=3, min_samples_leaf=20,
                                                      categorical_features=[feat_cols.index("family")],
                                                      random_state=0)
                    m.fit(Xtr[ok].values, ytr[ok].values)
                    gbm_models[q] = m
                last_fit = t
            Xt = X.loc[t]
            gbm_pred = {q: pd.Series(gbm_models[q].predict(Xt[feat_cols].values), index=Xt.index) for q in QS}

            for s in TARGET_IDS:
                y_t = p.at[t, s]
                actual = p.at[t + pd.DateOffset(months=h), s]
                hist = Y[s].loc[:train_end]
                lo, hi = empirical_spread(hist)
                preds = {}
                # naive (random walk): tomorrow = today
                preds["naive"] = (lo, 0.0, hi)
                # drift: recent 36-month average monthly trend continues
                drift = np.log(p[s]).diff().loc[:t].tail(36).mean() * h
                preds["drift"] = (drift + lo, drift, drift + hi)
                # ETS damped trend on log level, per series
                try:
                    ets = ExponentialSmoothing(np.log(p[s].loc[:t].values), trend="add", damped_trend=True,
                                               initialization_method="estimated").fit()
                    e = float(ets.forecast(h)[-1] - np.log(y_t))
                except Exception:
                    e = 0.0
                preds["ets_local"] = (e + lo, e, e + hi)
                # ridge, one model PER series, same features as the global model
                xs = X.xs(s, level="series").loc[:train_end].drop(columns=["family"])
                ys = Y[s].reindex(xs.index)
                ok = ys.notna() & xs.notna().all(axis=1)
                if ok.sum() >= 24:
                    rg = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(xs[ok].values, ys[ok].values)
                    res = ys[ok].values - rg.predict(xs[ok].values)
                    mu = float(rg.predict(X.loc[(t, s)].drop("family").values.reshape(1, -1).astype(float))[0])
                    preds["ridge_local"] = (mu + np.quantile(res, 0.1), mu, mu + np.quantile(res, 0.9))
                # global GBM quantile model
                g = sorted(gbm_pred[q].get(s, np.nan) for q in QS)
                preds["gbm_global"] = tuple(g)
                # simple ensemble: average of naive and global GBM medians, naive spread
                preds["ens_naive_gbm"] = (lo + g[1] / 2, g[1] / 2, hi + g[1] / 2)

                for name, (a, b, c) in preds.items():
                    q10, q50, q90 = np.sort([a, b, c])
                    rows.append({"model": name, "series": s, "h": h, "origin": t, "y_t": y_t, "actual": actual,
                                 "p10": y_t * np.exp(q10), "p50": y_t * np.exp(q50), "p90": y_t * np.exp(q90)})
        print(f"h={h}: {len(origins)} origins done ({time.time() - t0:.0f}s)")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "backtest_predictions.csv", index=False)
    return res


# ------------------------------------------------------------------------------------------ scoring
def pinball(y, q, tau):
    d = y - q
    return np.maximum(tau * d, (tau - 1) * d)


def score(res: pd.DataFrame):
    r = res.copy()
    r["ape"] = (r["p50"] - r["actual"]).abs() / r["actual"]
    r["ae"] = (r["p50"] - r["actual"]).abs()
    r["covered"] = (r["actual"] >= r["p10"]) & (r["actual"] <= r["p90"])
    r["pinball"] = sum(pinball(r["actual"], r[f"p{int(q*100)}"], q) for q in QS) / 3 / r["actual"]
    naive_ae = r[r.model == "naive"].set_index(["series", "h", "origin"])["ae"]
    r["naive_ae"] = naive_ae.reindex(pd.MultiIndex.from_frame(r[["series", "h", "origin"]])).values

    per = r.groupby(["model", "series", "h"]).agg(mape=("ape", "mean"), mae=("ae", "mean"),
                                                   naive_mae=("naive_ae", "mean"), coverage80=("covered", "mean"),
                                                   pinball=("pinball", "mean"), n=("ape", "size")).reset_index()
    per["rel_mae"] = per["mae"] / per["naive_mae"]
    per.to_csv(OUT / "scores_per_series.csv", index=False, float_format="%.4f")

    summ = per.groupby(["model", "h"]).agg(
        mean_mape=("mape", "mean"), median_rel_mae=("rel_mae", "median"),
        series_beating_naive=("rel_mae", lambda x: int((x < 0.97).sum())),
        n_series=("rel_mae", "size"), coverage80=("coverage80", "mean"), pinball=("pinball", "mean"),
    ).reset_index()
    summ.to_csv(OUT / "scores_summary.csv", index=False, float_format="%.4f")
    return per, summ


if __name__ == "__main__":
    res = run()
    per, summ = score(res)
    pd.set_option("display.width", 200)
    print(summ.pivot(index="model", columns="h", values=["mean_mape", "median_rel_mae", "series_beating_naive",
                                                          "coverage80"]).round(3).to_string())
