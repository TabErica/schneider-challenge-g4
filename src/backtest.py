"""Rolling-origin backtest at 3 / 6 / 9 months -> outputs/backtest/.

Targets come from targets.py. At every origin t the panel is cut with preprocess.as_of(): each series loses the same
number of months it lacks at the real origin (Sep-2026), so stainless 304 always ends at t - 1. Every model forecasts
log(y[t+h] / y[a]) with P10/P50/P90, where a = the target's last month known at t; the effective horizon is
h_eff = h + (t - a) (stainless 304 and the market-price BOM indices: h + 1). The BOM indices are rebuilt at every
origin from the cut panel, so they inherit the ragged edge of their inputs.
Univariate models (naive, drift, ETS) use each target's full history; Ridge and the global GBM use the months where
every input is known. No forward fill anywhere.

    python backtest.py [--origin 2026-09] [--nowcast] [--drop-provisional] [--bom-weights base|origin]

--origin            last origin (default: the data cutoff, Sep-2026). An earlier origin is a held-back check: the run
                    only sees data as of that origin, and holdout_check.csv compares its forecasts with the months after.
--nowcast           fill stainless 304's missing month at every origin (preprocess.nowcast) instead of h + 1.
--drop-provisional  remove the last 2 months of the Eurostat PPIs at every origin (sensitivity; baseline keeps them).
--bom-weights       base = Sep-2026 cost shares (submission, default); origin = shares from prices known at each origin.
Non-default options write to outputs/backtest/<tag>/ so the baseline is never overwritten.
"""
from __future__ import annotations

import argparse
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
import preprocess as P
import targets as T

warnings.filterwarnings("ignore")
OUT = D.OUT / "backtest"

HORIZONS = (3, 6, 9)
QS = (0.1, 0.5, 0.9)
FIRST_ORIGIN = pd.Timestamp("2020-01-01")
TARGET_IDS = T.ids()
BOM_IDS = T.ids("bom_index")
MATERIAL_IDS = T.ids("material")
DRIVER_IDS = ["ED01", "ED02", "E09", "S04", "S03", "S01", "S06"]
FAMILY = {t.id: t.family for t in T.REGISTRY}
GBM_REFIT_EVERY = 3  # months


def months(a, b):
    """Whole months from a to b."""
    return (b.to_period("M") - a.to_period("M")).n


def build_features(p):
    """(month, series) frame of features known at month t."""
    L = np.log(p[list(dict.fromkeys(TARGET_IDS + DRIVER_IDS))])
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


def empirical_spread(hist_ret):
    """P10 / P90 of past h-month log-returns, centred on their median."""
    r = hist_ret.dropna()
    if len(r) < 12:
        return -0.1, 0.1
    return r.quantile(0.1) - r.median(), r.quantile(0.9) - r.median()


class Bom:
    """BOM index columns for an as_of() panel. base: Sep-2026 cost shares and references (submission);
    origin: cost shares and references from the latest prices known at the origin (no look-ahead)."""

    def __init__(self, full, mode):
        self.mode = mode
        if mode == "base":
            self.base_costs = D.bom_cost_table(full, full)
            self.base_refs = {v: D.bom_refs(full, self.base_costs, v) for v in ("proxy", "market")}
        elif mode != "origin":
            raise ValueError(f"bom weights must be 'base' or 'origin', not {mode!r}")

    def weights(self, pt, t):
        if self.mode == "base":
            return self.base_costs, self.base_refs
        costs = D.bom_cost_table(pt, pt, at=t)
        return costs, {v: D.bom_refs(pt, costs, v, at=t) for v in ("proxy", "market")}

    def columns(self, prices, costs, refs):
        return pd.concat([D.bom_index(prices, costs, v, refs[v]) for v in ("proxy", "market")], axis=1)[BOM_IDS]


def check_history(full):
    """targets.history_start must be the first month of each target in the data."""
    for t in T.REGISTRY:
        first = full[t.id].first_valid_index()
        if first is None or f"{first:%Y-%m}" != t.history_start:
            raise ValueError(f"{t.id}: registry history_start {t.history_start}, data starts {first}")


def run(origin=P.CUTOFF, nowcast=False, drop_provisional=False, bom_weights="base"):
    """Predictions for every origin FIRST_ORIGIN..origin and horizon, plus the nowcast log and training sizes.
    Rows are kept when the outcome is known as of `origin`, and always at `origin` itself (the forecast)."""
    t0, origin = time.time(), pd.Timestamp(origin)
    cut = P.load()  # full history, cut at the data cutoff (Sep-2026)
    lags = P.relative_lags(cut)
    bom = Bom(cut, bom_weights)
    base = bom if bom_weights == "base" else Bom(cut, "base")
    full = P.as_of(cut, origin, lags)  # everything the run may see: data as of `origin` (outcomes come from here)
    full = full.join(base.columns(full, base.base_costs, base.base_refs))
    check_history(full)

    rows, nowlog, gbm, last_fit = [], [], {}, {}
    for t in pd.date_range(FIRST_ORIGIN, origin, freq="MS"):
        pt = P.as_of(cut, t, lags, drop_provisional=drop_provisional)
        if nowcast:
            pt, nl = P.nowcast(pt, t)
            nowlog += nl
        costs, refs = bom.weights(pt, t)
        pt = pt.join(bom.columns(pt, costs, refs))
        # outcomes, with the same BOM weights as the inputs (data as of `origin`, never later)
        actual = full if bom_weights == "base" else full[MATERIAL_IDS].join(bom.columns(full, costs, refs))
        prov = P.provisional_months(pt)
        X = build_features(pt)
        cols = list(X.columns)
        L = np.log(pt[TARGET_IDS])
        anchor = {s: pt[s].last_valid_index() for s in TARGET_IDS}
        lag = {s: months(anchor[s], t) for s in TARGET_IDS}

        for h in HORIZONS:
            if h not in gbm or months(last_fit[h], t) >= GBM_REFIT_EVERY:
                # every (month, series) whose h_eff-month outcome is known at t
                ytr = pd.concat({s: (L[s].shift(-(h + lag[s])) - L[s]) for s in TARGET_IDS}, axis=1).stack()
                Xtr = X.reindex(ytr.index)
                ok = ytr.notna() & Xtr.notna().all(axis=1)
                gbm[h] = {q: HistGradientBoostingRegressor(
                    loss="quantile", quantile=q, max_iter=150, learning_rate=0.05, max_depth=3,
                    min_samples_leaf=20, categorical_features=[cols.index("family")], random_state=0,
                ).fit(Xtr[ok].values, ytr[ok].values) for q in QS}
                gbm[h]["n"], last_fit[h] = int(ok.sum()), t

        for s in TARGET_IDS:
            a = anchor[s]
            hist = L[s].loc[:a].dropna()
            y_a = pt.at[a, s]
            try:
                ets = ExponentialSmoothing(hist.values, trend="add", damped_trend=True,
                                           initialization_method="estimated").fit()
                ets_path = ets.forecast(max(HORIZONS) + lag[s])
            except Exception:
                ets_path = None
            xs = X.xs(s, level="series").loc[:a].drop(columns=["family"])
            xa = X.loc[(a, s)].drop("family").values.reshape(1, -1).astype(float)

            for h in HORIZONS:
                target = t + pd.DateOffset(months=h)
                y_true = actual.at[target, s] if target in actual.index else np.nan
                if pd.isna(y_true) and t != origin:
                    continue
                he = h + lag[s]
                ys = L[s].shift(-he) - L[s]  # h_eff-month log changes known at t
                lo, hi = empirical_spread(ys)
                preds = {"naive": ((lo, 0.0, hi), len(hist))}
                drift = hist.diff().tail(36).mean() * he
                preds["drift"] = ((drift + lo, drift, drift + hi), len(hist))
                e = float(ets_path[he - 1] - np.log(y_a)) if ets_path is not None else 0.0
                preds["ets_local"] = ((e + lo, e, e + hi), len(hist))
                yy = ys.reindex(xs.index)
                ok = yy.notna() & xs.notna().all(axis=1)
                if ok.sum() >= 24 and np.isfinite(xa).all():
                    rg = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(xs[ok].values, yy[ok].values)
                    res = yy[ok].values - rg.predict(xs[ok].values)
                    mu = float(rg.predict(xa)[0])
                    preds["ridge_local"] = ((mu + np.quantile(res, 0.1), mu, mu + np.quantile(res, 0.9)), int(ok.sum()))
                xg = X.loc[[(a, s)], cols].values
                if np.isfinite(np.delete(xg, cols.index("family"), axis=1)).all():
                    preds["gbm_global"] = (tuple(float(gbm[h][q].predict(xg)[0]) for q in QS), gbm[h]["n"])

                for name, (q, n_train) in preds.items():
                    q10, q50, q90 = np.sort(q)
                    rows.append({"model": name, "series": s, "h": h, "h_eff": he, "origin": t, "anchor": a,
                                 "y_t": y_a, "actual": y_true, "p10": y_a * np.exp(q10), "p50": y_a * np.exp(q50),
                                 "p90": y_a * np.exp(q90), "n_train": n_train,
                                 "anchor_provisional": a in prov.get(s, [])})
        if t.month == 1 or t == origin:
            print(f"  origin {t:%Y-%m} ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(nowlog)


def score(res, save_dir=None):
    """Per-series and summary scores on the rows whose outcome is known."""
    r = res[res["actual"].notna()].copy()
    r["ae"] = (r["p50"] - r["actual"]).abs()
    r["ape"] = r["ae"] / r["actual"]
    r["covered"] = (r["actual"] >= r["p10"]) & (r["actual"] <= r["p90"])
    r["pinball"] = sum(np.maximum(q * (r["actual"] - r[f"p{int(q * 100)}"]), (q - 1) * (r["actual"] - r[f"p{int(q * 100)}"]))
                       for q in QS) / 3 / r["actual"]
    naive_ae = r[r.model == "naive"].set_index(["series", "h", "origin"])["ae"]
    r["naive_ae"] = naive_ae.reindex(pd.MultiIndex.from_frame(r[["series", "h", "origin"]])).values
    per = r.groupby(["model", "series", "h"]).agg(
        mape=("ape", "mean"), mae=("ae", "mean"), naive_mae=("naive_ae", "mean"),
        coverage80=("covered", "mean"), pinball=("pinball", "mean"), n=("ape", "size"),
        h_eff=("h_eff", "max"), n_train_first=("n_train", "first"), n_train_last=("n_train", "last")).reset_index()
    per["rel_mae"] = per["mae"] / per["naive_mae"]
    per["in_submission"] = per["series"].map(lambda s: T.BY_ID[s].in_submission)
    summ = summarise(per[per.in_submission])
    if save_dir is not None:
        per.to_csv(save_dir / "scores_per_series.csv", index=False, float_format="%.4f")
        summ.to_csv(save_dir / "scores_summary.csv", index=False, float_format="%.4f")
    return per, summ


def summarise(per):
    return per.groupby(["model", "h"]).agg(
        mean_mape=("mape", "mean"), median_rel_mae=("rel_mae", "median"),
        series_beating_naive=("rel_mae", lambda x: int((x < 0.97).sum())),
        coverage80=("coverage80", "mean"), pinball=("pinball", "mean"), n_series=("series", "nunique")).reset_index()


def training_samples(res, origin):
    """Per target (h = 3): months of history for the univariate models at the first and last origin, and Ridge
    training rows (window where every input is known) at the first origin Ridge could run and at the last one."""
    r = res.sort_values("origin")
    nv = r[(r.model == "naive") & (r.h == 3)].groupby("series")
    rg = r[(r.model == "ridge_local") & (r.h == 3)].groupby("series")
    out = pd.DataFrame({
        "history_start": {t.id: t.history_start for t in T.REGISTRY},
        "relative_lag_m": (r["h_eff"] - r["h"]).groupby(r["series"]).max(),
        f"n_hist_{FIRST_ORIGIN:%Y-%m}": nv["n_train"].first(), f"n_hist_{origin:%Y-%m}": nv["n_train"].last(),
        "ridge_first_origin": rg["origin"].first().dt.strftime("%Y-%m"), "n_ridge_first": rg["n_train"].first(),
        f"n_ridge_{origin:%Y-%m}": rg["n_train"].last(),
    })
    out.index.name = "series"
    return out.loc[TARGET_IDS]


def holdout_check(res, origin, bom_weights="base"):
    """Forecasts made at an earlier origin against the outcomes published after it (up to the data cutoff).
    BOM rows only with the Sep-2026 weights (the outcome index of other weights is not stored)."""
    full = D.panel()  # full history up to the cutoff, BOM indices with Sep-2026 weights, no fill
    f = res[res.origin == origin].copy()
    if bom_weights != "base":
        f = f[~f.series.isin(BOM_IDS)]
    f["target_month"] = f["origin"] + f["h"].map(lambda h: pd.DateOffset(months=h))
    f["actual"] = [full.at[m, s] if m in full.index else np.nan for m, s in zip(f.target_month, f.series)]
    f = f[f["actual"].notna()]
    f["error"] = f["p50"] / f["actual"] - 1
    f["naive_error"] = f["y_t"] / f["actual"] - 1
    f["covered"] = (f["actual"] >= f["p10"]) & (f["actual"] <= f["p90"])
    return f[["model", "series", "h", "h_eff", "origin", "target_month", "y_t", "p10", "p50", "p90", "actual",
              "error", "naive_error", "covered"]]


def out_dir(origin, nowcast, drop_provisional, bom_weights):
    tag = [f"origin_{origin:%Y-%m}"] if origin != P.CUTOFF else []
    tag += ["nowcast"] * nowcast + ["drop_provisional"] * drop_provisional
    tag += [f"bom_weights_{bom_weights}"] if bom_weights != "base" else []
    d = OUT / "_".join(tag) if tag else OUT
    d.mkdir(parents=True, exist_ok=True)
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--origin", default=f"{P.CUTOFF:%Y-%m}")
    ap.add_argument("--nowcast", action="store_true")
    ap.add_argument("--drop-provisional", action="store_true")
    ap.add_argument("--bom-weights", default="base", choices=["base", "origin"])
    a = ap.parse_args(argv)
    origin = pd.Timestamp(a.origin)
    d = out_dir(origin, a.nowcast, a.drop_provisional, a.bom_weights)
    print(f"backtest -> {d.relative_to(D.ROOT)}")
    res, nowlog = run(origin, a.nowcast, a.drop_provisional, a.bom_weights)
    res.to_csv(d / "backtest_predictions.csv", index=False)
    res[res.origin == origin].to_csv(d / "forecasts_at_origin.csv", index=False)
    if len(nowlog):
        nowlog.to_csv(d / "nowcast_log.csv", index=False)
    training_samples(res, origin).to_csv(d / "training_samples.csv")
    per, summ = score(res, d)
    if origin < P.CUTOFF:
        hc = holdout_check(res, origin, a.bom_weights)
        hc.to_csv(d / "holdout_check.csv", index=False, float_format="%.4f")
        print("held-back check (outcomes after the origin):")
        print(hc.groupby(["model", "h"]).agg(mae_pct=("error", lambda x: x.abs().mean()),
                                             naive_mae_pct=("naive_error", lambda x: x.abs().mean()),
                                             coverage80=("covered", "mean"), n=("error", "size")).round(3).to_string())
    print(summ.pivot(index="model", columns="h", values=["median_rel_mae", "coverage80"]).round(3).to_string())
    return res, per, summ


if __name__ == "__main__":
    main()
