"""Interval layer: P10 / P90 around any point model's P50 (log returns), from information known at each origin.

Methods (label "<point>+<method>"):
  emp     P50 + P10 / P90 of the target's past h_eff-month log changes, centred on their median (the first
          backtest's band, kept as the reference)
  conf    split conformal on the model's own out-of-sample residuals e = realised log return - P50: lower and upper
          bounds estimated separately (asymmetric), finite-sample conformal ranks
  aci     adaptive conformal inference (Gibbs & Candes 2021): the same residuals at level alpha_t, with
          alpha_{t+1} = alpha_t + gamma * (0.2 - miss) for every outcome revealed (delayed feedback, h months later);
          one gamma per point model (all horizons and targets pooled), chosen on the warm-up window only (bands
          where ACI was active and whose outcome was revealed before the first scored origin); DEFAULT_GAMMA if that
          window is too thin
  ewma    conformal on residuals standardised by sigma * sqrt(h_eff), sigma = EWMA volatility of the target's monthly
          log changes known at the origin (lambda = EWMA_LAMBDA); bounds scaled back with today's sigma
  native  the model's own quantiles (Ridge, GBM)

No leakage: at origin t only residuals whose outcome is published at t are used (outcome month + data lag <= t,
i.e. residual origin <= t - h for lag-0 targets). Fewer than MIN_RES residuals -> the emp band (flag fallback).
Every band is clipped so that P10 <= P50 <= P90.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd

ALPHA = 0.2  # 80% band
MIN_RES = 8
EWMA_LAMBDA = 0.9  # monthly data: half-life about 6.6 months
GAMMAS = (0.0, 0.005, 0.01, 0.02, 0.05)
DEFAULT_GAMMA = 0.005  # Gibbs & Candes (2021)
MIN_TUNE = 200  # active ACI bands needed to tune gamma
METHODS = ("emp", "conf", "aci", "ewma")


def ewma_sigma(log_hist):
    """EWMA volatility of monthly log changes (lambda = EWMA_LAMBDA)."""
    r = log_hist.diff().dropna()
    if len(r) < 2:
        return np.nan
    return float(np.sqrt((r ** 2).ewm(alpha=1 - EWMA_LAMBDA).mean().iloc[-1]))


def bounds(sorted_res, a_lo, a_hi):
    """Conformal lower / upper bound of sorted residuals at tail levels a_lo, a_hi (finite-sample ranks).
    A level beyond the sample (a -> 0) gives the min / max residual."""
    n = len(sorted_res)
    k_hi = min(math.ceil((n + 1) * (1 - a_hi)), n)
    k_lo = max(math.floor((n + 1) * a_lo), 1)
    return sorted_res[k_lo - 1], sorted_res[k_hi - 1]


def _group(g, gammas, clock):
    """One (point model, target, h) stream in origin order -> bands of every method (and ACI per gamma)."""
    n = len(g)
    q50, r = g["q50"].to_numpy(), g["r"].to_numpy()
    lo_e, hi_e = g["emp_lo"].to_numpy(), g["emp_hi"].to_numpy()
    scale = (g["sigma"] * np.sqrt(g["h_eff"])).to_numpy()
    origin, reveal = g["origin"].to_numpy(), g["reveal"].to_numpy()
    # residuals revealed at origin i: prefix j < k[i] (reveal increases with the origin at a fixed h)
    k = np.searchsorted(reveal, origin, side="right")
    e = r - q50
    z = e / scale
    out = {m: np.empty((n, 2)) for m in ("emp", "conf", "ewma")}
    fb = np.zeros(n, bool)
    aci = {gm: np.empty((n, 2)) for gm in gammas}
    alpha = {gm: ALPHA for gm in gammas}
    done = 0
    for i in range(n):
        out["emp"][i] = lo_e[i], hi_e[i]
        ki = k[i]
        ok = ki >= MIN_RES and np.isfinite(e[:ki]).all()
        fb[i] = not ok
        c0 = time.perf_counter()
        if ok:
            es = np.sort(e[:ki])
            out["conf"][i] = bounds(es, ALPHA / 2, ALPHA / 2)
        else:
            out["conf"][i] = lo_e[i], hi_e[i]
        c1 = time.perf_counter()
        zs = z[:ki]
        if ok and np.isfinite(zs).all() and np.isfinite(scale[i]):
            out["ewma"][i] = np.array(bounds(np.sort(zs), ALPHA / 2, ALPHA / 2)) * scale[i]
        else:
            out["ewma"][i] = lo_e[i], hi_e[i]
        c2 = time.perf_counter()
        clock["conf"] += c1 - c0
        clock["ewma"] += c2 - c1
        for gm in gammas:
            for j in range(done, ki):  # outcomes revealed since the last origin, in the order they were issued
                miss = not (aci[gm][j, 0] <= e[j] <= aci[gm][j, 1])
                alpha[gm] += gm * (ALPHA - miss)
            a = min(max(alpha[gm], 0.0), 0.98)
            aci[gm][i] = bounds(es, a / 2, a / 2) if ok else (lo_e[i], hi_e[i])
        done = max(done, ki)
        clock["aci"] += time.perf_counter() - c2
    return out, aci, fb


def apply(points, warm_end):
    """points: one row per (point model, series, h, origin) with q50, r (realised log return or NaN), emp_lo,
    emp_hi, sigma, h_eff, reveal (month the outcome is published). Returns (bands, gamma table, seconds per method).
    warm_end: outcomes revealed up to this month tune the ACI gamma. The aci loop reuses the residuals sorted for
    conf, so it can only run when conf does (same `ok`)."""
    streams, frames = {}, []
    clock = {"emp": 0.0, "conf": 0.0, "ewma": 0.0, "aci": 0.0}
    for key, g in points.sort_values("origin").groupby(["point", "series", "h"], sort=False):
        streams[key] = (g, *_group(g, GAMMAS, clock))

    # ACI gamma per point model (pooled over horizons and targets, so it does not depend on which other models run):
    # coverage closest to 80% (then narrowest) on the bands where ACI was active (not the emp fallback) and whose
    # outcome was revealed by warm_end. Fewer than MIN_TUNE such bands -> DEFAULT_GAMMA.
    tune = []
    for (pt, s, h), (g, out, aci, fb) in streams.items():
        w = (g["reveal"] <= warm_end).to_numpy() & g["r"].notna().to_numpy() & ~fb
        e = (g["r"] - g["q50"]).to_numpy()[w]
        for gm, b in aci.items():
            tune.append(pd.DataFrame({"point": pt, "h": h, "gamma": gm, "covered": (b[w, 0] <= e) & (e <= b[w, 1]),
                                      "width": b[w, 1] - b[w, 0]}))
    tune = pd.concat(tune)
    agg = dict(coverage=("covered", "mean"), width=("width", "mean"), n=("covered", "size"))
    pooled = tune.groupby(["point", "gamma"]).agg(**agg).reset_index().assign(h="all")
    tune = pd.concat([pooled, tune.groupby(["point", "h", "gamma"]).agg(**agg).reset_index()], ignore_index=True)
    tune["gap"] = (tune["coverage"] - (1 - ALPHA)).abs().round(3)
    pooled["gap"] = (pooled["coverage"] - (1 - ALPHA)).abs().round(3)
    best, rules = {}, {}
    for pt in points["point"].unique():
        p = pooled[pooled["point"] == pt]
        if len(p) and p["n"].min() >= MIN_TUNE:
            best[pt], rules[pt] = float(p.sort_values(["gap", "width"])["gamma"].iloc[0]), "tuned"
        else:
            best[pt], rules[pt] = DEFAULT_GAMMA, f"default (fewer than {MIN_TUNE} active warm-up bands)"
    tune["chosen"] = (tune["h"] == "all") & (tune["gamma"] == tune["point"].map(best))
    missing = [pt for pt in best if not tune.loc[tune["chosen"], "point"].eq(pt).any()]
    if missing:
        tune = pd.concat([tune, pd.DataFrame([{"point": pt, "h": "all", "gamma": best[pt], "n": 0, "chosen": True}
                                              for pt in missing])], ignore_index=True)
    tune["rule"] = np.where(tune["chosen"], tune["point"].map(rules), "")

    for (pt, s, h), (g, out, aci, fb) in streams.items():
        bands = dict(out) | {"aci": aci[best[pt]]}
        for m in METHODS:
            frames.append(pd.DataFrame({"point": pt, "interval": m, "series": s, "h": h, "origin": g["origin"].values,
                                        "q50": g["q50"].values, "lo": bands[m][:, 0], "hi": bands[m][:, 1],
                                        "fallback": fb if m != "emp" else False}))
    bands = pd.concat(frames, ignore_index=True)
    bands["q10"] = np.minimum(bands["q50"] + bands["lo"], bands["q50"])
    bands["q90"] = np.maximum(bands["q50"] + bands["hi"], bands["q50"])
    # seconds per method over every stream (conf / ewma include sorting; aci = all gammas of the grid)
    return bands.drop(columns=["lo", "hi"]), tune, clock
