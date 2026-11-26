"""Backtest scores (backtest.score): point accuracy, interval quality, Diebold-Mariano vs naive, by family and period.

Rows: origins >= SCORE_START with a known outcome. A label is "<point model>+<interval method>"; point metrics
(MAE, MASE, MAPE, DM) depend only on the point model.
  rel_mae       MAE / MAE of naive on the same rows
  mase          mean of |error| / scale, scale = mean |y[m + h_eff] - y[m]| over the history known at the origin;
                naive_mase = naive's on the same rows, rel_mase = mase / naive_mase
  complete      the label has a forecast at every origin where naive has one (best_by_family only ranks these)
  pinball       mean pinball loss of P10 / P50 / P90 relative to the outcome; rel_pinball = / naive+emp's
  coverage80    share of outcomes inside [P10, P90]; above / below = share above P90 / below P10 (target 10% / 10%)
  width         mean (P90 - P10) / y at the anchor
  winkler       mean Winkler score (alpha = 0.2) / y at the anchor: width + 10 x distance outside the band
  dm_stat, dm_p Diebold-Mariano on absolute errors vs naive, Newey-West variance with lag h_eff - 1 (negative stat =
                better than naive), two-sided p
  dm_q          Benjamini-Hochberg q value of dm_p. One family = every DM test of one per_series() call on the
                submitted targets (all point models x targets x horizons of that period); the non-submitted targets
                (epoxy inputs) form a family of their own. A test with no p (loss differential all 0, e.g. a model
                whose P50 is naive) is not counted in the family.
  dm_better / dm_worse        targets with q < 0.05 in each direction (the significance used in every conclusion)
  dm_better_raw / dm_worse_raw  the same with the uncorrected p < 0.05 (for comparison only)
Periods by origin (PERIODS, they overlap): 2020-01..2023-06 (stress: Covid, the 2021-22 rally, the energy crisis),
2023-07..end (end = last origin with a known outcome), 2022 alone; plus all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import targets as T
from models.base import QS

ALPHA = 0.2
SCORE_START = pd.Timestamp("2020-01-01")
P1, P2, Y2022 = "2020-01..2023-06", "2023-07..end", "2022"
PERIODS = {P1: ("2020-01-01", "2023-06-01"), P2: ("2023-07-01", None), Y2022: ("2022-01-01", "2022-12-01")}
Q_LEVEL = 0.05
NAIVE = "naive+emp"
FAMILY = {t.id: t.family for t in T.REGISTRY}


def dm_test(d, lag):
    """Diebold-Mariano statistic and two-sided p of a loss differential series d (time order), Newey-West (Bartlett)
    variance with `lag` lags."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10 or np.allclose(d, 0):
        return np.nan, np.nan
    dc = d - d.mean()
    v = dc @ dc / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    if v <= 0:
        return np.nan, np.nan
    stat = d.mean() / np.sqrt(v / n)
    return stat, 2 * stats.norm.sf(abs(stat))


def bh_qvalues(p):
    """Benjamini-Hochberg (1995) q values of the p values `p` (one family); NaN p -> NaN q, not counted in the family."""
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    n = ok.sum()
    if n == 0:
        return q
    pv = p[ok]
    o = np.argsort(pv)
    adj = pv[o] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1].clip(max=1.0)
    out = np.empty(n)
    out[o] = adj
    q[ok] = out
    return q


def in_period(origin, p):
    """Mask of the origins in period p of PERIODS (end None = up to the last origin)."""
    a, b = PERIODS[p]
    m = origin >= pd.Timestamp(a)
    return m & (origin <= pd.Timestamp(b)) if b is not None else m


def scored_rows(res):
    r = res[(res["origin"] >= SCORE_START) & res["actual"].notna()].copy()
    y, a = r["actual"], r["y_t"]
    r["ae"] = (r["p50"] - y).abs()
    r["ape"] = r["ae"] / y
    r["sae"] = r["ae"] / r["mase_scale"]
    r["covered"] = (y >= r["p10"]) & (y <= r["p90"])
    r["above"], r["below"] = y > r["p90"], y < r["p10"]
    r["pinball"] = sum(np.maximum(q * (y - r[f"p{int(q * 100)}"]), (q - 1) * (y - r[f"p{int(q * 100)}"]))
                       for q in QS) / 3 / y
    r["width"] = (r["p90"] - r["p10"]) / a
    r["winkler"] = (r["p90"] - r["p10"] + 2 / ALPHA * ((r["p10"] - y).clip(lower=0) + (y - r["p90"]).clip(lower=0))) / a
    key = pd.MultiIndex.from_frame(r[["series", "h", "origin"]])
    naive = r[r["model"] == NAIVE].set_index(["series", "h", "origin"])
    r["naive_ae"] = naive["ae"].reindex(key).values
    r["naive_pinball"] = naive["pinball"].reindex(key).values
    r["naive_sae"] = naive["sae"].reindex(key).values
    r["family"] = r["series"].map(FAMILY)
    return r


def per_series(r):
    per = r.groupby(["model", "point", "interval", "series", "h"], observed=True).agg(
        mape=("ape", "mean"), mae=("ae", "mean"), naive_mae=("naive_ae", "mean"), mase=("sae", "mean"),
        naive_mase=("naive_sae", "mean"), pinball=("pinball", "mean"), naive_pinball=("naive_pinball", "mean"),
        coverage80=("covered", "mean"),
        above=("above", "mean"), below=("below", "mean"), width=("width", "mean"), winkler=("winkler", "mean"),
        fallback=("fallback", "mean"), n=("ape", "size"), h_eff=("h_eff", "max"),
        n_train_first=("n_train", "first"), n_train_last=("n_train", "last")).reset_index()
    per["rel_mae"] = per["mae"] / per["naive_mae"]
    per["rel_pinball"] = per["pinball"] / per["naive_pinball"]
    per["rel_mase"] = per["mase"] / per["naive_mase"]
    # complete = forecasts at every origin naive has (Ridge needs 24 training rows, lme_fwd covers 4 targets)
    n_naive = per[per["model"] == NAIVE].set_index(["series", "h"])["n"]
    per["complete"] = per["n"].values == n_naive.reindex(pd.MultiIndex.from_frame(per[["series", "h"]])).values
    # DM on the point forecast (same for every interval method of a point model)
    dm = []
    pts = r.drop_duplicates(["point", "series", "h", "origin"]).sort_values("origin")
    for (p, s, h), g in pts.groupby(["point", "series", "h"]):
        st, pv = dm_test(g["ae"] - g["naive_ae"], int(g["h_eff"].max()) - 1)
        dm.append({"point": p, "series": s, "h": h, "dm_stat": st, "dm_p": pv})
    dm = pd.DataFrame(dm, columns=["point", "series", "h", "dm_stat", "dm_p"])
    # Benjamini-Hochberg per family of tests: submitted targets / other targets; naive vs itself is not a test
    dm["dm_q"] = np.nan
    sub = dm["series"].map(lambda s: T.BY_ID[s].in_submission)
    for fam_mask in (sub, ~sub):
        m = fam_mask & (dm["point"] != "naive")
        dm.loc[m, "dm_q"] = bh_qvalues(dm.loc[m, "dm_p"])
    per = per.merge(dm, on=["point", "series", "h"], how="left")
    per["family"] = per["series"].map(FAMILY)
    per["in_submission"] = per["series"].map(lambda s: T.BY_ID[s].in_submission)
    return per


def aggregate(per, by):
    sig, raw = per["dm_q"] < Q_LEVEL, per["dm_p"] < Q_LEVEL
    per = per.assign(dm_better=sig & (per["dm_stat"] < 0), dm_worse=sig & (per["dm_stat"] > 0),
                     dm_better_raw=raw & (per["dm_stat"] < 0), dm_worse_raw=raw & (per["dm_stat"] > 0),
                     beats=per["rel_mae"] < 0.97)
    return per.groupby(by, observed=True).agg(
        mean_mape=("mape", "mean"), median_rel_mae=("rel_mae", "median"), mean_mase=("mase", "mean"),
        mean_naive_mase=("naive_mase", "mean"), complete=("complete", "all"),
        series_beating_naive=("beats", "sum"), dm_better=("dm_better", "sum"), dm_worse=("dm_worse", "sum"),
        dm_better_raw=("dm_better_raw", "sum"), dm_worse_raw=("dm_worse_raw", "sum"),
        coverage80=("coverage80", "mean"), above=("above", "mean"), below=("below", "mean"), width=("width", "mean"),
        winkler=("winkler", "mean"), pinball=("pinball", "mean"), median_rel_pinball=("rel_pinball", "median"),
        fallback=("fallback", "mean"), n_series=("series", "nunique")).reset_index()


def best_by_family(fam, tol=0.03):
    """Per family x h (period all), among labels with a forecast at every scored origin of every target of the
    family (so all are scored on the same rows): the point model with the lowest mean MASE, and the band whose
    coverage is within `tol` of 80% with the smallest width (else the coverage closest to 80%)."""
    rows = []
    g_all = fam[(fam["period"] == "all")]
    n_fam = g_all[g_all["model"] == NAIVE].set_index(["family", "h"])["n_series"]
    for (f, h), g in g_all.groupby(["family", "h"]):
        g = g[g["complete"] & (g["n_series"] == n_fam[f, h])]
        pts = g[g["interval"] == "emp"].sort_values("mean_mase")
        bp, nv = pts.iloc[0], pts[pts["point"] == "naive"].iloc[0]
        g = g.assign(gap=(g["coverage80"] - (1 - ALPHA)).abs())
        cand = g[g["gap"] <= tol]
        bb = cand.sort_values("width").iloc[0] if len(cand) else g.sort_values(["gap", "width"]).iloc[0]
        rows.append({"family": f, "h": h, "n_series": int(bp["n_series"]), "best_point": bp["point"],
                     "mase": bp["mean_mase"], "naive_mase": nv["mean_mase"], "median_rel_mae": bp["median_rel_mae"],
                     "dm_better": int(bp["dm_better"]), "dm_worse": int(bp["dm_worse"]), "best_band": bb["model"], "coverage80": bb["coverage80"],
                     "above": bb["above"], "below": bb["below"], "width": bb["width"], "winkler": bb["winkler"],
                     "band_within_tol": len(cand) > 0})
    return pd.DataFrame(rows)


def score(res, save_dir=None):
    """Per-series scores (all periods), summary of the submitted targets, family x period table and best models.
    DM q values are per period (one BH family per period)."""
    r = scored_rows(res)
    per = per_series(r)
    sub = per[per["in_submission"]]
    summ = aggregate(sub, ["model", "point", "interval", "h"])
    fam = [aggregate(sub, ["model", "point", "interval", "family", "h"]).assign(period="all")]
    rs = r[r["series"].map(lambda s: T.BY_ID[s].in_submission)]
    for p in PERIODS:
        m = in_period(rs["origin"], p)
        if not m.any():
            continue
        fam.append(aggregate(per_series(rs[m]), ["model", "point", "interval", "family", "h"]).assign(period=p))
    fam = pd.concat(fam, ignore_index=True)
    best = best_by_family(fam)
    if save_dir is not None:
        per.to_csv(save_dir / "scores_per_series.csv", index=False, float_format="%.4f")
        summ.to_csv(save_dir / "scores_summary.csv", index=False, float_format="%.4f")
        fam.to_csv(save_dir / "scores_by_family_period.csv", index=False, float_format="%.4f")
        best.to_csv(save_dir / "best_by_family.csv", index=False, float_format="%.4f")
    return per, summ, fam, best
