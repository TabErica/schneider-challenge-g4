"""Rolling-origin backtest at 3 / 6 / 9 months -> outputs/backtest/.

Targets come from targets.py, models from models/ (one interface, models/base.py). At every origin t the panel is cut
with preprocess.as_of(): each series loses the same number of months it lacks at the real origin (Sep-2026), so
stainless 304 always ends at t - 1. By default (also for the submission) that missing month is nowcast
(preprocess.nowcast: nickel + EU steel PPI), so every target is known at t. Every model forecasts
log(y[t+h] / y[a]), where a = the target's last month known at t; the effective horizon is h_eff = h + (t - a)
(h, or h + 1 for stainless 304 and the market-price BOM with --no-nowcast). The BOM indices are rebuilt at every
origin from the cut panel, so they inherit the ragged edge of their inputs, and are rescaled so that the run's
origin month = 100 (bom_rebase.csv).

Two passes:
  1. point pass, origins WARMUP_START..origin: every model of models.make_registry() gives its P50 (Ridge and GBM
     also their own P10 / P90). Univariate models use each target's full history; Ridge / GBM the months where every
     input is known.
  2. interval layer (models/intervals.py): emp / conf / aci / ewma bands around every point model, from the
     residuals already published at each origin. The warm-up origins (2018-19) only build that residual history:
     scores (scoring.py) use origins >= SCORE_START (2020-01).
The point forecasts of the warm-up origins are written to backtest_warmup.csv (one row per point model, target, h,
origin): selection.py needs the errors known before 2020-01 for its rolling window. Every row carries `reveal`, the
month its outcome is published.
No forward fill anywhere.

    python backtest.py [--origin 2026-09] [--no-nowcast] [--drop-provisional] [--bom-weights base|origin]

--origin            last origin (default: the data cutoff, Sep-2026). An earlier origin is a held-back check: the run
                    only sees data as of that origin, and holdout_check.csv compares its forecasts with the months after.
--no-nowcast        sensitivity: no stainless nowcast; stainless 304 and the market-price BOM forecast from t - 1 (h + 1).
--drop-provisional  sensitivity: remove the last 2 months of the Eurostat PPIs at every origin (their h_eff = h + 2).
--bom-weights       base = Sep-2026 cost shares (submission, default); origin = shares from prices known at each origin.
--models            comma-separated point models (default: all of models.make_registry(); naive is always run).
Non-default options write to outputs/backtest/<tag>/ so the baseline is never overwritten.
"""
from __future__ import annotations

import argparse
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

import data as D
import models
import preprocess as P
import scoring
import targets as T
from models import intervals
from models.base import HORIZONS, Context, empirical_spread, months
from models.lme import quote_basis

warnings.filterwarnings("ignore")
OUT = D.OUT / "backtest"

WARMUP_START = pd.Timestamp("2018-01-01")  # residual history for the interval layer; not scored
SCORE_START = scoring.SCORE_START  # first scored origin (2020-01)
FIRST_ORIGIN = SCORE_START
TARGET_IDS = T.ids()
BOM_IDS = T.ids("bom_index")
MATERIAL_IDS = T.ids("material")


class Bom:
    """BOM index columns for an as_of() panel. base: Sep-2026 cost shares and references (submission);
    origin: cost shares and references from the latest prices known at the origin (no look-ahead)."""

    def __init__(self, full, mode):
        self.mode, self.scale = mode, {}
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
        out = pd.concat([D.bom_index(prices, costs, v, refs[v]) for v in ("proxy", "market")], axis=1)[BOM_IDS]
        return out * pd.Series({c: self.scale.get(c, 1.0) for c in BOM_IDS})

    def rebase(self, seen, origin):
        """Base mode: one constant per index so that its value in the origin month, as seen at the origin (`seen`,
        with the stainless nowcast when it is on), is 100. Cost shares are unchanged (Sep-2026); a constant does not
        change any log change, so it only sets the scale of the forecast levels. An index with no value in the origin
        month (market version with --no-nowcast) keeps its Sep-2026 references (factor 1)."""
        self.scale = {}
        raw = self.columns(seen, self.base_costs, self.base_refs).loc[origin]
        self.scale = {c: (100 / v if pd.notna(v) else 1.0) for c, v in raw.items()}
        return pd.DataFrame({"index_at_origin_before": raw, "factor": pd.Series(self.scale)}).rename_axis("series")


def check_history(full):
    """targets.history_start must be the first month of each target in the data."""
    for t in T.REGISTRY:
        first = full[t.id].first_valid_index()
        if first is None or f"{first:%Y-%m}" != t.history_start:
            raise ValueError(f"{t.id}: registry history_start {t.history_start}, data starts {first}")


def run(origin=P.CUTOFF, nowcast=True, drop_provisional=False, bom_weights="base", model_names=None):
    """Point pass over origins WARMUP_START..origin, then the interval layer. Rows are kept when the outcome is known
    as of `origin`, and always at `origin` itself (the forecast). Returns (predictions of the scored origins, nowcast
    log, BOM rebase factors, ACI gamma table, runtime table, point forecasts of the warm-up origins). model_names: run
    only these point models (naive is always kept as the reference)."""
    t0, origin = time.time(), pd.Timestamp(origin)
    if origin < SCORE_START:
        raise ValueError(f"origin must be >= {SCORE_START:%Y-%m}")
    cut = P.load()  # full history, cut at the data cutoff (Sep-2026)
    lags = P.relative_lags(cut)
    bom = Bom(cut, bom_weights)
    base = bom if bom_weights == "base" else Bom(cut, "base")
    full = P.as_of(cut, origin, lags)  # everything the run may see: data as of `origin` (outcomes come from here)
    rebase = base.rebase(P.nowcast(full, origin)[0] if nowcast else full, origin)  # origin month = 100
    full = full.join(base.columns(full, base.base_costs, base.base_refs))
    check_history(full)
    # months after its date that an outcome is published (the residual of origin j is usable at t >= target + lag)
    seen = P.as_of(cut, origin, lags, drop_provisional=drop_provisional)
    seen = seen.join(base.columns(seen, base.base_costs, base.base_refs))
    data_lag = {s: months(seen[s].last_valid_index(), origin) for s in TARGET_IDS}

    registry = models.make_registry()
    if model_names:
        keep = set(model_names) | {"naive"}
        unknown = keep - {m.name for m in registry}
        if unknown:
            raise ValueError(f"unknown models {sorted(unknown)}")
        registry = [m for m in registry if m.name in keep]
    clock = defaultdict(float)
    rows, nowlog = [], []
    for t in pd.date_range(WARMUP_START, origin, freq="MS"):
        pt = P.as_of(cut, t, lags, drop_provisional=drop_provisional)
        if nowcast:
            pt, nl = P.nowcast(pt, t)
            nowlog += nl
        costs, refs = bom.weights(pt, t)
        pt = pt.join(bom.columns(pt, costs, refs))
        # outcomes, with the same BOM weights as the inputs (data as of `origin`, never later)
        actual = full if bom_weights == "base" else full[MATERIAL_IDS].join(bom.columns(full, costs, refs))
        prov = P.provisional_months(pt)
        ctx = Context(t, pt)
        for m in registry:
            c = time.perf_counter()
            m.prepare(ctx)
            clock[m.name] += time.perf_counter() - c

        for s in TARGET_IDS:
            a, hist, y_a = ctx.anchor[s], ctx.hist(s), ctx.y_a(s)
            preds = {}
            for m in registry:
                if m.applies(s):
                    c = time.perf_counter()
                    preds[m.name] = m.predict(ctx, s)
                    clock[m.name] += time.perf_counter() - c
                    ctx.q50[m.name, s] = {h: p.q50 for h, p in preds[m.name].items()}
            sigma = intervals.ewma_sigma(hist)
            sigma_fast = intervals.ewma_sigma(hist, intervals.POOL_LAMBDA)
            lv = pt[s].loc[:a]
            for h in HORIZONS:
                target = t + pd.DateOffset(months=h)
                y_true = actual.at[target, s] if target in actual.index else np.nan
                if pd.isna(y_true) and t != origin:
                    continue
                he = ctx.h_eff(s, h)
                lo, hi = empirical_spread(ctx.ys(s, he))
                meta = {"series": s, "h": h, "h_eff": he, "origin": t, "anchor": a, "y_t": y_a, "actual": y_true,
                        "anchor_provisional": a in prov.get(s, []), "emp_lo": lo, "emp_hi": hi, "sigma": sigma,
                        "sigma_fast": sigma_fast,
                        "mase_scale": (lv.shift(-he) - lv).abs().mean(),
                        "reveal": target + pd.DateOffset(months=data_lag[s])}
                for name, ph in preds.items():
                    if h in ph:
                        p = ph[h]
                        rows.append(meta | {"point": name, "q50": p.q50, "q10n": p.q10, "q90n": p.q90,
                                            "n_train": p.n_train})
        if t.month == 1 or t == origin:
            print(f"  origin {t:%Y-%m} ({time.time() - t0:.0f}s)", flush=True)
    for m in registry:
        m.finish()

    pts = pd.DataFrame(rows)
    pts["r"] = np.log(pts["actual"] / pts["y_t"])
    bands, gamma, iclock = intervals.apply(pts, warm_end=SCORE_START)
    native = pts[pts["q10n"].notna()].assign(interval="native", q10=lambda d: d["q10n"], q90=lambda d: d["q90n"],
                                             fallback=False)
    meta_cols = ["point", "series", "h", "origin", "h_eff", "anchor", "y_t", "actual", "n_train",
                 "anchor_provisional", "mase_scale", "reveal"]
    res = pd.concat([bands.merge(pts[meta_cols], on=["point", "series", "h", "origin"]),
                     native[meta_cols + ["interval", "q10", "q50", "q90", "fallback"]]], ignore_index=True)
    warm = pts.loc[pts["origin"] < SCORE_START, ["point", "series", "h", "origin", "h_eff", "y_t", "actual", "q50",
                                                  "reveal"]].reset_index(drop=True)
    warm["p50"] = warm["y_t"] * np.exp(warm["q50"])
    res = res[res["origin"] >= SCORE_START]
    for q in ("10", "50", "90"):
        res[f"p{q}"] = res["y_t"] * np.exp(res[f"q{q}"])
    res.insert(0, "model", res["point"] + "+" + res["interval"])
    res = res.sort_values(["model", "series", "h", "origin"]).reset_index(drop=True)
    runtime = pd.DataFrame([{"kind": "point model", "name": k, "seconds": v} for k, v in clock.items()]
                           + [{"kind": "interval method", "name": k, "seconds": v} for k, v in iclock.items()])
    runtime["origins"] = len(pd.date_range(WARMUP_START, origin, freq="MS"))
    print(f"  done ({time.time() - t0:.0f}s)", flush=True)
    return res, pd.DataFrame(nowlog), rebase, gamma, runtime, warm


def score(res, save_dir=None):
    """See scoring.py. Returns (per series, summary, family x period, best per family)."""
    return scoring.score(res, save_dir)


def training_samples(res, origin):
    """Per target (h = 3): months of history for the univariate models at the first and last origin, and Ridge
    training rows (window where every input is known) at the first origin Ridge could run and at the last one."""
    r = res.sort_values("origin")
    nv = r[(r.model == "naive+emp") & (r.h == 3)].groupby("series")
    rg = r[(r.model == "ridge_local+native") & (r.h == 3)].groupby("series")
    out = pd.DataFrame({
        "history_start": {t.id: t.history_start for t in T.REGISTRY},
        "relative_lag_m": (r["h_eff"] - r["h"]).groupby(r["series"]).max(),
        f"n_hist_{SCORE_START:%Y-%m}": nv["n_train"].first(), f"n_hist_{origin:%Y-%m}": nv["n_train"].last(),
        "ridge_first_origin": rg["origin"].first().dt.strftime("%Y-%m"), "n_ridge_first": rg["n_train"].first(),
        f"n_ridge_{origin:%Y-%m}": rg["n_train"].last(),
    })
    out.index.name = "series"
    return out.loc[TARGET_IDS]


def holdout_check(res, origin, bom_weights="base", rebase=None):
    """Forecasts made at an earlier origin against the outcomes published after it (up to the data cutoff).
    BOM rows only with the Sep-2026 weights (the outcome index of other weights is not stored), on the run's scale."""
    full = D.panel()  # full history up to the cutoff, BOM indices with Sep-2026 weights, no fill
    for c, k in ({} if rebase is None else rebase["factor"]).items():
        full[c] = full[c] * k
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
    tag += ["no_nowcast"] * (not nowcast) + ["drop_provisional"] * drop_provisional
    tag += [f"bom_weights_{bom_weights}"] if bom_weights != "base" else []
    d = OUT / "_".join(tag) if tag else OUT
    d.mkdir(parents=True, exist_ok=True)
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--origin", default=f"{P.CUTOFF:%Y-%m}")
    ap.add_argument("--no-nowcast", action="store_true")
    ap.add_argument("--drop-provisional", action="store_true")
    ap.add_argument("--bom-weights", default="base", choices=["base", "origin"])
    ap.add_argument("--models", default="", help="comma-separated point models to run (default: all)")
    a = ap.parse_args(argv)
    origin = pd.Timestamp(a.origin)
    nowcast = not a.no_nowcast
    d = out_dir(origin, nowcast, a.drop_provisional, a.bom_weights)
    print(f"backtest -> {d.relative_to(D.ROOT)}")
    res, nowlog, rebase, gamma, runtime, warm = run(origin, nowcast, a.drop_provisional, a.bom_weights,
                                                    [m for m in a.models.split(",") if m])
    warm.to_csv(d / "backtest_warmup.csv", index=False)
    rebase.to_csv(d / "bom_rebase.csv", float_format="%.6f")
    gamma.to_csv(d / "aci_gamma.csv", index=False, float_format="%.4f")
    runtime.to_csv(d / "runtime.csv", index=False, float_format="%.2f")
    res.to_csv(d / "backtest_predictions.csv", index=False)
    res[res.origin == origin].to_csv(d / "forecasts_at_origin.csv", index=False)
    if len(nowlog):
        nowlog.to_csv(d / "nowcast_log.csv", index=False)
    training_samples(res, origin).to_csv(d / "training_samples.csv")
    qb = quote_basis(P.as_of(P.load(), origin), origin, {"all": (None, None), "2020-": (SCORE_START, None)})
    qb.to_csv(d / "quote_basis.csv", index=False, float_format="%.4f")
    per, summ, fam, best = score(res, d)
    if origin < P.CUTOFF:
        hc = holdout_check(res, origin, a.bom_weights, rebase)
        hc.to_csv(d / "holdout_check.csv", index=False, float_format="%.4f")
        print("held-back check (outcomes after the origin), P50 error by point model and 80% coverage by band:")
        e = hc[hc.model.str.endswith("+emp")]
        print(e.groupby(["model", "h"]).agg(mae_pct=("error", lambda x: x.abs().mean()),
                                            naive_mae_pct=("naive_error", lambda x: x.abs().mean()),
                                            n=("error", "size")).round(3).to_string())
        print(hc.assign(interval=hc.model.str.split("+").str[1]).groupby(["interval", "h"])["covered"]
              .mean().round(3).to_string())
    print_compact(best, gamma)
    if "lme_fwd+emp" in set(per["model"]):
        print_lme(per, qb)
    if set(per["point"]) & set(models.multi.NAMES):
        models.multi.write_reports(res, d)
    if set(per["point"]) & set(models.fm.NAMES):
        models.fm.write_reports(res, d)
    return res, per, summ


def print_lme(per, qb):
    """LME 3-month price as P50 vs naive on its 4 targets, and the quote-basis table (origins 2020-01+)."""
    p = per[per["model"].isin(["lme_fwd+emp", "naive+emp"])]
    print("\nlme_fwd vs naive (rel. MAE, DM stat / p), origins 2020-01+:")
    print(p[p.model == "lme_fwd+emp"].pivot_table(index="series", columns="h",
                                                   values=["rel_mae", "dm_stat", "dm_p", "dm_q"]).round(3).to_string())
    q = qb[qb["period"] == "2020-"].pivot_table(index=["metal", "currency", "h"], columns="basis",
                                                 values=["bias", "mae"])
    print("\nQuote basis: cash vs 3-month LME price at the origin as forecast of the cash average h months later "
          "(pred / actual - 1), origins 2020-01+:")
    print(q.round(4).to_string())


def print_compact(best, gamma):
    pd.set_option("display.width", 220)
    print("\nACI gamma per point model (all horizons pooled, from the warm-up window only):")
    print(gamma[gamma.chosen][["point", "gamma", "coverage", "width", "n", "rule"]].round(3).to_string(index=False))
    print("\nBest point model (lowest mean MASE) and band (coverage within 3 pts of 80%, narrowest), origins 2020-01+:")
    cols = ["family", "h", "n_series", "best_point", "mase", "naive_mase", "median_rel_mae", "dm_better",
            "best_band", "coverage80", "above", "below", "width", "winkler"]
    print(best[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
