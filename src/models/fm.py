"""Pretrained time-series foundation model: Chronos-2 (amazon/chronos-2), zero-shot, CPU, as a benchmark.

Two point + native-quantile models (label "<name>+native" = Chronos' own P10 / P90; conf / aci / ewma wrap its P50
like any other point model):
  chronos2       univariate: context = the target's whole history up to its anchor a (as_of + nowcast, as every
                 model), prediction of max(h_eff) steps (9 with the default nowcast), P10 / P50 / P90 read at step h_eff
  chronos2_cov   the same + past covariates only (no known-future covariates): 1-3 source series of the target's
                 leading indicators (features.TARGET_FEATURES, step-3 family models), aligned on the target's context
                 months (values up to a only). Only the 11 family targets.
The target is fed as y / y[a] (rounded to 10 decimals), so the BOM indices give the same input whatever their scale
(origin = 100) and the forecast quantiles are ratios to y[a]: log ratio = log return of the backtest. Series are
forecast independently (cross_learning off), weights pinned to REVISION, deterministic on CPU.

Optional dependency (uv sync --extra fm: chronos-forecasting, torch). Forecasts are cached in
outputs/backtest/fm_cache/<name>.csv, one row per (origin, target, input hash, step): a cached forecast is used only
when the input built at that origin hashes to the same key, so a change of data or preprocessing is recomputed, never
silently reused. Without the package, the backtest uses the cache and skips what is missing (printed).

Pretraining-leakage risk: Chronos-2 is trained on part of the Chronos and GIFT-Eval pretraining corpora (real data
incl. M4 monthly) plus synthetic data; the model card gives no cutoff of the real data. The weights were published
on 2025-10-30 (paper 2025-10-17), so origins from POST_RELEASE (2025-11) on are scored separately: their outcomes
cannot be in the training data.
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pandas as pd

import data as D
import features as F
import targets as T
from models.base import HORIZONS, QS, TARGET_IDS, Model, Pred

MODEL_ID = "amazon/chronos-2"
REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"  # main on 2026-06-05; weights unchanged since 2025-10-30
PACKAGE = "chronos-forecasting"
MODEL_CARD = "https://huggingface.co/amazon/chronos-2"
PAPER = "https://arxiv.org/abs/2510.15821"
WEIGHTS_DATE = pd.Timestamp("2025-10-30")  # commit "Add model weights and config" on the Hugging Face hub
POST_RELEASE = pd.Timestamp("2025-11-01")  # first origin whose outcomes postdate the weights
CACHE = D.OUT / "backtest" / "fm_cache"
ROUND = 10
NAMES = ["chronos2", "chronos2_cov"]

# Past covariates per target: sources of its step-3 leading indicators (features.TARGET_FEATURES), at most 3.
# Dropped to keep 3: TTF for HDG coil (zinc is the HDG-specific input) and for LME aluminium (power is the larger
# smelting cost). LME basis = 3-month / cash - 1.
BASIS = {"cu_basis": ("WM_CU_3M", "WM_CU_CASH"), "al_basis": ("WM_AL_3M", "WM_AL_CASH")}
COVARIATES = {
    "S09": ["cru_cr_de", "S04_EUR"], "S10": ["cru_cr_de", "S04_EUR"],
    "cru_cr_de": ["S04_EUR", "E08_EUR", "ED02"], "cru_hdg_de": ["S04_EUR", "E08_EUR", "S05_EUR"],
    "ss304_de": ["S03_EUR"], "S11": ["ED01"], "S08": ["D02", "ED02", "E09"],
    "D02": ["al_basis", "WM_AL_STOCK", "E09"], "S02": ["al_basis", "WM_AL_STOCK", "E09"],
    "D01": ["cu_basis", "WM_CU_STOCK"], "S01": ["cu_basis", "WM_CU_STOCK"],
}


def _sources(s):
    return {c for f in F.TARGET_FEATURES[s] for c in F.BY_NAME[f].source}


for _s, _c in COVARIATES.items():  # every covariate is a source of one of the target's step-3 indicators
    assert set(_c) <= _sources(_s) | {b for b, src in BASIS.items() if set(src) <= _sources(_s)}, _s
    assert 1 <= len(_c) <= 3, _s
assert set(COVARIATES) == set(F.TARGET_FEATURES)


def covariate(pt, c):
    return pt[BASIS[c][0]] / pt[BASIS[c][1]] - 1 if c in BASIS else pt[c]


def build_input(pt, s, a, covs=()):
    """Chronos input of target s with anchor a from panel pt: (dict, hash key, context length). Months up to a
    only; target relative to its value at a; covariates as published (NaN kept: Chronos treats them as missing)."""
    y = pt[s].loc[:a]
    y = y.loc[y.first_valid_index():]
    x = {"target": np.round((y / y.iloc[-1]).to_numpy(float), ROUND)}
    h = hashlib.sha256(f"{s}|{y.index[0]:%Y-%m}|{a:%Y-%m}".encode())
    h.update(x["target"].tobytes())
    if covs:
        x["past_covariates"] = {}
        for i, c in enumerate(covs):  # positional names: one schema per covariate count (batching)
            v = np.round(covariate(pt, c).reindex(y.index).to_numpy(float), ROUND)
            x["past_covariates"][f"cov_{i}"] = v
            h.update(c.encode() + v.tobytes())
    return x, h.hexdigest()[:20], len(y)


_PIPE = {}


def pipeline():
    """Chronos-2 on CPU, or None when the optional dependency is not installed."""
    if "p" not in _PIPE:
        try:
            from chronos import Chronos2Pipeline
        except ImportError:
            _PIPE["p"], _PIPE["load_s"] = None, 0.0
        else:
            c = time.perf_counter()
            _PIPE["p"] = Chronos2Pipeline.from_pretrained(MODEL_ID, revision=REVISION, device_map="cpu")
            _PIPE["load_s"] = time.perf_counter() - c
    return _PIPE["p"]


def versions():
    import importlib.metadata as md
    out = {"model": MODEL_ID, "revision": REVISION}
    for p in (PACKAGE, "torch", "transformers"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


class Cache:
    """(origin, series, key) -> array (steps x 3 quantiles) of log ratios to y[a]; CSV in CACHE."""

    def __init__(self, name):
        self.path, self.rt_path = CACHE / f"{name}.csv", CACHE / f"{name}_runtime.csv"
        self.d, self.new, self.runtime = {}, [], []
        if self.path.exists():
            c = pd.read_csv(self.path, parse_dates=["origin"])
            for (t, s, k), g in c.sort_values("step").groupby(["origin", "series", "key"]):
                self.d[t, s, k] = g[["q10", "q50", "q90"]].to_numpy()
        if self.rt_path.exists():
            self.runtime = pd.read_csv(self.rt_path, parse_dates=["origin"]).to_dict("records")

    def get(self, t, s, k):
        return self.d.get((t, s, k))

    def put(self, t, s, k, n, q):
        self.d[t, s, k] = q
        self.new += [{"origin": t, "series": s, "key": k, "n_context": n, "step": i + 1, "q10": r[0], "q50": r[1],
                      "q90": r[2]} for i, r in enumerate(q)]

    def save(self):
        if not self.new:
            return
        CACHE.mkdir(parents=True, exist_ok=True)
        old = pd.read_csv(self.path, parse_dates=["origin"]) if self.path.exists() else None
        pd.concat([old, pd.DataFrame(self.new)]).to_csv(self.path, index=False, float_format="%.10g")
        pd.DataFrame(self.runtime).to_csv(self.rt_path, index=False, float_format="%.4f")
        meta = versions() | {"model_load_s": round(_PIPE.get("load_s", float("nan")), 1), "device": "cpu"}
        (CACHE / "meta.json").write_text(json.dumps(meta, indent=2))
        self.new = []


class Chronos2(Model):
    native = True

    def __init__(self, name, with_covariates):
        self.name, self.cov = name, with_covariates
        self.series = set(COVARIATES) if with_covariates else None
        self.cache, self.inputs, self.skipped = Cache(name), {}, 0

    def prepare(self, ctx):
        ids = sorted(self.series) if self.cov else TARGET_IDS
        self.inputs = {s: build_input(ctx.pt, s, ctx.anchor[s], COVARIATES[s] if self.cov else ())
                       for s in ids if ctx.anchor[s] is not None}
        todo = [s for s, (_, k, _) in self.inputs.items() if self.cache.get(ctx.t, s, k) is None]
        if not todo:
            return
        pipe = pipeline()
        if pipe is None:
            self.skipped += len(todo)
            return
        steps = max(HORIZONS) + max(ctx.lag[s] for s in todo)
        c = time.perf_counter()
        groups = {}
        for s in todo:  # one call per input schema (number of covariates)
            groups.setdefault(len(self.inputs[s][0].get("past_covariates", {})), []).append(s)
        for g in groups.values():
            q, _ = pipe.predict_quantiles([self.inputs[s][0] for s in g], prediction_length=steps,
                                          quantile_levels=list(QS))
            for s, qs in zip(g, q):
                x, k, n = self.inputs[s]
                self.cache.put(ctx.t, s, k, n, np.log(np.clip(qs[0].numpy().astype(float), 1e-6, None)))
        self.cache.runtime.append({"origin": ctx.t, "n_series": len(todo), "steps": steps,
                                   "seconds": time.perf_counter() - c})

    def predict(self, ctx, s):
        if s not in self.inputs:
            return {}
        _, k, n = self.inputs[s]
        q = self.cache.get(ctx.t, s, k)
        if q is None:
            return {}
        out = {}
        for h in HORIZONS:
            he = ctx.h_eff(s, h)
            if he <= len(q):
                lo, mid, hi = np.sort(q[he - 1])
                out[h] = Pred(lo, mid, hi, n)
        return out

    def finish(self):
        self.cache.save()
        if self.skipped:
            print(f"  {self.name}: {self.skipped} (origin, target) inputs not in {CACHE.relative_to(D.ROOT)} and "
                  f"{PACKAGE} not installed -> skipped (uv sync --extra fm to compute them)", flush=True)


def models():
    return [Chronos2("chronos2", False), Chronos2("chronos2_cov", True)]


# ---------------------------------------------------------------------------------------------------------------------
# Leakage check of the inputs


def check_inputs(cut=None, start="2018-01-01"):
    """At every origin, the Chronos inputs of every material target built from the panel as seen at the origin
    (as_of + nowcast, as in the backtest) equal those built from the full panel (computed on all months, then read
    up to the anchor, so an input that looks ahead fails).
    Stainless 304: its anchor value at t < cutoff is a nowcast, published later: only the target may differ there."""
    import preprocess as P
    cut = P.load() if cut is None else cut
    lags = P.relative_lags(cut)
    full = P.nowcast(P.as_of(cut, P.CUTOFF, lags), P.CUTOFF)[0]
    n = 0
    for t in pd.date_range(start, P.CUTOFF, freq="MS"):
        pt = P.nowcast(P.as_of(cut, t, lags), t)[0]
        for s in T.ids("material"):
            a = pt[s].last_valid_index()
            for covs in [()] + ([COVARIATES[s]] if s in COVARIATES else []):
                got, ref = build_input(pt, s, a, covs)[0], build_input(full, s, a, covs)[0]
                for c, v in got.get("past_covariates", {}).items():
                    assert np.array_equal(v, ref["past_covariates"][c], equal_nan=True), (t, s, c)
                if s == "ss304_de" and t < P.CUTOFF:  # nowcast at a: compare the published months before it
                    assert np.array_equal(pt[s].loc[:a].iloc[:-1].to_numpy(), full[s].loc[:a].iloc[:-1].to_numpy(),
                                          equal_nan=True), (t, s)
                else:
                    assert np.array_equal(got["target"], ref["target"], equal_nan=True), (t, s)
                n += 1
    return n


# ---------------------------------------------------------------------------------------------------------------------
# Report: Chronos-2 against naive, the best univariate and the best multivariate model


PERIODS = {"all": (None, None), "pre_release": (None, POST_RELEASE - pd.DateOffset(months=1)),
           "post_release": (POST_RELEASE, None)}
BANDS = ["native", "conf", "aci", "ewma"]


def _period(r, p):
    a, b = PERIODS[p]
    m = pd.Series(True, index=r.index)
    if a is not None:
        m &= r["origin"] >= a
    if b is not None:
        m &= r["origin"] <= b
    return r[m]


def compare(res):
    """Per (Chronos model, target, h, period): rel. MAE and Diebold-Mariano (absolute errors, Newey-West lag
    h_eff - 1) against naive, the best univariate and the best multivariate model of that target and h (lowest MAE
    over all scored origins, chosen with hindsight: favours the benchmarks), and coverage / Winkler of the native
    and calibrated bands."""
    import scoring
    from models import multi

    r = scoring.scored_rows(res)
    pts = r[r["interval"].isin(["emp", "native"]) & ~(r["point"].isin(NAMES) & (r["interval"] == "emp"))]
    pts = pts.drop_duplicates(["point", "series", "h", "origin"])
    bench_sets = {"naive": ["naive"], "best_univariate": multi.UNIVARIATE,
                  "best_multivariate": [m for m in multi.NAMES if m != "gbm_family"]}
    rows = []
    for (s, h), g in pts.groupby(["series", "h"]):
        bench = {}
        for b, cands in bench_sets.items():
            u = g[g["point"].isin(cands)]
            if len(u):
                best = u.groupby("point")["ae"].mean().idxmin()
                bench[b] = (best, u[u["point"] == best].set_index("origin")["ae"])
        for m in NAMES:
            gm = g[g["point"] == m].set_index("origin").sort_index()
            if gm.empty:
                continue
            for p in PERIODS:
                gp = _period(gm.reset_index(), p).set_index("origin")
                if gp.empty:
                    continue
                row = {"model": m, "series": s, "family": T.BY_ID[s].family, "in_submission": T.BY_ID[s].in_submission,
                       "h": h, "period": p, "n": len(gp)}
                for b, (name, ae) in bench.items():
                    common = gp.index.intersection(ae.index)
                    d = gp.loc[common, "ae"] - ae.loc[common]
                    st, pv = scoring.dm_test(d.values, int(gp["h_eff"].max()) - 1)
                    row |= {f"{b}": name, f"rel_mae_vs_{b}": gp.loc[common, "ae"].mean() / ae.loc[common].mean(),
                            f"dm_stat_vs_{b}": st, f"dm_p_vs_{b}": pv}
                for band in BANDS:
                    gb = _period(r[(r["model"] == f"{m}+{band}") & (r["series"] == s) & (r["h"] == h)], p)
                    row[f"coverage80_{band}"] = gb["covered"].mean()
                    row[f"winkler_{band}"] = gb["winkler"].mean()
                rows.append(row)
    out = pd.DataFrame(rows)
    for b in bench_sets:  # benchmark not run (e.g. --models subset): empty columns
        for c, v in ((b, ""), (f"rel_mae_vs_{b}", np.nan), (f"dm_stat_vs_{b}", np.nan), (f"dm_p_vs_{b}", np.nan)):
            if c not in out:
                out[c] = v
    return out


def by_family(cmp):
    """Family x h x model x period, submitted targets: median rel. MAE and DM counts (p < 0.05) per benchmark, mean
    coverage / Winkler; family "all" pools them."""
    cmp = cmp[cmp["in_submission"]]
    agg = {"n_series": ("series", "nunique")}
    for b in ("naive", "best_univariate", "best_multivariate"):
        c = cmp.assign(**{f"better_{b}": (cmp[f"dm_p_vs_{b}"] < 0.05) & (cmp[f"dm_stat_vs_{b}"] < 0),
                          f"worse_{b}": (cmp[f"dm_p_vs_{b}"] < 0.05) & (cmp[f"dm_stat_vs_{b}"] > 0)})
        cmp = c
        agg |= {f"median_rel_mae_vs_{b}": (f"rel_mae_vs_{b}", "median"), f"dm_better_{b}": (f"better_{b}", "sum"),
                f"dm_worse_{b}": (f"worse_{b}", "sum")}
    for band in BANDS:
        agg |= {f"coverage80_{band}": (f"coverage80_{band}", "mean"), f"winkler_{band}": (f"winkler_{band}", "mean")}
    fam = cmp.groupby(["model", "period", "family", "h"]).agg(**agg).reset_index()
    tot = cmp.assign(family="all").groupby(["model", "period", "family", "h"]).agg(**agg).reset_index()
    return pd.concat([fam, tot], ignore_index=True)


def intervals_table(res):
    """Raw (native) vs calibrated bands around the Chronos P50, per model x band x h x period (submitted targets)."""
    import scoring
    r = scoring.scored_rows(res)
    r = r[r["point"].isin(NAMES) & r["series"].map(lambda s: T.BY_ID[s].in_submission)]
    out = []
    for p in PERIODS:
        rp = _period(r, p)
        out.append(rp.groupby(["point", "interval", "h"]).agg(
            coverage80=("covered", "mean"), above=("above", "mean"), below=("below", "mean"), width=("width", "mean"),
            winkler=("winkler", "mean"), n=("covered", "size")).reset_index().assign(period=p))
    return pd.concat(out, ignore_index=True)


SPLIT_POINTS = ["chronos2", "chronos2_cov", "ets_local", "auto_ets", "combo_mean", "ardl", "ridge_fam"]


def release_split(res):
    """Pooled MAE / naive MAE (sum of absolute errors over sum of naive's, same rows) before and after the release
    of the weights, for Chronos and a few benchmarks: split by origin (origin >= POST_RELEASE: nothing of the
    forecast window can be in the training data) and by outcome month (outcome >= POST_RELEASE: the outcome cannot
    be; the context can). Memorisation would show as an edge that disappears after the release."""
    import scoring
    r = scoring.scored_rows(res)
    r = r[r["interval"].isin(["emp", "native"])].drop_duplicates(["point", "series", "h", "origin"])
    outcome = pd.DatetimeIndex([o + pd.DateOffset(months=int(h)) for o, h in zip(r["origin"], r["h"])])
    r = r.assign(by_origin=np.where(r["origin"] >= POST_RELEASE, "post", "pre"),
                 by_outcome=np.where(outcome >= POST_RELEASE, "post", "pre"))
    fam = set(COVARIATES)
    out = []
    for scope, d in (("22 submitted targets", r[r["series"].map(lambda s: T.BY_ID[s].in_submission)]),
                     ("11 family targets", r[r["series"].isin(fam)])):
        for split in ("by_origin", "by_outcome"):
            g = d[d["point"].isin(SPLIT_POINTS)].groupby(["point", split, "h"])
            t = (g["ae"].sum() / g["naive_ae"].sum()).rename("mae_over_naive").reset_index()
            t = t.merge(g.size().rename("n").reset_index(), on=["point", split, "h"])
            out.append(t.rename(columns={split: "period"}).assign(scope=scope, split=split))
    return pd.concat(out, ignore_index=True)[["scope", "split", "period", "point", "h", "n", "mae_over_naive"]]


def runtime_table():
    """Inference time of the cache build (CPU), per Chronos model, with versions and model load time."""
    rows = []
    for m in NAMES:
        p = CACHE / f"{m}_runtime.csv"
        if p.exists():
            rt = pd.read_csv(p)
            meta = json.loads((CACHE / "meta.json").read_text()) if (CACHE / "meta.json").exists() else {}
            rows.append({"model": m, "origins_computed": len(rt), "series_forecasts": int(rt["n_series"].sum()),
                         "inference_s": rt["seconds"].sum(), "s_per_origin": rt["seconds"].mean()}
                        | {f"meta_{k}": v for k, v in meta.items()})
    return pd.DataFrame(rows)


def write_reports(res, d):
    cmp = compare(res)
    fam = by_family(cmp)
    itv = intervals_table(res)
    cmp.to_csv(d / "fm_vs_benchmarks.csv", index=False, float_format="%.4f")
    fam.to_csv(d / "fm_by_family.csv", index=False, float_format="%.4f")
    itv.to_csv(d / "fm_intervals.csv", index=False, float_format="%.4f")
    rs = release_split(res)
    rs.to_csv(d / "fm_release_split.csv", index=False, float_format="%.4f")
    rt = runtime_table()
    if len(rt):
        rt.to_csv(d / "fm_runtime.csv", index=False, float_format="%.2f")
    pd.set_option("display.width", 250)
    sub = fam[fam["family"] != "all"]
    for p in ("all", "post_release"):
        f = sub[sub["period"] == p]
        print(f"\nChronos-2 by family, origins {'2020-01' if p == 'all' else f'{POST_RELEASE:%Y-%m}'}+ "
              "(median rel. MAE vs naive / best univariate / best multivariate; DM better-worse vs best univariate):")
        f = f.assign(cell=f["median_rel_mae_vs_naive"].map("{:.2f}".format) + " / "
                     + f["median_rel_mae_vs_best_univariate"].map("{:.2f}".format) + " / "
                     + f["median_rel_mae_vs_best_multivariate"].map(lambda v: "-" if pd.isna(v) else f"{v:.2f}")
                     + "  " + f["dm_better_best_univariate"].astype(int).astype(str) + "-"
                     + f["dm_worse_best_univariate"].astype(int).astype(str))
        print(f.pivot_table(index=["model", "family"], columns="h", values="cell", aggfunc="first").to_string())
    print("\nChronos-2 bands, raw (native) vs calibrated, submitted targets:")
    print(itv.pivot_table(index=["period", "point", "interval"], columns="h",
                          values=["coverage80", "winkler"]).round(3).to_string())
    print(f"\nPooled MAE / naive MAE before / after the weights' release (origin >= {POST_RELEASE:%Y-%m}), "
          "22 submitted targets:")
    x = rs[(rs["scope"] == "22 submitted targets") & (rs["split"] == "by_origin")]
    print(x.pivot_table(index="point", columns=["period", "h"], values="mae_over_naive").round(3).to_string())
    print("rows per point:", x[x["point"] == "chronos2"].set_index(["period", "h"])["n"].to_dict())
    if len(rt):
        print("\nChronos-2 runtime (CPU, from the cache build):")
        print(rt[["model", "origins_computed", "series_forecasts", "inference_s", "s_per_origin"]].round(2)
              .to_string(index=False))
    return cmp
