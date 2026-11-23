"""Family models: a few leading indicators with an economic reason (features.py), one direct model per horizon.

Every model forecasts the h_eff-month log return of a target from its anchor a, trained on the months whose outcome
is known at the origin (Context.ys is NaN otherwise). Point models; the interval layer (models/intervals.py) adds
emp / conf / aci / ewma bands, GBM also keeps its own quantiles (native).

  ardl          OLS per target and h: own 1- and 3-month changes + the target's leading indicators (an ARDL in
                direct form, h-month change on the right-hand side known at a)
  ridge_fam     the same regressors, standardised Ridge, alpha chosen per fit by expanding time-series CV inside the
                training window (folds separated by h_eff months so no training outcome overlaps a validation row)
  ecm           steel PPI (S09, S10) <- CRU CR coil: error-correction term log(PPI) - c - b log(CRU), b estimated on
                the months known at the origin, + CRU changes (this month, last month) + own change
  vecm          chain iron ore (EUR) -> CRU CR coil -> S09 on log levels: Johansen trace test (5%) at every origin;
                rank >= 1 -> VECM(k_ar_diff = 1, constant in the cointegration relation), else VAR(1) on differences;
                forecasts CRU CR coil and S09. Test results per origin in vecm_johansen.csv
  gbm_family    one quantile GBM per horizon over the family targets: own changes, deviation from the 12-month mean,
                12-month volatility, every leading indicator (NaN where a target does not use it) and the family;
                longer history (CRU from 2012, stainless from 2008), shallow trees and L2 penalty. P50 shrunk towards
                naive: q50 = w x q50_gbm (band shifted with it), w chosen once on warm-up origins (2018-19) whose
                outcome is known at the first scored origin (gbm_shrink.csv)
  epoxy_struct  epoxy rebuilt from its cost columns: h_eff-month log change of epoxy = sum of beta_i x log change of
                benzene, acetone, ECH and other costs (no constant; beta fit on the months known at the origin); each
                input is forecast with mean(naive, AutoETS), the rule of combo_mean (the steadiest univariate model of
                the previous task). The source file is not an exact identity (features.epoxy_identity)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsforecast.models as M
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

import features as F
import targets as T
from models.base import HORIZONS, QS, Model, Pred, months, point

FAMILY_TARGETS = set(F.TARGET_FEATURES)
MIN_ROWS = 24  # training rows of a per-target regression
SCORE_START = pd.Timestamp("2020-01-01")  # = scoring.SCORE_START (not imported: scoring imports models.base)
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
CV_FOLDS = 4
CHAIN = ["S04_EUR", "cru_cr_de", "S09"]
VECM_MIN = 48  # months on the common window
GBM_REFIT_EVERY = 3
GBM_MIN_ROWS = 100
SHRINK_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
SHRINK_DEFAULT = 0.5  # warm-up origins only (not scored)

LOG = {"johansen": [], "gbm_shrink": [], "epoxy_beta": [], "ridge_alpha": []}  # reset by models()


def _feats(ctx):
    if not hasattr(ctx, "_fam_feats"):
        ctx._fam_feats, ctx._fam_design = F.build(ctx.pt), {}
    return ctx._fam_feats


def _design(ctx, s):
    _feats(ctx)
    if s not in ctx._fam_design:
        ctx._fam_design[s] = F.design(ctx.pt, s, ctx._fam_feats).loc[:ctx.anchor[s]]
    return ctx._fam_design[s]


def _rows(ctx, s, h):
    """Training rows (features complete, outcome known at t) and the anchor row."""
    X = _design(ctx, s)
    y = ctx.ys(s, ctx.h_eff(s, h)).reindex(X.index)
    ok = y.notna() & X.notna().all(axis=1)
    return X[ok], y[ok], X.loc[ctx.anchor[s]]


def _ols(A, y):
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return b


class Ardl(Model):
    name = "ardl"
    series = FAMILY_TARGETS

    def predict(self, ctx, s):
        out = {}
        for h in HORIZONS:
            X, y, xa = _rows(ctx, s, h)
            if len(y) < MIN_ROWS or not np.isfinite(xa.values).all():
                continue
            b = _ols(np.column_stack([np.ones(len(X)), X.values]), y.values)
            out[h] = point(float(b[0] + xa.values @ b[1:]), len(y))
        return out


def ts_cv_alpha(X, y, gap):
    """Ridge alpha by expanding-window CV on the last half of the rows (CV_FOLDS folds, `gap` rows dropped before
    each validation fold so that no training outcome reaches into it). Ties -> the larger alpha."""
    n = len(y)
    edges = np.linspace(n // 2, n, CV_FOLDS + 1).astype(int)
    err = {a: [] for a in ALPHAS}
    for k in range(CV_FOLDS):
        v0, v1 = edges[k], edges[k + 1]
        tr = slice(0, max(v0 - gap, 0))
        if v0 - gap < 20 or v1 <= v0:
            continue
        for a in ALPHAS:
            m = make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(X[tr], y[tr])
            err[a].append(np.abs(m.predict(X[v0:v1]) - y[v0:v1]).mean())
    scored = {a: np.mean(e) for a, e in err.items() if e}
    if not scored:
        return 10.0
    best = min(scored.values())
    return max(a for a, e in scored.items() if np.isclose(e, best))


class RidgeFamily(Model):
    name = "ridge_fam"
    series = FAMILY_TARGETS

    def predict(self, ctx, s):
        out = {}
        for h in HORIZONS:
            X, y, xa = _rows(ctx, s, h)
            if len(y) < MIN_ROWS or not np.isfinite(xa.values).all():
                continue
            he = ctx.h_eff(s, h)
            alpha = ts_cv_alpha(X.values, y.values, he)
            m = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X.values, y.values)
            out[h] = point(float(m.predict(xa.values.reshape(1, -1))[0]), len(y))
            if ctx.t.month == 1 or ctx.t == ctx.pt.index[-1]:
                LOG["ridge_alpha"].append({"origin": ctx.t, "series": s, "h": h, "alpha": alpha, "n": len(y)})
        return out


class Ecm(Model):
    name = "ecm"
    series = {"S09", "S10"}
    driver = "cru_cr_de"

    def predict(self, ctx, s):
        a = ctx.anchor[s]
        L = np.log(ctx.pt[[s, self.driver]].loc[:a])
        lv = L.dropna()
        if len(lv) < 36:
            return {}
        c, b = _ols(np.column_stack([np.ones(len(lv)), lv[self.driver].values]), lv[s].values)
        d = L[self.driver].diff()
        Z = pd.DataFrame({"ect": L[s] - c - b * L[self.driver], "drv_r1": d, "drv_r1_l1": d.shift(1),
                          "own_r1": L[s].diff()})
        za = Z.loc[a]
        if not np.isfinite(za.values).all():
            return {}
        out = {}
        for h in HORIZONS:
            y = ctx.ys(s, ctx.h_eff(s, h)).reindex(Z.index)
            ok = y.notna() & Z.notna().all(axis=1)
            if ok.sum() < MIN_ROWS:
                continue
            g = _ols(np.column_stack([np.ones(ok.sum()), Z[ok].values]), y[ok].values)
            out[h] = point(float(g[0] + za.values @ g[1:]), int(ok.sum()))
        return out


class VecmChain(Model):
    name = "vecm"
    series = {"cru_cr_de", "S09"}

    def __init__(self):
        self.paths = {}

    def prepare(self, ctx):
        self.paths = {}
        Y = np.log(ctx.pt[CHAIN]).dropna()
        if len(Y) < VECM_MIN:
            return
        end = Y.index[-1]
        steps = max(HORIZONS) + months(end, ctx.t)
        jo = coint_johansen(Y.values, det_order=0, k_ar_diff=1)
        rank = int(np.cumprod(jo.lr1 > jo.cvt[:, 1]).sum())  # trace test at 5%
        try:
            if rank >= 1:
                fit = VECM(Y.values, k_ar_diff=1, coint_rank=rank, deterministic="ci").fit()
                path, kind = fit.predict(steps=steps) - Y.values[-1], "vecm"
            else:
                dY = np.diff(Y.values, axis=0)
                fit = VAR(dY).fit(1)
                path, kind = np.cumsum(fit.forecast(dY[-1:], steps), axis=0), "var_diff"
        except Exception:  # noqa: BLE001 - no forecast at this origin
            return
        self.paths = {v: (path[:, i], end, len(Y)) for i, v in enumerate(CHAIN)}
        LOG["johansen"].append({"origin": ctx.t, "n": len(Y), "rank": rank, "model": kind,
                                "trace_r0": jo.lr1[0], "cv95_r0": jo.cvt[0, 1],
                                "trace_r1": jo.lr1[1], "cv95_r1": jo.cvt[1, 1],
                                "trace_r2": jo.lr1[2], "cv95_r2": jo.cvt[2, 1]})

    def predict(self, ctx, s):
        if s not in self.paths:
            return {}
        path, end, n = self.paths[s]
        base = np.log(ctx.pt.at[end, s]) - np.log(ctx.y_a(s))  # forecasts start from the common end month
        off = months(end, ctx.t)
        return {h: point(float(base + path[h + off - 1]), n) for h in HORIZONS}


GBM_OWN = ["own_r1", "own_r3", "own_r6", "dev12", "vol12"]


class GbmFamily(Model):
    name = "gbm_family"
    native = True
    series = FAMILY_TARGETS

    def __init__(self):
        self.fits, self.last_fit, self.raw, self.w = {}, {}, [], None
        self.fam_code = {f: i for i, f in enumerate(sorted({T.BY_ID[s].family for s in FAMILY_TARGETS}))}

    def _stack(self, ctx):
        feats = _feats(ctx)
        frames = []
        for s in sorted(FAMILY_TARGETS):
            lg = ctx.L[s]
            x = pd.DataFrame({"own_r1": lg.diff(), "own_r3": lg - lg.shift(3), "own_r6": lg - lg.shift(6),
                              "dev12": lg - lg.rolling(12).mean(), "vol12": lg.diff().rolling(12).std()})
            for f in F.FEATURES:
                x[f.name] = feats[f.name] if f.name in F.TARGET_FEATURES[s] else np.nan
            x["family"] = self.fam_code[T.BY_ID[s].family]
            x["series"] = s
            frames.append(x)
        return pd.concat(frames).set_index("series", append=True).sort_index()

    def prepare(self, ctx):
        X = self._stack(ctx)
        ctx._gbm_family_X = X
        cols = list(X.columns)
        for h in HORIZONS:
            if h in self.last_fit and months(self.last_fit[h], ctx.t) < GBM_REFIT_EVERY:
                continue
            y = pd.concat({s: ctx.ys(s, ctx.h_eff(s, h)) for s in sorted(FAMILY_TARGETS)}, axis=1).stack()
            y.index = y.index.set_names([X.index.names[0], "series"])
            Xtr = X.reindex(y.index)
            ok = y.notna() & Xtr[["own_r1", "own_r3"]].notna().all(axis=1)
            self.last_fit[h] = ctx.t
            if ok.sum() < GBM_MIN_ROWS:
                self.fits[h] = None
                continue
            self.fits[h] = {q: HistGradientBoostingRegressor(
                loss="quantile", quantile=q, max_iter=100, learning_rate=0.05, max_depth=2, min_samples_leaf=40,
                l2_regularization=1.0, categorical_features=[cols.index("family")], random_state=0,
            ).fit(Xtr[ok].values, y[ok].values) for q in QS}
            self.fits[h]["n"] = int(ok.sum())
        if self.w is None and ctx.t >= SCORE_START:
            self._choose_shrink(ctx)

    def _choose_shrink(self, ctx):
        """w minimising the MAE of w x q50 on warm-up forecasts whose outcome is known at this origin."""
        rows = []
        for t0, s, a0, he, q50 in self.raw:
            if t0 >= SCORE_START:
                continue
            r = ctx.ys(s, he).get(a0, np.nan)
            if pd.notna(r):
                rows.append((q50, r))
        if len(rows) < 50:
            self.w = SHRINK_DEFAULT
            LOG["gbm_shrink"].append({"origin": ctx.t, "w": self.w, "n": len(rows), "rule": "default (too few)"})
            return
        q, r = np.array(rows).T
        mae = {w: np.abs(w * q - r).mean() for w in SHRINK_GRID}
        self.w = min(mae, key=mae.get)
        LOG["gbm_shrink"].append({"origin": ctx.t, "w": self.w, "n": len(rows), "rule": "warm-up MAE",
                                  **{f"mae_w{w}": v for w, v in mae.items()}})

    def predict(self, ctx, s):
        X = ctx._gbm_family_X
        a = ctx.anchor[s]
        if (a, s) not in X.index:
            return {}
        xg = X.loc[[(a, s)]].values
        out = {}
        w = SHRINK_DEFAULT if self.w is None else self.w
        for h in HORIZONS:
            f = self.fits.get(h)
            if f is None:
                continue
            q10, q50, q90 = np.sort([float(f[q].predict(xg)[0]) for q in QS])
            if ctx.t < SCORE_START:
                self.raw.append((ctx.t, s, a, ctx.h_eff(s, h), q50))
            c = w * q50
            out[h] = Pred(min(q10 - q50 + c, c), c, max(q90 - q50 + c, c), f["n"])
        return out


class EpoxyStructural(Model):
    """Input forecasts = mean(naive, AutoETS) of each input, computed here in prepare() (the same rule as combo_mean,
    which cannot be read: the inputs come after epoxy in the target loop, and "other costs" is not a target)."""
    name = "epoxy_struct"
    series = {"epoxy"}

    def __init__(self):
        self.inputs = {}

    def prepare(self, ctx):
        self.inputs = {}
        for i in F.EPOXY_INPUTS:
            hist = np.log(ctx.pt[i]).dropna()
            path = self._auto_ets(hist, max(HORIZONS) + 2)
            if path is None:
                return
            self.inputs[i] = (hist.index[-1], 0.5 * path)  # mean of naive (0) and AutoETS log returns

    def predict(self, ctx, s):
        a = ctx.anchor[s]
        cols = F.EPOXY_INPUTS + [s]
        L = np.log(ctx.pt[cols].loc[:a])
        if L[cols].loc[a].isna().any() or len(self.inputs) < len(F.EPOXY_INPUTS) \
                or any(end != a for end, _ in self.inputs.values()):
            return {}
        out = {}
        for h in HORIZONS:
            he = ctx.h_eff(s, h)
            dY = (L.shift(-he) - L).dropna()
            if len(dY) < MIN_ROWS:
                continue
            beta = _ols(dY[F.EPOXY_INPUTS].values, dY[s].values)
            f = np.array([self.inputs[i][1][he - 1] for i in F.EPOXY_INPUTS])
            out[h] = point(float(beta @ f), len(dY))
            if ctx.t == ctx.pt.index[-1] or ctx.t.month == 1:
                LOG["epoxy_beta"].append({"origin": ctx.t, "h": h, "n": len(dY),
                                          **dict(zip(F.EPOXY_INPUTS, np.round(beta, 4)))})
        return out

    @staticmethod
    def _auto_ets(hist, steps):
        if len(hist) < 12:
            return None
        try:
            path = M.AutoETS(season_length=1, model="ZZZ", damped=None).forecast(
                y=hist.values.astype(float), h=steps)["mean"]
        except Exception:  # noqa: BLE001
            return None
        return np.asarray(path) - hist.iloc[-1] if np.isfinite(path).all() else None


def models():
    for v in LOG.values():
        v.clear()
    return [Ardl(), RidgeFamily(), Ecm(), VecmChain(), GbmFamily(), EpoxyStructural()]


NAMES = ["ardl", "ridge_fam", "ecm", "vecm", "gbm_family", "epoxy_struct"]


# ---------------------------------------------------------------------------------------------------------------------
# Report: family models against the best univariate model of each target
UNIVARIATE = ["naive", "drift", "ets_local", "auto_ets", "auto_arima", "theta", "otheta", "combo_mean", "combo_median"]
BANDS = ["aci", "ewma"]


def compare(res):
    """Per (target, h, family model): rel. MAE and Diebold-Mariano vs the best univariate model of that target and h
    (lowest MAE over the scored origins, chosen with hindsight, which favours the benchmark), rel. MAE vs naive, and
    coverage / Winkler of the aci and ewma bands (and the native GBM band)."""
    import scoring  # local: scoring imports models.base

    r = scoring.scored_rows(res)
    pts = r[r["interval"] == "emp"]
    rows = []
    for (s, h), g in pts[pts["point"].isin(NAMES)].groupby(["series", "h"]):
        uni = pts[(pts["series"] == s) & (pts["h"] == h) & pts["point"].isin(UNIVARIATE)]
        mae_uni = uni.groupby("point")["ae"].mean()
        best = mae_uni.idxmin()
        b = uni[uni["point"] == best].set_index("origin")
        for m, gm in g.groupby("point"):
            gm = gm.set_index("origin").sort_index()
            common = gm.index.intersection(b.index)
            d = gm.loc[common, "ae"] - b.loc[common, "ae"]
            stat, p = scoring.dm_test(d.values, int(gm["h_eff"].max()) - 1)
            row = {"series": s, "h": h, "model": m, "n": len(common), "best_univariate": best,
                   "rel_mae_vs_best_uni": gm.loc[common, "ae"].mean() / b.loc[common, "ae"].mean(),
                   "dm_stat": stat, "dm_p": p, "significantly_better": bool(p < 0.05 and stat < 0),
                   "significantly_worse": bool(p < 0.05 and stat > 0),
                   "rel_mae_vs_naive": gm["ae"].mean() / gm["naive_ae"].mean()}
            for band in BANDS + (["native"] if m == "gbm_family" else []):
                gb = r[(r["model"] == f"{m}+{band}") & (r["series"] == s) & (r["h"] == h)]
                row[f"coverage80_{band}"] = gb["covered"].mean()
                row[f"winkler_{band}"] = gb["winkler"].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def write_reports(res, d):
    """Family-model tables next to the backtest scores; returns the comparison table."""
    cmp = compare(res)
    cmp.to_csv(d / "multivariate_vs_univariate.csv", index=False, float_format="%.4f")
    for k, v in LOG.items():
        if v:
            pd.DataFrame(v).to_csv(d / {"johansen": "vecm_johansen.csv", "gbm_shrink": "gbm_shrink.csv",
                                        "epoxy_beta": "epoxy_beta.csv", "ridge_alpha": "ridge_alpha.csv"}[k],
                                   index=False, float_format="%.4f")
    pd.set_option("display.width", 220)
    print("\nFamily models vs the best univariate model per target (rel. MAE < 1 = better; * = DM p < 0.05 better, "
          "! = significantly worse), origins 2020-01+:")
    t = cmp.assign(cell=cmp["rel_mae_vs_best_uni"].map("{:.2f}".format)
                   + np.where(cmp["significantly_better"], "*", np.where(cmp["significantly_worse"], "!", "")))
    print(t.pivot_table(index=["series", "model"], columns="h", values="cell", aggfunc="first").to_string())
    sig = cmp[cmp["significantly_better"]]
    print("\nSignificantly better than the best univariate model:",
          "none" if sig.empty else "; ".join(f"{r.series} h={r.h} {r.model} ({r.rel_mae_vs_best_uni:.2f}, "
                                             f"p={r.dm_p:.3f})" for r in sig.itertuples()))
    return cmp
