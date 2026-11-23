"""The models of the first backtest, moved here unchanged: naive, drift, damped ETS (statsmodels), Ridge per series,
one global GBM. naive / drift / ets_local are point models (their old P10 / P90 = the "emp" interval method);
Ridge and GBM keep their own quantiles (native)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from models.base import HORIZONS, QS, TARGET_IDS, Model, Pred, months, point

GBM_REFIT_EVERY = 3  # months
GBM_MIN_ROWS = 50  # warm-up origins with fewer training rows get no GBM forecast


class Naive(Model):
    name = "naive"

    def predict(self, ctx, s):
        return {h: point(0.0, len(ctx.hist(s))) for h in HORIZONS}


class Drift(Model):
    """Mean monthly log change of the last 36 months, times h_eff."""
    name = "drift"

    def predict(self, ctx, s):
        hist = ctx.hist(s)
        m = hist.diff().tail(36).mean()
        return {h: point(m * ctx.h_eff(s, h), len(hist)) for h in HORIZONS}


class EtsLocal(Model):
    """Additive damped-trend Holt on the log level (statsmodels); 0 if the fit fails."""
    name = "ets_local"

    def predict(self, ctx, s):
        hist = ctx.hist(s)
        try:
            fit = ExponentialSmoothing(hist.values, trend="add", damped_trend=True,
                                       initialization_method="estimated").fit()
            path = fit.forecast(max(HORIZONS) + ctx.lag[s])
        except Exception:
            path = None
        la = np.log(ctx.y_a(s))
        return {h: point(float(path[ctx.h_eff(s, h) - 1] - la) if path is not None else 0.0, len(hist))
                for h in HORIZONS}


class RidgeLocal(Model):
    """Ridge per series on own returns + driver returns; P10 / P90 = P50 + in-sample residual quantiles."""
    name = "ridge_local"
    native = True

    def predict(self, ctx, s):
        a = ctx.anchor[s]
        xs = ctx.X.xs(s, level="series").loc[:a].drop(columns=["family"])
        xa = ctx.X.loc[(a, s)].drop("family").values.reshape(1, -1).astype(float)
        out = {}
        for h in HORIZONS:
            yy = ctx.ys(s, ctx.h_eff(s, h)).reindex(xs.index)
            ok = yy.notna() & xs.notna().all(axis=1)
            if ok.sum() >= 24 and np.isfinite(xa).all():
                rg = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(xs[ok].values, yy[ok].values)
                res = yy[ok].values - rg.predict(xs[ok].values)
                mu = float(rg.predict(xa)[0])
                q10, q50, q90 = np.sort((mu + np.quantile(res, 0.1), mu, mu + np.quantile(res, 0.9)))
                out[h] = Pred(q10, q50, q90, int(ok.sum()))
        return out


class GbmGlobal(Model):
    """One quantile GBM per horizon and quantile over every (month, target), refit every 3 months."""
    name = "gbm_global"
    native = True

    def __init__(self):
        self.fits, self.last_fit = {}, {}

    def prepare(self, ctx):
        X, L, t = ctx.X, ctx.L, ctx.t
        cols = list(X.columns)
        for h in HORIZONS:
            if h in self.last_fit and months(self.last_fit[h], t) < GBM_REFIT_EVERY:
                continue
            # every (month, series) whose h_eff-month outcome is known at t
            ytr = pd.concat({s: (L[s].shift(-(h + ctx.lag[s])) - L[s]) for s in TARGET_IDS}, axis=1).stack()
            Xtr = X.reindex(ytr.index)
            ok = ytr.notna() & Xtr.notna().all(axis=1)
            self.last_fit[h] = t
            if ok.sum() < GBM_MIN_ROWS:
                self.fits[h] = None
                continue
            self.fits[h] = {q: HistGradientBoostingRegressor(
                loss="quantile", quantile=q, max_iter=150, learning_rate=0.05, max_depth=3,
                min_samples_leaf=20, categorical_features=[cols.index("family")], random_state=0,
            ).fit(Xtr[ok].values, ytr[ok].values) for q in QS}
            self.fits[h]["n"] = int(ok.sum())

    def predict(self, ctx, s):
        cols = list(ctx.X.columns)
        xg = ctx.X.loc[[(ctx.anchor[s], s)], cols].values
        if not np.isfinite(np.delete(xg, cols.index("family"), axis=1)).all():
            return {}
        out = {}
        for h in HORIZONS:
            f = self.fits.get(h)
            if f is not None:
                q10, q50, q90 = np.sort([float(f[q].predict(xg)[0]) for q in QS])
                out[h] = Pred(q10, q50, q90, f["n"])
        return out
