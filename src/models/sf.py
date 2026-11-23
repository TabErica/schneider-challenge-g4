"""statsforecast point models on the log level: AutoETS (damped trend allowed), AutoARIMA, Theta, OptimizedTheta.

season_length = 1: monthly prices with no stable seasonality and 3-10 years of history; the Eurostat indices are
not seasonally adjusted, but a seasonal model would add 11 parameters to fit on 40-120 points. Each model is fit
once per (origin, target) and forecast to h_eff = 9 + lag; P50 log return = forecast(h_eff) - log(y at anchor).
"""
from __future__ import annotations

import numpy as np
import statsforecast.models as M

from models.base import HORIZONS, Model, point

MIN_HIST = 12  # months of history needed to fit (warm-up origins of the 2018 series get no forecast)


class StatsForecastModel(Model):
    def __init__(self, name, make):
        self.name, self.make = name, make

    def predict(self, ctx, s):
        hist = ctx.hist(s)
        if len(hist) < MIN_HIST:
            return {}
        try:
            path = self.make().forecast(y=hist.values.astype(float), h=max(HORIZONS) + ctx.lag[s])["mean"]
        except Exception:
            return {}
        if not np.isfinite(path).all():
            return {}
        la = np.log(ctx.y_a(s))
        return {h: point(float(path[ctx.h_eff(s, h) - 1] - la), len(hist)) for h in HORIZONS}


def models():
    return [
        StatsForecastModel("auto_ets", lambda: M.AutoETS(season_length=1, model="ZZZ", damped=None)),
        StatsForecastModel("auto_arima", lambda: M.AutoARIMA(season_length=1)),
        StatsForecastModel("theta", lambda: M.Theta(season_length=1)),
        StatsForecastModel("otheta", lambda: M.OptimizedTheta(season_length=1)),
    ]
