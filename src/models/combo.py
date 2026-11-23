"""Combinations of point forecasts (log returns), computed from the P50 of models run earlier at the same origin.
A combination is only made when every component has a forecast."""
from __future__ import annotations

import numpy as np

from models.base import HORIZONS, Model, point


class Combination(Model):
    def __init__(self, name, components, how):
        self.name, self.components, self.how = name, components, how

    def predict(self, ctx, s):
        parts = [ctx.q50.get((c, s), {}) for c in self.components]
        out = {}
        for h in HORIZONS:
            v = [p[h] for p in parts if h in p]
            if len(v) == len(self.components):
                out[h] = point(np.mean(v) if self.how == "mean" else np.median(v), len(ctx.hist(s)))
        return out


def models():
    return [
        Combination("combo_mean", ["naive", "auto_ets"], "mean"),
        Combination("combo_median", ["naive", "auto_ets", "theta", "auto_arima"], "median"),
    ]
