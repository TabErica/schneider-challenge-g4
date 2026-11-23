"""Model interface of the backtest.

A model sees one origin t through a Context: the panel as of t (preprocess.as_of, nowcast, BOM indices), the log
history of every target up to its anchor a (its last month known at t) and helpers. For a target s it returns, per
horizon h, the quantiles of the log return log(y[t+h] / y[a]) (effective horizon h_eff = h + (t - a)):

    model.predict(ctx, s) -> {h: Pred(q10, q50, q90, n_train)}

Point models return q10 = q90 = NaN: their P10 / P90 come from the interval layer (models/intervals.py), which can
wrap any point model. Models with their own quantiles (native = True: Ridge residuals, GBM quantile loss) also keep
them as the interval method "native". Global models fit across targets once per origin in prepare(ctx).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd

import targets as T

HORIZONS = (3, 6, 9)
QS = (0.1, 0.5, 0.9)
TARGET_IDS = T.ids()
DRIVER_IDS = ["ED01", "ED02", "E09", "S04", "S03", "S01", "S06"]
FAMILY = {t.id: t.family for t in T.REGISTRY}


@dataclass(frozen=True)
class Pred:
    q10: float
    q50: float
    q90: float
    n_train: int


def months(a, b):
    """Whole months from a to b."""
    return (b.to_period("M") - a.to_period("M")).n


def empirical_spread(hist_ret):
    """P10 / P90 of past h-month log-returns, centred on their median."""
    r = hist_ret.dropna()
    if len(r) < 12:
        return -0.1, 0.1
    return r.quantile(0.1) - r.median(), r.quantile(0.9) - r.median()


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


class Context:
    """Everything known at origin t. `pt` is the as_of() panel (with nowcast and BOM indices)."""

    def __init__(self, t, pt):
        self.t, self.pt = t, pt
        self.L = np.log(pt[TARGET_IDS])
        self.anchor = {s: pt[s].last_valid_index() for s in TARGET_IDS}
        self.lag = {s: months(self.anchor[s], t) for s in TARGET_IDS}
        self.q50 = {}  # (model, series) -> {h: q50}, read by combination models
        self._hist, self._ys = {}, {}

    @cached_property
    def X(self):
        return build_features(self.pt)

    def hist(self, s):
        """Log level of s up to its anchor (full history, no leading NaN)."""
        if s not in self._hist:
            self._hist[s] = self.L[s].loc[:self.anchor[s]].dropna()
        return self._hist[s]

    def y_a(self, s):
        return self.pt.at[self.anchor[s], s]

    def h_eff(self, s, h):
        return h + self.lag[s]

    def ys(self, s, he):
        """he-month log changes of s known at t (NaN where the outcome is not known yet)."""
        if (s, he) not in self._ys:
            self._ys[s, he] = self.L[s].shift(-he) - self.L[s]
        return self._ys[s, he]


class Model:
    name = ""
    native = False  # True: predict() returns its own q10 / q90
    series = None  # None = every target; else the targets the model applies to

    def applies(self, s):
        return self.series is None or s in self.series

    def prepare(self, ctx):
        """Once per origin, before predict() (global models fit here)."""

    def predict(self, ctx, s):
        raise NotImplementedError


def point(q50, n_train):
    return Pred(np.nan, float(q50), np.nan, int(n_train))
