"""Forecasting models of the backtest (interface: models/base.py) and the interval layer (models/intervals.py).

make_registry() returns fresh instances in run order: combinations read the P50 of models earlier in the list.
"""
from __future__ import annotations

from models import classic, combo, lme, sf


def make_registry():
    return ([classic.Naive(), classic.Drift(), classic.EtsLocal(), classic.RidgeLocal(), classic.GbmGlobal()]
            + sf.models() + combo.models() + lme.models())


LEGACY = {  # first-backtest model name -> label in the new backtest (identical forecasts)
    "naive": "naive+emp", "drift": "drift+emp", "ets_local": "ets_local+emp",
    "ridge_local": "ridge_local+native", "gbm_global": "gbm_global+native",
}
