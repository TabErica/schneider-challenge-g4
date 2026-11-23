"""Every modelling assumption in one table -> outputs/assumptions.csv (written by preprocess.py). README lists them.

Data transformations (calendar, cutoff, corrections, fills, FX) are logged row by row in transform_log.csv; this
table holds the decisions that are not a single transformation of one value.
"""
from __future__ import annotations

import pandas as pd

# (id, area, assumption, where in the code)
ASSUMPTIONS = [
    ("A01", "data", "Forecast origin = Sep-2026 = last month of data used. Values after Sep-2026 (Oct-2026 FX, power, "
     "CRU, MEPS, epoxy, Westmetall) are dropped before any other step and never used, also not as a nowcast input.",
     "preprocess.CUTOFF"),
    ("A02", "data", "Ragged edge, relative to the origin: at every backtest origin each series is cut by the number "
     "of months it lacks at the real origin (stainless 304 and its parts: 1, E20 / E21 / ED06: 2, all others: 0).",
     "preprocess.relative_lags, preprocess.as_of"),
    ("A03", "data", "A target whose last value is before the origin is forecast from its last value: the effective "
     "horizon is h + lag (stainless 304 and the market-price BOM indices: h + 1).", "backtest.run"),
    ("A04", "data", "No forward fill in any data used for forecasting. data.TRAIL_FILL (carry stainless Aug-2026 to "
     "Sep-2026) is kept only for the descriptive EDA charts and tables.", "data.panel(trail_fill=...)"),
    ("A05", "data", "Optional nowcast (off by default): the missing stainless 304 month(s) at an origin are filled with "
     "an OLS of its monthly log change on nickel in EUR (S03_EUR) and the EU steel PPI (S09), fitted on the months "
     "known at that origin. Each fill is logged (nowcast_log.csv).", "preprocess.nowcast, backtest --nowcast"),
    ("A06", "data", "Eurostat PPIs (S07-S11, E13-E16): the last 2 months known at an origin are provisional (Eurostat "
     "revises them). Baseline keeps them; the sensitivity run drops them. Only the final vintage is available, so "
     "the backtest sees revised values (slightly optimistic).", "preprocess.PROVISIONAL, backtest --drop-provisional"),
    ("A07", "model", "Univariate models use each target's full history (CRU from 2012, stainless from 2008, epoxy from "
     "2018); models with drivers use the months where every input is known.", "targets.py, backtest.py"),
    ("A08", "BOM", "RM6 largest unlabelled slice = carbon steel at 45% of the implied mass (the share the older BOM "
     "label gave).", "data.ASSUMED_UNLABELLED_SHARE"),
    ("A09", "BOM", "Polymer levels assumed at 1 300 EUR/t (PET) and 2 700 EUR/t (engineering plastics), constant over "
     "time (also for origin weights).", "data.ASSUMED_PRICE_EUR_T"),
    ("A10", "BOM", "Submission BOM indices: base Sep-2026 = 100 (Schneider's common base). Cost shares = kg x latest "
     "published EUR/t at or before Sep-2026 (stainless 304: Aug-2026); the market-price index has no Sep-2026 value "
     "until stainless is published, so it is forecast from Aug-2026.", "data.bom_cost_table, data.bom_index"),
    ("A11", "BOM", "Backtest option --bom-weights origin: cost shares from the prices known at each origin "
     "(kg x latest EUR/t at or before the origin) instead of Sep-2026 shares, to measure the look-ahead in the fixed "
     "weights. Not used for the submission.", "backtest --bom-weights"),
    ("A12", "BOM", "Fixed quantities (kg per unit) over the whole period: no design or sourcing change.", "data.load_bom"),
    ("A13", "data", "ECH and epoxy May-2026 are a decimal-shift typo: set to NaN, then log-linear interpolation.",
     "preprocess.CORRECTIONS"),
]


def frame():
    return pd.DataFrame(ASSUMPTIONS, columns=["id", "area", "assumption", "where"])


def write(path):
    frame().to_csv(path, index=False)
