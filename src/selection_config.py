"""Configuration of the model selection (selection.py). Every rule below was fixed before selection.py was run; the
submission model map (submission_models.csv) follows from these rules and the backtest, with no manual choice.

Why a small candidate set: on the 22 submitted targets, the Diebold-Mariano tests of every point model against naive
(885 model x target x horizon cells, 852 with a p value) give no model significantly better than naive after
Benjamini-Hochberg, and choosing the best model per
target on 2020-01..2023-06 is worse than naive afterwards. Choosing among many models on a short, regime-dependent
backtest mostly selects noise, so the choice is limited to a few candidates, each with a reason to be there that
does not come from the backtest ranking. Every other model stays in the backtest report (outputs/backtest/), not in
the selection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PPI_IDS = ["S07", "S08", "S09", "S10", "S11", "S12", "S13", "RM6_NE_IQI_433520", "RMAirSeT_IIQ_24kV"]
LME_IDS = ["D01", "D02", "S01", "S02"]

# point model -> (targets it may be selected for, None = every submitted target; reason)
CANDIDATES = {
    "naive": (None, "Reference and fallback of every rule: P50 = last known value. No point model is significantly "
                    "better than it after the Benjamini-Hochberg correction."),
    "lme_fwd": (LME_IDS, "Copper and aluminium only (D01, D02, S01, S02): the LME 3-month price is the market's own "
                         "price for later delivery. Nothing is estimated, so nothing can be over-fitted; it exists "
                         "only for these two metals."),
    "combo_median": (None, "Median of naive, AutoETS, Theta and AutoARIMA: a robust combination, the steadiest "
                           "model across the two periods (geometric-mean MAE / naive below 1 after 2023-07 at 3 and "
                           "6 months). One rule for every target."),
    "ets_local": (PPI_IDS, "PPI-type series only (S07-S13 and the PPI-proxy BOM indices): a PPI averages contract "
                           "prices, so its monthly changes are autocorrelated and damped ETS can carry the momentum. "
                           "On exchange and market prices it has no such mechanism. Its edge depends on the regime "
                           "(MAE / naive, 2020-01..2023-06 -> 2023-07..end, h = 3 / 9: PPI-proxy BOM 0.81 / 0.90 -> "
                           "0.92 / 1.10, plastics 0.84 / 1.03 -> 1.42 / 1.37), which is why the rolling window "
                           "decides when to use it."),
    "chronos2": (None, "Pretrained foundation model, zero-shot: no parameter is fitted on these series, so it adds a "
                       "different kind of forecaster without adding parameters chosen on the backtest."),
}
NOT_CANDIDATES = {  # kept in the backtest report only
    "auto_ets, theta, auto_arima": "components of combo_median",
    "combo_mean": "near duplicate of combo_median (two of its four components)",
    "drift, ridge_local, gbm_global, otheta": "worse than naive on most targets at 6-9 months",
    "ardl, ridge_fam, ecm, vecm, epoxy_struct": "family models: none significantly better than the best univariate "
                                                "model; few targets each",
    "gbm_family": "shrink weight 0 from the warm-up: its P50 is naive",
    "chronos2_cov": "worse after the release of the weights (2025-11); 11 targets only",
}

# Rolling selection ("selected"): at origin t, for horizon h, the point errors whose outcome is published at or before
# t (same rule as the interval layer) and in the last WINDOW_M months (reveal in (t - WINDOW_M, t]). Per target:
# MAE of the candidate / MAE of naive on the same rows (at least MIN_ERRORS rows). Per family: geometric mean of
# these ratios over the family's submitted targets where the candidate applies. Each target takes the family's best
# candidate among those that apply to it; if that ratio is above 1 - MIN_GAIN (gain < 3%), naive.
WINDOW_M = 24
MIN_GAIN = 0.03
MIN_ERRORS = 12

# Report periods (by origin): scoring.PERIODS = 2020-01..2023-06 (stress), 2023-07..end, 2022 alone; plus all.

# Intervals: the four bands of the interval layer, plus an asymmetric version of each ("<method>_up"): the upper
# distance P90 - P50 (log) times k >= 1, P10 unchanged. One k per (point model, method, family, h), the smallest of
# ASYM_GRID whose share of outcomes above P90 is <= ASYM_TARGET_ABOVE, on the origins of 2020-01..2023-06 whose
# outcome is published by ASYM_TUNE_BY (= first origin of the second period). So the second period tests k out of
# sample; the first-period coverage of a "_up" band is in sample.
# ewmap (pooled residuals, faster EWMA) added after an offline check: around naive it gave 71 / 70 / 68% coverage in
# 2020-01..2023-06 and 86 / 85 / 90% after (ewma: 69 / 67 / 61% and 88 / 90 / 95%).
# k is capped at ASYM_K_MAX: k multiplies a LOG distance, so k = 3 at 6-9 months turned P90 into +120% (copper) to
# +475% (plastics PPI) at Sep-2026, far beyond any 9-month move in the data. The cap keeps "_up" a skew, not a blow-up.
BASE_METHODS = ("emp", "conf", "aci", "ewma", "ewmap")
UP = "_up"
METHODS = BASE_METHODS + tuple(m + UP for m in BASE_METHODS)
ASYM_K_MAX = 1.5
ASYM_GRID = np.round(np.arange(1.0, ASYM_K_MAX + 0.0001, 0.05), 2)
ASYM_TARGET_ABOVE = 0.10
ASYM_TUNE_BY = pd.Timestamp("2023-07-01")

# Interval choice per family x h, around the point strategy chosen for that family x h: a band is admissible when its
# 80% coverage is within COVERAGE_RANGE in both periods (2020-01..2023-06 and 2023-07..end); among admissible bands
# the lowest Winkler score (all scored origins). If none is admissible: the highest stress-period (2020-01..2023-06)
# coverage, capped at the upper bound of COVERAGE_RANGE (so over-coverage earns nothing), then the lowest Winkler.
COVERAGE_RANGE = (0.70, 0.90)

# Point strategy per family x h for the submission: a strategy of POINT_STRATEGIES replaces naive only if its
# geometric-mean MAE / naive's MAE over the family's targets is below 1 in BOTH periods; if several do, the one with
# the lower worse-period ratio. "selected" then means the rolling rule's choice at the last origin (Sep-2026).
POINT_STRATEGIES = ("selected", "combo_median")
