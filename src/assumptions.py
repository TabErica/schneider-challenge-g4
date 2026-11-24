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
     "horizon is h + lag. With the default stainless nowcast (A05) every submitted target is known at the origin "
     "(h_eff = h); the sensitivity variant --no-nowcast forecasts stainless 304 and the market-price BOM indices from "
     "t - 1 (h + 1).", "backtest.run"),
    ("A04", "data", "No forward fill in any data used for forecasting. data.TRAIL_FILL (carry stainless Aug-2026 to "
     "Sep-2026) is kept only for the descriptive EDA charts and tables.", "data.panel(trail_fill=...)"),
    ("A05", "data", "Stainless 304 nowcast, ON by default (baseline and submission): the missing month(s) at an origin "
     "are filled with an OLS of its monthly log change on nickel in EUR (S03_EUR) and the EU steel PPI (S09), fitted "
     "on the months known at that origin; uses nothing after the origin. Chosen because the backtest showed lower "
     "errors for stainless and the market-price BOM with no leakage. Each fill is logged (nowcast_log.csv); "
     "--no-nowcast is the sensitivity variant.", "preprocess.nowcast, backtest --no-nowcast"),
    ("A06", "data", "Eurostat PPIs (S07-S11, E13-E16): the last 2 months known at an origin are provisional (Eurostat "
     "revises them); they are flagged and kept in the baseline. Revision risk is NOT measured: the files hold only "
     "the latest vintage, so the backtest sees revised values (optimistic by an unknown amount). The "
     "--drop-provisional variant removes those months, which moves the anchor back 2 months (h_eff = h + 2): its "
     "higher error reflects the longer horizon, not revisions.", "preprocess.PROVISIONAL, backtest --drop-provisional"),
    ("A07", "model", "Univariate models use each target's full history (CRU from 2012, stainless from 2008, epoxy from "
     "2018); models with drivers use the months where every input is known.", "targets.py, backtest.py"),
    ("A08", "BOM", "RM6 largest unlabelled slice = carbon steel at 45% of the implied mass (the share the older BOM "
     "label gave).", "data.ASSUMED_UNLABELLED_SHARE"),
    ("A09", "BOM", "Polymer levels assumed at 1 300 EUR/t (PET) and 2 700 EUR/t (engineering plastics), constant over "
     "time (also for origin weights).", "data.ASSUMED_PRICE_EUR_T"),
    ("A10", "BOM", "Submission BOM indices: Sep-2026 base (Schneider's common base). Cost shares = kg x latest "
     "published EUR/t at or before Sep-2026 (stainless 304: Aug-2026).", "data.bom_cost_table, data.bom_index"),
    ("A11", "BOM", "Backtest option --bom-weights origin: cost shares from the prices known at each origin "
     "(kg x latest EUR/t at or before the origin) instead of Sep-2026 shares, to measure the look-ahead in the fixed "
     "weights. Not used for the submission.", "backtest --bom-weights"),
    ("A12", "BOM", "Fixed quantities (kg per unit) over the whole period: no design or sourcing change.", "data.load_bom"),
    ("A13", "data", "ECH and epoxy May-2026 are a decimal-shift typo: set to NaN, then log-linear interpolation.",
     "preprocess.CORRECTIONS"),
    ("A14", "BOM", "Submitted BOM indices are scaled so that the origin month = 100 (format 'index (origin = 100)'): "
     "one constant per index, from its value in the origin month as seen at the origin (market version: with the "
     "nowcast stainless value). The PPI-proxy index is already 100 in Sep-2026 (factor 1). A constant changes no "
     "log change and no cost share. Factors in bom_rebase.csv; with --no-nowcast the market version keeps factor 1.",
     "backtest.Bom.rebase"),
    ("A15", "model", "Backtest warm-up from Jan-2018: origins 2018-19 only build the residual history of the interval "
     "layer; every score uses origins from Jan-2020. Point models need 12 months of history (statsforecast) or 24 "
     "training rows (Ridge); GBM needs 50 rows.", "backtest.WARMUP_START, scoring.SCORE_START"),
    ("A16", "model", "statsforecast models (AutoETS with damped trend allowed, AutoARIMA, Theta, OptimizedTheta) on "
     "the log level with season_length = 1 (no seasonal terms on 40-120 points). Combinations: mean(naive, "
     "AutoETS) and median(naive, AutoETS, Theta, AutoARIMA) of the P50 log returns.", "models/sf.py, models/combo.py"),
    ("A17", "model", "lme_fwd: P50 = LME 3-month price at the origin (Westmetall monthly average; EUR version = "
     "3-month USD / ECB spot rate, not a forward FX rate), flat for h = 6 and 9; the IMF / Westmetall cash basis "
     "(< 1%) is not corrected.", "models/lme.py"),
    ("A18", "interval", "Conformal bands (conf, aci, ewma) use only residuals published at the origin (outcome month + "
     "data lag <= origin); at least 8 residuals, else the emp band (flag fallback). Finite-sample conformal ranks; "
     "lower and upper bounds estimated separately. P10 <= P50 <= P90 enforced by clipping (Ridge / GBM native "
     "quantiles are sorted, as in the first backtest).", "models/intervals.py"),
    ("A19", "interval", "ACI: target 80%, delayed feedback (an outcome updates alpha when it is published). One gamma "
     "per point model (horizons and targets pooled) from {0, 0.005, 0.01, 0.02, 0.05}, chosen on warm-up bands only "
     "(outcomes published by Jan-2020, ACI active); 0.005 if fewer than 200 such bands. h = 9 has almost no active "
     "warm-up bands, so the choice rests on h = 3 and 6.", "models/intervals.py"),
    ("A20", "interval", "EWMA volatility: lambda = 0.9 on squared monthly log changes known at the origin; residuals "
     "standardised by sigma x sqrt(h_eff).", "models/intervals.EWMA_LAMBDA"),
    ("A21", "features", "Family models use 1-4 leading indicators per target, each with an economic reason (steel PPI "
     "<- CRU CR coil this and last month + iron ore; CRU coil <- iron ore, coal, TTF (+ zinc for HDG); stainless <- "
     "nickel this and last month; plastics PPI <- Brent 3-month change; aluminium PPI <- LME aluminium, TTF, power; "
     "LME copper / aluminium <- 3-month / cash basis and stock change, + TTF and power for aluminium), plus the "
     "target's own 1- and 3-month changes. Every feature at month m uses data up to m only; the leakage test rebuilds "
     "every design row at every origin 2018-01..2026-09 from the panel as seen at the origin and requires it to equal "
     "the full-panel row. Separate USD price x EUR/USD forecasts (optional in the plan) are not built.",
     "features.py (feature_availability.csv)"),
    ("A22", "model", "ardl / ridge_fam: one direct regression per target and horizon on the months whose h_eff-month "
     "outcome is known at the origin (at least 24 rows), with a constant. Ridge alpha from {0.1, 1, 10, 100, 1000} by "
     "expanding time-series CV on the last half of the training rows (4 folds, h_eff rows dropped before each "
     "validation fold; ties -> larger alpha).", "models/multi.py"),
    ("A23", "model", "ecm (S09, S10 <- CRU CR coil): Engle-Granger two-step, re-estimated at every origin on the months "
     "known then: log(PPI) = c + b log(CRU), then the h_eff-month PPI change on the lagged error-correction term, the "
     "CRU change this and last month and the own change. The ECM term is used whether or not a cointegration test "
     "rejects (the Johansen results of the chain are reported with vecm).", "models/multi.py"),
    ("A24", "model", "vecm (iron ore EUR -> CRU CR coil -> S09, log levels, common window, at least 48 months): "
     "Johansen trace test at 5% (constant, 1 lagged difference) at every origin; rank >= 1 -> VECM with the constant "
     "in the cointegration relation, else VAR(1) on first differences. Forecasts CRU CR coil and S09.",
     "models/multi.py (vecm_johansen.csv)"),
    ("A25", "model", "gbm_family: one quantile GBM per horizon over the 11 family targets (own changes, deviation from "
     "the 12-month mean, 12-month volatility, all leading indicators with NaN where a target does not use them, "
     "family); every month with a known outcome, so CRU from 2012 and stainless from 2008; max depth 2, min 40 rows "
     "per leaf, L2 = 1, 100 trees, refit every 3 months. P50 = w x GBM P50 (band shifted with it); w from {0, 0.25, "
     "0.5, 0.75, 1} minimising the MAE of the 2018-19 warm-up forecasts known at Jan-2020, then fixed.",
     "models/multi.py (gbm_shrink.csv)"),
    ("A26", "model", "Epoxy: the source file is not an exact identity of benzene, acetone, ECH and 'other costs' "
     "(best fit R2 0.98, max residual about 290 EUR/t). epoxy_struct: h_eff-month log change of epoxy = sum of "
     "beta_i x log change of the four inputs (no constant, beta on the months known at the origin); each input is "
     "forecast with mean(naive, AutoETS), the combo_mean rule.", "features.epoxy_identity, models/multi.py"),
    ("A27", "evaluation", "Family models are compared with the best univariate model per target and horizon (lowest "
     "MAE over the scored origins among naive, drift, ETS, AutoETS, AutoARIMA, Theta, OptimizedTheta and the two "
     "combinations), chosen with hindsight, which favours the benchmark; Diebold-Mariano on absolute errors, "
     "Newey-West lag h_eff - 1, p < 0.05.", "models/multi.compare (multivariate_vs_univariate.csv)"),
    ("A28", "model", "chronos2 (amazon/chronos-2, revision 29ec3766, chronos-forecasting 2.3.2, CPU), zero-shot: "
     "context = the target's whole history up to its anchor as seen at the origin (as_of + nowcast, as every model), "
     "fed as y / y[anchor] rounded to 10 decimals (scale-free, so the BOM indices give the same input whatever their "
     "origin = 100 factor); max(h_eff) steps (9 with the nowcast); P10 / P50 / P90 at step h_eff, log of the "
     "ratio = the log return. Series forecast independently (no cross-learning); no fine-tuning.", "models/fm.py"),
    ("A29", "model", "chronos2_cov: as chronos2 + past covariates only (never known-future values): 1-3 source series "
     "of the target's step-3 leading indicators, as published, aligned on the target's context months (values up to "
     "the anchor). TTF dropped for HDG coil and LME aluminium to keep 3. Only the 11 family targets.",
     "models/fm.COVARIATES"),
    ("A30", "evaluation", "Pretraining leakage: Chronos-2's training corpus (parts of the Chronos and GIFT-Eval "
     "pretraining data, incl. M4 monthly, plus synthetic series) has no published cutoff and may contain some public "
     "series of this panel. Weights published on the Hugging Face hub on 2025-10-30 (paper 2025-10-17): origins from "
     "2025-11 are reported separately (post_release); their outcomes cannot be in the training data.",
     "models/fm.POST_RELEASE"),
    ("A31", "pipeline", "Chronos forecasts are cached (outputs/backtest/fm_cache/) per origin, target and sha256 of "
     "the exact input: a cached forecast is reused only for an identical input. Without the optional fm dependencies "
     "the backtest reads the cache and skips what is missing.", "models/fm.Cache"),
    ("A32", "evaluation", "Multiple testing: every Diebold-Mariano conclusion (vs naive, vs the best univariate model) "
     "uses the Benjamini-Hochberg q value, q < 0.05. One family = all point models x submitted targets x horizons of "
     "one table and period; tests without a p value (P50 = naive) are left out of the family. The raw p < 0.05 "
     "counts are kept for comparison only.", "scoring.bh_qvalues, scoring.per_series"),
    ("A33", "evaluation", "Every score is reported for two periods by origin, 2020-01..2023-06 (stress: Covid, the "
     "2021-22 rally, the energy crisis) and 2023-07..last origin with a known outcome, plus 2022 alone and all.",
     "scoring.PERIODS"),
    ("A34", "selection", "Candidate point models fixed in advance with a reason each: naive, lme_fwd (copper and "
     "aluminium only), combo_median, ets_local (S07-S13 and the PPI-proxy BOM indices only), chronos2. All other "
     "models stay in the backtest report only.", "selection_config.CANDIDATES"),
    ("A35", "selection", "Rolling selection 'selected': at origin t, errors published at or before t in the last 24 "
     "months (at least 12 per target); per family the geometric mean over its targets of MAE / naive's MAE; each "
     "target takes the best candidate that applies to it, naive if the gain is below 3%. Its band is the band of the "
     "model it selected.", "selection.rolling_selection"),
    ("A36", "interval", "Asymmetric band '<method>_up': P90 - P50 (log) times k >= 1 (grid 1.00-3.00, step 0.05), "
     "one k per point model, method, family and h: the smallest k with at most 10% of outcomes above P90 on the "
     "2020-01..2023-06 origins whose outcome is published by 2023-07. The second period tests k out of sample.",
     "selection.tune_asym"),
    ("A37", "selection", "Submission map, per family x h: the point strategy (selected or combo_median) replaces "
     "naive only if its geometric-mean MAE / naive is below 1 in both periods (lower worse-period ratio if both "
     "are); 'selected' = the rolling rule's choice at Sep-2026. Band: 80% coverage within [0.70, 0.90] in both "
     "periods, then the lowest Winkler; if no band qualifies, the highest stress-period coverage (capped at 0.90), "
     "then Winkler. No manual choice.", "selection.point_strategy, selection.interval_choice"),
]


def frame():
    return pd.DataFrame(ASSUMPTIONS, columns=["id", "area", "assumption", "where"])


def write(path):
    frame().to_csv(path, index=False)
