# X-HEC × Schneider Electric Data Challenge 2026: raw-material forecasting

Forecast the raw-material prices behind the RM6 and RM AirSeT RMUs at **+3 / +6 / +9 months**, turn them into a
BOM cost index, then into ETO quote rules and an SF6-free migration strategy.

> **Data is not in this repo** (challenge rule). `Data/`, all `csv/xlsx/zip/pdf/pptx/png` files and `outputs/`
> are git-ignored, and two pre-commit hooks refuse them (see §3). Schneider-internal files (CRU / MEPS price levels,
> the BOM, supplier financials) and every output derived from them must never be committed.

## Setup

### 1. Data

Put the challenge files, unchanged, in `Data/Data/` (git-ignored; nothing in it is ever committed):

| File | Content | Status |
|---|---|---|
| `data.zip` (or its extracted `data/` folder) | Public prices and energy (IMF, Eurostat, ECB, BLS, Ember, ...) | public, challenge data |
| `LME_Copper_Aluminium_EU_Monthly_2016_2026.xlsx` | Westmetall LME cash / 3-month / stocks | challenge data |
| `Epoxy for student.xlsx` | Epoxy cost build-up (benzene, acetone, ECH, epoxy) | challenge data |
| `RMU_IQI_EU_Price_Benchmark_2017-2026_Synthetic.xlsx` | Synthetic RMU selling prices | challenge data |
| `Copia de Europe Steel INDEXES.xlsx` | CRU / MEPS EUR/t steel prices | **internal, licensed** |
| `RM_Prices_XHEC_Challenge.xlsx` (standalone) | BOM_Materials sheet (replaces the older BOM in `data.zip`) | **internal** |
| `Supplier_balancesheet.zip` | Supplier financials (PitchBook) | **internal**, not used yet |

### 2. Python environment (uv)

The environment is managed with [uv](https://docs.astral.sh/uv/). `pyproject.toml` lists the dependencies,
`uv.lock` pins every package version and `.python-version` pins Python 3.12, so every machine gets the same setup.

```bash
# install uv once (macOS / Linux; on Windows see the uv docs)
curl -LsSf https://astral.sh/uv/install.sh | sh     # or: brew install uv

uv sync                         # creates .venv/ with the exact locked versions (downloads Python 3.12 if needed)
uv run python main.py                   # everything, origin Sep-2026: ~9 min (backtest ~7 min) with cached sensitivity
uv run python main.py --skip-sensitivity  # skip the sensitivity variants; --skip-backtest also reuses the baseline
uv run python main.py --origin 2026-06  # held-back check: forecasting path only, data as of Jun-2026
uv run python src/preprocess.py         # any single step
uv run python src/backtest.py --help    # options (--no-nowcast, --drop-provisional, --bom-weights origin)
```

Data-quality checks: open `notebooks/01_data_quality.ipynb` in VS Code with the `.venv` kernel (`uv sync` installs it).

- No need to activate `.venv/`: `uv run` uses it automatically (or `source .venv/bin/activate`).
- Add or remove a dependency with `uv add <pkg>` / `uv remove <pkg>`, then commit both `pyproject.toml` and `uv.lock`.
- After pulling changes to `uv.lock`, run `uv sync` again.
- Without uv: `pip install -r requirements.txt` (same versions; regenerate with
  `uv export --no-hashes --no-dev -o requirements.txt` after changing dependencies).

### 3. Pre-commit hooks (once per clone)

The repo must never contain challenge data, and notebook outputs are data too. [pre-commit](https://pre-commit.com/)
checks every commit (config in `.pre-commit-config.yaml`; `uv sync` already installed the tool):

```bash
uv run pre-commit install          # once per clone: run the hooks on every `git commit`
uv run pre-commit run --all-files  # optional: run them on the whole repo now
```

| Hook | What it does |
|---|---|
| `nbstripout` | Removes outputs and execution counts from committed notebooks |
| `no-data-files` | Blocks `csv / xlsx / zip / parquet / pdf / pptx / png / jpg` files, even when forced with `git add -f` |
| `no-internal-data` | Blocks any path naming internal data (Steel INDEXES, BOM workbook / tables, supplier / PitchBook files, `outputs/`, `Data/`, `__MACOSX`, `.DS_Store`) |
| `pre-commit-hooks` | Large files (> 1 MB), merge-conflict markers, YAML / TOML syntax, end-of-file and trailing whitespace |

If a hook changes a file (e.g. strips a notebook), the commit stops: check the change, `git add` it and commit again.
nbstripout also clears the outputs in your local copy, so run the notebook again afterwards.

**CI** (`.github/workflows/ci.yml`, on every push to `main` and every pull request) runs the same hooks, checks that
`uv.lock` matches `pyproject.toml` and that the code imports. It cannot run the pipeline: the data is not in the repo.

### 4. Outputs

Outputs: `outputs/assumptions.csv` (every modelling assumption, see below), `outputs/preprocessed/` (clean panel,
series catalog with relative lags and provisional months, transformation log), `outputs/eda/` (charts, tables) and
`outputs/backtest/` (predictions of every `point+interval` label, forecasts at the origin, scores per series /
summary / family × period, `best_by_family.csv`, `quote_basis.csv`, `aci_gamma.csv`, `runtime.csv`, training sizes,
`sensitivity.csv`; one sub-folder per sensitivity variant or held-back origin).
`DECK=1` renders charts in slide style to `outputs/slides/`. Colours: **RM6 green `#16884A`, RM AirSeT blue `#3478B8`**.

| File | Role |
|---|---|
| `src/targets.py` | **Target registry**: the one list of what is forecast (id, name, family, region, unit, material / bom_index, in_submission, history_start). Every forecasting script reads it |
| `src/assumptions.py` | Every modelling assumption (A01–A27) → `outputs/assumptions.csv` |
| `src/data.py` | Reads every source as published; BOM (from the standalone workbook); BOM cost index in two versions (PPI proxy, market prices), with weights and references at any date; `panel()` keeps each series' full history |
| `src/preprocess.py` | One monthly panel: month-start dates, cut at Sep-2026, typo corrections, gap fill, USD → EUR. Writes `panel_monthly.csv`, `series_catalog.csv`, `transform_log.csv`. `as_of()` = the panel as seen at an origin (relative ragged edge, optional provisional drop); `nowcast()` = stainless fill of the missing month (on by default) |
| `notebooks/01_data_quality.ipynb` | Coverage, missing values, wrong values, derived-column checks, IMF vs Westmetall, steel PPI vs CRU / MEPS |
| `src/eda.py` | Price families, key material prices, BOM cost index and cost per unit, cost structure, volatility by horizon, price benchmark (figures and the tables behind them) |
| `src/eda_risk.py` | Risk EDA: unit roots, fat tails, volatility clustering, band coverage, cost-index rises (3/6/9 m) and their drivers, regimes, correlations (fig11–fig15, `risk_*.csv`) |
| `notebooks/02_eda_risk.ipynb` | Conclusions of the EDA: risk (§1–4), key prices, cost per unit, risk by horizon, supply-chain links, selling-price benchmark (§5–9) |
| `src/backtest.py` | Rolling backtest, warm-up Jan-2018, scored Jan-2020 → origin: loops over the model registry (point pass), then the interval layer; data as of each origin (`as_of`); BOM indices rescaled to origin = 100; options `--origin`, `--no-nowcast`, `--drop-provisional`, `--bom-weights`, `--models` |
| `src/models/` | One interface (`base.py`: origin context in, log-return quantiles out). `classic.py` naive / drift / damped ETS / Ridge / global GBM (the first backtest, unchanged); `sf.py` AutoETS, AutoARIMA, Theta, OptimizedTheta (statsforecast); `combo.py` mean(naive, AutoETS), median(naive, AutoETS, Theta, AutoARIMA); `lme.py` LME 3-month price as P50 + quote-basis table; `intervals.py` emp / split conformal / ACI / EWMA-scaled bands around any point model; `multi.py` family models with leading indicators: ARDL, Ridge (time-series CV alpha), ECM (steel PPI ← CRU), VECM chain (iron ore → CRU CR → S09, Johansen), slim global quantile GBM shrunk to naive, epoxy structural model, and the comparison with the best univariate model |
| `src/features.py` | Leading indicators of the family models (1-4 per target, each with an economic reason) → `outputs/features/feature_availability.csv` (source, transformation, lag vs the origin, first month, meaning); **leakage test** (every design row at every origin rebuilt from the panel as seen at the origin = full-panel row); epoxy identity check (`epoxy_identity.csv`) |
| `src/scoring.py` | Scores: rel. MAE, MASE, pinball, coverage (above / below), width, Winkler, Diebold-Mariano vs naive; by family × period (2020-21, 2022, 2023-26); best point model and band per family × horizon |
| `src/sensitivity.py` | Backtest variants against the baseline (no nowcast, provisional dropped, origin BOM weights) → `sensitivity.csv`; a variant is re-run only when its inputs change (cache key) |
| `src/plot_backtest.py` | Error relative to naive per series / model / horizon |
| `src/diagrams.py` | Production chain (predictive links, `link_table`) and methodology diagrams |

## Findings so far

- **Exchange-traded prices** (LME metals, iron ore): no model clearly beats naive, so P50 = today's price and the
  work goes into the range.
- **EU steel and aluminium PPIs, AirSeT cost index**: ETS beats naive by about 15–27% at 3 months. EU plastics and
  copper PPIs are about even with naive; US PPIs are worse.
- **ML models lose to naive at 6–9 months** (Ridge is the worst at 6 months, the global GBM at 9 months): about
  120 monthly points per series is too little.
- **Upstream prices help predict the next 3 months** for aluminium, steel, stainless and plastics (r 0.4–0.6).
  Copper LME and the EU copper index move in the same month.
- **Ranges are too narrow**: about 68% coverage against an 80% target. Why (risk EDA): misses are mostly upwards
  (18% above vs 7% below the band) and the 2020-26 volatility is about 2× the pre-2020 level; fat tails only matter
  beyond P95.
- **The PPI proxy halves the steel risk**: with CRU market prices, the 3-month cost rise of one unit has P90 +12% /
  P95 +17% (RM6) and P90 +14% / P95 +24% (AirSeT); the PPI-proxy index gives P95 +12% / +15%.
- **Correlations double in stress** (mean pairwise 0.22 calm vs 0.45 in 2020-22): the cost P10 / P90 needs joint
  scenarios; summing material quantiles is off by 15–36%.

### Backtest on data as of each origin (22 submitted targets + 3 epoxy inputs)

- **Refactor checked**: on the 16 original series, naive / drift / ETS / Ridge predictions are identical to the
  previous backtest (same data at every origin); only the global GBM moves, because its pool now has 25 series.
- **Baseline = stainless nowcast on** (A05): every submitted target is forecast from the origin month (h_eff = h),
  and the BOM indices are scaled to origin = 100 (A14; market version ×1.0004 / ×1.0008, PPI proxy ×1).
- **New targets are hard**: on the CRU CR / HDG coil prices, ETS is 8–10% *worse* than naive at 3 months and
  34–43% worse at 6–9 months (momentum then sharp reversals), and the naive 80% band covers only 58–63% even with
  history from 2012. Stainless 304 is the one new market price where ETS beats naive (rel. MAE 0.91–0.97).
  Epoxy: ETS 0.92 at 3 months. Market-price BOM indices: no model clearly beats naive; ETS ≈ naive for RM6
  (1.00–1.06), 4–18% worse for AirSeT.
- **Sensitivities** (`outputs/backtest/sensitivity.csv`; variants are cached and re-run only when their inputs change):
  - *No nowcast* (stainless and the market-price BOM from t - 1, h + 1): MAPE rises by 0.1–2.2 points
    (e.g. RM6 market BOM, 3 months: 7.8% → 9.1%), which is why the nowcast is the baseline.
  - *Dropping the provisional Eurostat months* moves the anchor of S07–S11 and the PPI-proxy BOM back 2 months
    (h_eff 3 → 5, 6 → 8, 9 → 11). Their 2–3 point higher 3-month MAPE (S09 ETS: 3.7% → 6.5%) is mostly the longer
    horizon, **not** revision risk. Revision risk cannot be measured here: the files hold only the latest Eurostat
    vintage (A06). The baseline keeps the provisional months.
  - *Origin BOM weights*: changes MAPE by < 0.3 point and 80% coverage by at most 3 points; the Sep-2026 weights add
    no material look-ahead.
- **Held-back origin Jun-2026** (3 months → Sep-2026, 22 series with a Sep-2026 outcome): every model is within ±1
  point of naive's 6.3% error; 80% coverage 77–91% on this single, calm quarter (too few points to rank models).

### Univariate baselines and calibrated bands (22 submitted targets, origins Jan-2020 → Sep-2026)

- **Refactor checked**: the five first-backtest models, moved to `src/models/`, give exactly the same P10 / P50 / P90
  on every row (`naive` = `naive+emp`, ..., `gbm_global` = `gbm_global+native`).
- **P50: nothing significantly beats naive.** Diebold-Mariano (absolute errors, Newey-West lag h - 1): at most one
  target per model and horizon is significantly better than naive; Theta / OptimizedTheta are significantly *worse*
  on 7–9 targets, the global GBM on 15 at 9 months.
  - Best mean MASE per family × horizon (`best_by_family.csv`): damped ETS at 3 months, mean(naive, AutoETS) at
    6 months, naive at 9 months in most families. Gains over naive are 0–11% in MASE.
  - AutoETS picks a random walk on most exchange-traded prices (rel. MAE exactly 1.00). The mean(naive, AutoETS)
    combination is the steadiest: rel. MAE 0.88–0.98 on the EU steel / aluminium / plastics PPIs and the PPI-proxy
    BOM (aluminium PPI 1.02 at 9 months).
  - **LME 3-month price as P50** (D01, D02, S01, S02): rel. MAE 0.97–1.01, only S02 at 6 months significant
    (p = 0.03). **Quote basis**: the 3-month price beats cash as the forecast of future cash by 0–0.4 MAE points
    (copper in EUR at 6 months is the one case where cash is slightly better) and has less bias (e.g. aluminium,
    3 months: −0.9% vs −1.7%). Both under-forecast in 2020-26 (copper −2.6% to −8.5%), because prices rose.
- **Bands: conformal fixes the average, not the regimes.** Mean 80% coverage over all point models
  (3 / 6 / 9 months): emp 67 / 65 / 62%, split conformal 74 / 73 / 67%, ACI 79 / 77 / 72%, EWMA-scaled
  82 / 78 / 76%.
  - Misses stay lopsided (above P90 12–17%, below P10 8–12% with ACI).
  - The Winkler score still prefers the narrow emp band at 6–9 months, because the wider conformal bands pay for
    their width.
  - By period: on steel, ACI around damped ETS covers 50–61% in 2020-21 and 96–100% in 2023-26. The 80% average
    hides a band that is too narrow in the rally and too wide afterwards.
- ACI gamma per point model from the 2018-19 warm-up only (`aci_gamma.csv`). Ridge, GBM and lme_fwd have too few
  warm-up bands and use the default 0.005.
- **Held-back Jun-2026** (3 months, 22 targets): every point model is within 1 point of naive's 6.3% error
  (AutoARIMA 5.4%); coverage emp 86%, conf 91%, ACI 91%, EWMA 95% on this one calm quarter.
- Runtime of a full backtest, 105 origins (`runtime.csv`): AutoARIMA 244 s, GBM 114 s, damped ETS 13 s,
  Ridge 12 s, AutoETS 9 s, Theta about 1 s; the whole interval layer < 1 s.

### Family models with leading indicators (11 targets + epoxy, origins Jan-2020 → Sep-2026)

- **Leakage test passes** (`features.py`): 1,155 design rows (11 targets × 105 origins) are identical whether built
  from the panel as seen at the origin or from the full panel. A negative control (a feature shifted one month into
  the future) is caught at the first origin. Sources, lags and economic meaning in `feature_availability.csv`.
- **Epoxy is not an exact identity** of benzene, acetone, ECH and "other costs" (`epoxy_identity.csv`): best fit
  R² 0.98, max residual about 290 EUR/t. Estimated elasticities at Sep-2026 (9 months): other costs 0.58, ECH 0.24,
  acetone 0.10, benzene 0.09.
- **No family model is significantly better than the best univariate model, on any target or horizon** (Diebold-Mariano,
  `multivariate_vs_univariate.csv`), and none is significantly better than naive. Closest: VECM on S09 at 3 months
  (0.97 × the best univariate MAE), the epoxy structural model at 6 months (0.96), ARDL / Ridge on S09 at 3 months (0.98).
  - ARDL is significantly *worse* than the best univariate model on 2 / 4 / 7 targets at 3 / 6 / 9 months, Ridge on
    1 / 1 / 7 (mostly LME metals and CRU coil at 9 months). Ridge picks the largest penalty (alpha = 1000) in 38% of
    its fits: the indicators carry little signal once the target's own changes are in.
  - On the steel PPI at 3 months ECM (S09 0.84, S10 0.86 × naive's MAE) and VECM (S09 0.77) beat naive, but not
    damped ETS (0.97-1.05 × its MAE, DM p > 0.7), which already captures the PPI's momentum. ECM loses to naive at
    9 months (1.25-1.29); VECM adds nothing for CRU CR coil (1.00 / 1.19 / 1.26 × naive). Johansen finds at least one cointegrating relation in
    iron ore → CRU CR → S09 at every origin from 2020 (rank 1 at 70 of 72 origins).
  - The slim global GBM gets shrink weight w = 0 from the warm-up (any weight on the GBM raised the warm-up MAE from
    0.049 to up to 0.056): its P50 is naive.
  - The in-sample correlations of the EDA (r 0.4-0.6) do not survive out of sample: about 40 independent 3-month
    windows per target are too few to estimate even 3-5 coefficients reliably.
- Bands: around the family models, ewma / aci give the same coverage pattern as around the univariate models
  (ARDL + ewma 82 / 82 / 74%).
- Runtime: the six family models add about 3 minutes to the backtest (Ridge with CV 99 s, GBM 72 s).
- **Consequence for the submission**: keep univariate P50s (naive / damped ETS / combo_mean per family). The family
  models stay in the backtest as documented evidence; VECM / ECM on the steel PPI and epoxy_struct are the only
  candidates worth an ensemble slot in step 5.

## Next steps

1. Forecast CSV `<team>_forecast_202609.csv`: per-series model choice by backtest, calibrated intervals (wider,
   asymmetric, from a window that includes 2020-22).
2. Market prices: LME 3-month forward and the Oct-2026 analyst view as P50 adjustments.
3. ~~Indices: upstream drivers~~ done (family models): none significantly beats the best univariate model; VECM / ECM (steel PPI) and epoxy_struct as optional ensemble members.
4. Pretrained model (Chronos-Bolt) as a benchmark.
5. ~~Publication lags in the backtest~~ done: `preprocess.as_of()` cuts every origin by the relative lag (`rel_lag_m`).
6. Business layer: joint scenarios (resample whole months) → BOM cost (market-price index) → margin at risk → quote rule.

## Assumptions and limits

Every assumption is in `src/assumptions.py`, written to `outputs/assumptions.csv` by `preprocess.py`; each data
transformation is a row of `outputs/preprocessed/transform_log.csv` (steps calendar, cutoff, drop, correct, fill,
derive, fx, ragged_edge, provisional).

| id | area | assumption | where |
|---|---|---|---|
| A01 | data | Forecast origin = Sep-2026 = last month of data used. Values after Sep-2026 (Oct-2026 FX, power, CRU, MEPS, epoxy, Westmetall) are dropped before any other step and never used, also not as a nowcast input. | `preprocess.CUTOFF` |
| A02 | data | Ragged edge, relative to the origin: at every backtest origin each series is cut by the number of months it lacks at the real origin (stainless 304 and its parts: 1, E20 / E21 / ED06: 2, all others: 0). | `preprocess.relative_lags, preprocess.as_of` |
| A03 | data | A target whose last value is before the origin is forecast from its last value: the effective horizon is h + lag. With the default stainless nowcast (A05) every submitted target is known at the origin (h_eff = h); the sensitivity variant --no-nowcast forecasts stainless 304 and the market-price BOM indices from t - 1 (h + 1). | `backtest.run` |
| A04 | data | No forward fill in any data used for forecasting. data.TRAIL_FILL (carry stainless Aug-2026 to Sep-2026) is kept only for the descriptive EDA charts and tables. | `data.panel(trail_fill=...)` |
| A05 | data | Stainless 304 nowcast, ON by default (baseline and submission): the missing month(s) at an origin are filled with an OLS of its monthly log change on nickel in EUR (S03_EUR) and the EU steel PPI (S09), fitted on the months known at that origin; uses nothing after the origin. Chosen because the backtest showed lower errors for stainless and the market-price BOM with no leakage. Each fill is logged (nowcast_log.csv); --no-nowcast is the sensitivity variant. | `preprocess.nowcast, backtest --no-nowcast` |
| A06 | data | Eurostat PPIs (S07-S11, E13-E16): the last 2 months known at an origin are provisional (Eurostat revises them); they are flagged and kept in the baseline. Revision risk is NOT measured: the files hold only the latest vintage, so the backtest sees revised values (optimistic by an unknown amount). The --drop-provisional variant removes those months, which moves the anchor back 2 months (h_eff = h + 2): its higher error reflects the longer horizon, not revisions. | `preprocess.PROVISIONAL, backtest --drop-provisional` |
| A07 | model | Univariate models use each target's full history (CRU from 2012, stainless from 2008, epoxy from 2018); models with drivers use the months where every input is known. | `targets.py, backtest.py` |
| A08 | BOM | RM6 largest unlabelled slice = carbon steel at 45% of the implied mass (the share the older BOM label gave). | `data.ASSUMED_UNLABELLED_SHARE` |
| A09 | BOM | Polymer levels assumed at 1 300 EUR/t (PET) and 2 700 EUR/t (engineering plastics), constant over time (also for origin weights). | `data.ASSUMED_PRICE_EUR_T` |
| A10 | BOM | Submission BOM indices: Sep-2026 base (Schneider's common base). Cost shares = kg x latest published EUR/t at or before Sep-2026 (stainless 304: Aug-2026). | `data.bom_cost_table, data.bom_index` |
| A11 | BOM | Backtest option --bom-weights origin: cost shares from the prices known at each origin (kg x latest EUR/t at or before the origin) instead of Sep-2026 shares, to measure the look-ahead in the fixed weights. Not used for the submission. | `backtest --bom-weights` |
| A12 | BOM | Fixed quantities (kg per unit) over the whole period: no design or sourcing change. | `data.load_bom` |
| A13 | data | ECH and epoxy May-2026 are a decimal-shift typo: set to NaN, then log-linear interpolation. | `preprocess.CORRECTIONS` |
| A14 | BOM | Submitted BOM indices are scaled so that the origin month = 100 (format 'index (origin = 100)'): one constant per index, from its value in the origin month as seen at the origin (market version: with the nowcast stainless value). The PPI-proxy index is already 100 in Sep-2026 (factor 1). A constant changes no log change and no cost share. Factors in bom_rebase.csv; with --no-nowcast the market version keeps factor 1. | `backtest.Bom.rebase` |
| A15 | model | Backtest warm-up from Jan-2018: origins 2018-19 only build the residual history of the interval layer; every score uses origins from Jan-2020. Point models need 12 months of history (statsforecast) or 24 training rows (Ridge); GBM needs 50 rows. | `backtest.WARMUP_START, scoring.SCORE_START` |
| A16 | model | statsforecast models (AutoETS with damped trend allowed, AutoARIMA, Theta, OptimizedTheta) on the log level with season_length = 1 (no seasonal terms on 40-120 points). Combinations: mean(naive, AutoETS) and median(naive, AutoETS, Theta, AutoARIMA) of the P50 log returns. | `models/sf.py, models/combo.py` |
| A17 | model | lme_fwd: P50 = LME 3-month price at the origin (Westmetall monthly average; EUR version = 3-month USD / ECB spot rate, not a forward FX rate), flat for h = 6 and 9; the IMF / Westmetall cash basis (< 1%) is not corrected. | `models/lme.py` |
| A18 | interval | Conformal bands (conf, aci, ewma) use only residuals published at the origin (outcome month + data lag <= origin); at least 8 residuals, else the emp band (flag fallback). Finite-sample conformal ranks; lower and upper bounds estimated separately. P10 <= P50 <= P90 enforced by clipping (Ridge / GBM native quantiles are sorted, as in the first backtest). | `models/intervals.py` |
| A19 | interval | ACI: target 80%, delayed feedback (an outcome updates alpha when it is published). One gamma per point model (horizons and targets pooled) from {0, 0.005, 0.01, 0.02, 0.05}, chosen on warm-up bands only (outcomes published by Jan-2020, ACI active); 0.005 if fewer than 200 such bands. h = 9 has almost no active warm-up bands, so the choice rests on h = 3 and 6. | `models/intervals.py` |
| A20 | interval | EWMA volatility: lambda = 0.9 on squared monthly log changes known at the origin; residuals standardised by sigma x sqrt(h_eff). | `models/intervals.EWMA_LAMBDA` |
| A21 | features | Family models use 1-4 leading indicators per target, each with an economic reason (steel PPI <- CRU CR coil this and last month + iron ore; CRU coil <- iron ore, coal, TTF (+ zinc for HDG); stainless <- nickel this and last month; plastics PPI <- Brent 3-month change; aluminium PPI <- LME aluminium, TTF, power; LME copper / aluminium <- 3-month / cash basis and stock change, + TTF and power for aluminium), plus the target's own 1- and 3-month changes. Every feature at month m uses data up to m only; the leakage test rebuilds every design row at every origin 2018-01..2026-09 from the panel as seen at the origin and requires it to equal the full-panel row. Separate USD price x EUR/USD forecasts (optional in the plan) are not built. | `features.py (feature_availability.csv)` |
| A22 | model | ardl / ridge_fam: one direct regression per target and horizon on the months whose h_eff-month outcome is known at the origin (at least 24 rows), with a constant. Ridge alpha from {0.1, 1, 10, 100, 1000} by expanding time-series CV on the last half of the training rows (4 folds, h_eff rows dropped before each validation fold; ties -> larger alpha). | `models/multi.py` |
| A23 | model | ecm (S09, S10 <- CRU CR coil): Engle-Granger two-step, re-estimated at every origin on the months known then: log(PPI) = c + b log(CRU), then the h_eff-month PPI change on the lagged error-correction term, the CRU change this and last month and the own change. The ECM term is used whether or not a cointegration test rejects (the Johansen results of the chain are reported with vecm). | `models/multi.py` |
| A24 | model | vecm (iron ore EUR -> CRU CR coil -> S09, log levels, common window, at least 48 months): Johansen trace test at 5% (constant, 1 lagged difference) at every origin; rank >= 1 -> VECM with the constant in the cointegration relation, else VAR(1) on first differences. Forecasts CRU CR coil and S09. | `models/multi.py (vecm_johansen.csv)` |
| A25 | model | gbm_family: one quantile GBM per horizon over the 11 family targets (own changes, deviation from the 12-month mean, 12-month volatility, all leading indicators with NaN where a target does not use them, family); every month with a known outcome, so CRU from 2012 and stainless from 2008; max depth 2, min 40 rows per leaf, L2 = 1, 100 trees, refit every 3 months. P50 = w x GBM P50 (band shifted with it); w from {0, 0.25, 0.5, 0.75, 1} minimising the MAE of the 2018-19 warm-up forecasts known at Jan-2020, then fixed. | `models/multi.py (gbm_shrink.csv)` |
| A26 | model | Epoxy: the source file is not an exact identity of benzene, acetone, ECH and 'other costs' (best fit R2 0.98, max residual about 290 EUR/t). epoxy_struct: h_eff-month log change of epoxy = sum of beta_i x log change of the four inputs (no constant, beta on the months known at the origin); each input is forecast with mean(naive, AutoETS), the combo_mean rule. | `features.epoxy_identity, models/multi.py` |
| A27 | evaluation | Family models are compared with the best univariate model per target and horizon (lowest MAE over the scored origins among naive, drift, ETS, AutoETS, AutoARIMA, Theta, OptimizedTheta and the two combinations), chosen with hindsight, which favours the benchmark; Diebold-Mariano on absolute errors, Newey-West lag h_eff - 1, p < 0.05. | `models/multi.compare (multivariate_vs_univariate.csv)` |

Limits:
- BOM from the standalone `RM_Prices_XHEC_Challenge.xlsx`; the older BOM in `data.zip` (`bom_materials.csv`) is not used.
- No data vintages: the backtest sees today's (revised) values of past months, so it is optimistic for the Eurostat
  PPIs by an amount that cannot be measured with these files (A06). The drop-provisional variant does not measure it.
- Price benchmark is synthetic and not like-for-like (Xiria 630 A / 21 kA vs RM6 400 A / 12.5 kA).
- Rough publication lags (`lag_m` in `series_catalog.csv`, from the Nov-2026 download) are informative only; the
  backtest uses the relative lag at the origin (`rel_lag_m`).

## AI-tool disclosure

Code written with the help of Claude Code (Anthropic); results run and checked by the team.
