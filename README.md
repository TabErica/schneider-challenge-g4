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

Optional: the Chronos-2 benchmark needs PyTorch (about 1 GB), kept out of the main install in the `fm` extra.
`uv sync` alone gives the full pipeline without it: the backtest then reads the Chronos forecasts from
`outputs/backtest/fm_cache/` and skips the ones it does not find (printed).

```bash
uv sync --extra fm                      # + chronos-forecasting, torch (CPU); a later plain `uv sync` removes them
uv run python main.py                   # computes only the Chronos forecasts missing from the cache (~30 s on CPU,
                                        # + model download of 480 MB the first time)
```

Data-quality checks: open `notebooks/01_data_quality.ipynb` in VS Code with the `.venv` kernel (`uv sync` installs it).

- No need to activate `.venv/`: `uv run` uses it automatically (or `source .venv/bin/activate`).
- Add or remove a dependency with `uv add <pkg>` / `uv remove <pkg>`, then commit both `pyproject.toml` and `uv.lock`.
- After pulling changes to `uv.lock`, run `uv sync` again.
- Without uv: `pip install -r requirements.txt` (same versions; regenerate with
  `uv export --no-hashes --no-dev -o requirements.txt` after changing dependencies), plus
  `pip install chronos-forecasting==2.3.2` for the optional Chronos-2 benchmark.

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
`sensitivity.csv`; one sub-folder per sensitivity variant or held-back origin). Chronos-2: `fm_vs_benchmarks.csv`,
`fm_by_family.csv`, `fm_intervals.csv`, `fm_runtime.csv`, and the forecast cache `fm_cache/` (one row per origin,
target, input hash and step; reused only for an identical input). `backtest_warmup.csv`: point forecasts of the
2018-19 warm-up origins (input of the rolling selection).
`outputs/selection/` (`selection.py`): `candidates.csv`, `selected_choices.csv` (the rolling rule's choice at every
origin), `asym_k.csv`, `selection_report.csv`, `selection_summary.csv`, `interval_choice.csv`, `submission_models.csv`, `band_plausibility.csv`.
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
| `src/models/` | One interface (`base.py`: origin context in, log-return quantiles out). `classic.py` naive / drift / damped ETS / Ridge / global GBM (the first backtest, unchanged); `sf.py` AutoETS, AutoARIMA, Theta, OptimizedTheta (statsforecast); `combo.py` mean(naive, AutoETS), median(naive, AutoETS, Theta, AutoARIMA); `lme.py` LME 3-month price as P50 + quote-basis table; `intervals.py` emp / split conformal / ACI / EWMA-scaled / pooled EWMA-scaled (`ewmap`) bands around any point model; `multi.py` family models with leading indicators: ARDL, Ridge (time-series CV alpha), ECM (steel PPI ← CRU), VECM chain (iron ore → CRU CR → S09, Johansen), slim global quantile GBM shrunk to naive, epoxy structural model, and the comparison with the best univariate model |
| `src/models/fm.py` | Chronos-2 (pretrained foundation model, zero-shot, CPU): univariate and with past covariates (sources of the step-3 leading indicators), forecast cache keyed by input hash, input leakage check, comparison with naive / best univariate / best multivariate by family, raw vs calibrated bands, runtime |
| `src/features.py` | Leading indicators of the family models (1-4 per target, each with an economic reason) → `outputs/features/feature_availability.csv` (source, transformation, lag vs the origin, first month, meaning); **leakage test** (every design row at every origin rebuilt from the panel as seen at the origin = full-panel row); epoxy identity check (`epoxy_identity.csv`) |
| `src/scoring.py` | Scores: rel. MAE, MASE, pinball, coverage (above / below), width, Winkler, Diebold-Mariano vs naive with Benjamini-Hochberg q values (significance = q < 0.05); by family × period (2020-01..2023-06, 2023-07..end, 2022 alone, all); best point model and band per family × horizon |
| `src/selection_config.py` | Rules of the model selection, fixed before it was run: the 5 candidates and why, the rolling window (24 months, 3% minimum gain), the asymmetric band grid, the coverage constraint [0.70, 0.90] and the submission rules |
| `src/selection.py` | Rolling selection meta-model `selected`, asymmetric bands `<method>_up`, scores of the candidates by family × h × period (`selection_report.csv`), interval choice, submission model map (`submission_models.csv`) |
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
- **P50: nothing significantly beats naive.** Diebold-Mariano (absolute errors, Newey-West lag h - 1) with the
  Benjamini-Hochberg correction (A32): of 885 point model × target × horizon cells (852 with a p value; gbm_family's
  P50 is naive), **none is significantly better or worse than naive** (smallest q 0.054). Without the correction
  (p < 0.05) 7 would count as better and 137 as worse (Theta / OptimizedTheta on 7–9 targets, the global GBM on 15
  at 9 months): the "worse" signal is real in the raw p values, but no single cell survives the correction.
  - Best mean MASE per family × horizon (`best_by_family.csv`): damped ETS at 3 months, mean(naive, AutoETS) at
    6 months, naive at 9 months in most families. Gains over naive are 0–11% in MASE.
  - AutoETS picks a random walk on most exchange-traded prices (rel. MAE exactly 1.00). The mean(naive, AutoETS)
    combination is the steadiest: rel. MAE 0.88–0.98 on the EU steel / aluminium / plastics PPIs and the PPI-proxy
    BOM (aluminium PPI 1.02 at 9 months).
  - **LME 3-month price as P50** (D01, D02, S01, S02): rel. MAE 0.97–1.01; S02 at 6 months has p = 0.03 but q = 0.25:
    not significant after the correction. **Quote basis**: the 3-month price beats cash as the forecast of future cash by 0–0.4 MAE points
    (copper in EUR at 6 months is the one case where cash is slightly better) and has less bias (e.g. aluminium,
    3 months: −0.9% vs −1.7%). Both under-forecast in 2020-26 (copper −2.6% to −8.5%), because prices rose.
- **Bands: conformal fixes the average, not the regimes.** Mean 80% coverage over all point models
  (3 / 6 / 9 months): emp 67 / 65 / 62%, split conformal 74 / 73 / 67%, ACI 79 / 77 / 72%, EWMA-scaled
  82 / 78 / 76%.
  - Misses stay lopsided (above P90 12–17%, below P10 8–12% with ACI).
  - The Winkler score still prefers the narrow emp band at 6–9 months, because the wider conformal bands pay for
    their width.
  - By period: on steel, ACI around damped ETS covers 66–72% in 2020-01..2023-06 and 95–100% from 2023-07. The 80%
    average hides a band that is too narrow in the rally and too wide afterwards.
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
  BH q, `multivariate_vs_univariate.csv`), and none is significantly better than naive. Closest: VECM on S09 at 3 months
  (0.97 × the best univariate MAE), the epoxy structural model at 6 months (0.96), ARDL / Ridge on S09 at 3 months (0.98).
  - Uncorrected, ARDL is *worse* than the best univariate model (p < 0.05) on 2 / 4 / 7 targets at 3 / 6 / 9 months,
    Ridge on 1 / 1 / 7 (mostly LME metals and CRU coil at 9 months); after the BH correction none of the 23 is
    significant (smallest q 0.10). Ridge picks the largest penalty (alpha = 1000) in 38% of
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
- **Deviations from the step-3 plan** (not expected to change the result: Ridge already shrinks most indicator
  coefficients towards 0):
  - Lags: nickel uses this and last month's change (plan: 1-2 months); TTF and power use the 3-month change up to
    the anchor (plan: 1-month lag); Brent uses the 3-month change up to the anchor, i.e. lags 0-2 (plan: 1-3). For a
    forecast from the anchor, the latest month's move is the part that passes through over the next months, so it
    is kept.
  - The error-correction term log(S09) - b log(CRU) is used in `ecm` only, not as an ARDL / Ridge regressor.
  - ARDL in direct form only (no ARIMAX); the optional USD price × EUR/USD split is not built.
- **Consequence for the submission**: keep univariate P50s (naive / damped ETS / combo_mean per family). The family
  models stay in the backtest as documented evidence; VECM / ECM on the steel PPI and epoxy_struct are the only
  candidates worth an ensemble slot in step 5.

### Pretrained foundation model: Chronos-2 zero-shot (22 submitted targets, origins Jan-2020 → Sep-2026)

Setup (A28–A31): `amazon/chronos-2` (120M parameters, Apache-2.0), revision `29ec3766`, `chronos-forecasting` 2.3.2,
CPU, no fine-tuning. `chronos2` = univariate, context = whole history up to the anchor (as_of + nowcast, as every
model); `chronos2_cov` = + 1–3 past covariates (sources of the step-3 leading indicators, never future values), 11
family targets. Same interval layer (conf / aci / ewma) around its P50; `+native` = Chronos' own P10 / P90.
Comparison tables: `fm_vs_benchmarks.csv` (per target, vs naive / best univariate / best multivariate, each chosen
with hindsight on MAE, so the benchmarks are favoured), `fm_by_family.csv`, `fm_intervals.csv`, `fm_release_split.csv`.

- **Pretraining leakage.** Model card: trained on "a subset of Chronos Datasets (excluding test portion of datasets
  that overlap with GIFT-Eval)", "a subset of GIFT-Eval Pretrain" and synthetic data; the paper (arXiv 2510.15821,
  17 Oct 2025) lists real datasets incl. M4 monthly and gives no cutoff date of the real data. Whether a public series
  of this panel (IMF / Eurostat / LME prices) is in it cannot be ruled out. Weights on the Hugging Face hub since
  2025-10-30 (unchanged since; GitHub release note 20 Oct 2025), so **origins from 2025-11 are reported separately**:
  their outcomes cannot be in the training data. Only 8 / 5 / 2 such origins at 3 / 6 / 9 months (173 / 107 / 41
  target-origins): too few for a Diebold-Mariano test (n < 10 per target), so pooled ratios only.
- **No sign of memorisation**: pooled MAE / naive MAE of `chronos2` is 0.99 / 1.05 / 1.07 (3 / 6 / 9 months)
  before 2025-11 and 0.99 / 1.02 / 1.05 after: no edge that disappears after the release. `chronos2_cov` gets worse
  after it (1.05 / 1.07 / 1.08 → 1.13 / 1.26 / 1.19) while ARDL / Ridge improve over the same months.
- **Point forecasts: Chronos-2 is a naive-like forecaster, no better than the univariate models.** Median rel. MAE
  vs naive by family (3 / 6 / 9 months): BOM 1.01 / 1.04 / 1.02, base metals 1.02 / 1.04 / 1.09, epoxy 0.95 / 1.02 /
  1.07, plastics 0.93 / 0.98 / 1.02, steel 0.99 / 1.01 / 1.02; vs the best univariate model of each target 0.96–1.10.
  Diebold-Mariano with BH q: vs the best univariate model no cell is significant (uncorrected p < 0.05: 1 better,
  zinc 9 m, 6 worse). Against naive none is significantly better (uncorrected: S08 at 3 months, 0.70, and zinc 9 m);
  copper D01 and S01 at 9 months are significantly *worse* (1.13 / 1.12 × naive, q = 0.03).
- **With covariates**: better on the PPIs (median rel. MAE vs naive: steel 0.93 / 0.94 / 0.94, plastics 0.85 /
  0.87 / 0.94; S08 at 3 months 0.63, 0.85 × the best univariate model's MAE,
  p = 0.03 but not significant after BH), worse on LME copper (1.10–1.18 × naive; uncorrected p < 0.05 vs the best
  univariate at 3 and 9 months, not significant after BH). Against the best multivariate model of each target
  (hindsight) it is 0.81–0.97 by family; uncorrected p < 0.05 on S10 / S11 / stainless at 9 months and HDG coil,
  none significant after BH.
- **Raw Chronos bands are too narrow and lopsided**: coverage of `+native` 72 / 70 / 69% (3 / 6 / 9 months), misses
  above P90 20 / 24 / 27% vs below P10 8 / 6 / 4%; after 2025-11 only 70 / 50 / 29%. Calibrated: conf = aci (gamma 0
  chosen on the warm-up for chronos2) 74 / 72 / 69%, ewma 81 / 79 / 76% at 1.4–1.9 × the native Winkler at 6–9 months
  (wider bands). `chronos2_cov + aci` 77 / 73 / 72%. Same pattern as around the univariate models: the calibration
  fixes the level of coverage, not the upward skew.
- **Family ranking** (`best_by_family.csv`, lowest mean MASE): `chronos2` now ranks first for base metals 3 / 6 m,
  plastics 3 / 6 / 9 m and steel 3 / 9 m, by 0.4–3% of MASE over the previous best, none of it significant.
- **Runtime** (CPU, Apple M4, 10 cores): inference 14 s (`chronos2`, 105 origins × 25 series) and 18 s (`chronos2_cov`,
  105 × 11); first model download 480 MB (about 50 s). With the cache, the backtest only reads it (< 1 s).
- **Consequence for the submission**: Chronos-2 adds no accuracy that the univariate models do not already give;
  keep it as a documented benchmark. `chronos2_cov` on the EU PPIs (S08 / S09 / S10 / S11) is the only candidate for
  an ensemble slot, and its post-release deterioration argues against relying on it.

### Model selection and submission map (22 submitted targets, origins Jan-2020 → Sep-2026)

Rules fixed before the run in `src/selection_config.py` (A34–A37); `src/selection.py` applies them, no manual
choice. Periods by origin: **2020-01..2023-06** (stress: Covid, the 2021-22 rally, the energy crisis) and
**2023-07..end**, plus 2022 alone (`selection_report.csv`: every candidate and `selected`, every band, family × h ×
period: geometric-mean MAE / naive, BH q, coverage with the shares above / below, Winkler).

- **Point forecasts as a whole do not significantly beat naive.** After the Benjamini-Hochberg correction no point
  model is significantly better than naive on any target and horizon (885 cells, 852 tests, smallest q 0.054). In
  `selection_report.csv` no candidate and not `selected` is significantly better than naive at the family level
  (pooled DM) in any family, horizon or period; the significant family-level results are all *worse* (chronos2 and
  ets_local at 9 months after 2023-07, combo_median on plastics at 9 months). Per target, 2 sub-period cells pass
  q < 0.05 as better (combo_median on aluminium S02 at 9 months after 2023-07, 0.98 × naive; lme_fwd on copper D01
  at 9 months in 2022, 0.95, 12 origins), against 26 as worse.
- **Candidates** (A34, reasons in `candidates.csv`): naive, lme_fwd (copper and aluminium only), combo_median,
  ets_local (S07–S13 and the PPI-proxy BOM indices only), chronos2. The other models stay in the backtest report.
- **Choosing by past performance does not carry over.** Picking the best model per target on 2020-01..2023-06 (all
  models, hindsight) gives 1.027 / 1.047 / 1.062 × naive's MAE after 2023-07 (3 / 6 / 9 months, geometric mean over
  the 22 targets). The rolling meta-model `selected` (A35: family ranking on the last 24 months of published errors,
  naive unless the gain is ≥ 3%) is out of sample in both periods:

  | MAE / naive (3 / 6 / 9 m) | 2020-01..2023-06 | 2022 | 2023-07..end |
  |---|---|---|---|
  | `selected` | 0.948 / 1.008 / 1.035 | 1.072 / 1.167 / 1.246 | 1.016 / 1.056 / 1.036 |
  | `combo_median` | 0.977 / 1.020 / 1.047 | 1.042 / 1.150 / 1.243 | 0.967 / 0.982 / 1.000 |

  `selected` lags the regime: ets_local wins in the stress period (PPI families 0.79–0.89 × naive at 3 months), the
  window keeps choosing it after 2023-07, when it loses (plastics 1.42 at 3 months). Only combo_median stays at or
  below naive after 2023-07; neither is significant.
- **Bands depend on the regime, not only on the method.** Around naive, 80% coverage is 45–57% (emp / conf / aci)
  and 64–75% (ewma) in 2020-01..2023-06, with misses mostly *above* P90 (18–33% above vs 6–23% below). From 2023-07
  the same bands cover 88–98%. In 2022 the misses are *below* P10 (prices fell in the second half): the stress is not
  only upward.
- **Asymmetric band `<method>_up`** (A36: P90 - P50 × k, k tuned on 2020-01..2023-06 only): at 3 months k = 1.25–3.0
  brings the stress-period share above P90 to about 10% (in sample), and the coverage after 2023-07 rises to 95–97%
  (out of sample: over-coverage). At 6–9 months even k = 3.0 (the grid ceiling) leaves 10–35% above P90 in the
  stress period, and the band covers 98–100% after 2023-07.
- **Interval choice** (A37: coverage within [0.70, 0.90] in both periods, then the lowest Winkler): only 4 of 15
  family × horizon cells have such a band (BOM 3 m, base metals 3 m, plastics 3 / 6 m, all `ewma`). In the other 11
  the fallback (highest stress-period coverage, capped at 0.90) picks an `_up` band, mostly at k = 3.0, with 98–100%
  coverage after 2023-07: wide enough for a 2021-22 rally, too wide in a calm market.
- **`submission_models.csv`** (66 target × horizon rows, with the reason of each choice): P50 = combo_median on 23
  rows (base metals and steel at 3 months, BOM indices at 3 and 6 months), naive on 43 (plastics at 9 months chose
  the `selected` strategy, whose rule picks naive at Sep-2026). Bands: ewma_up 39, ewma 16, emp_up 10, conf_up 1.

- **Pooled bands and a cap on k** (A38, A39; `ewmap`, `ASYM_K_MAX = 1.5`, `band_plausibility.csv`). Two problems of
  the band rule above: per-target residual pools have a few dozen points, so their tails miss the stress months of
  the other targets; and k multiplies a *log* distance, so k = 3 at 6–9 months turned P90 at Sep-2026 into +100–490%
  (plastics PPI S11 +492%, US steel PPI +404%, AirSeT market index +214%), far beyond any move of that length in the
  data. `ewmap` pools the standardised residuals of every target of a point model (faster EWMA, lambda 0.8) and k is
  capped at 1.5. Same backtest, old vs new rules (run without the Chronos-2 cache, so chronos2 is not a candidate in
  either):

  | | old rules | new rules |
  |---|---|---|
  | family × h cells with a band within [0.70, 0.90] in both periods | 4 / 15 | 9 / 15 |
  | submitted P90 above the largest past rise of the same length (Sep-2026) | 38 / 66 | 15 / 66 |
  | P90 at 9 months: S11 / S12 / AirSeT market index | +492 / +404 / +214% | +81 / +124 / +31% |
  | worst Winkler of a chosen band (steel 9 m) | 312 | 7.4 |
  | chosen bands | ewma_up 39, ewma 16, emp_up 10, conf_up 1 | ewmap_up 31, ewmap 16, ewma 12, ewma_up 7 |

  Around naive, `ewmap` vs `ewma` over all scored origins: coverage 77 / 76 / 76% vs 82 / 80 / 78%, mean width
  0.29 / 0.49 / 0.77 vs 0.32 / 0.68 / 1.30, Winkler 0.42 / 0.83 / 1.41 vs 0.44 / 1.00 / 1.88 (3 / 6 / 9 months).
  The 15 remaining flags are mostly just above the past maximum (copper 9 m: +54% vs +52%); the large ones are the
  plastics PPIs (S11 +81% vs +30%: its EWMA volatility is high after the 2026 oil shock) and aluminium (k = 1.5).
  They are left to judgement, not clipped. The stress-period misses stay mostly *above* P90: no backward-looking
  band widens before a rally starts.

## Next steps

1. Forecast CSV `<team>_forecast_202609.csv` from `submission_models.csv` (point model and band per target × h,
   chosen by the rules of `selection_config.py`; see "Model selection" above). Check `band_plausibility.csv`
   before sending: 15 of 66 P90s are above the largest past rise of the same length (see "Pooled bands" above).
2. Market prices: LME 3-month forward and the Oct-2026 analyst view as P50 adjustments.
3. ~~Indices: upstream drivers~~ done (family models): none significantly beats the best univariate model; VECM / ECM (steel PPI) and epoxy_struct as optional ensemble members.
4. ~~Pretrained model as a benchmark~~ done (Chronos-2, zero-shot): see the findings above.
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
| A28 | model | chronos2 (amazon/chronos-2, revision 29ec3766, chronos-forecasting 2.3.2, CPU), zero-shot: context = the target's whole history up to its anchor as seen at the origin (as_of + nowcast, as every model), fed as y / y[anchor] rounded to 10 decimals (scale-free, so the BOM indices give the same input whatever their origin = 100 factor); max(h_eff) steps (9 with the nowcast); P10 / P50 / P90 at step h_eff, log of the ratio = the log return. Series forecast independently (no cross-learning); no fine-tuning. | `models/fm.py` |
| A29 | model | chronos2_cov: as chronos2 + past covariates only (never known-future values): 1-3 source series of the target's step-3 leading indicators, as published, aligned on the target's context months (values up to the anchor). TTF dropped for HDG coil and LME aluminium to keep 3. Only the 11 family targets. | `models/fm.COVARIATES` |
| A30 | evaluation | Pretraining leakage: Chronos-2's training corpus (parts of the Chronos and GIFT-Eval pretraining data, incl. M4 monthly, plus synthetic series) has no published cutoff and may contain some public series of this panel. Weights published on the Hugging Face hub on 2025-10-30 (paper 2025-10-17): origins from 2025-11 are reported separately (post_release); their outcomes cannot be in the training data. | `models/fm.POST_RELEASE` |
| A31 | pipeline | Chronos forecasts are cached (outputs/backtest/fm_cache/) per origin, target and sha256 of the exact input: a cached forecast is reused only for an identical input. Without the optional fm dependencies the backtest reads the cache and skips what is missing. | `models/fm.Cache` |
| A32 | evaluation | Multiple testing: every Diebold-Mariano conclusion (vs naive, vs the best univariate model) uses the Benjamini-Hochberg q value, q < 0.05. One family = all point models x submitted targets x horizons of one table and period; tests without a p value (P50 = naive) are left out of the family. The raw p < 0.05 counts are kept for comparison only. | `scoring.bh_qvalues, scoring.per_series` |
| A33 | evaluation | Every score is reported for two periods by origin, 2020-01..2023-06 (stress: Covid, the 2021-22 rally, the energy crisis) and 2023-07..last origin with a known outcome, plus 2022 alone and all. | `scoring.PERIODS` |
| A34 | selection | Candidate point models fixed in advance with a reason each: naive, lme_fwd (copper and aluminium only), combo_median, ets_local (S07-S13 and the PPI-proxy BOM indices only), chronos2. All other models stay in the backtest report only. | `selection_config.CANDIDATES` |
| A35 | selection | Rolling selection 'selected': at origin t, errors published at or before t in the last 24 months (at least 12 per target); per family the geometric mean over its targets of MAE / naive's MAE; each target takes the best candidate that applies to it, naive if the gain is below 3%. Its band is the band of the model it selected. | `selection.rolling_selection` |
| A36 | interval | Asymmetric band '<method>_up': P90 - P50 (log) times k >= 1 (grid 1.00-1.50, step 0.05; capped at 1.5 by A39), one k per point model, method, family and h: the smallest k with at most 10% of outcomes above P90 on the 2020-01..2023-06 origins whose outcome is published by 2023-07. The second period tests k out of sample. | `selection.tune_asym` |
| A37 | selection | Submission map, per family x h: the point strategy (selected or combo_median) replaces naive only if its geometric-mean MAE / naive is below 1 in both periods (lower worse-period ratio if both are); 'selected' = the rolling rule's choice at Sep-2026. Band: 80% coverage within [0.70, 0.90] in both periods, then the lowest Winkler; if no band qualifies, the highest stress-period coverage (capped at 0.90), then Winkler. No manual choice. | `selection.point_strategy, selection.interval_choice` |
| A38 | interval | ewmap: residuals standardised by an EWMA volatility with lambda = 0.8 (half-life about 3 months, faster than ewma's 0.9) and pooled over every target of the point model at the same horizon; at origin t only residuals published by t; conformal bounds of the pool x the target's own volatility today. Pooling assumes the standardised residuals of different targets share one distribution: the pool's tails then include every target's stress months, not only the few dozen of one target. Fewer than 8 pooled residuals -> emp. | `models/intervals._pooled` |
| A39 | interval | Cap of the asymmetric factor: k <= 1.5. k multiplies a log distance, so with k = 3 the 6-9 month P90s at Sep-2026 reached +120% (copper) to +475% (plastics PPI), beyond any move of that length in the data. band_plausibility.csv compares every submitted P90 with the largest past rise of the same length. | `selection_config.ASYM_K_MAX, selection.plausibility` |

Limits:
- BOM from the standalone `RM_Prices_XHEC_Challenge.xlsx`; the older BOM in `data.zip` (`bom_materials.csv`) is not used.
- No data vintages: the backtest sees today's (revised) values of past months, so it is optimistic for the Eurostat
  PPIs by an amount that cannot be measured with these files (A06). The drop-provisional variant does not measure it.
- Price benchmark is synthetic and not like-for-like (Xiria 630 A / 21 kA vs RM6 400 A / 12.5 kA).
- Rough publication lags (`lag_m` in `series_catalog.csv`, from the Nov-2026 download) are informative only; the
  backtest uses the relative lag at the origin (`rel_lag_m`).

## Software and model licences

Third-party code and models used by the pipeline (data licences: see §1). Licences as declared in each package's
metadata; Chronos-2 as stated on its official model card (`license: apache-2.0`, checked 2026-11-23).

| Component | Version | Licence | Source |
|---|---|---|---|
| pandas | 3.0.5 | BSD-3-Clause | https://pandas.pydata.org |
| NumPy | 2.5.3 | BSD-3-Clause (bundled parts: 0BSD, MIT, Zlib, CC0-1.0) | https://numpy.org |
| statsmodels | 0.15.0 | BSD-3-Clause | https://www.statsmodels.org |
| scikit-learn | 1.9.1 | BSD-3-Clause | https://scikit-learn.org |
| statsforecast (Nixtla) | 2.0.1 | Apache-2.0 | https://github.com/Nixtla/statsforecast |
| matplotlib | 3.11.2 | matplotlib licence (PSF-based) | https://matplotlib.org |
| openpyxl | 3.1.5 | MIT | https://openpyxl.readthedocs.io |
| chronos-forecasting (optional `fm`) | 2.3.2 | Apache-2.0 | https://github.com/amazon-science/chronos-forecasting |
| PyTorch (optional `fm`) | 2.15.0 | BSD-3-Clause (package metadata: Apache-2.0 AND BSD-3-Clause AND MIT AND others for bundled parts) | https://pytorch.org |
| transformers (optional `fm`, Chronos dependency) | 5.24.0 | Apache-2.0 | https://github.com/huggingface/transformers |
| **Chronos-2** model weights (amazon/chronos-2) | Hugging Face revision `29ec3766` (weights of 2025-10-30), 120M parameters | Apache-2.0 | https://huggingface.co/amazon/chronos-2 |

## AI-tool disclosure

Code written with the help of Claude Code (Anthropic); results run and checked by the team.
