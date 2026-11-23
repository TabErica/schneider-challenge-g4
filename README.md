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
uv run python main.py                   # ~9 min: everything, origin Sep-2026; --skip-backtest reuses the backtests
uv run python main.py --origin 2026-06  # held-back check: forecasting path only, data as of Jun-2026
uv run python src/preprocess.py         # any single step
uv run python src/backtest.py --help    # backtest options (--nowcast, --drop-provisional, --bom-weights origin)
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
`outputs/backtest/` (predictions, forecasts at the origin, scores, training sizes, `sensitivity.csv`; one sub-folder
per sensitivity variant or held-back origin).
`DECK=1` renders charts in slide style to `outputs/slides/`. Colours: **RM6 green `#16884A`, RM AirSeT blue `#3478B8`**.

| File | Role |
|---|---|
| `src/targets.py` | **Target registry**: the one list of what is forecast (id, name, family, region, unit, material / bom_index, in_submission, history_start). Every forecasting script reads it |
| `src/assumptions.py` | Every modelling assumption (A01–A13) → `outputs/assumptions.csv` |
| `src/data.py` | Reads every source as published; BOM (from the standalone workbook); BOM cost index in two versions (PPI proxy, market prices), with weights and references at any date; `panel()` keeps each series' full history |
| `src/preprocess.py` | One monthly panel: month-start dates, cut at Sep-2026, typo corrections, gap fill, USD → EUR. Writes `panel_monthly.csv`, `series_catalog.csv`, `transform_log.csv`. `as_of()` = the panel as seen at an origin (relative ragged edge, optional provisional drop); `nowcast()` = optional stainless fill |
| `notebooks/01_data_quality.ipynb` | Coverage, missing values, wrong values, derived-column checks, IMF vs Westmetall, steel PPI vs CRU / MEPS |
| `src/eda.py` | Price families, key material prices, BOM cost index and cost per unit, cost structure, volatility by horizon, price benchmark (figures and the tables behind them) |
| `src/eda_risk.py` | Risk EDA: unit roots, fat tails, volatility clustering, band coverage, cost-index rises (3/6/9 m) and their drivers, regimes, correlations (fig11–fig15, `risk_*.csv`) |
| `notebooks/02_eda_risk.ipynb` | Conclusions of the EDA: risk (§1–4), key prices, cost per unit, risk by horizon, supply-chain links, selling-price benchmark (§5–9) |
| `src/backtest.py` | Rolling backtest Jan-2020 → origin: naive, drift, ETS, Ridge, global GBM; P10/P50/P90; data as of each origin (`as_of`); options `--origin`, `--nowcast`, `--drop-provisional`, `--bom-weights` |
| `src/sensitivity.py` | Backtest variants against the baseline (nowcast, provisional dropped, origin BOM weights) → `sensitivity.csv` |
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
- **New targets are hard**: on the CRU CR / HDG coil prices, ETS is 8–10% *worse* than naive at 3 months and
  34–43% worse at 6–9 months (momentum then sharp reversals), and the naive 80% band covers only 58–63% even with
  history from 2012. Stainless 304 (from h + 1) is the one new market price where ETS beats naive (rel. MAE 0.93–0.96).
  Epoxy: ETS 0.92 at 3 months. Market-price BOM indices: no model beats naive; ETS ≈ naive for RM6 (1.00–1.07), 4–19% worse for AirSeT.
- **Sensitivities** (`outputs/backtest/sensitivity.csv`):
  - *Nowcast of stainless*: MAPE of stainless and the market-price BOM indices falls by 0.5–2 points
    (e.g. RM6 market BOM, 3 months: 9.1% → 7.8%) — worth using in the submission model.
  - *Dropping the provisional Eurostat months* costs 2–3 MAPE points at 3 months on S07–S11 and the PPI-proxy BOM
    (S09: 3.7% → 6.5% for ETS): the last two months carry real information; the baseline keeps them.
  - *Origin BOM weights*: changes MAPE by < 0.3 point and 80% coverage by at most 3 points; the Sep-2026 weights add no material look-ahead.
- **Held-back origin Jun-2026** (3 months → Sep-2026, 22 series): every model is within ±1 point of naive's 6.3% error;
  80% coverage 77–95% on this single, calm quarter (too few points to rank models).

## Next steps

1. Forecast CSV `<team>_forecast_202609.csv`: per-series model choice by backtest, calibrated intervals (wider,
   asymmetric, from a window that includes 2020-22).
2. Market prices: LME 3-month forward and the Oct-2026 analyst view as P50 adjustments.
3. Indices: add only the upstream drivers that predict 3 months ahead (a Ridge with all drivers did not beat naive).
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
| A03 | data | A target whose last value is before the origin is forecast from its last value: the effective horizon is h + lag (stainless 304 and the market-price BOM indices: h + 1). | `backtest.run` |
| A04 | data | No forward fill in any data used for forecasting. data.TRAIL_FILL (carry stainless Aug-2026 to Sep-2026) is kept only for the descriptive EDA charts and tables. | `data.panel(trail_fill=...)` |
| A05 | data | Optional nowcast (off by default): the missing stainless 304 month(s) at an origin are filled with an OLS of its monthly log change on nickel in EUR (S03_EUR) and the EU steel PPI (S09), fitted on the months known at that origin. Each fill is logged (nowcast_log.csv). | `preprocess.nowcast, backtest --nowcast` |
| A06 | data | Eurostat PPIs (S07-S11, E13-E16): the last 2 months known at an origin are provisional (Eurostat revises them). Baseline keeps them; the sensitivity run drops them. Only the final vintage is available, so the backtest sees revised values (slightly optimistic). | `preprocess.PROVISIONAL, backtest --drop-provisional` |
| A07 | model | Univariate models use each target's full history (CRU from 2012, stainless from 2008, epoxy from 2018); models with drivers use the months where every input is known. | `targets.py, backtest.py` |
| A08 | BOM | RM6 largest unlabelled slice = carbon steel at 45% of the implied mass (the share the older BOM label gave). | `data.ASSUMED_UNLABELLED_SHARE` |
| A09 | BOM | Polymer levels assumed at 1 300 EUR/t (PET) and 2 700 EUR/t (engineering plastics), constant over time (also for origin weights). | `data.ASSUMED_PRICE_EUR_T` |
| A10 | BOM | Submission BOM indices: base Sep-2026 = 100 (Schneider's common base). Cost shares = kg x latest published EUR/t at or before Sep-2026 (stainless 304: Aug-2026); the market-price index has no Sep-2026 value until stainless is published, so it is forecast from Aug-2026. | `data.bom_cost_table, data.bom_index` |
| A11 | BOM | Backtest option --bom-weights origin: cost shares from the prices known at each origin (kg x latest EUR/t at or before the origin) instead of Sep-2026 shares, to measure the look-ahead in the fixed weights. Not used for the submission. | `backtest --bom-weights` |
| A12 | BOM | Fixed quantities (kg per unit) over the whole period: no design or sourcing change. | `data.load_bom` |
| A13 | data | ECH and epoxy May-2026 are a decimal-shift typo: set to NaN, then log-linear interpolation. | `preprocess.CORRECTIONS` |

Limits:
- BOM from the standalone `RM_Prices_XHEC_Challenge.xlsx`; the older BOM in `data.zip` (`bom_materials.csv`) is not used.
- No data vintages: the backtest sees today's (revised) values of past months, so it is slightly optimistic for the
  Eurostat PPIs (A06 gives the size of the effect of their last two months).
- Price benchmark is synthetic and not like-for-like (Xiria 630 A / 21 kA vs RM6 400 A / 12.5 kA).
- Rough publication lags (`lag_m` in `series_catalog.csv`, from the Nov-2026 download) are informative only; the
  backtest uses the relative lag at the origin (`rel_lag_m`).

## AI-tool disclosure

Code written with the help of Claude Code (Anthropic); results run and checked by the team.
