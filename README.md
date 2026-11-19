# X-HEC × Schneider Electric Data Challenge 2026: raw-material forecasting

Forecast the raw-material prices behind the RM6 and RM AirSeT RMUs at **+3 / +6 / +9 months**, turn them into a
BOM cost index, then into ETO quote rules and an SF6-free migration strategy.

> **Data is not in this repo** (challenge rule). `Data/`, all `csv/xlsx/zip/pdf/pptx/png` files and `outputs/`
> are git-ignored.

## Setup

### 1. Data

Put the challenge files, unchanged, in `Data/Data/`: `data.zip` (read in place; an already-extracted `Data/Data/data/`
folder also works), `Copia de Europe Steel INDEXES.xlsx`, `Epoxy for student.xlsx`,
`LME_Copper_Aluminium_EU_Monthly_2016_2026.xlsx`, `RMU_IQI_EU_Price_Benchmark_2017-2026_Synthetic.xlsx`.

### 2. Python environment (uv)

The environment is managed with [uv](https://docs.astral.sh/uv/). `pyproject.toml` lists the dependencies,
`uv.lock` pins every package version and `.python-version` pins Python 3.12, so every machine gets the same setup.

```bash
# install uv once (macOS / Linux; on Windows see the uv docs)
curl -LsSf https://astral.sh/uv/install.sh | sh     # or: brew install uv

uv sync                         # creates .venv/ with the exact locked versions (downloads Python 3.12 if needed)
uv run python main.py           # ~2 min; --skip-backtest reuses existing backtest results
uv run python src/preprocess.py # any single step
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
| `pre-commit-hooks` | Large files (> 1 MB), merge-conflict markers, YAML / TOML syntax, end-of-file and trailing whitespace |

If a hook changes a file (e.g. strips a notebook), the commit stops: check the change, `git add` it and commit again.
nbstripout also clears the outputs in your local copy, so run the notebook again afterwards.

**CI** (`.github/workflows/ci.yml`, on every push to `main` and every pull request) runs the same hooks, checks that
`uv.lock` matches `pyproject.toml` and that the code imports. It cannot run the pipeline: the data is not in the repo.

### 4. Outputs

Outputs: `outputs/preprocessed/` (clean panel, series catalog, transformation log), `outputs/eda/` (charts, tables)
and `outputs/backtest/` (predictions, scores).
`DECK=1` renders charts in slide style to `outputs/slides/`. Colours: **RM6 green `#16884A`, RM AirSeT blue `#3478B8`**.

| File | Role |
|---|---|
| `src/data.py` | Reads every source as published; targets; BOM (from `data.zip` or its extracted folder) and BOM cost index |
| `src/preprocess.py` | One monthly panel: month-start dates, cut at Sep-2026, typo corrections, gap fill, USD → EUR. Writes `panel_monthly.csv`, `series_catalog.csv`, `transform_log.csv` |
| `notebooks/01_data_quality.ipynb` | Coverage, missing values, wrong values, derived-column checks, IMF vs Westmetall, steel PPI vs CRU / MEPS |
| `src/eda.py` | Price families, key material prices, BOM cost index, cost structure, volatility, price benchmark |
| `src/backtest.py` | Rolling backtest Jan-2020 → Sep-2026: naive, drift, ETS, Ridge, global GBM; P10/P50/P90 |
| `src/plot_backtest.py` | Error relative to naive per series / model / horizon |
| `src/diagrams.py` | Production chain (predictive links) and methodology diagrams |

## Findings so far

- **Exchange-traded prices** (LME metals, iron ore): no model clearly beats naive, so P50 = today's price and the
  work goes into the range.
- **EU steel and aluminium PPIs, AirSeT cost index**: ETS beats naive by about 15–27% at 3 months. EU plastics and
  copper PPIs are about even with naive; US PPIs are worse.
- **ML models lose to naive at 6–9 months** (Ridge is the worst at 6 months, the global GBM at 9 months): about
  120 monthly points per series is too little.
- **Upstream prices help predict the next 3 months** for aluminium, steel, stainless and plastics (r 0.4–0.6).
  Copper LME and the EU copper index move in the same month.
- **Ranges are too narrow**: about 68% coverage against an 80% target.

## Next steps

1. Forecast CSV `<team>_forecast_202609.csv`: per-series model choice by backtest, calibrated intervals.
2. Market prices: LME 3-month forward and the Oct-2026 analyst view as P50 adjustments.
3. Indices: add only the upstream drivers that predict 3 months ahead (a Ridge with all drivers did not beat naive).
4. Pretrained model (Chronos-Bolt) as a benchmark.
5. Publication lags in the backtest: `preprocess.as_of()` cuts each series by its lag (`lag_m` in `series_catalog.csv`).
6. Business layer: forecast → BOM cost → margin at risk → quote rule.

## Assumptions and limits

- RM6 largest unlabelled BOM slice (~45%) = carbon steel; polymer EUR/t assumed (`src/data.py`). Two BOM versions
  exist; the `data.zip` one is used.
- BOM index: fixed Sep-2026 weights; stainless follows the EU steel PPI (BOM proxy), not nickel.
- Backtest assumes month-t data is known at origin t.
- Price benchmark is synthetic and not like-for-like (Xiria 630 A / 21 kA vs RM6 400 A / 12.5 kA).
- Epoxy file: ECH (and so epoxy) May-2026 is a typo, about 10× too low; set to NaN and interpolated (`preprocess.py`).
- Publication lags in `series_catalog.csv` are rough: read from the last month in each file at the Nov-2026 download.

## AI-tool disclosure

Code written with the help of Claude Code (Anthropic); results run and checked by the team.
