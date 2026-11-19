# X-HEC × Schneider Electric Data Challenge 2026: raw-material forecasting

Forecast the raw-material prices behind the RM6 and RM AirSeT RMUs at **+3 / +6 / +9 months**, turn them into a
BOM cost index, then into ETO quote rules and an SF6-free migration strategy.

> **Data is not in this repo** (challenge rule). `Data/`, all `csv/xlsx/zip/pdf/pptx/png` files and `outputs/`
> are git-ignored.

## Setup

Put the challenge files, unchanged, in `Data/Data/` (zips are read in place): `data.zip`,
`Copia de Europe Steel INDEXES.xlsx`, `Epoxy for student.xlsx`, `RMU_IQI_EU_Price_Benchmark_2017-2026_Synthetic.xlsx`.

```bash
pip install -r requirements.txt
python main.py                  # ~2 min; --skip-backtest reuses existing backtest results
```

Outputs: `outputs/eda/` (charts, tables) and `outputs/backtest/` (predictions, scores).
`DECK=1` renders charts in slide style to `outputs/slides/`. Colours: **RM6 green `#16884A`, RM AirSeT blue `#3478B8`**.

| File | Role |
|---|---|
| `src/data.py` | Loads all sources; targets; BOM (read from `data.zip`) and BOM cost index |
| `src/eda.py` | Price families, BOM cost index, cost structure, volatility, price benchmark |
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
5. Publication lags (Eurostat 1–2 months) in the backtest.
6. Business layer: forecast → BOM cost → margin at risk → quote rule.

## Assumptions and limits

- RM6 largest unlabelled BOM slice (~45%) = carbon steel; polymer EUR/t assumed (`src/data.py`). Two BOM versions
  exist; the `data.zip` one is used.
- BOM index: fixed Sep-2026 weights; stainless follows the EU steel PPI (BOM proxy), not nickel.
- Backtest assumes month-t data is known at origin t.
- Price benchmark is synthetic and not like-for-like (Xiria 630 A / 21 kA vs RM6 400 A / 12.5 kA).
- Epoxy file: ECH May-2026 is a typo (about 10× too low).

## AI-tool disclosure

Code written with the help of Claude Code (Anthropic); results run and checked by the team.
