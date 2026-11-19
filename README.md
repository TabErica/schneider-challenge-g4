# X-HEC × Schneider Electric Data Challenge 2026: raw-material forecasting

Forecast raw-material prices for the two RMU products (RM6 and RM AirSeT) at **+3 / +6 / +9 months**, propagate them
into a BOM cost index, then turn that into (A) margin-protecting ETO quote rules and (B) a strategy for winning
SF6-free migration in non-regulated markets.

> **Data is NOT in this repo** (challenge rule: never publish data files). Everything under `Data/`, plus all
> `*.csv / *.xlsx / *.zip / *.pdf / *.pptx / *.png` and `outputs/`, is git-ignored. Keep it that way.

## Setup

1. Get the challenge data from the team drive and place it **unchanged** like this (zips are read in place, no
   need to extract):

   ```
   Data/Data/data.zip
   Data/Data/Copia de Europe Steel INDEXES.xlsx
   Data/Data/Epoxy for student.xlsx
   Data/Data/LME_Copper_Aluminium_EU_Monthly_2016_2026.xlsx
   Data/Data/RMU_IQI_EU_Price_Benchmark_2017-2026_Synthetic.xlsx
   Data/Data/benchmark-modules-fonctionnels-rm6-ne-iqi-3-fonctions-400a-appareillage-isolement-et-coupure-dans-le-gaz-sf6.xlsx
   Data/Data/Supplier_balancesheet.zip
   ```

2. Install and run (Python 3.13):

   ```bash
   pip install -r requirements.txt
   python main.py                 # everything, ~2 min
   python main.py --skip-backtest # charts only
   ```

Outputs land in `outputs/eda/` (charts + tables) and `outputs/backtest/` (predictions + scores).

## Code map

| File | What it does |
|---|---|
| `src/data.py` | Loads every source into monthly EUR series; target catalogue; BOM (read from `data.zip`) and BOM cost index |
| `src/eda.py` | EDA charts `fig01`–`fig07` + coverage / volatility / lead-lag tables |
| `src/backtest.py` | Rolling-origin backtest (origins Jan-2020 → Sep-2026−h), 6 forecasters, P10/P50/P90, scoring |
| `src/plot_backtest.py` | `fig08`: each model's error relative to the naive forecast |
| `src/diagrams.py` | `fig09` production chain with measured lead/lag, `fig10` methodology |
| `main.py` | Runs everything in order |

## Status (19 Nov)

**Done**
- Data loading and alignment: 14 BOM-proxy targets (S01–S05, S07–S13, D01, D02), 2 BOM cost indices, energy drivers,
  internal CRU/MEPS steel prices, epoxy cost build-up, competitor price benchmark.
- EDA (fig01–fig10).
- Backtest of naive, drift, ETS (per series), Ridge (per series), GBM (one global quantile model), and a naive+GBM blend.

**What the backtest says** (MAE relative to naive; < 1 = better)
- **Exchange-traded prices** (LME metals, iron ore): nothing beats naive. Use naive/futures for P50 and put the
  effort into the P10–P90 range.
- **Smooth indices** (EU steel / aluminium PPI, BOM indices): per-series ETS beats naive by 10–25% at 3 months.
- **One global GBM is the worst** at 6–9 months (about 120 monthly points per series is too little).
- **Intervals are too narrow**: about 68% coverage against an 80% target.

**Next steps**
1. Submission CSV generator `<team>_forecast_202609.csv`: per-series model selection by backtest and calibrated
   (volatility-scaled / conformal) intervals. Gets us Level 2–3.
2. Market prices: add LME 3-month forward (LME file) and analyst view (Oct-2026 RM deck) as P50 adjustments.
3. Indices: add upstream drivers with the measured lags (fig09) to the trend model.
4. Try a pretrained model (Chronos-Bolt) as a bonus benchmark.
5. Handle publication lags (Eurostat 1–2 months) in the backtest.
6. Business layer: forecast → BOM cost → margin at risk → quote rule (adjustment, validity, indexation trigger, buffer).

**Assumptions to confirm with Schneider**
- RM6: the largest unlabelled BOM slice (~45%) is treated as carbon steel.
- Polymer EUR/t levels are assumed (no free EUR/t source), see `ASSUMED_PRICE_EUR_T` in `src/data.py`.
- Two BOM versions exist (in `data.zip` and standalone). Code uses the `data.zip` one.
- Backtest assumes month-t data is known at origin t (same convention as the challenge CSV).

**Known data issues**
- Epoxy file: ECH May-2026 is about 10× too low (likely a typo), and dates drift from the 1st to the 5th of the month.
- The competitor teardown file is unreliable (e.g. one BOM totals 236 g).

## AI-tool disclosure

Code in this repo was written with the help of Claude Code (Anthropic). All results were run and checked by the team.
