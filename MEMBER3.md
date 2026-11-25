# Member 3

This additive package targets the foundation-model branch with the interval-pooling API at GitHub revision ca5b971. It uses the branch's own preprocessing, backtest and selection functions. Member 3 CSVs and figures go to outputs/member3.

## Forecast inputs

Place the authorized challenge files in Data/Data, including the Member 3 v1 BOM workbook RM_Prices_XHEC_Challenge_v1.xlsx. If the branch's default backtest outputs have not yet been generated, use its normal pipeline from the repository root.

    uv run --locked python main.py

Member 3 reads the existing backtest_predictions.csv, backtest_warmup.csv and forecasts_at_origin.csv. Run it as a separate step from the repository root.

    uv run --locked python src/member3.py --team-id team_04

The normal project dependencies are sufficient. Optional Chronos results present in the upstream predictions can enter the branch selection. For the traditional benchmark comparison, generate the upstream predictions with its standard environment without the optional foundation-model extra.

## Outputs

- team_04_forecast_202609.csv has 54 material forecasts and six v1 BOM-index forecasts.
- product_holdout_scores.csv compares four point methods on paired dates after the selection freeze in December 2024. The full historical comparison with the final selection reapplied is descriptive.
- eto_quote_policy.csv records the planning basis, downside driver, quote and required protection. The simultaneous material stress uses each material's individual tail forecast. The planning P90 is fitted on errors of the flat product basket.
- submission_target_trace.csv records forecast provenance and unit conversion.
- notebooks/03_member3.ipynb opens the resulting tables and figures.

The flat product P50 and validated material exposure use the priced portion of the v1 BOM. Default quotes use synthetic benchmarks. Optional commercial inputs and quote constraints are listed under src/member3.py --help.
