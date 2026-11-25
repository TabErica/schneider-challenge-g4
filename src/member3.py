"""Run Member 3 after the default preprocessing and pooling-compatible backtest.

Run `uv run python src/member3.py` for CSVs and figures in outputs/member3/.
Optionally pass `--commercial-inputs PATH` for a CSV keyed by product and target_month
(YYYY-MM), with any of units, base_price_eur_ex_vat, additional_direct_cost_eur_per_unit.
Units are the target month's volume at the base price. Additional direct cost should
cover omitted materials and any other direct costs needed for the contribution scenario.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import data as D
import preprocess as P
from member3_backtest import (
    COMPARISON_METHODS, EVALUATION_CUTOFF, add_product_bands, branch_material_forecasts, product_rows, release_lags,
    score_products,
)
from member3_bom import COST, assumption_sensitivity, bom_audit, build_costs, load_bom
from member3_charts import save_charts
from member3_commercial import commercial_scenarios, decision_table, elasticity_scenarios, load_commercial_inputs
from member3_decision_analysis import material_exposure, pricing_frontier, validate_forecast_choice
from member3_decision_brief import save_decision_brief
from member3_eto import OfferAssumptions, basket_history, make_offers, validity_choices, waiting_risk
from member3_eto_charts import save_eto_brief
from member3_submission import DEFAULT_TEAM_ID, add_planning_reference, build_submission, coverage_register, material_calibration
from member3_handoff import migration_thresholds
import targets as T

OUT = D.OUT / "member3"


def run(commercial_path=None, out=OUT, team_id=DEFAULT_TEAM_ID, offer_assumptions=OfferAssumptions()):
    bom = load_bom()
    series = sorted(bom["series"].dropna().unique())
    forecast_series = sorted(set(series) | set(T.ids("material")))
    lags = release_lags(forecast_series)
    forecasts, material_plausibility = branch_material_forecasts(forecast_series, lags)
    panel = P.load()
    current = forecasts.loc[forecasts["origin"].eq(D.BOM_BASE)]
    build_columns = ["series", "h", "origin", "anchor", "y_t", "actual", "point", "interval", "model",
                     "p10", "p50", "p90", "forecast_source", "selection_revision", "selection_n"]
    detail, summary = build_costs(bom, current[build_columns])
    reconciliation, coverage = bom_audit(bom, detail)
    sensitivity = assumption_sensitivity(summary, detail)

    raw_roll = product_rows(bom, forecasts, panel, lags)
    roll = add_product_bands(raw_roll)
    scores = score_products(roll, COMPARISON_METHODS).assign(
        evaluation="final_strategy_reapplied_retrospectively")
    holdout = raw_roll.assign(bottom_up=raw_roll["heldout_bottom_up"])
    holdout = add_product_bands(holdout).loc[lambda d: d["origin"].gt(EVALUATION_CUTOFF)]
    holdout_scores = score_products(holdout, COMPARISON_METHODS).assign(
        evaluation="strategy_frozen_2024_12")
    if holdout_scores.groupby("method").size().reindex(COMPARISON_METHODS).ne(len(summary)).any():
        raise ValueError("Expected a held-out MAE for each product, horizon and method")
    current_costs = roll.loc[roll["origin"].eq(D.BOM_BASE)]
    bands = current_costs[["product", "h", "bottom_up", "bottom_up_p10", "bottom_up_p90", "band_n", "direct_autoets"]]
    summary = summary.merge(bands, on=["product", "h"], how="left", validate="one_to_one")
    if summary["bottom_up"].isna().any() or not summary["bottom_up"].sub(summary[f"central_{COST}"]).abs().lt(1e-6).all():
        raise ValueError("Current BOM rollup and rolling validation use different cost bases")
    summary = summary.drop(columns="bottom_up").rename(columns={
        "bottom_up_p10": "empirical_p10_material_cost_eur_per_unit",
        "bottom_up_p90": "empirical_p90_material_cost_eur_per_unit",
        "band_n": "empirical_band_history_n",
        "direct_autoets": "direct_autoets_material_cost_eur_per_unit",
    })
    summary, planning_calibration = add_planning_reference(summary, raw_roll)

    pricing = commercial_scenarios(summary, D.load_price_benchmark(), load_commercial_inputs(commercial_path, summary))
    elasticity = elasticity_scenarios(pricing)
    decisions = decision_table(pricing, holdout_scores)
    tests = pd.concat([
        validate_forecast_choice(roll, methods=("direct_autoets",)).assign(evaluation="full_sample"),
        validate_forecast_choice(holdout, methods=("bottom_up", "direct_autoets", "legacy_bottom_up"))
        .assign(evaluation="strategy_frozen_2024_12"),
    ], ignore_index=True)
    frontier = pricing_frontier(pricing)
    exposure = material_exposure(detail)
    index_history, index_weights = basket_history(detail, panel)
    validity = validity_choices(pricing, waiting_risk(index_history), offer_assumptions)
    offers = make_offers(pricing, summary, validity, offer_assumptions)
    tco = migration_thresholds(offers)
    submission, submission_trace = build_submission(current, summary, panel, team_id)
    precursors = current.loc[current["series"].isin(T.ids("material", in_submission=False)),
                            ["series", "h", "origin", "model", "forecast_source", "p10", "p50", "p90"]].assign(unit="EUR/t")
    tables = {
        "bom_reconciliation": reconciliation, "bom_coverage": coverage, "bom_sensitivity": sensitivity,
        "material_costs": detail, "product_costs": summary, "product_backtest": roll,
        "product_backtest_scores": scores, "product_holdout_scores": holdout_scores,
        "material_band_plausibility": material_plausibility, "pricing_revenue": pricing,
        "elasticity_scenarios": elasticity,
        "pricing_decisions": decisions,
        "forecast_decision_tests": tests, "member4_pricing_frontier": frontier, "member4_material_exposure": exposure,
        "eto_quote_policy": offers, "eto_validity_backtest": validity, "eto_bom_index": index_history,
        "eto_bom_index_weights": index_weights, "member4_tco_thresholds": tco,
        "submission_target_trace": submission_trace, "submission_coverage_register": coverage_register(bom),
        "submission_material_calibration": material_calibration(forecasts, lags).assign(
            evaluation="final_strategy_reapplied_retrospectively"),
        "submission_bom_calibration": planning_calibration, "epoxy_input_forecasts": precursors,
        f"{team_id}_forecast_{D.BOM_BASE:%Y%m}": submission,
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False, encoding="utf-8" if frame is submission else "utf-8-sig")
    save_charts(out, coverage, detail, summary, holdout_scores, sensitivity, pricing, elasticity)
    save_decision_brief(out, tests.loc[tests["evaluation"].eq("strategy_frozen_2024_12")],
                        frontier, exposure, sensitivity)
    save_eto_brief(out, offers, validity, tco)
    return summary, holdout_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--commercial-inputs", type=Path)
    parser.add_argument("--team-id", default=DEFAULT_TEAM_ID)
    parser.add_argument("--revenue-loss-limit", type=float, default=0.01)
    parser.add_argument("--elasticity-floor", type=float, default=-2.0)
    parser.add_argument("--waiting-risk-budget", type=float, default=0.005)
    parser.add_argument("--margin-erosion-limit", type=float, default=0.005)
    args = parser.parse_args()
    assumptions = OfferAssumptions(args.revenue_loss_limit, args.elasticity_floor, args.waiting_risk_budget, args.margin_erosion_limit)
    summary, scores = run(args.commercial_inputs, team_id=args.team_id, offer_assumptions=assumptions)
    print(summary[["product", "target_month", "priced_mass_share", f"central_{COST}",
                   "central_material_cost_change_eur_per_unit"]].round(4).to_string(index=False))
    print("Held-out backtest MAE in EUR per unit (strategy frozen in December 2024)")
    print(scores.pivot(index=["product", "h"], columns="method", values="mae_eur_per_unit").round(2).to_string())
    print(f"Wrote Member 3 tables and figures to {OUT}")


if __name__ == "__main__":
    main()
