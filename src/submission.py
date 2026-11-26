"""Build the challenge CSV and trace its scope, currency and origin conventions."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

import data as D
import targets as T
from cost_backtest import add_product_bands
from bom_cost import COST
from models.base import HORIZONS

DEFAULT_TEAM_ID = "team_04"
AI_DISCLOSURE = "OpenAI Codex; Claude Code (Anthropic) (code and analysis assistance; no LLM-generated predictions)"
INDEX_UNIT = "index (origin=100)"
QUANTILES = ["p10", "p50", "p90"]
SUBMISSION_COLUMNS = ["team_id", "target_type", "target_id", "region", "origin_month", "horizon_m",
                      "target_month", "unit", *QUANTILES, "model_name", "ai_tools"]


def add_planning_reference(summary, raw_backtest):
    """Use the flat planning point and an interval calibrated on that same point method."""
    rows = add_product_bands(raw_backtest, "flat_cost")
    current = rows.loc[rows["origin"].eq(D.BOM_BASE), ["product", "h", "flat_cost_p10", "flat_cost", "flat_cost_p90", "band_n"]]
    columns = {"flat_cost_p10": "planning_p10_eur_per_unit", "flat_cost": "planning_p50_eur_per_unit",
               "flat_cost_p90": "planning_p90_eur_per_unit", "band_n": "planning_band_n"}
    output = summary.merge(current.rename(columns=columns), on=["product", "h"], how="left", validate="one_to_one")
    if output[list(columns.values())].isna().any().any():
        raise ValueError("Flat planning forecast needs a calibrated interval at all submission horizons")
    if not np.allclose(output["planning_p50_eur_per_unit"], output[f"origin_{COST}"]):
        raise ValueError("Planning reference and current BOM cost must agree")
    scored = rows.loc[rows["outcome_available"].le(D.BOM_BASE)].dropna(
        subset=["actual_proxy_cost", "flat_cost_p10", "flat_cost_p90"]).copy()
    calibration = interval_calibration(scored, ["product", "h"], "actual_proxy_cost", "flat_cost_p10", "flat_cost_p90")
    return output, calibration


def interval_calibration(rows, keys, actual, low, high):
    checks = rows.assign(covered=rows[actual].between(rows[low], rows[high]), below=rows[actual].lt(rows[low]),
                         above=rows[actual].gt(rows[high]), width=rows[high] - rows[low])
    return (checks.groupby(keys, as_index=False).agg(n=("covered", "size"), empirical_coverage=("covered", "mean"),
             below_p10=("below", "mean"), above_p90=("above", "mean"), mean_interval_width_in_original_units=("width", "mean")))


def material_calibration(forecasts, lags):
    dates = forecasts["origin"].dt.to_period("M")
    delay = forecasts["h"] + forecasts["series"].map(lags)
    revealed = dates + delay
    rows = forecasts.loc[revealed.le(D.BOM_BASE.to_period("M"))].dropna(subset=["actual"])
    checks = interval_calibration(rows, ["series", "h"], "actual", "p10", "p90")
    return checks.assign(original_unit=checks["series"].map({key: t.unit for key, t in T.BY_ID.items()}))


def build_submission(current, summary, panel, team_id):
    if not re.fullmatch(r"team_\d{2}", team_id):
        raise ValueError("team_id must use the team_NN format from the challenge brief")
    required = T.ids("material", in_submission=True)
    selected = current.loc[current["series"].isin(required)].copy()
    if "forecast_source" not in selected:
        selected["forecast_source"] = "provided_material_forecast"
    if "selection_revision" not in selected:
        selected["selection_revision"] = ""
    units = {key: T.BY_ID[key].unit for key in required}
    fx = float(panel.at[D.BOM_BASE, "S06"])
    if not np.isfinite(fx) or fx <= 0:
        raise ValueError("A positive USD-per-EUR rate at the forecast origin is required")
    source = selected["series"].map(units)
    if not source.isin(["index", "USD/t", "EUR/t"]).all():
        raise ValueError("Forecast contains a unit without a defined transformation")
    transform = np.select([source.eq("index"), source.eq("USD/t")], [100 / selected["y_t"], 1 / fx], default=1)
    selected[QUANTILES] = selected[QUANTILES].mul(transform, axis=0)
    selected["original_unit"] = source
    selected["unit"] = np.where(source.eq("index"), INDEX_UNIT, "EUR/t")
    selected["conversion"] = np.select([source.eq("index"), source.eq("USD/t")],
                                       ["rebase_to_100_at_origin", "fixed_origin_fx_usd_per_eur"], default="identity")
    selected["usd_per_eur_at_origin"] = np.where(source.eq("USD/t"), fx, np.nan)
    selected["region"] = selected["series"].map({key: T.BY_ID[key].region for key in required})
    selected["target_type"] = "material"
    selected = selected.rename(columns={"series": "target_id", "h": "horizon_m", "model": "model_name"})

    bom = summary[["product", "h", f"origin_{COST}", "planning_p10_eur_per_unit", "planning_p50_eur_per_unit",
                   "planning_p90_eur_per_unit"]].copy()
    for quantile in QUANTILES:
        bom[quantile] = 100 * bom[f"planning_{quantile}_eur_per_unit"] / bom[f"origin_{COST}"]
    bom = bom.rename(columns={"product": "target_id", "h": "horizon_m"}).assign(
        target_type="bom_index", original_unit="EUR/unit", unit=INDEX_UNIT, region="EU",
        conversion="v1_basket_divided_by_origin_cost", model_name="v1_flat_cost+empirical_product_errors",
        forecast_source="v1_bom_flat_cost", selection_revision="",
    )
    out = pd.concat([selected, bom], ignore_index=True)
    out["team_id"] = team_id
    out["ai_tools"] = AI_DISCLOSURE
    origin = D.BOM_BASE.to_period("M")
    out["origin_month"] = str(origin)
    out["target_month"] = out["horizon_m"].map(lambda h: str(origin + int(h)))
    expected = pd.MultiIndex.from_product([required + sorted(D.PRODUCT_ID.values()), HORIZONS],
                                          names=["target_id", "horizon_m"])
    actual = pd.MultiIndex.from_frame(out[["target_id", "horizon_m"]])
    if actual.has_duplicates or not expected.difference(actual).empty or not actual.difference(expected).empty:
        raise ValueError("Submission must contain one row for every registered target and horizon")
    if (not np.isfinite(out[QUANTILES]).all().all() or not out[QUANTILES].gt(0).all().all()
            or not (out["p10"].le(out["p50"]) & out["p50"].le(out["p90"])).all()):
        raise ValueError("Submission requires finite, positive and ordered quantiles")
    trace = out[["target_type", "target_id", "horizon_m", "model_name", "original_unit", "unit", "conversion",
                 "usd_per_eur_at_origin", "forecast_source", "selection_revision"]].sort_values(
                     ["target_type", "target_id", "horizon_m"])
    return out[SUBMISSION_COLUMNS].sort_values(["target_type", "target_id", "horizon_m"]), trace


def coverage_register(bom):
    gaps = bom[["product", "material_label_source", "mass_kg", "series", "price_basis"]].copy()
    gaps["status"] = gaps["series"].notna().map({True: "submitted_proxy", False: "unpriced_bom_line"})
    gaps["evidence"] = gaps["price_basis"].map({
        "market_price_proxy": "Market proxy; validate actual purchase prices and sourcing basis",
        "assumed_level_indexed": "Assumed purchase-price level scaled by the plastics PPI",
        "unpriced": "No validated line-level price proxy or quoted benchmark in the project inputs",
    })
    scope = pd.DataFrame([
        {"product": "both", "material_label_source": "Electronics / control components", "status": "brief_category_gap",
         "evidence": "No separately mapped quantity and usable price series in the current v1 inputs"},
        {"product": "both", "material_label_source": "Insulation outside mapped epoxy/polymers", "status": "brief_category_gap",
         "evidence": "Epoxy and indexed resins are represented; other insulating or dielectric items need price and quantity"},
    ])
    return pd.concat([gaps, scope], ignore_index=True)
