"""Shared readers of the decision tool. They read the pipeline outputs; nothing here trains or backtests a model."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUT = ROOT / "outputs"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))  # as the pipeline runs (cwd = src); the app has no module named like one in src/

import data as D  # (src/data.py: constants only, no file is read at import)
import targets as T

ORIGIN = D.BOM_BASE
HORIZONS = (3, 6, 9)
RUN_FIRST = "Run `uv run python main.py` first."
PRODUCTS = {  # product id -> display name, colour (repo convention)
    "RM6_NE_IQI_433520": ("RM6", "#16884A"),
    "RMAirSeT_IIQ_24kV": ("RM AirSeT", "#3478B8"),
}
BOM_LABELS = {pid: f"{name} material cost index (v1 BOM)" for pid, (name, _) in PRODUCTS.items()}
NEUTRAL = "#5B6670"  # single-series colour for the material charts

FILES = {
    "forecast": f"cost_pricing/team_04_forecast_{ORIGIN:%Y%m}.csv",
    "trace": "cost_pricing/submission_target_trace.csv",
    "panel": "preprocessed/panel_monthly.csv",
    "origin_forecasts": "backtest/forecasts_at_origin.csv",
    "models": "selection/submission_models.csv",
    "plausibility": "selection/band_plausibility.csv",
    "bom_index": "cost_pricing/eto_bom_index.csv",
    "product_costs": "cost_pricing/product_costs.csv",
    "material_costs": "cost_pricing/material_costs.csv",
    "material_exposure": "cost_pricing/material_exposure.csv",
    "pricing_revenue": "cost_pricing/pricing_revenue.csv",
    "holdout_scores": "cost_pricing/product_holdout_scores.csv",
}


def missing(*keys):
    return [FILES[k] for k in keys if not (OUT / FILES[k]).is_file()]


def require(*keys):
    """Stop the page with a clear message when a pipeline output is missing."""
    absent = missing(*keys)
    if absent:
        st.error(f"Pipeline outputs not found. {RUN_FIRST}")
        st.caption("Missing: " + ", ".join(f"`outputs/{p}`" for p in absent))
        st.stop()


@st.cache_data(show_spinner=False)
def read(key, parse_month=None):
    kw = {"parse_dates": [parse_month]} if parse_month else {}
    return pd.read_csv(OUT / FILES[key], dtype={"target_month": str}, **kw)


# ---------- page 1: material forecasts ----------

def forecast_targets():
    """The 18 submitted materials (registry order) + the 2 submitted v1 BOM indices, as {id: label}."""
    submitted = read("forecast")
    materials = [i for i in T.ids("material", in_submission=True) if i in set(submitted["target_id"])]
    boms = [i for i in PRODUCTS if i in set(submitted.loc[submitted["target_type"].eq("bom_index"), "target_id"])]
    return {**{i: T.name(i) for i in materials}, **{i: BOM_LABELS[i] for i in boms}}


@st.cache_data(show_spinner=False)
def material_history(target_id):
    """History in the unit of the submitted forecast, plus the anchor point at the origin.

    Conversion follows submission_target_trace.csv: identity, index rebased to 100 at the origin anchor, or USD -> EUR.
    USD history uses the monthly ECB rate of preprocess.convert_fx (S06; WM_FX before S06 starts), so past values are
    the EUR prices of their month; the submitted forecast uses the fixed origin rate, equal to the monthly rate at
    the origin, so both meet at the anchor.
    """
    trace = read("trace").query("target_id == @target_id and horizon_m == 3").iloc[0]
    panel = read("panel", "month").set_index("month")
    anchor_row = read("origin_forecasts", "anchor").query("series == @target_id").iloc[0]
    y_t, anchor_month = float(anchor_row["y_t"]), anchor_row["anchor"]
    values = panel[target_id]
    if trace["conversion"] == "rebase_to_100_at_origin":
        values, anchor = 100 * values / y_t, 100.0
    elif trace["conversion"] == "fixed_origin_fx_usd_per_eur":
        fx = panel["S06"].combine_first(panel["WM_FX"])
        values, anchor = values / fx, y_t / float(trace["usd_per_eur_at_origin"])
    elif trace["conversion"] == "identity":
        anchor = y_t
    else:
        raise ValueError(f"Unknown conversion {trace['conversion']!r} for {target_id}")
    history = values.dropna().rename("value").rename_axis("month").reset_index()
    return history, pd.Timestamp(anchor_month), anchor, trace["conversion"]


@st.cache_data(show_spinner=False)
def bom_history(product):
    """v1 BOM basket index (origin = 100) from eto_bom_index.csv; the origin row is the offer anchor (with nowcast)."""
    rows = read("bom_index", "month").query("product == @product")
    return rows[["month", "bom_index"]].rename(columns={"bom_index": "value"}), ORIGIN, 100.0


# ---------- page 3: quote simulator ----------

def synthetic_benchmark():
    """Benchmark price table in the shape of data.load_price_benchmark(), rebuilt from pricing_revenue.csv."""
    revenue = read("pricing_revenue")
    quantiles = {f"synthetic_benchmark_{q}_eur_ex_vat": q for q in ("p10", "p50", "p90")}
    return (revenue[["benchmark_id", "benchmark_brand", "benchmark_year", *quantiles]].drop_duplicates()
            .rename(columns={"benchmark_id": "id", "benchmark_brand": "brand", "benchmark_year": "year", **quantiles})
            .reset_index(drop=True))


@st.cache_data(show_spinner=False)
def waiting_reserves():
    from quote_policy import waiting_risk

    return waiting_risk(read("bom_index", "month"))


CUSTOM_POLICY = "custom_pass_through"


def quote_tables(inputs, pass_through=None, assumptions=None):
    """Re-run the pipeline's pricing and quote-policy functions (cost_pricing.run) on the saved product costs.

    inputs: output of pricing_scenarios.load_commercial_inputs. pass_through: optional extra policy
    `custom_pass_through` (share of the P50 cost change passed to the price). Returns pricing, validity, offers.
    """
    from pricing_scenarios import commercial_scenarios, default_policies
    from quote_policy import OfferAssumptions, make_offers, validity_choices

    assumptions = OfferAssumptions() if assumptions is None else assumptions
    policies = default_policies()
    if pass_through is not None:
        policies = pd.concat([policies, pd.DataFrame({"policy": [CUSTOM_POLICY], "pass_through": [pass_through]})],
                             ignore_index=True)
    summary = read("product_costs")
    pricing = commercial_scenarios(summary, synthetic_benchmark(), inputs, policies)
    validity = validity_choices(pricing, waiting_reserves(), assumptions)
    return pricing, validity, make_offers(pricing, summary, validity, assumptions)
