"""Page 3: quote decision for one product, horizon and selling price, with the pipeline's quote policy."""
import io
import math

import numpy as np
import pandas as pd
import streamlit as st

import loaders as L

st.set_page_config(page_title="Quote simulator", layout="wide")
st.title("Quote simulator")
st.caption("What price, validity, reserve and indexation clause should a fixed-price offer carry, and what margin is "
           "at risk?")
L.require("product_costs", "pricing_revenue", "bom_index")

from pricing_scenarios import BENCHMARK_ID, COMMERCIAL_COLUMNS, KEY, load_commercial_inputs
from quote_policy import OfferAssumptions

SCENARIOS = {
    "downside": "Stress (max of planning P90 and all-materials P90)",
    "baseline": "Baseline (P50)",
    "upside": "Upside (P10)",
}
summary = L.read("product_costs")

# ---------- inputs ----------
with st.sidebar:
    st.header("Offer")
    product = st.radio("Product", list(L.PRODUCTS), format_func=lambda p: L.PRODUCTS[p][0])
    h = st.radio("Delivery horizon", L.HORIZONS, format_func=lambda m: f"+{m} months")
    scenario = st.radio("Cost scenario priced into the offer", ["upside", "baseline", "downside"], index=1,
                        format_func=SCENARIOS.get)
    uploaded = st.file_uploader(
        "Commercial inputs (optional CSV)", type="csv",
        help="Same format as `src/cost_pricing.py --commercial-inputs`: columns product, target_month (YYYY-MM) and any "
             "of units, base_price_eur_ex_vat, additional_direct_cost_eur_per_unit. Its values fill the fields below.")

target_month = summary.query("product == @product and h == @h")["target_month"].iat[0]
csv_inputs = None
if uploaded is not None:
    try:
        csv_inputs = load_commercial_inputs(uploaded, summary)
    except ValueError as error:
        st.sidebar.error(f"Commercial inputs rejected: {str(error).split(': ', 1)[-1]}")
csv_row = {}
if csv_inputs is not None:
    match = csv_inputs.loc[csv_inputs["product"].eq(product) & csv_inputs["target_month"].eq(target_month)]
    csv_row = {} if match.empty else match.iloc[0].dropna().to_dict()

benchmark = L.synthetic_benchmark().set_index("id")["p50"]
benchmark_price = round(float(benchmark[BENCHMARK_ID[product]]), 2)
key = f"{product}_{h}_{uploaded.file_id if uploaded is not None else 'none'}"  # new defaults when these change
with st.sidebar:
    base_price = st.number_input("Base selling price (€ per unit, ex VAT)", min_value=0.01, step=100.0, format="%.2f",
                                 value=float(csv_row.get("base_price_eur_ex_vat", benchmark_price)), key=f"price_{key}",
                                 help=f"Default: synthetic benchmark P50 ({benchmark_price:,.2f} €), not like-for-like.")
    units = st.number_input("Quantity (units)", min_value=0, step=1, key=f"units_{key}",
                            value=int(csv_row["units"]) if "units" in csv_row else None,
                            help="Optional. Needed for totals.")
    extra_cost = st.number_input("Additional direct cost (€ per unit)", min_value=0.0, step=10.0, key=f"extra_{key}",
                                 value=csv_row.get("additional_direct_cost_eur_per_unit"),
                                 help="Optional: direct costs outside the priced BOM lines.")
    pass_through = st.slider("Pass-through of the expected (P50) cost change", 0, 100, 50, 5, format="%d%%") / 100
    with st.expander("Quote-policy limits"):
        default = OfferAssumptions()
        revenue_loss = st.number_input("Max worst-case revenue loss from the uplift (%)", 0.0, 99.0,
                                       100 * default.revenue_loss_limit, 0.1)
        elasticity = st.number_input("Worst price elasticity", -10.0, 0.0, default.elasticity_floor, 0.1)
        wait_budget = st.number_input("Waiting-risk budget (% of price)", 0.0, 99.0, 100 * default.waiting_risk_budget,
                                      0.1)
        erosion = st.number_input("Margin erosion before indexation (% of price)", 0.0, 99.0,
                                  100 * default.margin_erosion_limit, 0.1)

# ---------- pipeline functions on these inputs ----------
inputs = csv_inputs.copy() if csv_inputs is not None else pd.DataFrame(columns=KEY + COMMERCIAL_COLUMNS)
inputs = inputs.loc[~(inputs["product"].eq(product) & inputs["target_month"].eq(target_month))]
row = {"product": product, "target_month": target_month, "units": units,
       "base_price_eur_ex_vat": np.nan if math.isclose(base_price, benchmark_price) else base_price,
       "additional_direct_cost_eur_per_unit": extra_cost}
inputs = pd.concat([inputs, pd.DataFrame([row])], ignore_index=True).dropna(how="all", subset=COMMERCIAL_COLUMNS)
try:
    # Through the pipeline's own reader: same validation as --commercial-inputs.
    commercial = load_commercial_inputs(io.StringIO(inputs.to_csv(index=False)) if len(inputs) else None, summary)
    assumptions = OfferAssumptions(revenue_loss / 100, elasticity, wait_budget / 100, erosion / 100)
    pricing, validity, offers = L.quote_tables(commercial, pass_through, assumptions)
except ValueError as error:
    st.error(f"These inputs cannot be priced: {str(error).split(': ', 1)[-1]}")
    st.stop()

offer = offers.query("product == @product and h == @h and scenario == @scenario").iloc[0]
name = L.PRODUCTS[product][0]


def total(per_unit):
    return f"€{per_unit * units:,.0f} in total" if units else "enter a quantity for the total"


st.markdown(f"**{name}, delivery in {target_month} (+{h} months)**: base price €{offer['base_price_eur_ex_vat']:,.2f} "
            f"({offer['base_price_source'].replace('_', ' ')}), material cost at origin "
            f"€{offer['base_bom_cost_eur_per_unit']:,.0f}, scenario cost €{offer['scenario_bom_cost_eur_per_unit']:,.0f}.")

# ---------- recommended offer ----------
st.subheader("Recommended offer (quote policy)")
c = st.columns(4)
c[0].metric("Price uplift", f"€{offer['price_adjustment_eur']:+,.2f}",
            f"{offer['price_adjustment_eur'] / offer['base_price_eur_ex_vat']:+.2%} → €{offer['illustrative_offer_eur_ex_vat']:,.2f}",
            delta_color="off")
c[1].metric("Quotation validity", f"{offer['selected_validity_calendar_months']:.0f} months",
            f"issue to delivery ≤ {offer['maximum_calendar_months_issue_to_delivery']:.0f} months", delta_color="off")
c[2].metric("Indexation trigger", f"index {offer['index_trigger_level']:.1f}",
            f"cost +€{offer['index_trigger_eur_per_unit']:,.2f} per unit vs origin", delta_color="off")
c[3].metric("Margin at risk", f"€{offer['cost_stress_left_on_fixed_offer_eur_per_unit']:,.2f} per unit",
            total(offer["cost_stress_left_on_fixed_offer_eur_per_unit"]), delta_color="off")
c = st.columns(4)
c[0].metric("Risk buffer: cost above P50", f"€{offer['horizon_reserve_above_planning_eur_per_unit']:,.2f}",
            help="Scenario cost minus the planning P50 cost (0 for the baseline and upside scenarios).")
c[1].metric("Risk buffer: waiting reserve", f"€{offer['wait_risk_reserve_eur_per_unit']:,.2f}",
            help="P90 of past cost rises over the validity window: the customer may accept late.")
c[2].metric("Uncovered after indexation", f"€{offer['cost_stress_left_after_safe_surcharge_eur_per_unit']:,.2f}",
            total(offer["cost_stress_left_after_safe_surcharge_eur_per_unit"]), delta_color="off")
c[3].metric("Worst-case revenue loss", f"{offer['worst_case_revenue_loss_fraction']:.2%}",
            f"limit {offer['revenue_loss_limit_fraction']:.2%}", delta_color="off")
stress_rise = offer["downside_cost_change_with_wait_reserve_eur_per_unit"]
st.info(
    f"**Margin at risk** is the part of the stress cost rise that this offer price does not cover: stress cost "
    f"(€{max(offer['planning_product_p90_eur_per_unit'], offer['simultaneous_material_p90_eur_per_unit']):,.0f}) + "
    f"waiting reserve (€{offer['wait_risk_reserve_eur_per_unit']:,.2f}) − cost at origin "
    f"(€{offer['base_bom_cost_eur_per_unit']:,.0f}) = €{stress_rise:,.2f} per unit, minus the price uplift "
    f"(€{offer['price_adjustment_eur']:,.2f}). It is measured **before** the indexation clause applies (price fixed). "
    f"If the BOM index (origin = 100) reaches {offer['index_trigger_level']:.1f}, the clause adds a revenue-safe "
    f"surcharge of €{offer['revenue_safe_surcharge_at_downside_eur_per_unit']:,.2f} at the stress level, which leaves "
    f"€{offer['cost_stress_left_after_safe_surcharge_eur_per_unit']:,.2f} per unit uncovered."
)
st.caption("Uplift: the smallest whole-cent increase that covers the scenario cost rise plus the waiting reserve, "
           "capped so that worst-case revenue (elasticity down to the floor) falls by no more than the limit. "
           "Validity: the longest window whose waiting reserve fits the budget. Trigger: cost rise beyond the "
           "uplift plus the tolerated margin erosion.")

# ---------- pass-through ----------
st.subheader("Alternative: simple pass-through")
alt_row = pricing.loc[pricing["product"].eq(product) & pricing["h"].eq(h) & pricing["policy"].eq(L.CUSTOM_POLICY)].iloc[0]
uncovered = alt_row["high_stress_cost_change_net_of_price_eur_per_unit"]
c = st.columns(4)
c[0].metric("Price change", f"€{alt_row['selling_price_change_eur']:+,.2f}",
            f"{pass_through:.0%} of the P50 change (€{alt_row['central_material_cost_change_eur_per_unit']:+,.2f})",
            delta_color="off")
c[1].metric("Selling price", f"€{alt_row['selling_price_eur_ex_vat']:,.2f}")
c[2].metric("Margin at risk", f"€{max(uncovered, 0):,.2f} per unit", total(max(uncovered, 0)), delta_color="off")
c[3].metric("Volume loss that keeps revenue flat", f"{-alt_row['break_even_volume_change_for_revenue']:.2%}")
st.caption(
    f"Margin at risk here: the all-materials P90 cost increase "
    f"(€{offer['simultaneous_material_p90_eur_per_unit'] - offer['base_bom_cost_eur_per_unit']:,.2f}) minus the price "
    "change, with no waiting reserve and no indexation clause. Its stress basis differs from the quote policy's, so "
    "compare the two with care."
    + (f" The price change covers that increase with €{-uncovered:,.2f} to spare." if uncovered < 0 else "")
)

with st.expander("All quote-policy fields for this offer"):
    st.dataframe(offer.rename("value").astype(str), width="stretch")
