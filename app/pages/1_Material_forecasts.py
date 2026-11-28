"""Page 1: history and submitted P10 / P50 / P90 forecast of one target."""
import altair as alt
import pandas as pd
import streamlit as st

import loaders as L

st.set_page_config(page_title="Material forecasts", layout="wide")
st.title("Material forecasts")
st.caption("How far could a raw-material price move in 3, 6 or 9 months?")
L.require("forecast", "trace", "panel", "origin_forecasts", "models", "plausibility", "bom_index")

targets = L.forecast_targets()
target = st.selectbox("Target", list(targets), format_func=lambda t: f"{targets[t]} ({t})")
is_bom = target in L.PRODUCTS

rows = L.read("forecast").query("target_id == @target").sort_values("horizon_m")
unit = rows["unit"].iat[0]
if is_bom:
    history, anchor_month, anchor = L.bom_history(target)
    conversion = "v1_basket_divided_by_origin_cost"
    colour = L.PRODUCTS[target][1]
else:
    history, anchor_month, anchor, conversion = L.material_history(target)
    colour = L.NEUTRAL

# ---------- chart ----------
start = st.slider("History from", 2008, L.ORIGIN.year - 1, 2018)
history = history.loc[history["month"].dt.year.ge(start)]
fan = pd.concat([
    pd.DataFrame({"month": [anchor_month], "p10": [anchor], "p50": [anchor], "p90": [anchor], "horizon": ["origin"]}),
    rows.assign(month=pd.to_datetime(rows["target_month"]), horizon="+" + rows["horizon_m"].astype(str) + " m")
    [["month", "p10", "p50", "p90", "horizon"]],
], ignore_index=True)

fmt = ",.1f" if unit.startswith("index") else ",.0f"
x = alt.X("month:T", title=None)
history_line = alt.Chart(history).mark_line(color=colour, strokeWidth=2).encode(
    x, alt.Y("value:Q", title=unit, scale=alt.Scale(zero=False)),
    tooltip=[alt.Tooltip("month:T", format="%b %Y"), alt.Tooltip("value:Q", format=fmt, title=unit)])
band = alt.Chart(fan).mark_area(color=colour, opacity=0.18).encode(x, alt.Y("p10:Q"), alt.Y2("p90:Q"))
median = alt.Chart(fan).mark_line(color=colour, strokeWidth=2, strokeDash=[5, 3], point=alt.OverlayMarkDef(size=60))
median = median.encode(x, alt.Y("p50:Q"), tooltip=[
    "horizon:N", alt.Tooltip("month:T", format="%b %Y"),
    *[alt.Tooltip(f"{q}:Q", format=fmt, title=q.upper()) for q in ("p10", "p50", "p90")]])
origin_rule = alt.Chart(pd.DataFrame({"month": [L.ORIGIN]})).mark_rule(color="#9AA0A6", strokeDash=[2, 2]).encode(x)
st.altair_chart((band + history_line + median + origin_rule).properties(height=380), width="stretch")
st.caption(f"Solid line: history. Dashed line: P50. Shaded fan: P10 to P90. Vertical line: origin {L.ORIGIN:%b %Y}.")
if conversion == "fixed_origin_fx_usd_per_eur":
    fx = L.read("trace").query("target_id == @target")["usd_per_eur_at_origin"].iat[0]
    st.caption(f"USD price shown in EUR: history at each month's ECB rate, forecast at the {L.ORIGIN:%b %Y} rate "
               f"({fx:.4f} USD per EUR). The forecast range carries no exchange-rate risk.")
elif conversion == "rebase_to_100_at_origin":
    st.caption(f"Index rebased to 100 at the {L.ORIGIN:%b %Y} value.")

# ---------- plausibility ----------
if is_bom:
    st.info("No plausibility check for the BOM index: its band comes from past errors of the product cost itself.")
else:
    check = L.read("plausibility").query("series == @target").sort_values("h")
    for r in check.loc[check["p90_above_max_rise"].astype(bool)].itertuples():
        st.warning(f"**+{r.h} months: P90 is above every past move of that length.** The forecast P90 is "
                   f"{r.p90_pct:+.0%} vs the origin; the largest {r.h}-month rise in the data is "
                   f"{r.hist_max_rise:+.0%} ({r.n_windows} windows). Treat this upper bound with judgement.")

# ---------- table ----------
table = rows[["horizon_m", "target_month", "p10", "p50", "p90", "unit", "model_name"]].copy()
if is_bom:
    table["why this point forecast"] = "Flat material prices (P50 = cost at the origin) on the v1 bill of materials"
    table["why this range"] = "Empirical P10 / P90 of past product-cost errors at the same horizon"
else:
    reasons = L.read("models").query("series == @target")[["h", "point_reason", "interval_reason"]]
    table = table.merge(reasons, left_on="horizon_m", right_on="h", how="left").drop(columns="h").rename(
        columns={"point_reason": "why this point forecast", "interval_reason": "why this range"})
    check = L.read("plausibility").query("series == @target")[["h", "p90_pct", "hist_max_rise"]]
    table = table.merge(check, left_on="horizon_m", right_on="h", how="left").drop(columns="h").rename(
        columns={"p90_pct": "P90 vs origin", "hist_max_rise": "largest past rise"})
table = table.rename(columns={"horizon_m": "horizon (months)", "target_month": "month", "p10": "P10", "p50": "P50",
                              "p90": "P90", "model_name": "model (point + range)"})
pct = st.column_config.NumberColumn(format="percent")
st.dataframe(table, hide_index=True, width="stretch", column_config={
    **{q: st.column_config.NumberColumn(format="%.1f" if unit.startswith("index") else "%.0f") for q in ("P10", "P50", "P90")},
    "P90 vs origin": pct, "largest past rise": pct})
