"""Page 2: material cost per unit at the origin and at +3 / +6 / +9 months, split by material."""
import altair as alt
import streamlit as st

import loaders as L

st.set_page_config(page_title="BOM cost", layout="wide")
st.title("BOM cost per unit")
st.caption("What does one unit cost in materials, and which material drives the risk?")
L.require("product_costs", "material_costs", "material_exposure")

c1, c2 = st.columns(2)
product = c1.radio("Product", list(L.PRODUCTS), format_func=lambda p: L.PRODUCTS[p][0], horizontal=True)
h = c2.radio("Horizon", L.HORIZONS, format_func=lambda m: f"+{m} months", horizontal=True)
name, colour = L.PRODUCTS[product]

cost = L.read("product_costs").query("product == @product and h == @h").iloc[0]
origin = cost["origin_material_cost_eur_per_unit"]
st.subheader(f"{name}, material cost per unit in {cost['target_month']} (EUR)")
m = st.columns(5)
m[0].metric(f"At origin ({L.ORIGIN:%b %Y})", f"€{origin:,.0f}")
for col, q in zip(m[1:4], ("p10", "p50", "p90"), strict=True):
    value = cost[f"planning_{q}_eur_per_unit"]
    col.metric(f"Planning {q.upper()}", f"€{value:,.0f}", f"{value - origin:+,.0f} €", delta_color="inverse")
stress = cost["high_stress_material_cost_eur_per_unit"]
m[4].metric("All materials at P90", f"€{stress:,.0f}", f"{stress - origin:+,.0f} €", delta_color="inverse")
st.caption(
    "Planning P10 / P50 / P90: the cost at flat prices with a range from past errors of this product's cost at the same "
    "horizon (the basis of the quote policy). All materials at P90: every material at its own P90 in the same month, "
    f"a joint stress. Priced lines cover {cost['priced_mass_share']:.0%} of the unit's {cost['total_mass_kg']:,.0f} kg; "
    f"{cost['unpriced_mass_kg']:,.0f} kg have no price series and are not in these costs."
)

# ---------- per-material split ----------
def priced(frame):
    return frame.loc[frame["product"].eq(product) & frame["h"].eq(h) & frame["series"].notna()]


lines, exposure = priced(L.read("material_costs")), priced(L.read("material_exposure"))
by_material = (lines.groupby("material_label_source", as_index=False)
               .agg(origin_cost=("origin_material_cost_eur_per_unit", "sum"), mass_kg=("mass_kg", "sum"))
               .merge(exposure.groupby("material_label_source", as_index=False)["stress_cost_change_eur_per_unit"].sum(),
                      on="material_label_source", validate="one_to_one")
               .rename(columns={"material_label_source": "material", "stress_cost_change_eur_per_unit": "p90_increase"})
               .sort_values("p90_increase", ascending=False))
total_increase = stress - origin
top = by_material.iloc[0]
st.markdown(
    f"#### {top['material']} drives €{top['p90_increase']:,.0f} of the €{total_increase:,.0f} cost increase per "
    f"{name} if all materials reach their P90 together in {cost['target_month']} "
    f"({top['p90_increase'] / total_increase:.0%} of it)."
)

order = by_material["material"].tolist()
bar = dict(color=colour, cornerRadiusEnd=4, height={"band": 0.7})
y = alt.Y("material:N", sort=order, title=None, axis=alt.Axis(labelLimit=260))
tooltip = ["material:N", alt.Tooltip("mass_kg:Q", format=",.1f", title="kg per unit"),
           alt.Tooltip("origin_cost:Q", format=",.0f", title="cost at origin (€)"),
           alt.Tooltip("p90_increase:Q", format=",.0f", title="increase at P90 (€)")]
left, right = st.columns(2)
left.altair_chart(alt.Chart(by_material).mark_bar(**bar).encode(
    alt.X("origin_cost:Q", title="Cost at origin, € per unit"), y, tooltip=tooltip)
    .properties(title="Cost per unit today", height=36 * len(by_material) + 40), width="stretch")
right.altair_chart(alt.Chart(by_material).mark_bar(**bar).encode(
    alt.X("p90_increase:Q", title="Increase at P90, € per unit"), y, tooltip=tooltip)
    .properties(title=f"Increase if each material reaches its P90 (+{h} m)", height=36 * len(by_material) + 40),
    width="stretch")
st.caption(
    f"The material increases add up to the all-materials P90 increase (€{total_increase:,.0f}), not to the planning "
    f"P90 increase (€{cost['planning_p90_eur_per_unit'] - origin:,.0f}), which comes from past errors of the whole "
    "product cost and cannot be split by material."
)
with st.expander("Table"):
    st.dataframe(by_material.rename(columns={"mass_kg": "kg per unit", "origin_cost": "cost at origin (€)",
                                             "p90_increase": "increase at P90 (€)"}),
                 hide_index=True, width="stretch")
