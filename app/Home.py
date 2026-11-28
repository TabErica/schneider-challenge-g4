"""Decision tool, home page. Run: uv run streamlit run app/Home.py"""
import streamlit as st

import loaders as L

st.set_page_config(page_title="RMU cost & quote tool", layout="wide")
st.title("RMU material cost and quote decision tool")
st.write(
    "Raw-material price forecasts for the RM6 and RM AirSeT ring main units, the material cost per unit they imply, "
    "and the quote rules that follow. Every number comes from the team's forecasting pipeline; this tool only reads "
    "its results and re-runs the quote rules on your inputs. It trains no model and runs no backtest."
)

absent = L.missing(*L.FILES)
if absent:
    st.error(f"Pipeline outputs not found. {L.RUN_FIRST}")
    st.caption("Missing: " + ", ".join(f"`outputs/{p}`" for p in absent))
    st.stop()

panel = L.read("panel", "month")
c1, c2, c3 = st.columns(3)
c1.metric("Forecast origin", f"{L.ORIGIN:%b %Y}")
c2.metric("Last month of data", f"{panel['month'].max():%b %Y}")
c3.metric("Horizons", " / ".join(f"+{h} m" for h in L.HORIZONS))

st.subheader("How to use it")
st.markdown(
    """
1. **Material forecasts**: how far could each raw-material price move in 3, 6 or 9 months? History, the P10 / P50 / P90
   range of the submitted forecast, the model behind it and why it was chosen. A warning flags a P90 above the
   largest rise of the same length seen in the data.
2. **BOM cost**: what does one RM6 or RM AirSeT unit cost in materials today and at each horizon, and which material
   drives the risk?
3. **Quote simulator**: for a product, a delivery horizon and your selling price, what price uplift, validity, risk
   reserve and indexation trigger does the quote policy recommend, and how much margin is at risk? Compare with a
   simple pass-through of the expected cost change.

P10 / P50 / P90: 10% / 50% / 90% quantiles of the forecast. There is a 1-in-10 chance the price ends above P90.
"""
)
st.caption(
    "Prices are monthly averages; material cost covers the priced lines of the v1 bill of materials only. Selling "
    "prices default to a synthetic benchmark (not like-for-like). Upload your own commercial inputs on page 3."
)
