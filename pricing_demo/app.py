import math

import streamlit as st

from pricing_model import (
    COSTS, PRICE_TICKS, PRODUCTS, Buyer, Offer, cents, constraints, evaluate,
    minimum_price, recommend, surplus, switch_prices,
)


CHOICE_COLORS = {"RM6": "#89bbf4", "RM AirSeT": "#71d4bb", "No buy": "#d9dde4"}
BUYER_NAMES = ("Buyer A", "Buyer B")

st.set_page_config(page_title="RM6 & AirSeT | Quote simulator", page_icon="📊", layout="wide")


def euros(value, scale=100):
    if value is None:
        return "—"
    amount = value / scale
    return f"−€{-amount:,.2f}" if amount < 0 else f"€{amount:,.2f}"


@st.cache_data(show_spinner=False, max_entries=128)
def best_quote(buyers, states, rate, gap):
    return recommend(buyers, states, rate, gap)


def set_demo(months=3, premium_b=120.0):
    st.session_state.update({"horizon": months, "a_base": 12250.0, "a_premium": 40.0,
                             "b_base": 12250.0, "b_premium": premium_b,
                             "a_units": 1, "b_units": 1, "a_eligible": "Both products",
                             "b_eligible": "Both products", "floor": 85.0, "gap": 1.0})


def buyer_inputs(name):
    prefix = name[-1].lower()
    st.markdown(f"**{name}**")
    base, premium = st.columns(2)
    b = base.number_input("B (€)", 0.0, 50000.0, step=50.0, format="%.2f",
                          key=f"{prefix}_base", help="Willingness to pay for RM6 per unit.")
    w = premium.number_input("W (€)", -10000.0, 10000.0, step=10.0, format="%.2f", key=f"{prefix}_premium",
                             help="Extra willingness to pay for AirSeT per unit. AirSeT WTP = B + W.")
    return name, prefix, b, w


def decision_map(buyer, offer):
    """Draw the continuous price regions through the algebraic breakpoints."""
    b, w = buyer.base / 100, buyer.premium / 100
    r = offer.rm6 / 100 if offer.rm6 is not None else None
    a = offer.airset / 100 if offer.airset is not None else None
    x_refs = [b, r if r is not None else b]
    y_refs = [b + w, a if a is not None else b + w]
    pad = max(140, 0.14 * max(max(x_refs) - min(x_refs), max(y_refs) - min(y_refs)))
    xlo, xhi = max(0, math.floor(min(x_refs) - pad)), math.ceil(max(x_refs) + pad)
    ylo, yhi = max(0, math.floor(min(y_refs) - pad)), math.ceil(max(y_refs) + pad)
    x_axis = {"field": "rm6", "type": "quantitative", "title": "RM6 quote Pᵣ (€ / unit)",
              "scale": {"domain": [xlo, xhi], "nice": False, "zero": False},
              "axis": {"format": ",.0f", "grid": True}}
    y_axis = {"field": "airset", "type": "quantitative", "title": "AirSeT quote Pₐ (€ / unit)",
              "scale": {"domain": [ylo, yhi], "nice": False, "zero": False},
              "axis": {"format": ",.0f", "grid": True}}
    layers = []

    def area(choice, start, stop, threshold, below):
        if start >= stop:
            return
        breaks = sorted({start, stop, *(v for v in (b, ylo - w, yhi - w) if start < v < stop)})
        points = []
        for x in breaks:
            cutoff = max(ylo, min(yhi, threshold(x)))
            points.append({"rm6": x, "airset": ylo if below else cutoff,
                           "other": cutoff if below else yhi, "choice": choice})
        layers.append({"data": {"values": points},
                       "mark": {"type": "area", "opacity": 0.55, "clip": True},
                       "encoding": {"y2": {"field": "other"}, "color": {
                           "field": "choice", "type": "nominal", "title": None,
                           "scale": {"domain": list(CHOICE_COLORS), "range": list(CHOICE_COLORS.values())},
                           "legend": {"orient": "top", "direction": "horizontal"}},
                           "tooltip": [{"field": "choice", "title": "Customer choice"}]}})

    if buyer.can_buy_rm6 and buyer.can_buy_airset:
        area("RM AirSeT", xlo, xhi, lambda x: w + min(b, x), below=True)
        area("RM6", xlo, min(b, xhi), lambda x: w + x, below=False)
        area("No buy", max(b, xlo), xhi, lambda x: b + w, below=False)
        boundaries = [[(xlo, w + xlo), (b, w + b), (xhi, w + b)], [(b, w + b), (b, yhi)]]
    elif buyer.can_buy_rm6:
        area("RM6", xlo, min(b, xhi), lambda x: ylo, below=False)
        area("No buy", max(b, xlo), xhi, lambda x: ylo, below=False)
        boundaries = [[(b, ylo), (b, yhi)]]
    else:
        area("RM AirSeT", xlo, xhi, lambda x: b + w, below=True)
        area("No buy", xlo, xhi, lambda x: b + w, below=False)
        boundaries = [[(xlo, b + w), (xhi, b + w)]]

    for boundary in boundaries:
        layers.append({"data": {"values": [{"rm6": x, "airset": y, "order": i}
                                             for i, (x, y) in enumerate(boundary)]},
                       "mark": {"type": "line", "color": "#475569", "strokeDash": [4, 4], "clip": True},
                       "encoding": {"order": {"field": "order"}}})
    if r is not None and a is not None:
        layers.append({"data": {"values": [{"rm6": r, "airset": a}]},
                       "mark": {"type": "point", "filled": True, "color": "#172238", "size": 135,
                                "stroke": "white", "strokeWidth": 1.5},
                       "encoding": {"tooltip": [{"field": "rm6", "title": "RM6 quote", "format": ",.2f"},
                                                {"field": "airset", "title": "AirSeT quote", "format": ",.2f"}]}})

    st.vega_lite_chart(spec={"encoding": {"x": x_axis, "y": y_axis}, "layer": layers,
                            "config": {"view": {"stroke": None}, "axis": {"gridOpacity": 0.22}}},
                       width="stretch", height=300)
    st.caption("The map assumes both quotes are posted and uses the selected buyer’s eligibility. "
               "The dot marks the displayed quote pair. At a positive tie, RM6 wins. "
               "At zero surplus, the buyer walks away.")


def show_quote(result, buyers, states, rate, gap):
    st.subheader(result.offer.menu)
    metrics = st.columns(5)
    metrics[0].metric("RM6 quote", euros(result.offer.rm6) if result.offer.rm6 is not None else "Not quoted")
    metrics[1].metric("AirSeT quote", euros(result.offer.airset) if result.offer.airset is not None else "Not quoted")
    metrics[2].metric("Revenue", euros(result.revenue))
    metrics[3].metric("P50 contribution", euros(result.contributions[0], 100 * PRICE_TICKS))
    metrics[4].metric("P90 contribution", euros(result.contributions[1], 100 * PRICE_TICKS))

    customers, chart = st.columns([0.95, 1.25], gap="large")
    with customers:
        st.subheader("Customer decisions")
        for buyer, choice in zip(buyers, result.choices):
            u_r, u_a = surplus(buyer, result.offer)
            with st.container(border=True):
                who, decision = st.columns([1, 1.15])
                who.markdown(f"**{buyer.name}**")
                decision.markdown(f"**{choice}** · {buyer.units if choice != 'No buy' else 0} sold")
                st.caption(f"RM6 surplus {euros(u_r)}  ·  AirSeT surplus {euros(u_a)}")
        st.caption(f"{result.units[0]} RM6 and {result.units[1]} AirSeT units sold. Quotes are per unit, excluding VAT.")
    with chart:
        st.subheader("Price and choice map")
        selected = st.radio("Buyer shown on map", BUYER_NAMES, horizontal=True, label_visibility="collapsed")
        decision_map(buyers[BUYER_NAMES.index(selected)], result.offer)

    with st.expander("BOM costs and seller constraints"):
        checks = constraints(result.offer, states, rate, gap)
        rows = []
        for label, state, passed in zip(("P50", "P90"), states, checks):
            margins = [100 * (1 - cost / (price * PRICE_TICKS)) if price is not None else None
                       for price, cost in zip(result.offer.prices, state)]
            spread = abs(margins[0] - margins[1]) if None not in margins else None
            rows.append({"Scenario": label, "RM6 BOM": euros(state[0], 100 * PRICE_TICKS),
                         "AirSeT BOM": euros(state[1], 100 * PRICE_TICKS),
                         "RM6 rate": f"{margins[0]:.2f}%" if margins[0] is not None else "—",
                         "AirSeT rate": f"{margins[1]:.2f}%" if margins[1] is not None else "—",
                         "Rate gap": f"{spread:.2f} pp" if spread is not None else "—",
                         "Checks": "Pass" if False not in passed else "Fail"})
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption("Contribution = revenue − priced BOM costs. Both scenarios use the same quotes and units.")


def show_switch(buyers, states, rate, gap, offer):
    with st.expander("AirSeT switch at a fixed RM6 quote · slide 10", expanded=True):
        anchor = offer.rm6 if offer.rm6 is not None else max(1, max(b.base for b in buyers) - 1)
        st.caption(f"RM6 held at {euros(anchor)}. Each row compares the premium needed for a feasible, "
                   "profitable switch with the highest premium the buyer would accept.")
        anchor_feasible = all(anchor >= minimum_price(state[0], rate) for state in states)
        if not anchor_feasible:
            st.warning("The RM6 anchor fails the minimum priced-BOM rate in at least one cost scenario.")
        rows = []
        for buyer in buyers:
            required, seller_max, buyer_max, margin_floor, recovery_floor = switch_prices(buyer, anchor, states, rate, gap)
            limit = "Rate constraints" if margin_floor >= recovery_floor else "Contribution gain"
            if buyer_max is None:
                status = "Ineligible for AirSeT"
            elif not anchor_feasible:
                status = "RM6 anchor fails minimum rate"
            elif seller_max is not None and required > seller_max:
                status = "Outside seller’s price band"
            else:
                status = "Switch possible" if buyer_max >= required else "Below required premium"
            rows.append({"Buyer": buyer.name, "Required premium": euros(required - anchor),
                         "Buyer ceiling": euros(buyer_max - anchor if buyer_max is not None else None),
                         "Lower bound set by": limit, "At this RM6 quote": status})
        st.dataframe(rows, hide_index=True, width="stretch")


if "horizon" not in st.session_state:
    set_demo()

with st.sidebar:
    st.header("Scenario")
    with st.expander("Presentation presets"):
        st.button("Slide baseline · 3 months", on_click=set_demo)
        st.button("Longer delivery · 6 months", on_click=set_demo, args=(6,))
        st.button("Higher AirSeT WTP · 6 months", on_click=set_demo, args=(6, 400.0))
    months = st.radio("Delivery horizon", (3, 6, 9), horizontal=True,
                      format_func=lambda value: f"{value} mo", key="horizon")
    st.subheader("Customer inputs")
    first = buyer_inputs(BUYER_NAMES[0])
    second = buyer_inputs(BUYER_NAMES[1])
    st.caption("B is RM6 willingness to pay. W is the extra willingness to pay for AirSeT.")
    with st.expander("Units and product eligibility"):
        details = []
        for name, prefix, _, _ in (first, second):
            st.markdown(f"**{name}**")
            u, e = st.columns([0.7, 1.3])
            units = u.number_input("Units", 1, 1000, key=f"{prefix}_units")
            eligibility = e.selectbox("Eligible offers", ("Both products", "RM6 only", "RM AirSeT only"),
                                      key=f"{prefix}_eligible")
            details.append((units, eligibility))
    with st.expander("Seller constraints"):
        rate = cents(st.number_input("Minimum priced-BOM rate (%)", 0.0, 99.99, step=0.5, key="floor"))
        gap = cents(st.number_input("Maximum rate gap (pp)", 0.0, 100.0, step=0.1, key="gap"))

buyers = tuple(Buyer(name, cents(b), cents(w), units, eligible)
               for (name, _, b, w), (units, eligible) in zip((first, second), details))
states = COSTS[months]

st.title("RM6 & AirSeT quote simulator")
st.markdown("Change either buyer’s willingness to pay and see how the quotes and purchases respond.")
if any(buyer.airset_value < 0 for buyer in buyers):
    st.error("For each buyer, enter B and W such that B + W is at least €0.")
    st.stop()

mode = st.radio("Quote mode", ("Optimize quotes", "Test a fixed quote"), horizontal=True,
                label_visibility="collapsed")
if mode == "Test a fixed quote":
    manual_menu, rm6_field, airset_field = st.columns(3)
    menu = manual_menu.selectbox("Products offered", ("Both products", "RM6 only", "RM AirSeT only"))
    rm6 = rm6_field.number_input("RM6 quote (€)", 0.01, 100000.0, 12249.99, 10.0, "%.2f",
                                 disabled=menu == "RM AirSeT only", key="manual_rm6")
    airset = airset_field.number_input("AirSeT quote (€)", 0.01, 100000.0, 12369.98, 10.0, "%.2f",
                                       disabled=menu == "RM6 only", key="manual_airset")
    offer = Offer(cents(rm6) if menu != "RM AirSeT only" else None,
                  cents(airset) if menu != "RM6 only" else None)
    result = evaluate(offer, buyers, states)
    failed = [label for label, checks in zip(("P50", "P90"), constraints(offer, states, rate, gap))
              if False in checks]
    if failed:
        st.warning(f"This quote fails the seller constraints in {' and '.join(failed)}. "
                   "Expand BOM costs and seller constraints for the scenario rates.")
else:
    result = best_quote(buyers, states, rate, gap)
    st.caption("One quote decision maximizes worst-case contribution and satisfies the seller constraints at both P50 and P90.")
    if result.offer.menu == "No sale":
        st.info("No eligible buyer accepts a quote that meets the seller constraints.")

show_quote(result, buyers, states, rate, gap)
show_switch(buyers, states, rate, gap, result.offer)
