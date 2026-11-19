"""Two explanatory diagrams for the slides:
  fig09  production chain: energy -> upstream commodities -> materials SE buys -> products,
         each link annotated with the measured lead time and correlation (3-month changes).
  fig10  methodology in one picture.

Run:  python src/diagrams.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import data as D
from eda import AXIS, INK, INK2, MUTED, SLOT, SURF, save

# ------------------------------------------------------------------------------------------ link stats
def measured_links():
    rm, en, st = D.load_rm(), D.load_energy(), D.load_steel_licensed()
    ep, _ = D.load_epoxy()
    ep = ep.copy()
    ep.loc[pd.Timestamp("2026-05-01"), ["ech", "epoxy"]] = np.nan  # known typo month
    df = rm.join(en[[c for c in en.columns if c not in rm.columns]]).join(st).join(ep).loc[:"2026-10-01"]
    l3 = np.log(df).diff(3)

    def best(a, b):
        cs = {k: l3[b].corr(l3[a].shift(k)) for k in range(0, 7)}
        k = max(cs, key=cs.get)
        return k, cs[k]

    pairs = {
        ("brent", "chem"): ("ED01", "benzene"), ("brent", "plast"): ("ED01", "S11"),
        ("chem", "epoxy"): ("benzene", "epoxy"), ("gas", "alu_m"): ("ED02", "S08"),
        ("power", "alu_m"): ("E09", "S08"), ("gas", "cs"): ("ED02", "S09"),
        ("ore", "cs"): ("S04", "cru_cr_de"), ("zinc", "hdg"): ("S05", "cru_hdg_de"),
        ("cs", "hdg"): ("cru_cr_de", "cru_hdg_de"), ("nickel", "ss"): ("S03", "ss304_alloy_de"),
        ("cu_lme", "cu_m"): ("D01", "S07"), ("al_lme", "alu_m"): ("D02", "S08"),
    }
    return {k: best(*v) for k, v in pairs.items()}


# ------------------------------------------------------------------------------------------ drawing helpers
def box(ax, x, y, text, w=1.9, h=0.78, fc="#ffffff", ec=AXIS, tc=INK, bold=False, fs=8.5):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.2, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc, zorder=4,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, p0, p1, lw=1.5, color=INK2, ls="-", label=None, rad=0.0, lpos=0.5):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=10, lw=lw, color=color, ls=ls,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2, zorder=2)
    ax.add_patch(a)
    if label:
        x = p0[0] + (p1[0] - p0[0]) * lpos
        y = p0[1] + (p1[1] - p0[1]) * lpos
        ax.text(x, y, label, ha="center", va="center", fontsize=7, color=INK, zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc=SURF, ec="none"))


def link_style(r):
    if r >= 0.6:
        return 2.6, INK, "-"
    if r >= 0.4:
        return 1.6, INK2, "-"
    return 1.0, MUTED, (0, (3, 2))


# ------------------------------------------------------------------------------------------ fig09
def fig_chain():
    L = measured_links()
    X = {"energy": 0.9, "up": 3.6, "mat": 6.7, "prod": 9.7}
    W = 1.9
    nodes = {
        "brent": (X["energy"], 9.2, "Crude oil\n(Brent)"),
        "gas": (X["energy"], 5.7, "Natural gas\n(EU TTF)"),
        "power": (X["energy"], 7.4, "Electricity\n(DE / FR / HU)"),
        "chem": (X["up"], 7.9, "Benzene, ECH\n(petrochemicals)"),
        "al_lme": (X["up"], 6.5, "Aluminium\n(LME)"),
        "cu_lme": (X["up"], 5.1, "Copper\n(LME)"),
        "nickel": (X["up"], 3.7, "Nickel\n(LME)"),
        "ore": (X["up"], 2.3, "Iron ore"),
        "zinc": (X["up"], 0.9, "Zinc\n(LME)"),
        "plast": (X["mat"], 9.2, "Plastics PET/PBT\n(EU PPI S11)"),
        "epoxy": (X["mat"], 7.9, "Epoxy resin\n(insulation)"),
        "alu_m": (X["mat"], 6.5, "Aluminium parts\n(EU PPI S08)"),
        "cu_m": (X["mat"], 5.1, "Copper busbars\n(EU PPI S07)"),
        "ss": (X["mat"], 3.7, "Stainless 304\n(CRU; alloy surcharge)"),
        "cs": (X["mat"], 2.3, "Carbon steel CR coil\n(CRU / EU PPI S09)"),
        "hdg": (X["mat"], 0.9, "Galvanised HDG\n(CRU / PPI S10)"),
    }
    fig, ax = plt.subplots(figsize=(15, 8.6))
    ax.set_xlim(-0.2, 11.0)
    ax.set_ylim(-0.6, 11.0)
    ax.axis("off")

    heads = [("energy", "1  ENERGY", "input to every process"),
             ("up", "2  UPSTREAM COMMODITIES", "exchange-traded: move first,\nhard to predict"),
             ("mat", "3  MATERIALS SCHNEIDER BUYS", "follow upstream\n0-2 months later"),
             ("prod", "4  PRODUCTS", "share of material cost")]
    for k, t, sub in heads:
        ax.text(X[k], 10.65, t, ha="center", fontsize=9.5, fontweight="bold", color=INK)
        ax.text(X[k], 10.25, sub, ha="center", va="top", fontsize=7.8, color=MUTED, linespacing=1.2)

    for k, (x, y, t) in nodes.items():
        box(ax, x, y, t, w=W)

    def edge(a, b, rad=0.0, lpos=0.5, label_extra=""):
        (x0, y0, _), (x1, y1, _) = nodes[a], nodes[b]
        k, r = L[(a, b)]
        lw, col, ls = link_style(r)
        lag = "same month" if k == 0 else f"+{k} m"
        arrow(ax, (x0 + W / 2, y0), (x1 - W / 2, y1), lw=lw, color=col, ls=ls, rad=rad, lpos=lpos,
              label=f"{lag} · r={r:.2f}{label_extra}")

    edge("brent", "chem", lpos=0.45)
    (x0, y0, _), (x1, y1, _) = nodes["brent"], nodes["plast"]
    k, r = L[("brent", "plast")]
    lw, col, ls = link_style(r)
    arrow(ax, (x0 + W / 2, y0), (x1 - W / 2, y1), lw=lw, color=col, ls=ls, label=f"+{k} m · r={r:.2f}", lpos=0.5)
    edge("chem", "epoxy")
    edge("al_lme", "alu_m")
    edge("gas", "alu_m", lpos=0.24)
    edge("power", "alu_m", lpos=0.24)
    edge("gas", "cs", lpos=0.14)
    edge("cu_lme", "cu_m")
    edge("nickel", "ss")
    edge("ore", "cs")
    edge("zinc", "hdg")
    # CR coil -> HDG (vertical, same column)
    k, r = L[("cs", "hdg")]
    lw, col, ls = link_style(r)
    x, y0, _ = nodes["cs"]
    arrow(ax, (x, y0 - 0.39), (x, nodes["hdg"][1] + 0.39), lw=lw, color=col, label=None)
    ax.text(x + 0.12, (y0 + nodes["hdg"][1]) / 2, f"HDG = CR coil + zinc coat\nsame month · r={r:.2f}",
            fontsize=7, color=INK, va="center", ha="left")

    # products
    costs = D.bom_cost_table(D.load_rm(), D.load_steel_licensed())
    mat_node = {"Carbon steel": "cs", "Stainless steel": "ss", "Copper": "cu_m", "Galvanised steel": "hdg",
                "Plastics": "plast", "Aluminium": "alu_m"}
    prods = [("RM6_NE_IQI_433520", "RM6  (SF6)", 6.6, SLOT[0]), ("RMAirSeT_IIQ_24kV", "RM AirSeT  (SF6-free)", 3.0, SLOT[1])]
    for pid, pname, py, pc in prods:
        g = costs[costs["product"] == pid].sort_values("cost_share", ascending=False)
        lines = "\n".join(f"{r.material}  {r.cost_share:.0%}" for r in g.itertuples() if r.cost_share >= 0.015)
        box(ax, X["prod"], py, f"{pname}\n\n{lines}", w=2.1, h=2.3, ec=pc, bold=False, fs=8)
        for r in g.itertuples():
            x0, y0, _ = nodes[mat_node[r.material]]
            arrow(ax, (x0 + W / 2, y0), (X["prod"] - 1.05, py + (y0 - py) * 0.15), lw=0.6 + 6 * r.cost_share,
                  color=pc, rad=0.0)
    # epoxy not in BOM
    ax.text(nodes["epoxy"][0], nodes["epoxy"][1] - 0.58, "insulation parts; not in BOM shares", fontsize=7, color=MUTED, ha="center")

    ax.text(-0.15, -0.45, "Lead and r = best correlation of 3-month % changes, upstream leading 0-6 months "
            "(Nov-2016 to Oct-2026). Thick solid r >= 0.6 · thin solid 0.4-0.6 · dashed < 0.4. "
            "Product arrows: width = share of material cost (Sep-2026 prices, BOM zip version).",
            fontsize=7.5, color=MUTED)
    ax.set_title("Production chain: price shocks travel downstream with a 0-2 month delay",
                 fontsize=14, fontweight="bold", color=INK, loc="left", pad=6)
    save(fig, "fig09_production_chain.png")


# ------------------------------------------------------------------------------------------ fig10
def fig_method():
    fig, ax = plt.subplots(figsize=(15, 6.4))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6.6)
    ax.axis("off")
    steps = [
        (1.25, 3.3, "1  DATA\n\n30+ monthly series\nprices, indices, energy\nall converted to EUR", "#ffffff", AXIS),
        (8.9, 3.3, "3  MOCK EXAM\n(backtest 2020-2026)\n\nevery month: forecast\nwith past data only,\ncompare with reality.\nKeep a model only if it\nbeats 'price stays\nthe same'", "#ffffff", AXIS),
        (11.25, 3.3, "4  FORECAST\n\nP10 / P50 / P90\nper material\nat +3 / +6 / +9 m", "#ffffff", AXIS),
        (13.6, 4.6, "5  COST\n\nx BOM kg\n-> RM6 / AirSeT\ncost index", "#ffffff", AXIS),
        (13.6, 1.75, "6  QUOTE RULE\n\nprice adjustment,\nvalidity, indexation\ntrigger, risk buffer", "#ffffff", AXIS),
    ]
    for x, y, t, fc, ec in steps:
        h = 4.6 if "MOCK" in t else (2.4 if x > 13 else 3.0)
        box(ax, x, y, t, w=2.1, h=h, fc=fc, ec=ec, fs=8.5)
    # two lanes
    lanes = [
        (4.95, 4.85, SLOT[0], "2a  UPSTREAM MARKET PRICES",
         "copper, aluminium, nickel, zinc,\niron ore, Brent (LME / exchanges)",
         "behave like a random walk:\nnobody beats 'no change' (efficient market)\n-> P50 = today's price  (+ futures / analysts)\n-> effort goes into the RANGE (P10-P90)"),
        (4.95, 1.75, SLOT[1], "2b  DOWNSTREAM INDICES",
         "EU steel, plastics, aluminium,\ncopper PPI · BOM cost index",
         "smooth, with inertia, and follow upstream\n1-2 months later\n-> trend model (ETS) beats 'no change'\n   by 10-25%  + upstream prices as inputs"),
    ]
    for x, y, c, title, who, how in lanes:
        ax.add_patch(FancyBboxPatch((x - 2.55, y - 1.35), 5.1, 2.7, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc="#ffffff", ec=c, lw=2, zorder=3))
        ax.text(x - 2.4, y + 1.12, title, fontsize=9.5, fontweight="bold", color=INK, va="top", zorder=4)
        ax.text(x - 2.4, y + 0.72, who, fontsize=8, color=INK2, va="top", zorder=4, linespacing=1.2)
        ax.text(x - 2.4, y - 0.05, how, fontsize=8, color=INK, va="top", zorder=4, linespacing=1.25)
    arrow(ax, (2.3, 3.6), (2.4, 4.6), lw=1.6)
    arrow(ax, (2.3, 3.0), (2.4, 2.0), lw=1.6)
    arrow(ax, (7.5, 4.85), (7.85, 4.0), lw=1.6)
    arrow(ax, (7.5, 1.75), (7.85, 2.6), lw=1.6)
    ax.text(7.62, 3.3, "upstream\nfeeds\ndownstream", fontsize=7, color=MUTED, ha="center", va="center")
    arrow(ax, (4.95, 3.5), (4.95, 3.1), lw=1.2, color=MUTED, ls=(0, (3, 2)))
    arrow(ax, (9.95, 3.3), (10.2, 3.3), lw=1.6)
    arrow(ax, (12.3, 3.7), (12.55, 4.3), lw=1.6)
    arrow(ax, (13.6, 3.4), (13.6, 2.95), lw=1.6)
    ax.text(0.15, 6.35, "Method: one pipeline, two kinds of series, the mock exam decides",
            fontsize=14, fontweight="bold", color=INK)
    ax.text(0.15, 0.1, "Same code path for every series; only the forecasting rule differs by type. "
            "The backtest picks the rule per series, so every choice is evidence-based.", fontsize=8, color=MUTED)
    save(fig, "fig10_methodology.png")


if __name__ == "__main__":
    fig_chain()
    fig_method()
