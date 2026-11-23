"""fig09 production chain with measured links, fig10 methodology."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import data as D
import preprocess as P
from eda import AXIS, DECK, INK, INK2, MUTED, PROD_COLOR, SLOT, SURF, save, title

AHEAD_MIN = 0.3


LINKS = {  # (upstream node, downstream node) -> (upstream series, downstream series, label)
    ("brent", "chem"): ("ED01", "benzene", "Brent -> benzene"),
    ("brent", "plast"): ("ED01", "S11", "Brent -> plastics PPI EU"),
    ("chem", "epoxy"): ("benzene", "epoxy", "benzene -> epoxy resin"),
    ("gas", "alu_m"): ("ED02", "S08", "gas TTF -> aluminium PPI EU"),
    ("power", "alu_m"): ("E09", "S08", "power DE -> aluminium PPI EU"),
    ("gas", "cs"): ("ED02", "S09", "gas TTF -> steel PPI EU"),
    ("ore", "cs"): ("S04", "cru_cr_de", "iron ore -> CR coil (CRU)"),
    ("zinc", "hdg"): ("S05", "cru_hdg_de", "zinc -> HDG coil (CRU)"),
    ("cs", "hdg"): ("cru_cr_de", "cru_hdg_de", "CR coil -> HDG coil (CRU)"),
    ("nickel", "ss"): ("S03", "ss304_alloy_de", "nickel -> stainless 304 alloy surcharge"),
    ("cu_lme", "cu_m"): ("D01", "S07", "copper LME -> copper PPI EU"),
    ("al_lme", "alu_m"): ("D02", "S08", "aluminium LME -> aluminium PPI EU"),
}


def link_table():
    """Both measures for every link of the chain. "next 3 m": corr(last 3-m upstream change, next 3-m downstream
    change), no overlap; "same month": corr of monthly changes. The chart shows "next 3 m" when it is >= AHEAD_MIN,
    else "same month". Each pair on its common months, up to the cutoff."""
    df = P.load()  # cut at the forecast origin, ECH typo corrected
    L = np.log(df.where(df > 0))
    rows = {}
    for k, (a, b, label) in LINKS.items():
        ahead = (L[b].shift(-3) - L[b]).corr(L[a] - L[a].shift(3))
        same = L[b].diff().corr(L[a].diff())
        both = L[[a, b]].dropna()
        rows[k] = {"link": label, "upstream": a, "downstream": b, "next 3 m r": ahead, "same month r": same,
                   "shown": "next 3 m" if ahead >= AHEAD_MIN else "same month",
                   "months": f"{both.index[0]:%b-%y} - {both.index[-1]:%b-%y}"}
    return pd.DataFrame.from_dict(rows, orient="index")


def measured_links():
    """(kind, r) per link, as drawn: ("ahead", next-3-m r) if it is >= AHEAD_MIN, else ("same", same-month r)."""
    t = link_table()
    return {k: ("ahead", r["next 3 m r"]) if r["shown"] == "next 3 m" else ("same", r["same month r"])
            for k, r in t.iterrows()}


K = 1.3 if DECK else 1.0  # larger text in slide style
BOX_H = 0.95 if DECK else 0.78

def box(ax, x, y, text, w=1.9, h=None, fc="#ffffff", ec=AXIS, tc=INK, bold=False, fs=8.5):
    h = BOX_H if h is None else h
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.2, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs * K, color=tc, zorder=4,
            fontweight="bold" if bold else "normal", linespacing=1.0 if DECK else 1.25)


def arrow(ax, p0, p1, lw=1.5, color=INK2, ls="-", label=None, lpos=0.5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=10, lw=lw, color=color, ls=ls,
                                 shrinkA=2, shrinkB=2, zorder=2))
    if label:
        x = p0[0] + (p1[0] - p0[0]) * lpos
        y = p0[1] + (p1[1] - p0[1]) * lpos
        ax.text(x, y, label, ha="center", va="center", fontsize=7 * K, color=INK, zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc=SURF, ec="none"))


def link_style(r):
    if r >= 0.6:
        return 2.6, INK, "-"
    if r >= 0.4:
        return 1.6, INK2, "-"
    return 1.0, MUTED, (0, (3, 2))


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
        "hdg": (X["mat"], 0.9, "Galvanised HDG = CR + zinc\n(CRU / PPI S10)" if DECK else "Galvanised HDG\n(CRU / PPI S10)"),
    }
    fig, ax = plt.subplots(figsize=(15, 6.2) if DECK else (15, 8.6))
    ax.set_xlim(-0.2, 11.0)
    ax.set_ylim(-0.6, 11.0)
    ax.axis("off")

    heads = [("energy", "1  ENERGY", "input to every process"),
             ("up", "2  UPSTREAM COMMODITIES", "exchange-traded: move first,\nhard to predict" if not DECK else "exchange-traded, move first"),
             ("mat", "3  MATERIALS SCHNEIDER BUYS", "react in the same month\nor over the next months" if not DECK else "react same month or later"),
             ("prod", "4  PRODUCTS", "share of material cost")]
    for k, t, sub in heads:
        ax.text(X[k], 10.65, t, ha="center", fontsize=9.5 * K, fontweight="bold", color=INK)
        ax.text(X[k], 10.25, sub.replace("\n", " ") if DECK else sub, ha="center", va="top", fontsize=7.8 * K, color=MUTED, linespacing=1.2)

    for k, (x, y, t) in nodes.items():
        box(ax, x, y, t, w=W)

    def edge(a, b, lpos=0.5):
        (x0, y0, _), (x1, y1, _) = nodes[a], nodes[b]
        kind, r = L[(a, b)]
        lw, col, ls = link_style(r)
        tag = "next 3 m" if kind == "ahead" else "same month"
        arrow(ax, (x0 + W / 2, y0), (x1 - W / 2, y1), lw=lw, color=col, ls=ls, lpos=lpos,
              label=f"{tag} · r={r:.2f}")

    edge("brent", "chem", lpos=0.45)
    edge("brent", "plast")
    edge("chem", "epoxy")
    edge("al_lme", "alu_m")
    edge("gas", "alu_m", lpos=0.24)
    edge("power", "alu_m", lpos=0.24)
    edge("gas", "cs", lpos=0.14)
    edge("cu_lme", "cu_m")
    edge("nickel", "ss")
    edge("ore", "cs")
    edge("zinc", "hdg")
    _, r = L[("cs", "hdg")]
    lw, col, ls = link_style(r)
    x, y0, _ = nodes["cs"]
    arrow(ax, (x, y0 - BOX_H / 2), (x, nodes["hdg"][1] + BOX_H / 2), lw=lw, color=col, label=None)
    if DECK:
        ax.text(x + 0.1, (y0 + nodes["hdg"][1]) / 2, f"same month · r={r:.2f}", fontsize=7 * K, color=INK,
                va="center", ha="left", zorder=5, bbox=dict(boxstyle="round,pad=0.1", fc=SURF, ec="none"))
    else:
        ax.text(x + 0.12, (y0 + nodes["hdg"][1]) / 2, f"HDG = CR coil + zinc coat\nsame month · r={r:.2f}",
                fontsize=7 * K, color=INK, va="center", ha="left")

    costs = D.bom_cost_table(D.load_rm(), D.load_steel_licensed())
    mat_node = {"Carbon steel": "cs", "Stainless steel": "ss", "Copper": "cu_m", "Galvanised steel": "hdg",
                "Plastics": "plast", "Aluminium": "alu_m"}
    prods = [("RM6_NE_IQI_433520", "RM6  (SF6)", 6.6, PROD_COLOR["RM6"]),
             ("RMAirSeT_IIQ_24kV", "RM AirSeT  (SF6-free)", 3.0, PROD_COLOR["AirSeT"])]
    for pid, pname, py, pc in prods:
        g = costs[costs["product"] == pid].sort_values("cost_share", ascending=False)
        lines = "\n".join(f"{r.material}  {r.cost_share:.0%}" for r in g.itertuples() if r.cost_share >= 0.015)
        box(ax, X["prod"], py, f"{pname}\n{lines}" if DECK else f"{pname}\n\n{lines}", w=2.1,
            h=2.9 if DECK else 2.3, ec=pc, fs=7.4 if DECK else 8)
        for r in g.itertuples():
            x0, y0, _ = nodes[mat_node[r.material]]
            arrow(ax, (x0 + W / 2, y0), (X["prod"] - 1.05, py + (y0 - py) * 0.15), lw=0.6 + 6 * r.cost_share,
                  color=pc)
    ax.text(nodes["epoxy"][0], nodes["epoxy"][1] - (0.7 if DECK else 0.58), "insulation parts; not in BOM shares", fontsize=7 * K, color=MUTED, ha="center")

    if not DECK:
        ax.text(-0.15, -0.45, "next 3 m: corr(last 3-month change upstream, next 3-month change downstream), no "
            "overlap. same month: corr of monthly changes. Nov-2016 to Sep-2026. Thick r >= 0.6 · thin 0.4-0.6 · "
            "dashed < 0.4. Product arrows: width = share of material cost.",
            fontsize=7.5, color=MUTED)
    title(ax, "Production chain: which upstream moves help predict the next 3 months?",
                 fontsize=14, fontweight="bold", color=INK, loc="left", pad=6)
    return fig


def method_numbers():
    per = pd.read_csv(D.OUT / "backtest" / "scores_per_series.csv")
    e3 = per[(per.model == "ets_local+emp") & (per.h == 3)].set_index("series")["rel_mae"]
    win = e3[["S08", "S09", "S10", "RMAirSeT_IIQ_24kV"]]
    ahead = [r for kind, r in measured_links().values() if kind == "ahead" and r >= 0.4]
    return 1 - win.max(), 1 - win.min(), min(ahead), max(ahead)


def fig_method():
    g_lo, g_hi, r_lo, r_hi = method_numbers()
    fig, ax = plt.subplots(figsize=(15, 6.4))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6.6)
    ax.axis("off")
    steps = [
        (1.25, 3.3, "1  DATA\n\n30+ monthly series\nprices, indices, energy\nall converted to EUR"),
        (8.9, 3.3, "3  MOCK EXAM\n(backtest 2020-2026)\n\nevery month: forecast\nwith past data only,\ncompare with reality.\nKeep a model only if it\nbeats 'price stays\nthe same'"),
        (11.25, 3.3, "4  FORECAST\n\nP10 / P50 / P90\nper material\nat +3 / +6 / +9 m"),
        (13.6, 4.6, "5  COST\n\nx BOM kg\n-> RM6 / AirSeT\ncost index"),
        (13.6, 1.75, "6  QUOTE RULE\n\nprice adjustment,\nvalidity, indexation\ntrigger, risk buffer"),
    ]
    for x, y, t in steps:
        box(ax, x, y, t, w=2.1, h=4.6 if "MOCK" in t else (2.4 if x > 13 else 3.0), fs=7.9 if DECK else 8.5)
    lanes = [
        (4.95, 4.85, SLOT[0], "2a  UPSTREAM MARKET PRICES",
         "copper, aluminium, nickel, zinc,\niron ore, Brent (LME / exchanges)",
         "behave like a random walk:\nnobody beats 'no change' (efficient market)\n-> P50 = today's price  (+ futures / analysts)\n-> effort goes into the RANGE (P10-P90)"),
        (4.95, 1.75, SLOT[1], "2b  DOWNSTREAM INDICES",
         "EU steel & aluminium PPIs, BOM cost indices\n(plastics, copper & US PPIs: decided by backtest)",
         f"smooth, with inertia (own momentum)\n-> trend model (ETS) beats 'no change'\n   by {g_lo:.0%}-{g_hi:.0%} at 3 months"
         f"\n-> next: add upstream drivers (r {r_lo:.1f}-{r_hi:.1f})"),
    ]
    for x, y, c, head, who, how in lanes:
        ax.add_patch(FancyBboxPatch((x - 2.55, y - 1.35), 5.1, 2.7, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc="#ffffff", ec=c, lw=2, zorder=3))
        ax.text(x - 2.4, y + 1.12, head, fontsize=9.5 * K, fontweight="bold", color=INK, va="top", zorder=4)
        ax.text(x - 2.4, y + 0.72, who, fontsize=8 * K, color=INK2, va="top", zorder=4, linespacing=1.2)
        ax.text(x - 2.4, y - 0.05, how, fontsize=8 * K, color=INK, va="top", zorder=4, linespacing=1.25)
    arrow(ax, (2.3, 3.6), (2.4, 4.6), lw=1.6)
    arrow(ax, (2.3, 3.0), (2.4, 2.0), lw=1.6)
    arrow(ax, (7.5, 4.85), (7.85, 4.0), lw=1.6)
    arrow(ax, (7.5, 1.75), (7.85, 2.6), lw=1.6)
    ax.text(5.12, 3.3, "predicts next 3 m" if DECK else
            "upstream changes help predict downstream indices 3 months ahead", fontsize=7.5 * K,
            color=MUTED, ha="left", va="center")
    arrow(ax, (4.95, 3.5), (4.95, 3.1), lw=1.2, color=MUTED, ls=(0, (3, 2)))
    arrow(ax, (9.95, 3.3), (10.2, 3.3), lw=1.6)
    arrow(ax, (12.3, 3.7), (12.55, 4.3), lw=1.6)
    arrow(ax, (13.6, 3.4), (13.6, 2.95), lw=1.6)
    if not DECK:
        ax.text(0.15, 6.35, "Method: one pipeline, two kinds of series, the mock exam decides",
            fontsize=14, fontweight="bold", color=INK)
        ax.text(0.15, 0.1, "Same code path for every series; only the forecasting rule differs by type. "
            "The backtest picks the rule per series, so every choice is evidence-based.", fontsize=8, color=MUTED)
    return fig


if __name__ == "__main__":
    save(fig_chain(), "fig09_production_chain.png")
    save(fig_method(), "fig10_methodology.png")
