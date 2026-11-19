"""EDA charts and tables -> outputs/eda/  (DECK=1: slide style -> outputs/slides/)."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

import data as D
import preprocess as P

DECK = os.environ.get("DECK") == "1"
OUT = D.OUT / ("slides" if DECK else "eda")
OUT.mkdir(parents=True, exist_ok=True)

_aptos = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/FontCache/4/CloudFonts/Aptos"
for _f in _aptos.glob("*.ttf") if _aptos.exists() else []:
    font_manager.fontManager.addfont(str(_f))

SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURF = "#ffffff" if DECK else "#fcfcfb"
PROD_COLOR = {"RM6": "#16884A", "AirSeT": "#3478B8"}
DIV_NEG, DIV_MID, DIV_POS = "#2a78d6", "#f0efec", "#e34948"

plt.rcParams.update({
    "font.family": ["Aptos", "Segoe UI", "DejaVu Sans"] if DECK else ["Segoe UI", "DejaVu Sans"],
    "font.size": 11 if DECK else 10,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "legend.frameon": False, "legend.fontsize": 9, "axes.axisbelow": True,
})
EVENTS = [("2020-03-01", "2020-06-01", "COVID"), ("2022-02-01", "2022-12-01", "Ukraine / energy shock")]


def shade(ax):
    for a, b, name in EVENTS:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=GRID, alpha=0.6, lw=0)
        ax.text(pd.Timestamp(a), 1.0, f" {name}", transform=ax.get_xaxis_transform(),
                va="top", fontsize=7.5, color=MUTED)


def end_labels(ax, series, colors):
    pts = sorted(((s.dropna().iloc[-1], k) for k, s in series.items()), reverse=True)
    ymin, ymax = ax.get_ylim()
    gap, last = (ymax - ymin) * 0.06, None
    for y, k in pts:
        if last is not None and last - y < gap:
            y = last - gap
        last = y
        x = series[k].dropna().index[-1]
        ax.plot([x], [series[k].dropna().iloc[-1]], "o", ms=4, color=colors[k])
        ax.annotate(k, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)


def title(ax, text, **kw):
    if not DECK:
        ax.set_title(text, **kw)


def note(fig, x, y, text, **kw):
    if not DECK:
        fig.text(x, y, text, **kw)


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def fig_families(rm, en, steel, epoxy):
    base = pd.Timestamp("2018-01-01")

    def reb(s):
        s = s.dropna()
        s = s[s.index >= base]
        return 100 * s / s.iloc[0]

    panels = [
        ("Base metals", {"Copper": rm["D01"], "Aluminium": rm["D02"], "Nickel": rm["S03"], "Zinc": rm["S05"]}, False),
        ("Steel", {"EU steel PPI": rm["S09"], "US steel PPI": rm["S12"], "Iron ore": rm["S04"],
                   "CR coil DE (CRU)": steel["cru_cr_de"]}, False),
        ("Plastics & insulation", {"EU plastics PPI": rm["S11"], "US plastics PPI": rm["S13"],
                                   "Epoxy resin": epoxy["epoxy"]}, False),
        ("Energy (log scale)", {"Brent": en["ED01"], "EU gas TTF": en["ED02"], "Power FR": en["E10"],
                                "Power HU": en["E11"]}, True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.2) if DECK else (13, 7.6), sharex=True)
    for ax, (name, ser, logy) in zip(axes.flat, panels):
        cols = {k: SLOT[i] for i, k in enumerate(ser)}
        reb_ser = {k: reb(v) for k, v in ser.items()}
        shade(ax)
        for k, s in reb_ser.items():
            ax.plot(s.index, s.values, color=cols[k], label=k)
        ax.axhline(100, color=AXIS, lw=1)
        if logy:
            ax.set_yscale("log")
        ax.set_title(name)
        ax.legend(loc="upper left", bbox_to_anchor=(0, 0.93), ncol=2)
        ax.set_xlim(base, pd.Timestamp("2027-04-01"))
        end_labels(ax, reb_ser, cols)
    if not DECK:
        fig.suptitle("Input prices since 2018 by family, rebased Jan-2018 = 100", x=0.01, ha="left",
                     fontsize=14, fontweight="bold", color=INK)
    note(fig, 0.01, 0.005, "Sources: IMF/FRED, Eurostat, US BLS, ECB, Ember, CRU (internal file), epoxy file. "
         "Epoxy May-2026 = ECH typo, interpolated (preprocess.py).", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    save(fig, "fig01_price_families.png")


def fig_bom_index(p):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    shade(ax)
    for col, name, c in [("RM6_NE_IQI_433520", "RM6 (SF6, copper-heavy)", PROD_COLOR["RM6"]),
                         ("RMAirSeT_IIQ_24kV", "RM AirSeT (SF6-free, steel-heavy)", PROD_COLOR["AirSeT"])]:
        ax.plot(p.index, p[col], color=c, label=name)
        pk = p[col].idxmax()
        ax.annotate(f"peak {p[col].max():.0f} ({pk:%b-%y})", (pk, p[col].max()), xytext=(8, -2),
                    textcoords="offset points", ha="left", va="top", fontsize=8, color=INK2)
    ax.axhline(100, color=AXIS, lw=1)
    title(ax, "Material cost index per unit (fixed Sep-2026 BOM weights, Sep-2026 = 100)")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.92))
    ax.set_ylabel("index")
    note(fig, 0.01, -0.02, "Weights = kg (BOM, data.zip) x EUR/t in Sep-2026. RM6 unlabelled 45% slice assumed "
         "carbon steel. Plastics EUR/t assumed.", fontsize=7.5, color=MUTED)
    save(fig, "fig02_bom_cost_index.png")


def fig_cost_structure(costs):
    order = ["Carbon steel", "Stainless steel", "Copper", "Galvanised steel", "Plastics", "Aluminium"]
    colors = dict(zip(order, SLOT))
    prods = [("RMAirSeT_IIQ_24kV", "RM AirSeT"), ("RM6_NE_IQI_433520", "RM6")]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    for yi, (pid, _) in enumerate(prods):
        g = costs[costs["product"] == pid].set_index("material")
        left = 0
        for m in [m for m in order if m in g.index]:
            v = g.at[m, "eur_per_unit"]
            ax.barh(yi, v, left=left, color=colors[m], height=0.55, edgecolor=SURF, linewidth=2, label=m)
            if v > 60:
                ax.text(left + v / 2, yi, f"{m.split()[0]}\n€{v:.0f} ({g.at[m, 'cost_share']:.0%})",
                        ha="center", va="center", fontsize=8, color=INK if colors[m] == SLOT[3] else "#ffffff")
            left += v
        ax.text(left + 8, yi, f"€{left:.0f}", va="center", fontsize=10, fontweight="bold", color=INK)
    ax.set_yticks([0, 1], [p[1] for p in prods])
    ax.set_xlabel("EUR per unit (raw-material content only)")
    ax.grid(axis="y", visible=False)
    uniq = dict(zip(*reversed(ax.get_legend_handles_labels())))
    ax.legend([uniq[m] for m in order], order, ncol=6, loc="lower left", bbox_to_anchor=(0, 1.0))
    title(ax, "Two products, two different risks: RM6 is a copper bet, AirSeT a steel bet", pad=28)
    note(fig, 0.01, -0.06, "Sep-2026 prices: copper/aluminium LME (EUR), steel CRU Germany, stainless CRU 304 (Aug-26). "
         "Plastics EUR/t assumed. RM6 unlabelled 45% slice assumed carbon steel.", fontsize=7.5, color=MUTED)
    save(fig, "fig03_cost_structure.png")


KEY_BASE = pd.Timestamp("2017-01-01")


def key_prices(rm, steel):
    """The four materials that drive BOM cost (colours match the cost-structure chart)."""
    return {"Carbon steel (CRU CR coil)": (steel["cru_cr_de"], SLOT[0]),
            "Stainless 304 (CRU)": (steel["ss304_de"], SLOT[1]),
            "Copper (LME, EUR)": (rm["D01"], SLOT[2]),
            "Plastics (EU PPI)": (rm["S11"], SLOT[4])}


def fig_key_prices(rm, steel):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    shade(ax)
    ser, cols = {}, {}
    for name, (s, c) in key_prices(rm, steel).items():
        s = s.dropna().loc[KEY_BASE:]
        ser[name], cols[name] = 100 * s / s.iloc[0], c
        ax.plot(ser[name].index, ser[name].values, color=c, label=name)
    ax.axhline(100, color=AXIS, lw=1)
    ax.set_xlim(KEY_BASE, pd.Timestamp("2027-09-01"))
    ax.set_ylabel(f"index, {KEY_BASE:%b-%Y} = 100")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.92))
    end_labels(ax, ser, {k: cols[k] for k in ser})
    title(ax, f"Key material prices, rebased {KEY_BASE:%b-%Y} = 100")
    note(fig, 0.01, -0.02, "Sources: CRU Germany (internal steel file), LME via IMF/FRED converted with ECB EUR/USD, "
         "Eurostat PPI.", fontsize=7.5, color=MUTED)
    save(fig, "fig05_key_prices.png")


def fig_volatility(p):
    ids = list(D.TARGETS) + ["RM6_NE_IQI_433520", "RMAirSeT_IIQ_24kV"]
    names = [D.TARGETS[i][0] if i in D.TARGETS else ("BOM index RM6" if "RM6" in i else "BOM index AirSeT") for i in ids]
    lp = np.log(p[ids])
    tab = pd.concat([(np.exp(lp.shift(-h) - lp) - 1).abs().median().rename(h) for h in (3, 6, 9)], axis=1)
    tab.index = names
    tab = tab.sort_values(9)
    fig, ax = plt.subplots(figsize=(9.5, 5.3) if DECK else (9, 6.2))
    y = np.arange(len(tab))
    for j, (h, c) in enumerate({3: "#86b6ef", 6: "#2a78d6", 9: "#104281"}.items()):
        ax.barh(y + (j - 1) * 0.26, tab[h] * 100, height=0.24, color=c, label=f"{h} months")
    ax.set_yticks(y, tab.index)
    ax.set_xlabel("median absolute price change over the horizon (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", title="horizon")
    title(ax, "How much does each series typically move in 3 / 6 / 9 months?")
    note(fig, 0.01, -0.01, "Nov-2016 to Sep-2026, all rolling windows.", fontsize=7.5, color=MUTED)
    save(fig, "fig04_volatility_by_horizon.png")
    tab.to_csv(OUT / "volatility_by_horizon.csv", float_format="%.4f")


def fig_price_benchmark(bm):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    hi = {"Schneider Electric": PROD_COLOR["RM6"], "Eaton": SLOT[1]}
    series = {}
    for brand, g in bm.groupby("brand"):
        g = g.sort_values("year")
        ax.plot(g["year"], g["p50"], color=hi.get(brand, AXIS), lw=2.2 if brand in hi else 1.4,
                zorder=3 if brand in hi else 2)
        series[brand] = g.set_index("year")["p50"]
    last = None
    for y, b in sorted(((s.iloc[-1], b) for b, s in series.items()), reverse=True):
        y = y if last is None or last - y > 330 else last - 330
        last = y
        ax.annotate(b + (" (SF6-free)" if b == "Eaton" else ""), (2026, y), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK if b in hi else INK2, fontweight="bold" if b in hi else "normal")
    se, et = series["Schneider Electric"], series["Eaton"]
    ax.text(0.01, 0.97, f"SF6-free premium (Eaton vs Schneider):  2025 {et[2025] / se[2025] - 1:+.0%}   ->   "
            f"2026 {et[2026] / se[2026] - 1:+.0%}", transform=ax.transAxes, va="top", fontsize=9.5, color=INK,
            fontweight="bold")
    ax.set_xlim(2017, 2027.6)
    ax.set_xticks(range(2017, 2027))
    ax.set_ylabel("EUR ex-VAT, median (P50)")
    title(ax, "3-function RMU selling price: SF6 products jump in 2026, the SF6-free premium disappears")
    note(fig, 0.01, -0.02, "Source: RMU_IQI_EU_Price_Benchmark_2017-2026_Synthetic (synthetic, yearly).",
         fontsize=7.5, color=MUTED)
    save(fig, "fig07_price_benchmark.png")


def main():
    c = P.load()  # every source in one panel: cut at 2026-09, in EUR, typos corrected (coverage: series_catalog.csv)
    p = D.panel()
    costs = D.bom_cost_table(c, c)
    costs.to_csv(OUT / "bom_cost_table.csv", index=False, float_format="%.4f")
    fig_families(c, c, c, c)
    fig_bom_index(p)
    fig_cost_structure(costs)
    fig_key_prices(c, c)
    fig_volatility(p)
    fig_price_benchmark(D.load_price_benchmark())


if __name__ == "__main__":
    main()
