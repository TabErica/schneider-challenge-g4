"""Exploratory data analysis: writes slide-ready PNGs + summary CSVs to outputs/eda/.

Run:  python src/eda.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data as D

OUT = D.OUT / "eda"
OUT.mkdir(parents=True, exist_ok=True)

# Reference palette (dataviz skill, light mode) - categorical slots in fixed order.
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
DIV_NEG, DIV_MID, DIV_POS = "#2a78d6", "#f0efec", "#e34948"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "legend.frameon": False, "legend.fontsize": 9, "axes.axisbelow": True,
})
EVENTS = [("2020-03-01", "2020-06-01", "COVID"), ("2022-02-01", "2022-12-01", "Ukraine / energy shock")]


def shade(ax, label=True):
    for a, b, name in EVENTS:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=GRID, alpha=0.6, lw=0)
        if label:
            ax.text(pd.Timestamp(a), 1.0, f" {name}", transform=ax.get_xaxis_transform(),
                    va="top", fontsize=7.5, color=MUTED)


def end_labels(ax, series: dict, colors: dict):
    """Direct labels at the right end of each line (text in ink, not series color)."""
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


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# ------------------------------------------------------------------------------------------
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
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.6), sharex=True)
    for ax, (title, ser, logy) in zip(axes.flat, panels):
        cols = {k: SLOT[i] for i, k in enumerate(ser)}
        reb_ser = {k: reb(v) for k, v in ser.items()}
        shade(ax)
        for k, s in reb_ser.items():
            ax.plot(s.index, s.values, color=cols[k], label=k)
        ax.axhline(100, color=AXIS, lw=1)
        if logy:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.legend(loc="upper left", bbox_to_anchor=(0, 0.93), ncol=2)
        ax.set_xlim(base, pd.Timestamp("2027-04-01"))
        end_labels(ax, reb_ser, cols)
    fig.suptitle("Ten years of input prices, rebased Jan-2018 = 100", x=0.01, ha="left",
                 fontsize=14, fontweight="bold", color=INK)
    fig.text(0.01, 0.005, "Sources: IMF/FRED, Eurostat, US BLS, ECB, Ember, CRU (internal file), epoxy file. "
             "Epoxy May-2026 ECH typo left in place.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    save(fig, "fig01_price_families.png")


def fig_bom_index(p):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    shade(ax)
    for i, (col, name) in enumerate([("RM6_NE_IQI_433520", "RM6 (SF6, copper-heavy)"),
                                     ("RMAirSeT_IIQ_24kV", "RM AirSeT (SF6-free, steel-heavy)")]):
        ax.plot(p.index, p[col], color=SLOT[i], label=name)
        pk = p[col].idxmax()
        ax.annotate(f"peak {p[col].max():.0f} ({pk:%b-%y})", (pk, p[col].max()), xytext=(8, -2),
                    textcoords="offset points", ha="left", va="top", fontsize=8, color=INK2)
    ax.axhline(100, color=AXIS, lw=1)
    ax.set_title("Material cost index per unit (fixed Sep-2026 BOM weights, Sep-2026 = 100)")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.92))
    ax.set_ylabel("index")
    fig.text(0.01, -0.02, "Weights = kg (BOM, zip version) x EUR/t level in Sep-2026. RM6 unlabelled 45% slice assumed "
             "carbon steel. Plastics EUR/t assumed.", fontsize=7.5, color=MUTED)
    save(fig, "fig02_bom_cost_index.png")


def fig_cost_structure(costs):
    order = ["Carbon steel", "Stainless steel", "Copper", "Galvanised steel", "Plastics", "Aluminium"]
    colors = dict(zip(order, SLOT))
    prods = [("RMAirSeT_IIQ_24kV", "RM AirSeT"), ("RM6_NE_IQI_433520", "RM6")]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    for yi, (pid, pname) in enumerate(prods):
        g = costs[costs["product"] == pid].set_index("material")
        left = 0
        for m in order:
            if m not in g.index:
                continue
            v = g.at[m, "eur_per_unit"]
            ax.barh(yi, v, left=left, color=colors[m], height=0.55, edgecolor=SURF, linewidth=2,
                    label=m if yi == 1 or m not in costs[costs["product"] == prods[1][0]]["material"].values else None)
            if v > 60:
                txt_col = INK if colors[m] == SLOT[3] else "#ffffff"
                ax.text(left + v / 2, yi, f"{m.split()[0]}\n€{v:.0f} ({g.at[m, 'cost_share']:.0%})",
                        ha="center", va="center", fontsize=8, color=txt_col)
            left += v
        ax.text(left + 8, yi, f"€{left:.0f}", va="center", fontsize=10, fontweight="bold", color=INK)
    ax.set_yticks([0, 1], [p[1] for p in prods])
    ax.set_xlabel("EUR per unit (raw-material content only)")
    ax.grid(axis="y", visible=False)
    h, l = ax.get_legend_handles_labels()
    uniq = dict(zip(l, h))
    ax.legend([uniq[m] for m in order if m in uniq], [m for m in order if m in uniq], ncol=6,
              loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.set_title("Two products, two different risks: RM6 is a copper bet, AirSeT a steel bet", pad=28)
    fig.text(0.01, -0.06, "Sep-2026 prices: copper/aluminium LME (EUR), steel CRU Germany, stainless CRU 304 (Aug-26). "
             "Plastics EUR/t assumed. RM6 unlabelled 45% slice assumed carbon steel.", fontsize=7.5, color=MUTED)
    save(fig, "fig03_cost_structure.png")


def fig_volatility(p):
    ids = list(D.TARGETS) + ["RM6_NE_IQI_433520", "RMAirSeT_IIQ_24kV"]
    names = [D.TARGETS[i][0] if i in D.TARGETS else ("BOM index RM6" if "RM6" in i else "BOM index AirSeT") for i in ids]
    lp = np.log(p[ids])
    rows = []
    for h in (3, 6, 9):
        ch = (np.exp(lp.shift(-h) - lp) - 1).abs()
        rows.append(ch.median().rename(h))
    tab = pd.concat(rows, axis=1)
    tab.index = names
    tab = tab.sort_values(9)
    fig, ax = plt.subplots(figsize=(9, 6.2))
    y = np.arange(len(tab))
    shades = {3: "#86b6ef", 6: "#2a78d6", 9: "#104281"}  # ordinal blue ramp
    for j, h in enumerate((3, 6, 9)):
        ax.barh(y + (j - 1) * 0.26, tab[h] * 100, height=0.24, color=shades[h], label=f"{h} months")
    ax.set_yticks(y, tab.index)
    ax.set_xlabel("median absolute price change over the horizon (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", title="horizon")
    ax.set_title("How much does each series typically move in 3 / 6 / 9 months?")
    fig.text(0.01, -0.01, "Nov-2016 to Sep-2026, all rolling windows. Wider = harder to forecast = bigger risk buffer.",
             fontsize=7.5, color=MUTED)
    save(fig, "fig04_volatility_by_horizon.png")
    tab.to_csv(OUT / "volatility_by_horizon.csv", float_format="%.4f")


def _diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("div", [DIV_NEG, DIV_MID, DIV_POS])


def fig_correlation(p):
    ids = list(D.TARGETS)
    r = np.log(p[ids]).diff().corr()
    names = [D.TARGETS[i][0] for i in ids]
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(r.values, cmap=_diverging_cmap(), vmin=-1, vmax=1)
    ax.set_xticks(range(len(ids)), names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(ids)), names, fontsize=8)
    ax.grid(False)
    for i in range(len(ids)):
        for j in range(len(ids)):
            v = r.values[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7, color="#ffffff" if abs(v) > 0.6 else INK)
    fig.colorbar(im, ax=ax, shrink=0.7, label="correlation of monthly % changes")
    ax.set_title("Which inputs move together? (monthly log-returns)")
    save(fig, "fig05_correlation.png")


def fig_leadlag(p):
    targets = ["D01", "D02", "S09", "S10", "S12", "S11", "S13", "S07", "S08"]
    drivers = {"ED01": "Brent", "ED02": "EU gas", "E09": "Power DE", "S04": "Iron ore", "S03": "Nickel",
               "S01": "Copper LME", "S06": "EUR/USD"}
    lr = np.log(p[list(set(targets) | set(drivers))]).diff()
    best = pd.DataFrame(index=targets, columns=list(drivers), dtype=float)
    lag = best.copy()
    for t in targets:
        for d in drivers:
            cs = {k: lr[t].corr(lr[d].shift(k)) for k in range(1, 7)}
            k = max(cs, key=lambda x: abs(cs[x]))
            best.at[t, d], lag.at[t, d] = cs[k], k
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(best.values.astype(float), cmap=_diverging_cmap(), vmin=-0.7, vmax=0.7)
    ax.set_xticks(range(len(drivers)), list(drivers.values()), rotation=30, ha="right")
    ax.set_yticks(range(len(targets)), [D.TARGETS[t][0] for t in targets])
    ax.grid(False)
    for i in range(len(targets)):
        for j in range(len(drivers)):
            v, k = best.values[i, j], int(lag.values[i, j])
            ax.text(j, i, f"{v:.2f}\n{k}m", ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if abs(v) > 0.45 else INK)
    fig.colorbar(im, ax=ax, shrink=0.75, label="best correlation, driver leading by 1-6 months")
    ax.set_title("Leading indicators: driver change k months earlier vs target change today")
    save(fig, "fig06_lead_lag.png")
    best.to_csv(OUT / "leadlag_corr.csv", float_format="%.3f")
    lag.to_csv(OUT / "leadlag_months.csv")


def fig_price_benchmark(bm):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    hi = {"Schneider Electric": SLOT[0], "Eaton": SLOT[1]}
    series = {}
    for brand, g in bm.groupby("brand"):
        g = g.sort_values("year")
        col = hi.get(brand, AXIS)
        lw = 2.2 if brand in hi else 1.4
        ax.plot(g["year"], g["p50"], color=col, lw=lw, zorder=3 if brand in hi else 2)
        series[brand] = g.set_index("year")["p50"]
    pts = sorted(((s.iloc[-1], b) for b, s in series.items()), reverse=True)
    last = None
    for y, b in pts:
        yy = y if last is None or last - y > 330 else last - 330
        last = yy
        lab = b + (" (SF6-free)" if b == "Eaton" else "")
        ax.annotate(lab, (2026, yy), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8.5,
                    color=INK if b in hi else INK2, fontweight="bold" if b in hi else "normal")
    se, et = series["Schneider Electric"], series["Eaton"]
    g25, g26 = et[2025] / se[2025] - 1, et[2026] / se[2026] - 1
    ax.text(0.01, 0.97, f"SF6-free premium (Eaton vs Schneider):  2025 {g25:+.0%}   ->   2026 {g26:+.0%}",
            transform=ax.transAxes, va="top", fontsize=9.5, color=INK, fontweight="bold")
    ax.set_xlim(2017, 2027.6)
    ax.set_xticks(range(2017, 2027))
    ax.set_ylabel("EUR ex-VAT, median (P50)")
    ax.set_title("3-function RMU selling price: SF6 products jump in 2026, the SF6-free premium disappears")
    fig.text(0.01, -0.02, "Source: RMU_IQI_EU_Price_Benchmark_2017-2026_Synthetic (synthetic, yearly P10/P50/P90). "
             "Lines = P50 per brand.", fontsize=7.5, color=MUTED)
    save(fig, "fig07_price_benchmark.png")


def coverage_table(rm, en, steel, epoxy):
    rows = []
    def add(src, key, name, s, unit):
        s = s.dropna()
        rows.append({"source": src, "id": key, "name": name, "unit": unit, "first": f"{s.index.min():%Y-%m}",
                     "last": f"{s.index.max():%Y-%m}", "n_obs": len(s)})
    for k, (n, fam, reg, u) in D.TARGETS.items():
        add("data.zip (RM)", k, n, rm[k], u)
    for k, n in D.DRIVERS.items():
        add("data.zip (energy)" if k != "S06" else "data.zip (RM)", k, n, en[k] if k in en else rm[k], "")
    for k in ["meps_cr_de", "cru_cr_de", "cru_hdg_de", "ss304_de"]:
        add("Steel INDEXES (internal)", k, k, steel[k], "EUR/t")
    add("Epoxy file", "epoxy", "Epoxy resin (calculated)", epoxy["epoxy"], "EUR/t")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "data_coverage.csv", index=False)
    print(df.to_string(index=False))


def main():
    rm, en, steel = D.load_rm(), D.load_energy(), D.load_steel_licensed()
    epoxy, notes = D.load_epoxy()
    p = D.panel()
    costs = D.bom_cost_table(rm, steel)
    costs.to_csv(OUT / "bom_cost_table.csv", index=False, float_format="%.4f")
    coverage_table(rm, en, steel, epoxy)
    print("data-quality notes:", notes)
    fig_families(rm, en, steel, epoxy)
    fig_bom_index(p)
    fig_cost_structure(costs)
    fig_volatility(p)
    fig_correlation(p)
    fig_leadlag(p)
    fig_price_benchmark(D.load_price_benchmark())


if __name__ == "__main__":
    main()
