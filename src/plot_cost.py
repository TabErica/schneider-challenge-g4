"""Ship the BOM cost and pricing figures as individual PNGs and a single vector PDF."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from bom_cost import COST
from pricing_scenarios import POLICIES

RM6 = "RM6_NE_IQI_433520"
AIRSET = "RMAirSeT_IIQ_24kV"
PRODUCTS = {RM6: "RM6", AIRSET: "RM AirSeT"}
COLORS = {RM6: "#216787", AIRSET: "#876120"}
INK = "#17242e"
MUTED = "#52656f"
GRID = "#dbe4e8"
BLUE = "#216787"
GOLD = "#b57e22"
BAR_DARK = "#334c57"
BAR_LIGHT = "#afc4ce"
POLICY_LABELS = dict(zip(POLICIES, ("Hold price", "50% pass-through", "100% pass-through", "Cover high stress")))
LARGEST_HORIZON = "Final forecast horizon"


def figure(title, subtitle, note, rows=1, cols=2):
    fig, axes = plt.subplots(rows, cols, figsize=(11.5, 7.2), squeeze=False)
    fig.subplots_adjust(left=0.105, right=0.95, top=0.80, bottom=0.16, wspace=0.34, hspace=0.75)
    fig.text(0.07, 0.955, "SCHNEIDER ELECTRIC DATA CHALLENGE / BOM COST AND PRICING", color=BLUE, fontsize=9, weight="bold")
    fig.text(0.07, 0.909, title, color=INK, fontsize=21, weight="bold")
    fig.text(0.07, 0.866, subtitle, color=MUTED, fontsize=10)
    fig.text(0.07, 0.052, note, color=MUTED, fontsize=9, linespacing=1.4)
    for ax in axes.ravel():
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color(GRID)
        ax.tick_params(axis="both", labelcolor=INK, color=GRID)
    return fig, axes


def coverage_chart(coverage):
    fig, axes = figure("Which BOM materials can be priced?",
                       "Mass composition under each workbook version, as a share of the current product weight.",
                       "Sources: legacy and v1 BOM workbooks. Legacy RM6 carbon steel uses the project's 112.5 kg imputation.\n"
                       "Named but unpriced includes the AirSeT ‘Other’ line, whose composition is unspecified.", cols=1)
    ax = axes[0, 0]
    order = [(product, version) for product in PRODUCTS for version in ("legacy", "v1")]
    wide = coverage.pivot(index=["product", "bom_version"], columns="coverage_status", values="mass_share")
    wide = wide.reindex(order).fillna(0)
    y = np.arange(len(order))
    left = np.zeros(len(order))
    specs = [("priced", "Priced", BLUE, "white"), ("assumed_carbon_steel", "Assumed carbon steel", GOLD, INK),
             ("identified_unpriced", "Named but unpriced", BAR_LIGHT, INK), ("unidentified", "Unidentified", "#e9eff1", INK)]
    for status, label, color, text_color in specs:
        values = wide.get(status, pd.Series(0, index=wide.index)).to_numpy()
        ax.barh(y, values, left=left, height=0.58, color=color, label=label)
        for i, share in enumerate(values):
            if share >= 0.05:
                ax.text(left[i] + share / 2, y[i], f"{share:.1%}", ha="center", va="center",
                        color=text_color, fontsize=10, weight="bold")
        left += values
    ax.set_yticks(y, [f"{PRODUCTS[p]} · {v}" for p, v in order])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=4, frameon=False, borderaxespad=0)
    fig.subplots_adjust(left=0.18, top=0.74)
    return fig


def cost_drivers_chart(detail):
    h = detail["h"].max()
    last = detail.loc[detail["h"].eq(h) & detail["series"].notna()]
    month = last["target_month"].iloc[0]
    fig, axes = figure("Which materials drive unit cost?",
                       f"{month} central forecast by priceable material group, with change from the September base.",
                       "Source: BOM v1 and selected material forecasts. The right-hand columns show signed changes;\n"
                       "individual changes sum to the modeled unit-cost change. Displayed figures are rounded.")
    fig.subplots_adjust(left=0.17, right=0.91, wspace=0.55, bottom=0.18)
    for ax, (product, label) in zip(axes[0], PRODUCTS.items()):
        lines = last.loc[last["product"].eq(product)].copy()
        lines.loc[lines["series"].eq("epoxy"), "material_family"] = "Epoxy resin"
        lines.loc[lines["material_family"].eq("Plastic"), "material_family"] = "PET"
        cost_col = f"central_{COST}"
        delta_col = "central_material_cost_change_eur_per_unit"
        group = lines.groupby("material_family")[[cost_col, delta_col]].sum().sort_values(cost_col, ascending=True)
        ax.barh(group.index, group[cost_col], color=COLORS[product], height=0.62)
        for i, (value, delta) in enumerate(group.itertuples(index=False, name=None)):
            ax.text(value + 8, i, f"{value:,.0f}", va="center", color=INK, fontsize=9)
            ax.text(1.02, i, f"{delta:+.1f}" if abs(delta) >= 0.05 else "0.0", va="center", ha="left", transform=ax.get_yaxis_transform(),
                    color=MUTED, fontsize=9)
        ax.text(1.02, 1.01, "Δ €/unit", transform=ax.transAxes, color=MUTED, fontsize=9)
        ax.set(xlim=(0, last[cost_col].max() * 1.16), xlabel="Modeled material cost (€/unit)")
        ax.set_title(label, loc="left", weight="bold", color=INK)
        ax.grid(axis="x", color=GRID)
    return fig


def forecast_chart(summary):
    fig, axes = figure("How far could material cost move?",
                       "Flat planning P50, product-level error band and simultaneous material stress across three horizons.",
                       "The shaded band uses historical basket errors and targets 80% coverage. Crosses move each material\n"
                       "to its individual low/high forecast together, as a stress test. All values cover priced BOM materials.")
    for ax, (product, label) in zip(axes[0], PRODUCTS.items()):
        rows = summary.loc[summary["product"].eq(product)].sort_values("h")
        h = rows["h"].to_numpy()
        x = np.r_[0, h]
        base = rows[f"origin_{COST}"].iloc[0]
        lo = rows["planning_p10_eur_per_unit"]
        hi = rows["planning_p90_eur_per_unit"]
        ax.fill_between(h, lo, hi, color=COLORS[product], alpha=0.17, label="Empirical product band")
        ax.plot(x, np.r_[base, rows["planning_p50_eur_per_unit"]], color=COLORS[product], marker="o", label="Flat planning P50")
        ax.plot(x, np.r_[base, rows[f"central_{COST}"]], color=MUTED, linestyle="--", marker=".", label="Selected bottom-up P50")
        ax.scatter(np.r_[h, h], pd.concat([rows[f"low_stress_{COST}"], rows[f"high_stress_{COST}"]]),
                   color=BAR_DARK, marker="x", label="Simultaneous stress")
        ax.set_xticks(x, [rows["origin_month"].iloc[0], *rows["target_month"]])
        ax.set_title(label, loc="left", weight="bold", color=INK)
        ax.grid(axis="y", color=GRID)
    limits = [ax.get_ylim() for ax in axes[0]]
    for ax in axes[0]:
        ax.set_ylim(min(a[0] for a in limits), max(a[1] for a in limits))
    axes[0, 0].set_ylabel("Modeled material cost (€/unit)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 0.84), ncol=4, frameon=False)
    fig.subplots_adjust(top=0.73, bottom=0.17)
    return fig


def backtest_chart(scores):
    fig, axes = figure("Do the cost forecasts beat a flat baseline?",
                       "Held-out MAE on identical origin dates after freezing the selection strategy in December 2024.",
                       "Validation targets are realized commodity-price proxies applied to current BOM weights.\n"
                       "n gives the number of evaluated months per method. Input data use the latest available historical vintage.")
    y_max = scores["mae_eur_per_unit"].max() * 1.22
    methods = [("flat_cost", "Flat cost", BAR_LIGHT), ("direct_autoets", "Direct AutoETS", GOLD),
               ("legacy_bottom_up", "Legacy bottom-up", MUTED), ("bottom_up", "Selected bottom-up", BLUE)]
    for ax, (product, label) in zip(axes[0], PRODUCTS.items()):
        rows = scores.loc[scores["product"].eq(product)]
        table = rows.pivot(index="h", columns="method", values="mae_eur_per_unit")
        counts = rows.groupby("h")["n"].first().reindex(table.index)
        x = np.arange(len(table))
        for offset, (method, method_label, color) in zip((-0.30, -0.10, 0.10, 0.30), methods):
            bars = ax.bar(x + offset, table[method], width=0.18, color=color, label=method_label)
            ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=8)
        ax.set_xticks(x, [f"{h} months\nn = {n}" for h, n in counts.items()])
        ax.set_ylim(0, y_max)
        ax.set_title(label, loc="left", weight="bold", color=INK)
        ax.grid(axis="y", color=GRID)
    axes[0, 0].set_ylabel("MAE (€/unit)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 0.84), ncol=4, frameon=False)
    fig.subplots_adjust(top=0.73, bottom=0.19)
    return fig


def assumption_chart(sensitivity):
    rows = sensitivity.loc[sensitivity["h"].eq(sensitivity["h"].max())]
    fig, axes = figure("How sensitive is cost to BOM assumptions?",
                       f"{LARGEST_HORIZON} ({rows['target_month'].iat[0]}), with one assumption changed at a time.",
                       "The missing-mass values are illustrative uniform cost assumptions. The polymer test moves the assumed\n"
                       "PET and engineering-plastics price anchors together. These scenarios are separate from forecast error.")
    low, high = rows["impact_eur_per_unit"].agg(["min", "max"])
    for ax, (product, label) in zip(axes[0], PRODUCTS.items()):
        block = rows.loc[rows["product"].eq(product)]
        names = [f"€{r.unpriced_assumed_eur_per_kg:.0f}/kg unpriced" if r.test == "unpriced_mass"
                 else f"Polymer {r.polymer_anchor_change:+.0%}" for r in block.itertuples()]
        y = np.arange(len(block))
        bars = ax.barh(y, block["impact_eur_per_unit"], color=[GOLD if r == "unpriced_mass" else BLUE for r in block["test"]])
        ax.bar_label(bars, fmt="%+.1f", padding=4, fontsize=9)
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.axvline(0, linewidth=0.8, color=INK)
        ax.set(xlim=(min(low * 6, -70), high * 1.17), xlabel="Change in cost (€/unit)")
        ax.set_title(label, loc="left", weight="bold", color=INK)
        ax.grid(axis="x", color=GRID)
    fig.subplots_adjust(left=0.15, wspace=0.50)
    return fig


def price_chart(pricing):
    rows = pricing.loc[pricing["h"].eq(pricing["h"].max())]
    fig, axes = figure("How much cost exposure does each price cover?",
                       f"{rows['target_month'].iat[0]} pricing policies and uncovered material-cost increases under joint stress.",
                       "Each policy starts from its product's synthetic price proxy unless a commercial price is supplied.\n"
                       "Each euro of price increase offsets one euro of material-cost increase per unit sold.")
    x = np.arange(len(POLICIES))
    limit = rows[["selling_price_change_eur", "high_stress_cost_change_net_of_price_eur_per_unit"]].max().max()
    minimum = rows[["selling_price_change_eur", "high_stress_cost_change_net_of_price_eur_per_unit"]].min().min()
    for ax, (product, label) in zip(axes[0], PRODUCTS.items()):
        block = rows.loc[rows["product"].eq(product)].set_index("policy").loc[list(POLICIES)]
        inc = ax.bar(x - 0.16, block["selling_price_change_eur"], 0.32, color=BLUE, label="Price change")
        risk = ax.bar(x + 0.16, block["high_stress_cost_change_net_of_price_eur_per_unit"].clip(lower=0), 0.32, color=GOLD,
                      label="Uncovered cost under stress")
        ax.bar_label(inc, fmt="%.0f", padding=3, fontsize=9)
        ax.bar_label(risk, fmt="%.0f", padding=3, fontsize=9)
        ax.set_xticks(x, ["Hold", "50% of\ncentral", "100% of\ncentral", "Cover high\nstress"])
        ax.set_ylim(min(0, minimum * 1.15), max(1, limit * 1.19))
        ax.set_title(label, loc="left", weight="bold", color=INK)
        ax.grid(axis="y", color=GRID)
    axes[0, 0].set_ylabel("€/unit versus base")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 0.84), ncol=2, frameon=False)
    fig.subplots_adjust(top=0.73, bottom=0.18)
    return fig


def revenue_chart(pricing, elasticity):
    horizon = pricing["h"].max()
    price = pricing.loc[pricing["h"].eq(horizon)]
    revenue = elasticity.loc[elasticity["h"].eq(horizon)]
    fig, axes = figure("When does a price increase stop lifting revenue?",
                       f"{price['target_month'].iat[0]} revenue change versus holding price, at four assumed elasticities.",
                       "Volume scales with (new price ÷ base price) raised to the assumed elasticity. Elasticities are illustrative.\n"
                       "The maximum unit loss shown for the stress policy is the volume decline that keeps revenue unchanged.")
    styles = [(BAR_DARK, "--", "s"), ("#5d8697", ":", "o"), (BLUE, "-.", "D"), (GOLD, "-", "o")]
    ymax = revenue["revenue_change_vs_hold_price"].abs().max()
    for ax, (product, label) in zip(axes[0], PRODUCTS.items()):
        for policy, (color, style, marker) in zip(POLICIES, styles):
            rows = revenue.loc[revenue["product"].eq(product) & revenue["policy"].eq(policy)]
            ax.plot(rows["assumed_price_elasticity"], rows["revenue_change_vs_hold_price"], color=color, linestyle=style,
                    marker=marker, markersize=4, label=POLICY_LABELS[policy])
        high = price.loc[price["product"].eq(product) & price["policy"].eq("cover_high_stress")].iloc[0]
        ax.text(0.04, 0.95, f"Max. unit loss at stress price  {-high['break_even_volume_change_for_revenue']:.2%}",
                transform=ax.transAxes, va="top", fontsize=9, color=MUTED)
        ax.set_xticks([0, -0.5, -1, -2])
        ax.set(xlabel="Assumed price elasticity", ylim=(-max(ymax, 0.001) * 1.28, max(ymax, 0.001) * 1.28))
        ax.set_title(label, loc="left", weight="bold", color=INK)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
        ax.grid(axis="y", color=GRID)
    axes[0, 0].set_ylabel("Revenue change")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 0.84), ncol=4, frameon=False)
    fig.subplots_adjust(top=0.73, bottom=0.17)
    return fig


def save_charts(out, coverage, detail, summary, scores, sensitivity, pricing, elasticity):
    charts = out / "figures"
    charts.mkdir(parents=True, exist_ok=True)
    queries = [
        ("01_bom_coverage", coverage_chart, (coverage,)),
        ("02_cost_drivers", cost_drivers_chart, (detail,)),
        ("03_unit_cost_forecast", forecast_chart, (summary,)),
        ("04_product_backtest", backtest_chart, (scores,)),
        ("05_bom_sensitivity", assumption_chart, (sensitivity,)),
        ("06_price_and_cost_risk", price_chart, (pricing,)),
        ("07_revenue_and_elasticity", revenue_chart, (pricing, elasticity)),
    ]
    with plt.rc_context({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK, "pdf.fonttype": 42}):
        with PdfPages(out / "cost_pricing_figures.pdf", metadata={"Title": "Unit Cost, Pricing and Revenue"}) as pdf:
            for name, build, args in queries:
                fig = build(*args)
                fig.savefig(charts / f"{name}.png", dpi=180, facecolor="white")
                pdf.savefig(fig, facecolor="white")
                plt.close(fig)
