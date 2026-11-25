"""Export presentation-ready evidence for the Member 3 decision and Member 4 handoff."""
from __future__ import annotations

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter
import numpy as np

from member3_charts import BAR_DARK, BLUE, GOLD, GRID, INK, MUTED, PRODUCTS, figure, plt
from member3_decision_analysis import optimal_quote

EXAMPLE_FLOOR = -2.0
EXAMPLE_BUDGET = 0.01
STRESS_LABELS = {"simultaneous_high": "Simultaneous material stress", "empirical_p90": "Empirical product P90"}


def prioritization_slide(frontier, exposure, sensitivity):
    fig, axes = figure("Prioritize the cost exposures we can quantify",
                       "9-month decision · copper exposure and the cost of missing BOM prices",
                       "Sources: BOM v1, selected forecasts and one-way missing-mass tests. The €1/kg copper shock is a unit sensitivity.\n"
                       "The missing-mass span uses the €1–€6/kg scenarios and measures uncertainty in the modeled cost level.", cols=1)
    ax = axes[0, 0]
    ax.axis("off")
    h = frontier["h"].max()
    values = []
    for product, label in PRODUCTS.items():
        card = frontier.loc[frontier["product"].eq(product) & frontier["h"].eq(h)].iloc[0]
        material = exposure.loc[exposure["product"].eq(product) & exposure["h"].eq(h) & exposure["series"].eq("D01"),
                                "cost_change_for_one_eur_per_kg_eur_per_unit"].sum()
        missing = sensitivity.loc[sensitivity["product"].eq(product) & sensitivity["h"].eq(h)
                                  & sensitivity["test"].eq("unpriced_mass"), "impact_eur_per_unit"]
        values.append([label, f"{card.priced_mass_share:.1%}", f"{card.unpriced_mass_kg:.2f} kg",
                       f"€{material:.2f}/unit", f"€{missing.max() - missing.min():.2f}/unit"])
    simple_table(ax, ["Product", "Priced mass", "Unpriced mass", "+€1/kg copper", "Unpriced cost span"], values, (0, 0.44, 1, 0.43))
    ax.text(0, 0.28, r"Material cost per unit:  $C=\sum_i w_i p_i$     Copper sensitivity:  $\partial C/\partial p_{\mathrm{Cu}}=w_{\mathrm{Cu}}$",
            fontsize=15, color=INK, transform=ax.transAxes)
    ax.text(0, 0.11, "Track copper for both products. Price the remaining AirSeT BOM first to reduce the larger cost-level uncertainty.",
            fontsize=10.2, color=MUTED, transform=ax.transAxes)
    return fig


def simple_table(ax, columns, rows, bbox):
    table = ax.table(cellText=rows, colLabels=columns, cellLoc="left", colLoc="left", bbox=bbox)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor(BLUE if row == 0 else ("#edf4f6" if row % 2 else "#f7f9fa"))
        cell.get_text().set_color("white" if row == 0 else INK)
        cell.get_text().set_weight("bold" if row == 0 else "normal")


def forecast_decision_slide(tests):
    significant = tests["p_bonferroni"].lt(0.05).sum()
    fig, axes = figure("Use flat cost as the planning reference",
                       "Held-out MAE difference versus flat cost after freezing the Member 2 strategy in December 2024.",
                       f"Paired Newey–West intervals with h − 1 lags and 95% familywise coverage (Bonferroni). "
                       f"{significant} of {len(tests)} comparisons reject at 5%.\n"
                       "All methods use the same dates and realized commodity proxies. The current BOM quantities are applied historically.")
    low, high = tests["ci_95_simultaneous_low"].min(), tests["ci_95_simultaneous_high"].max()
    pad = 0.06 * (high - low)
    for ax, (product, label) in zip(axes[0], PRODUCTS.items()):
        rows = tests.loc[tests["product"].eq(product)]
        methods = [("bottom_up", "Member 2 bottom-up", BLUE, "o"), ("direct_autoets", "Direct AutoETS", GOLD, "s"),
                   ("legacy_bottom_up", "Legacy bottom-up", MUTED, "^")]
        for offset, (method, method_label, color, marker) in zip((-0.20, 0, 0.20), methods):
            method_rows = rows.loc[rows["method"].eq(method)].sort_values("h")
            y = np.arange(len(method_rows)) + offset
            x = method_rows["mae_minus_flat_eur_per_unit"]
            errors = [x - method_rows["ci_95_simultaneous_low"], method_rows["ci_95_simultaneous_high"] - x]
            ax.errorbar(x, y, xerr=errors, marker=marker, linestyle="none", color=color, capsize=3,
                        label=method_label)
        ax.axvline(0, color=BAR_DARK, linewidth=1, linestyle="--")
        ax.set(xlim=(min(low, 0) - pad, max(high, 0) + pad), ylim=(2.5, -0.5), xlabel="MAE difference (€/unit)")
        ax.set_yticks([0, 1, 2], [f"{h} months" for h in sorted(rows["h"].unique())])
        ax.set_title(label, loc="left", weight="bold", color=INK)
        ax.grid(axis="y", color=GRID)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 0.84), ncol=3, frameon=False)
    fig.subplots_adjust(top=0.74)
    return fig


def proof_slide():
    fig, axes = figure("Derive the price from the revenue constraint",
                       "Decision: choose a nonnegative uplift x to minimize the stress cost left uncovered per unit.",
                       "H is the stress increase in modeled material cost. B is the allowed fractional revenue loss.\n"
                       "The implementation floors the revenue cap to cents and ceilings the full-cover price to cents.", cols=1)
    ax = axes[0, 0]
    ax.axis("off")
    blocks = [
        (0.93, "Demand response", r"$p=p_0+x\quad Q/Q_0=(p/p_0)^{\varepsilon}\quad R/R_0=(p/p_0)^{1+\varepsilon}$"),
        (0.70, "Worst case", r"$\varepsilon\geq\varepsilon_{\min},\quad p\geq p_0\quad\Longrightarrow\quad \min R/R_0=(p/p_0)^{1+\varepsilon_{\min}}$"),
        (0.47, "Revenue cap", r"$\varepsilon_{\min}<-1:\quad x\leq x_{\max}=p_0\left[(1-B)^{1/(1+\varepsilon_{\min})}-1\right]$"),
        (0.24, "Optimal uplift", r"$\min_{0\leq x\leq x_{\max}}\,\max(0,H-x)\quad\Longrightarrow\quad x^*=\min(H,x_{\max})$"),
    ]
    for y, label, equation in blocks:
        ax.text(0.0, y, label, fontsize=10, color=BLUE, weight="bold", transform=ax.transAxes)
        ax.text(0.23, y - 0.007, equation, fontsize=17, color=INK, transform=ax.transAxes)
        ax.axhline(y - 0.075, color=GRID, linewidth=0.7)
    ax.text(0.0, 0.025, "Once the full stress is covered, choose the smallest uplift. With an elasticity floor of −1 or higher, the revenue limit is slack.",
            fontsize=9.5, color=MUTED, transform=ax.transAxes)
    return fig


def price_frontier_slide(frontier):
    subset = example_rows(frontier)
    fig, axes = figure("Show the price–risk trade-off before choosing a quote",
                       "9-month decision · worst assumed price elasticity −2 · separate stress definitions",
                       "The curves recompute the optimal quote at each loss budget. Dots show the 1% illustration.\n"
                       "Empirical P90 comes from historical product errors. Simultaneous stress moves material price estimates together.")
    grid = np.linspace(0, 0.045, 181)
    ymax = subset["stress_increase_eur_per_unit"].max() * 1.15
    for ax, (product, label) in zip(axes[0], PRODUCTS.items()):
        rows = subset.loc[subset["product"].eq(product)]
        for option, color, style in [("simultaneous_high", BLUE, "-"), ("empirical_p90", GOLD, "--")]:
            r = rows.loc[rows["stress_basis"].eq(option)].iloc[0]
            tail = [optimal_quote(r.base_price_eur_ex_vat, r.stress_increase_eur_per_unit, EXAMPLE_FLOOR, b)
                    ["uncovered_cost_eur_per_unit"] for b in grid]
            ax.plot(grid, tail, color=color, linestyle=style, linewidth=2, label=STRESS_LABELS[option])
            ax.plot(EXAMPLE_BUDGET, r.uncovered_cost_eur_per_unit, color=color, marker="o")
        ax.axvline(EXAMPLE_BUDGET, linewidth=0.8, color=MUTED, linestyle=":")
        ax.set_title(label, loc="left", weight="bold", color=INK)
        ax.set(xlim=(0, grid.max()), ylim=(0, ymax), xlabel="Allowed worst-case revenue loss")
        ax.set_xticks([0, .01, .02, .03, .04])
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis="y", color=GRID)
    axes[0, 0].set_ylabel("Uncovered stress cost (€/unit)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 0.84), ncol=2, frameon=False)
    fig.subplots_adjust(top=0.74)
    return fig


def example_rows(frontier):
    return frontier.loc[frontier["h"].eq(frontier["h"].max()) & frontier["elasticity_floor"].eq(EXAMPLE_FLOOR)
                        & frontier["maximum_revenue_loss"].eq(EXAMPLE_BUDGET)]


def handoff_slide(frontier):
    rows = example_rows(frontier)
    fig, axes = figure("Give Member 4 the decision inputs and trade-offs",
                       "Illustration · 9 months · 1% revenue loss limit · elasticity floor −2 · simultaneous material stress",
                       "Both prices use synthetic market anchors. The spreadsheet retains both stress definitions and multiple elasticity floors.\n"
                       "Absolute revenue and contribution require commercial inputs for the relevant customer or product segment.", cols=1)
    ax = axes[0, 0]
    ax.axis("off")
    values = []
    for product, label in PRODUCTS.items():
        r = rows.loc[rows["product"].eq(product) & rows["stress_basis"].eq("simultaneous_high")].iloc[0]
        values.append([label, f"+€{r.optimal_uplift_eur:.2f}", f"€{r.uncovered_cost_eur_per_unit:.2f}/unit",
                       f"{r.minimum_revenue_loss_budget_for_full_cover:.2%}"])
    simple_table(ax, ["Product", "Optimal uplift", "Uncovered stress", "Loss limit for full cover"], values, (0, 0.53, 1, 0.39))
    ax.text(0, 0.37, "Set the limit and the demand assumption by customer or segment.", fontsize=13, color=INK, transform=ax.transAxes)
    ax.text(0, 0.23, "Use the uncovered €/unit to test procurement actions and contract terms.", fontsize=13, color=INK, transform=ax.transAxes)
    ax.text(0, 0.09, "Validate the selected policy with observed quotes, volumes and contribution margins.", fontsize=13, color=INK, transform=ax.transAxes)
    return fig


def save_decision_brief(out, tests, frontier, exposure, sensitivity):
    figures = out / "decision_figures"
    figures.mkdir(exist_ok=True)
    pages = [("01_material_priorities", prioritization_slide, (frontier, exposure, sensitivity)),
             ("02_forecast_evidence", forecast_decision_slide, (tests,)), ("03_optimal_quote_proof", proof_slide, ()),
             ("04_pricing_frontier", price_frontier_slide, (frontier,)), ("05_member4_handoff", handoff_slide, (frontier,))]
    with plt.rc_context({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK, "pdf.fonttype": 42}):
        with PdfPages(out / "member3_decision_brief.pdf", metadata={"Title": "Member 3 — Quantitative Decision Brief"}) as pdf:
            for name, build, args in pages:
                fig = build(*args)
                pdf.savefig(fig, facecolor="white")
                fig.savefig(figures / f"{name}.png", dpi=160, facecolor="white")
                plt.close(fig)
