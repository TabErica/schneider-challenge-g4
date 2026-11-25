"""Create the figures for the final ETO offer policy and the Goal B handoff."""
from __future__ import annotations

from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from plot_cost import BLUE, GOLD, GRID, INK, MUTED, PRODUCTS, figure, plt
from plot_decision_brief import simple_table


def validity_slide(validity):
    fig, axes = figure("Set quote validity from the cost of waiting",
                       "Choose the longest calendar-month window whose historical P90 cost rise fits the waiting-risk budget.",
                       "The chart uses the offer's 3-month delivery horizon to show the default price anchor and budget. The same\n"
                       "acceptance calculation is repeated for 6 and 9 months. A 0-month policy requires immediate confirmation.")
    start = validity.loc[validity["h"].eq(validity["h"].min())]
    peak = max(start["wait_risk_reserve_eur_per_unit"].max(), start["wait_risk_budget_eur_per_unit"].max())
    for ax, (product, label) in zip(axes[0], PRODUCTS.items()):
        block = start.loc[start["product"].eq(product)].sort_values("validity_calendar_months")
        budget = block["wait_risk_budget_eur_per_unit"].iloc[0]
        selected = block.loc[block["selected_validity"], "validity_calendar_months"].iat[0]
        bars = ax.bar(block["validity_calendar_months"], block["wait_risk_reserve_eur_per_unit"], width=0.56,
                      color=np.where(block["selected_validity"], BLUE, "#a8bec8"))
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
        ax.axhline(budget, color=GOLD, linestyle="--", label=f"Budget €{budget:.2f}")
        ax.set_xticks(block["validity_calendar_months"])
        ax.set(xlabel="Quote validity (calendar months)", ylim=(0, peak * 1.22))
        ax.set_title(f"{label} · {selected} month selected", loc="left", color=INK, weight="bold")
        ax.legend(frameon=False, loc="upper left")
        ax.grid(axis="y", color=GRID)
    axes[0, 0].set_ylabel("Waiting-risk reserve (€/unit)")
    return fig


def offer_slide(offers):
    fig, axes = figure("Longer ETO commitments need more protection",
                       "Downside cost reserve, initial quote and procurement protection after revenue-safe indexation.",
                       "The downside takes the higher of product P90 and simultaneous material P90, then adds waiting risk.\n"
                       "The resulting stress is a scenario in euros per unit. The benchmark and risk limits are illustrative.", cols=1)
    ax = axes[0, 0]
    ax.axis("off")
    values = []
    for product, label in PRODUCTS.items():
        rows = offers.loc[offers["product"].eq(product) & offers["scenario"].eq("downside")].sort_values("h")
        for r in rows.itertuples():
            values.append([f"{label} · {r.h}m", f"€{r.downside_cost_change_with_wait_reserve_eur_per_unit:.0f}",
                           f"€{r.illustrative_offer_eur_ex_vat:,.2f}", f"{r.index_trigger_level:.2f}",
                           f"€{r.required_non_price_protection_at_downside_eur_per_unit:.0f}"])
    simple_table(ax, ["Offer term", "Total stress", "Downside price", "Index trigger", "Extra protection"], values, (0, 0.16, 1, 0.8))
    ax.text(0, 0.04, "Extra protection is the cost relief still needed to stay within the modeled margin-erosion limit.",
            color=MUTED, fontsize=10, transform=ax.transAxes)
    return fig


def trigger_slide():
    fig, axes = figure("Make the index clause explicit before signing",
                       "Separate the economically required surcharge from the amount allowed by the revenue constraint.",
                       "The rule requires an agreed indexation clause. For a firm fixed price, use the trigger as a pre-award alert\n"
                       "and protect any excess exposure through purchasing, contract terms or a shorter commitment.", cols=1)
    ax = axes[0, 0]
    ax.axis("off")
    values = [
        (0.88, "Cost movement from the index", r"$\Delta C=C_0(I/100-1)$"),
        (0.66, "Trigger index level", r"$I^*=100[1+(x+M)/C_0]$"),
        (0.44, "Surcharge above the trigger", r"$S=\max(0,\Delta C-x-M)$"),
        (0.22, "Additional cost relief needed", r"$\max(0,\Delta C-x-S_{\mathrm{safe}}-M)$"),
    ]
    for y, label, math in values:
        ax.text(0, y, label, color=BLUE, weight="bold", fontsize=11, transform=ax.transAxes)
        ax.text(0.41, y, math, color=INK, fontsize=17, transform=ax.transAxes)
        ax.axhline(y - 0.065, color=GRID, linewidth=0.7)
    ax.text(0, 0.021, "x is the initial adjustment. M is the margin-erosion limit. The combined uplift must meet the revenue cap.",
            fontsize=10, color=MUTED, transform=ax.transAxes)
    return fig


def migration_slide(tco):
    h = tco["h"].max()
    rows = tco.loc[tco["h"].eq(h) & tco["assumed_service_years"].eq(20) & tco["assumed_discount_rate"].eq(0.08)]
    fig, axes = figure("Give Goal B a measurable adoption threshold",
                       "RM AirSeT vs RM6 · 20-year buyer horizon · 8% illustrative discount rate",
                       "Thresholds use the modeled cost gap when complete direct costs are unavailable. Validate the price and\n"
                       "lifecycle savings with the customer. The numbers provide break-even tests for Member 4's market strategy.", cols=1)
    ax = axes[0, 0]
    ax.axis("off")
    values = []
    for label in ("upside", "baseline", "downside"):
        r = rows.loc[rows["scenario"].eq(label)].iloc[0]
        values.append([label.title(), f"€{r.equal_unit_contribution_premium_eur:.2f}",
                       f"€{r.annual_tco_savings_needed_at_contribution_neutral_premium_eur_per_unit:.2f}/year"])
    simple_table(ax, ["Material scenario", "Equal-contribution premium", "Buyer savings needed"], values, (0, 0.40, 1, 0.54))
    ax.text(0, 0.24, r"Adoption test: annual savings $\times$ annuity factor $\geq$ upfront premium.",
            fontsize=13, color=INK, transform=ax.transAxes)
    ax.text(0, 0.08, "Member 4 tests buyer savings, procurement criteria and willingness to switch by segment.",
            fontsize=11, color=MUTED, transform=ax.transAxes)
    return fig


def save_eto_brief(out, offers, validity, tco):
    pages = [("01_quote_validity", validity_slide, (validity,)), ("02_offer_downside", offer_slide, (offers,)),
             ("03_index_clause", trigger_slide, ()), ("04_migration_threshold", migration_slide, (tco,))]
    figures = out / "eto_figures"
    figures.mkdir(exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "text.color": INK, "pdf.fonttype": 42}):
        with PdfPages(out / "quote_policy_brief.pdf", metadata={"Title": "Final ETO Policy"}) as pdf:
            for name, build, args in pages:
                fig = build(*args)
                pdf.savefig(fig, facecolor="white")
                fig.savefig(figures / f"{name}.png", dpi=160, facecolor="white")
                plt.close(fig)
