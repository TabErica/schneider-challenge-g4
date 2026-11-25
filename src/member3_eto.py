"""Quote validity, reserve and contractual indexation for the v1 BOM basket."""
from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd

import data as D
from member3_bom import COST
from member3_decision_analysis import optimal_quote

SCENARIOS = ("upside", "baseline", "downside")
VALIDITY_MONTHS = (0, 1, 2, 3)
WAIT_QUANTILE = 0.90
WAIT_VALIDATION_CUTOFF = pd.Timestamp("2024-12-01")


@dataclass(frozen=True)
class OfferAssumptions:
    revenue_loss_limit: float = 0.01
    elasticity_floor: float = -2.0
    waiting_risk_budget: float = 0.005
    margin_erosion_limit: float = 0.005

    def __post_init__(self):
        for name in ("revenue_loss_limit", "waiting_risk_budget", "margin_erosion_limit"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be a fraction in [0, 1)")
        if not math.isfinite(self.elasticity_floor) or self.elasticity_floor > 0:
            raise ValueError("elasticity_floor must be finite and <= 0")


def basket_history(detail, panel):
    """Observed historical market proxies, frozen v1 quantities, and current EUR price anchors."""
    columns = ["product", "material_label_source", "series", "mass_kg", "price_basis", "origin_price_eur_per_t"]
    lines = detail.loc[detail["h"].eq(detail["h"].min()) & detail["series"].notna(), columns].copy()
    lines["base_contribution_eur_per_unit"] = lines["mass_kg"] * lines["origin_price_eur_per_t"] / 1000
    lines["index_weight"] = (lines["base_contribution_eur_per_unit"]
                             / lines.groupby("product")["base_contribution_eur_per_unit"].transform("sum"))
    baskets = []
    for product, group in lines.groupby("product"):
        terms = []
        for row in group.itertuples(index=False):
            values = panel.loc["2020-01":D.BOM_BASE, row.series]
            if row.price_basis == "assumed_level_indexed":
                values = values * row.origin_price_eur_per_t / panel.at[D.BOM_BASE, row.series]
            terms.append(values * row.mass_kg / 1000)
        costs = pd.concat(terms, axis=1).sum(axis=1, min_count=len(terms))
        base = group["base_contribution_eur_per_unit"].sum()
        history = costs.rename("cost_eur_per_unit").rename_axis("month").reset_index()
        history["source"] = "historical_complete_basket"
        history.loc[history["month"].eq(D.BOM_BASE), ["cost_eur_per_unit", "source"]] = [base, "offer_anchor_with_nowcast"]
        history["bom_index"] = 100 * history["cost_eur_per_unit"] / base
        baskets.append(history.dropna(subset=["cost_eur_per_unit"]).assign(product=product))
    return pd.concat(baskets, ignore_index=True), lines


def waiting_risk(history):
    """Quantile of rising proxy costs over a candidate acceptance window, in EUR/unit."""
    estimates = []
    observed = history.loc[history["source"].eq("historical_complete_basket")]
    for product, group in observed.groupby("product"):
        cost = group.set_index("month")["cost_eur_per_unit"].sort_index().asfreq("MS")
        for months in VALIDITY_MONTHS:
            change = cost.diff(months).dropna()
            past = change.loc[:WAIT_VALIDATION_CUTOFF]
            check = change.loc[change.index > WAIT_VALIDATION_CUTOFF]
            historical_reserve = max(0, past.quantile(WAIT_QUANTILE))
            estimates.append({"product": product, "validity_calendar_months": months,
                              "wait_risk_reserve_eur_per_unit": max(0, change.quantile(WAIT_QUANTILE)),
                              "calibration_n": len(change), "held_back_calibration_cutoff": WAIT_VALIDATION_CUTOFF,
                              "held_back_reserve_eur_per_unit": historical_reserve,
                              "held_back_n": len(check), "held_back_coverage": check.le(historical_reserve).mean()})
    return pd.DataFrame(estimates)


def validity_choices(prices, wait, assumptions):
    keys = ["product", "h", "target_month", "base_price_eur_ex_vat"]
    base = prices.loc[prices["policy"].eq("hold_price"), keys]
    candidates = base.merge(wait, on="product", validate="many_to_many")
    candidates["wait_risk_budget_eur_per_unit"] = candidates["base_price_eur_ex_vat"] * assumptions.waiting_risk_budget
    candidates["feasible_validity"] = (
        candidates["wait_risk_reserve_eur_per_unit"].le(candidates["wait_risk_budget_eur_per_unit"])
        & candidates["validity_calendar_months"].le(candidates["h"])
    )
    longest = candidates.loc[candidates["feasible_validity"]].groupby(["product", "h"])["validity_calendar_months"].max()
    reference = candidates.set_index(["product", "h"]).index.map(longest)
    candidates["selected_validity"] = candidates["validity_calendar_months"].eq(reference)
    return candidates


def signed_price_change(base, net_cost, assumptions):
    if net_cost >= 0:
        return optimal_quote(base, net_cost, assumptions.elasticity_floor, assumptions.revenue_loss_limit)["optimal_uplift_eur"]
    # For a discount, zero elasticity is the most adverse volume response.
    discount = min(-net_cost, base * assumptions.revenue_loss_limit)
    return -math.floor(100 * discount + 1e-8) / 100


def indexation_impact(base_quote, initial_adjustment, base_cost, published_index, assumptions):
    """Evaluate a monthly index print under the chosen quote and pricing constraints."""
    if not math.isfinite(base_cost) or base_cost <= 0 or not math.isfinite(published_index) or published_index <= 0:
        raise ValueError("The reference cost and observed basket index must be positive and finite")
    movement = base_cost * (published_index / 100 - 1)
    erosion = base_quote * assumptions.margin_erosion_limit
    trigger_amount = initial_adjustment + erosion
    required = max(0, movement - trigger_amount)
    cents_needed = math.ceil(100 * required - 1e-9) / 100
    safe_total = signed_price_change(base_quote, max(movement - erosion, 0), assumptions)
    applied = max(0, min(round(safe_total - initial_adjustment, 2), cents_needed))
    fixed_exposure = max(0, movement - initial_adjustment)
    indexed_exposure = max(0, fixed_exposure - applied)
    return {"margin_erosion_limit_eur_per_unit": erosion, "index_reference": 100.0,
            "index_trigger_eur_per_unit": trigger_amount, "index_trigger_level": 100 * (1 + trigger_amount / base_cost),
            "index_scenario_level": published_index, "index_scenario_exceeds_trigger": required > 0,
            "cost_stress_left_on_fixed_offer_eur_per_unit": fixed_exposure,
            "required_protection_on_fixed_offer_eur_per_unit": max(0, fixed_exposure - erosion),
            "contractual_surcharge_at_downside_eur_per_unit": cents_needed,
            "revenue_safe_surcharge_at_downside_eur_per_unit": applied,
            "cost_stress_left_after_safe_surcharge_eur_per_unit": indexed_exposure,
            "required_non_price_protection_at_downside_eur_per_unit": max(0, indexed_exposure - erosion),
            "fixed_offer_requires_protection": fixed_exposure - erosion > 1e-8}


def make_offers(prices, summary, validity, assumptions):
    base = prices.loc[prices["policy"].eq("hold_price"), ["product", "h", "base_price_eur_ex_vat", "base_price_source",
                                                        "benchmark_brand", "units", "additional_direct_cost_eur_per_unit"]]
    wait = validity.loc[validity["selected_validity"], ["product", "h", "validity_calendar_months", "wait_risk_reserve_eur_per_unit",
                                                      "wait_risk_budget_eur_per_unit"]]
    bases = summary.merge(base, on=["product", "h"], validate="one_to_one").merge(wait, on=["product", "h"], validate="one_to_one")
    offers = []
    for row in bases.itertuples(index=False):
        quote_ref, cost_ref = row.base_price_eur_ex_vat, getattr(row, f"origin_{COST}")
        simultaneous = getattr(row, f"high_stress_{COST}")
        high = max(row.planning_p90_eur_per_unit, simultaneous)
        downside_basis = "simultaneous_material_p90" if simultaneous > row.planning_p90_eur_per_unit else "planning_product_p90"
        planning_costs = (row.planning_p10_eur_per_unit, row.planning_p50_eur_per_unit, high)
        waiting = row.wait_risk_reserve_eur_per_unit
        adverse_move = high - cost_ref + waiting
        stress_index = 100 * (1 + adverse_move / cost_ref)

        for scenario, cost in zip(SCENARIOS, planning_costs, strict=True):
            delta = cost - cost_ref + waiting
            change = signed_price_change(quote_ref, delta, assumptions)
            quote = round(quote_ref + change, 2)
            log_ratio = math.log1p(change / quote_ref)
            worst_elasticity = assumptions.elasticity_floor if change >= 0 else 0.0
            worst_rev_loss = max(0, -math.expm1((1 + worst_elasticity) * log_ratio))
            terms = indexation_impact(quote_ref, change, cost_ref, stress_index, assumptions)
            worst_quantity_ratio = math.exp(worst_elasticity * log_ratio)
            complete_cost = cost + waiting + row.additional_direct_cost_eur_per_unit
            offers.append({
                "product": row.product, "h": row.h, "target_month": row.target_month, "scenario": scenario,
                "decision_condition": {"upside": "verified_cost_saving_only", "baseline": "planning_reference",
                                       "downside": "firm_offer_risk_test"}[scenario],
                "base_price_eur_ex_vat": quote_ref, "base_price_source": row.base_price_source, "benchmark_brand": row.benchmark_brand,
                "base_bom_cost_eur_per_unit": cost_ref, "scenario_bom_cost_eur_per_unit": cost,
                "planning_product_p90_eur_per_unit": row.planning_p90_eur_per_unit,
                "simultaneous_material_p90_eur_per_unit": simultaneous, "downside_cost_basis": downside_basis,
                "horizon_reserve_above_planning_eur_per_unit": max(0, cost - row.planning_p50_eur_per_unit),
                "downside_reserve_above_planning_eur_per_unit": high - row.planning_p50_eur_per_unit,
                "selected_validity_calendar_months": row.validity_calendar_months,
                "maximum_calendar_months_issue_to_delivery": row.h + row.validity_calendar_months,
                "wait_risk_budget_eur_per_unit": row.wait_risk_budget_eur_per_unit,
                "wait_risk_reserve_eur_per_unit": waiting, "net_cost_change_with_wait_reserve_eur_per_unit": delta,
                "price_adjustment_eur": change, "illustrative_offer_eur_ex_vat": quote,
                "worst_case_revenue_loss_fraction": worst_rev_loss, "revenue_loss_limit_fraction": assumptions.revenue_loss_limit,
                "elasticity_floor": assumptions.elasticity_floor,
                "downside_cost_change_with_wait_reserve_eur_per_unit": adverse_move,
                **terms,
                "worst_case_revenue_per_100_base_units_eur_ex_vat": 100 * quote * worst_quantity_ratio,
                "worst_case_revenue_at_input_volume_eur_ex_vat": row.units * quote * worst_quantity_ratio,
                "scenario_direct_cost_with_input_eur_per_unit": complete_cost,
                "scenario_contribution_with_input_eur_per_unit": quote - complete_cost,
            })
    return pd.DataFrame(offers)
