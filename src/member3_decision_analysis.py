"""Test the forecast choice and solve the constrained price decision in euros and cents."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

import data as D
from member3_backtest import POINT_METHODS
from member3_bom import COST

STRESS_BASES = {"simultaneous_high": f"high_stress_{COST}", "empirical_p90": "planning_p90_eur_per_unit"}
ELASTICITY_FLOORS = (-1.0, -1.5, -2.0, -3.0)
REVENUE_LOSS_LIMITS = (0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.08)


def loss_difference_test(differential, lag):
    """Newey-West standard error of the paired mean, with Bartlett weights."""
    n = len(differential)
    mean = differential.mean()
    centered = differential - mean
    long_run_variance = centered @ centered / n
    for k in range(1, min(lag, n - 1) + 1):
        long_run_variance += 2 * (1 - k / (lag + 1)) * (centered[k:] @ centered[:-k]) / n
    se = np.sqrt(max(long_run_variance, 0) / n)
    p = 2 * norm.sf(abs(mean / se)) if se > 0 else np.nan
    return mean, se, p


def validate_forecast_choice(roll, methods=("bottom_up", "direct_autoets")):
    """Point estimates and paired intervals for alternative MAE minus flat-baseline MAE."""
    rows = roll.loc[roll["outcome_available"].le(D.BOM_BASE)].dropna(subset=["actual_proxy_cost", *POINT_METHODS, *methods])
    out = []
    for (product, h), block in rows.sort_values("origin").groupby(["product", "h"]):
        actual = block["actual_proxy_cost"]
        for method in methods:
            delta = ((block[method] - actual).abs() - (block["flat_cost"] - actual).abs()).to_numpy()
            mean, se, p = loss_difference_test(delta, h - 1)
            _, wider_se, wider_p = loss_difference_test(delta, 2 * h - 1)
            out.append({"product": product, "h": h, "method": method, "n": len(delta),
                        "mae_minus_flat_eur_per_unit": mean, "hac_lag": h - 1, "hac_se": se, "p_two_sided": p,
                        "hac_lag_sensitivity": 2 * h - 1, "hac_se_sensitivity": wider_se,
                        "p_two_sided_sensitivity": wider_p})
    result = pd.DataFrame(out)
    critical = norm.ppf(0.975)
    familywise = norm.ppf(1 - 0.025 / len(result))
    for label, z, se in [("95", critical, "hac_se"), ("95_simultaneous", familywise, "hac_se")]:
        result[f"ci_{label}_low"] = result["mae_minus_flat_eur_per_unit"] - z * result[se]
        result[f"ci_{label}_high"] = result["mae_minus_flat_eur_per_unit"] + z * result[se]
    result["p_bonferroni"] = result["p_two_sided"].mul(len(result)).clip(upper=1)
    result["p_bonferroni_sensitivity"] = result["p_two_sided_sensitivity"].mul(len(result)).clip(upper=1)
    return result


def optimal_quote(base_price, stress_increase, elasticity_floor, revenue_loss_limit):
    """Minimize worst-case uncovered cost per unit, then minimize the price uplift.

    Feasible decisions are non-negative integer cents. The constraint is a floor on
    revenue relative to base-price revenue for every elasticity in [floor, 0].
    """
    inputs = (base_price, stress_increase, elasticity_floor, revenue_loss_limit)
    if (not all(math.isfinite(x) for x in inputs) or base_price <= 0 or elasticity_floor > 0
            or not 0 <= revenue_loss_limit < 1):
        raise ValueError("A quote requires finite inputs, a positive base price and a revenue loss limit in [0, 1)")
    if not math.isclose(100 * base_price, round(100 * base_price), rel_tol=0, abs_tol=1e-8):
        raise ValueError("The base price must be specified to whole cents")
    exposure = max(0.0, stress_increase)
    cover_cents = math.ceil(exposure * 100 - 1e-9)
    exponent = 1 + elasticity_floor
    cover_log_ratio = math.log1p(cover_cents / (100 * base_price))
    required_loss = -math.expm1(exponent * cover_log_ratio) if exponent < 0 else 0.0
    uplift_cents = cover_cents
    if revenue_loss_limit < required_loss:
        cap = base_price * math.expm1(math.log1p(-revenue_loss_limit) / exponent)
        uplift_cents = math.floor(cap * 100 + 1e-8)
    uplift = uplift_cents / 100
    log_price_ratio = math.log1p(uplift / base_price)
    revenue_loss = max(0.0, -math.expm1(exponent * log_price_ratio))
    volume_loss = -math.expm1(elasticity_floor * log_price_ratio)
    return {"optimal_uplift_eur": uplift, "optimized_quote_eur_ex_vat": round(base_price + uplift, 2),
            "worst_case_revenue_loss": revenue_loss, "worst_case_volume_loss": volume_loss,
            "uncovered_cost_eur_per_unit": max(0.0, exposure - uplift),
            "minimum_revenue_loss_budget_for_full_cover": required_loss,
            "full_cover_uplift_eur": cover_cents / 100}


def pricing_frontier(pricing):
    """Scenario inputs remain editable by Member 4 when evidence or risk tolerance changes."""
    base = pricing.loc[pricing["policy"].eq("hold_price")]
    rows = []
    for r in base.itertuples(index=False):
        for stress_name, stress_field in STRESS_BASES.items():
            increase = getattr(r, stress_field) - getattr(r, f"origin_{COST}")
            if pd.isna(increase):
                continue
            for floor in ELASTICITY_FLOORS:
                for budget in REVENUE_LOSS_LIMITS:
                    solution = optimal_quote(r.base_price_eur_ex_vat, increase, floor, budget)
                    rows.append({"product": r.product, "target_month": r.target_month, "h": r.h,
                                 "base_price_source": r.base_price_source, "base_price_eur_ex_vat": r.base_price_eur_ex_vat,
                                 "priced_mass_share": r.priced_mass_share, "unpriced_mass_kg": r.unpriced_mass_kg,
                                 "stress_basis": stress_name, "stress_increase_eur_per_unit": increase,
                                 "elasticity_floor": floor, "maximum_revenue_loss": budget, **solution})
    return pd.DataFrame(rows)


def material_exposure(detail):
    fields = ["product", "target_month", "h", "material_label_source", "price_basis", "series", "mass_kg"]
    exposure = detail[fields].copy()
    priceable = exposure["series"].notna()
    exposure["cost_change_for_one_eur_per_kg_eur_per_unit"] = exposure["mass_kg"].where(priceable)
    exposure["central_cost_change_eur_per_unit"] = detail["central_material_cost_change_eur_per_unit"]
    exposure["stress_cost_change_eur_per_unit"] = detail[f"high_stress_{COST}"] - detail[f"origin_{COST}"]
    return exposure
