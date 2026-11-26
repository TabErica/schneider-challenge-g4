"""Price pass-through, break-even volumes and contribution scenarios."""
from __future__ import annotations

import numpy as np
import pandas as pd

import data as D
from bom_cost import COST


BENCHMARK_ID = {"RM6_NE_IQI_433520": "SE", "RMAirSeT_IIQ_24kV": "ETN"}
KEY = ["product", "target_month"]
COMMERCIAL_COLUMNS = ["units", "base_price_eur_ex_vat", "additional_direct_cost_eur_per_unit"]
PASS_THROUGH = (0.0, 0.5, 1.0)
ELASTICITIES = (0.0, -0.5, -1.0, -2.0)
POLICIES = ("hold_price", "half_central_cost", "full_central_cost", "cover_high_stress")


def load_commercial_inputs(path, summary):
    if path is None:
        return summary[KEY].assign(**dict.fromkeys(COMMERCIAL_COLUMNS, np.nan))
    inputs = pd.read_csv(path, dtype={c: str for c in KEY})
    if not set(KEY).issubset(inputs) or not set(COMMERCIAL_COLUMNS).intersection(inputs.columns):
        raise ValueError(f"{path}: requires {', '.join(KEY)} and at least one of {', '.join(COMMERCIAL_COLUMNS)}")
    if inputs.duplicated(KEY).any():
        raise ValueError(f"{path}: duplicate product and target_month")
    unknown = pd.MultiIndex.from_frame(inputs[KEY]).difference(pd.MultiIndex.from_frame(summary[KEY]))
    if len(unknown):
        raise ValueError(f"{path}: unknown product/month pairs {unknown.tolist()}")
    inputs = inputs.reindex(columns=KEY + COMMERCIAL_COLUMNS)
    for col in COMMERCIAL_COLUMNS:
        inputs[col] = pd.to_numeric(inputs[col], errors="raise")
        values = inputs[col].dropna()
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{path}: {col} must be finite and non-negative")
        if col == "base_price_eur_ex_vat" and values.eq(0).any():
            raise ValueError(f"{path}: base_price_eur_ex_vat must be positive")
        if col == "units" and values.mod(1).ne(0).any():
            raise ValueError(f"{path}: units must be whole numbers")
    return inputs


def default_policies():
    """Pricing policies and their pass-through of the central cost change (NaN = cover the high stress)."""
    return pd.DataFrame({"policy": POLICIES, "pass_through": (*PASS_THROUGH, np.nan)})


def commercial_scenarios(summary, benchmark, inputs, policies=None):
    """Price scenarios per product and month; `policies` (policy, pass_through) defaults to default_policies()."""
    policies = default_policies() if policies is None else policies
    benchmark = (benchmark.loc[benchmark["year"].eq(D.BOM_BASE.year), ["id", "brand", "year", "p10", "p50", "p90"]]
                 .rename(columns={"id": "benchmark_id", "brand": "benchmark_brand", "year": "benchmark_year",
                                  **{q: f"synthetic_benchmark_{q}_eur_ex_vat" for q in ("p10", "p50", "p90")}}))
    scenarios = (summary.assign(benchmark_id=summary["product"].map(BENCHMARK_ID))
                 .merge(benchmark, on="benchmark_id", how="left", validate="many_to_one")
                 .merge(inputs, on=KEY, how="left", validate="one_to_one"))
    benchmark_price = scenarios["synthetic_benchmark_p50_eur_ex_vat"]
    if not np.isfinite(benchmark_price).all() or not benchmark_price.gt(0).all():
        raise ValueError("Synthetic benchmark requires a positive price for each benchmark brand in the origin year")
    scenarios["base_price_source"] = np.where(scenarios["base_price_eur_ex_vat"].notna(),
                                               "commercial_input", "synthetic_benchmark")
    scenarios["base_price_eur_ex_vat"] = scenarios["base_price_eur_ex_vat"].fillna(benchmark_price).round(2)
    if not scenarios["base_price_eur_ex_vat"].gt(0).all():
        raise ValueError("Base selling prices must be positive when rounded to cents")
    scenarios["central_direct_cost_with_input_eur_per_unit"] = (
        scenarios[f"central_{COST}"] + scenarios["additional_direct_cost_eur_per_unit"]
    )
    scenarios = scenarios.merge(policies, how="cross")
    central_change = scenarios["central_material_cost_change_eur_per_unit"]
    stress_change = scenarios[f"high_stress_{COST}"] - scenarios[f"origin_{COST}"]
    stress_uplift = np.ceil(100 * stress_change.clip(lower=0)) / 100
    price_adjustment = (scenarios["pass_through"] * central_change).fillna(stress_uplift)
    scenarios["selling_price_eur_ex_vat"] = (
        scenarios["base_price_eur_ex_vat"] + price_adjustment
    ).round(2)
    if not scenarios["selling_price_eur_ex_vat"].gt(0).all():
        raise ValueError("A pricing scenario yielded a non-positive selling price. Check the input price anchors.")
    price_change = scenarios["selling_price_eur_ex_vat"] - scenarios["base_price_eur_ex_vat"]
    scenarios["selling_price_change_eur"] = price_change
    scenarios["high_stress_cost_change_net_of_price_eur_per_unit"] = stress_change - price_change
    band_change = scenarios["planning_p90_eur_per_unit"] - scenarios[f"origin_{COST}"]
    scenarios["empirical_p90_cost_change_net_of_price_eur_per_unit"] = band_change - price_change
    scenarios["break_even_volume_change_for_revenue"] = (scenarios["base_price_eur_ex_vat"]
                                                         / scenarios["selling_price_eur_ex_vat"] - 1)
    scenarios["break_even_units_for_revenue"] = np.ceil(
        scenarios["units"] * (1 + scenarios["break_even_volume_change_for_revenue"]) - 1e-9
    )
    scenarios["within_synthetic_reference_band"] = scenarios["selling_price_eur_ex_vat"].between(
        scenarios["synthetic_benchmark_p10_eur_ex_vat"], scenarios["synthetic_benchmark_p90_eur_ex_vat"]
    )
    scenarios["revenue_per_100_units_eur_ex_vat"] = scenarios["selling_price_eur_ex_vat"] * 100
    scenarios["revenue_at_input_volume_eur_ex_vat"] = scenarios["selling_price_eur_ex_vat"] * scenarios["units"]
    return scenarios


def elasticity_scenarios(pricing):
    cols = KEY + ["h", "policy", "units", "base_price_eur_ex_vat", "selling_price_eur_ex_vat",
                  "central_direct_cost_with_input_eur_per_unit"]
    scenario = pricing[cols].merge(pd.DataFrame({"assumed_price_elasticity": ELASTICITIES}), how="cross")
    p0, p1 = scenario["base_price_eur_ex_vat"], scenario["selling_price_eur_ex_vat"]
    ratio = (p1 / p0) ** scenario["assumed_price_elasticity"]
    scenario["volume_per_100_base_units"] = 100 * ratio
    scenario["revenue_per_100_base_units_eur_ex_vat"] = 100 * ratio * p1
    scenario["revenue_change_vs_hold_price"] = ratio * p1 / p0 - 1
    scenario["volume_at_input_base"] = scenario["units"] * ratio
    scenario["revenue_at_input_base_eur_ex_vat"] = scenario["volume_at_input_base"] * p1
    direct = scenario["central_direct_cost_with_input_eur_per_unit"]
    scenario["contribution_change_vs_hold_per_100_base_units_eur"] = 100 * (ratio * (p1 - direct) - (p0 - direct))
    return scenario


def decision_table(pricing, scores):
    """Report the adopted planning point alongside every tied held-out MAE winner."""
    keys = ["product", "h"]
    minimum = scores.groupby(keys)["mae_eur_per_unit"].transform("min")
    best = (scores.loc[np.isclose(scores["mae_eur_per_unit"], minimum, rtol=0, atol=1e-9)]
            .sort_values("method").groupby(keys, as_index=False)
            .agg(lowest_mae_method=("method", "|".join), mae_eur_per_unit=("mae_eur_per_unit", "min"), n=("n", "first")))
    metrics = ["selling_price_change_eur", "break_even_volume_change_for_revenue",
               "high_stress_cost_change_net_of_price_eur_per_unit"]
    compare = pricing.pivot(index=KEY + ["h"], columns="policy", values=metrics)
    compare.columns = [f"{policy}_{metric}" for metric, policy in compare.columns]
    base = pricing.loc[pricing["policy"].eq("hold_price"), KEY + ["h", "base_price_source", "base_price_eur_ex_vat",
                       "central_material_cost_change_eur_per_unit", "empirical_p90_cost_change_net_of_price_eur_per_unit",
                       "planning_p50_eur_per_unit"]]
    base = base.rename(columns={"planning_p50_eur_per_unit": "planning_cost_reference_eur_per_unit"})
    return (base.assign(planning_method="flat_cost", evaluation=scores["evaluation"].iat[0])
            .merge(best, on=keys, validate="one_to_one").merge(compare.reset_index(), on=KEY + ["h"], validate="one_to_one"))
