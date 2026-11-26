"""Translate scenario cost differences into commercial and TCO thresholds for Goal B."""
from __future__ import annotations

import math

import pandas as pd

OLD_PRODUCT = "RM6_NE_IQI_433520"
NEW_PRODUCT = "RMAirSeT_IIQ_24kV"


def migration_thresholds(offers):
    keys = ["h", "target_month", "scenario"]
    fields = ["scenario_bom_cost_eur_per_unit", "scenario_direct_cost_with_input_eur_per_unit", "illustrative_offer_eur_ex_vat"]
    paired = offers.pivot(index=keys, columns="product", values=fields)
    output = []
    for index, row in paired.iterrows():
        modeled = row[fields[0], NEW_PRODUCT] - row[fields[0], OLD_PRODUCT]
        complete = row[fields[1], NEW_PRODUCT] - row[fields[1], OLD_PRODUCT]
        price_gap = row[fields[2], NEW_PRODUCT] - row[fields[2], OLD_PRODUCT]
        premium = complete if pd.notna(complete) else modeled
        for years in (10, 20, 30):
            for rate in (0.04, 0.08, 0.12):
                annuity = -math.expm1(-years * math.log1p(rate)) / rate
                output.append(dict(zip(keys, index, strict=True)) | {
                    "modeled_bom_gap_airset_minus_rm6_eur_per_unit": modeled,
                    "direct_cost_gap_with_both_commercial_inputs_eur_per_unit": complete,
                    "illustrative_offer_gap_airset_minus_rm6_eur_per_unit": price_gap,
                    "equal_unit_contribution_premium_eur": premium,
                    "premium_basis": "both_direct_cost_inputs_plus_modeled_wait_reserve" if pd.notna(complete) else "equal_unmodeled_costs_assumption",
                    "assumed_service_years": years, "assumed_discount_rate": rate, "annuity_factor": annuity,
                    "annual_tco_savings_needed_at_contribution_neutral_premium_eur_per_unit": max(0, premium) / annuity,
                    "annual_tco_savings_needed_at_illustrative_offer_gap_eur_per_unit": max(0, price_gap) / annuity,
                })
    return pd.DataFrame(output)
