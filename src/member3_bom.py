"""Audit material weight and translate commodity prices to EUR per finished unit."""
from __future__ import annotations

import numpy as np
import pandas as pd

import data as D
from models.base import HORIZONS


BOM_FILE = D.DATA / "RM_Prices_XHEC_Challenge_v1.xlsx"
TOTAL_MASS_ROWS = {
    "RM6_NE_IQI_433520": "Source total mass – SFCME / RM6 (kg)",
    "RMAirSeT_IIQ_24kV": "Implied total mass – RMAirSeT (kg) = mass / share",
}
COST = "material_cost_eur_per_unit"


def material_price(label, family):
    """Return a forecast series and an optional EUR/t anchor for an index series."""
    if label == "Epoxy Resin EU":
        return "epoxy", np.nan
    if family == "Plastic" and label != "PET Polyethylene Terephthalate EU":
        return None, np.nan  # PE, PVC and PP have no grade-specific price anchor.
    _, proxy, level = D.FAMILY_MAP.get(family, (None, None, None))
    if level in D.ASSUMED_PRICE_EUR_T:
        return proxy, D.ASSUMED_PRICE_EUR_T[level]
    return level, np.nan


def workbook_rows(path):
    raw = pd.read_excel(path, sheet_name="BOM_Materials")
    raw.columns = raw.columns.str.strip()
    rows = raw.loc[raw["product_ref"].isin(D.PRODUCT_ID),
                   ["product_ref", "material_label_source", "material_family", "mass_kg", "share_pct"]].copy()
    rows.insert(0, "product", rows.pop("product_ref").map(D.PRODUCT_ID))
    return rows


def load_bom():
    raw = pd.read_excel(BOM_FILE, sheet_name="BOM_Materials")
    masses = raw.set_index("product_ref")["mass_kg"]
    total = pd.Series({product: masses.loc[row] for product, row in TOTAL_MASS_ROWS.items()})
    bom = workbook_rows(BOM_FILE)
    if bom.empty or set(bom["product"]) != set(total.index):
        raise ValueError(f"{BOM_FILE.name}: missing an expected product")
    if not np.isfinite(bom["mass_kg"]).all() or not bom["mass_kg"].gt(0).all():
        raise ValueError(f"{BOM_FILE.name}: each BOM line requires a positive mass")
    if not np.isfinite(total).all() or not total.gt(0).all():
        raise ValueError(f"{BOM_FILE.name}: invalid total product mass")
    remainder = total - bom.groupby("product")["mass_kg"].sum()
    if (remainder < -1e-6).any():
        raise ValueError(f"{BOM_FILE.name}: BOM masses exceed the reported product mass")
    unassigned = remainder[remainder > 1e-6].rename("mass_kg").rename_axis("product").reset_index()
    if not unassigned.empty:
        bom = pd.concat([bom, unassigned.assign(material_label_source="Mass not allocated in workbook",
                                                material_family="To confirm")], ignore_index=True)
    sources = [material_price(row.material_label_source, row.material_family) for row in bom.itertuples()]
    bom["series"], bom["assumed_base_eur_per_t"] = zip(*sources)
    bom["price_basis"] = np.select(
        [bom["series"].isna(), bom["assumed_base_eur_per_t"].notna()],
        ["unpriced", "assumed_level_indexed"], default="market_price_proxy",
    )
    return bom


def bom_audit(bom, detail):
    """Compare source versions using the same current product mass and EUR/t reference levels."""
    total = bom.groupby("product")["mass_kg"].sum()
    base = detail.loc[detail["h"].eq(min(HORIZONS))]
    origin_indices = base.loc[base["price_basis"].eq("market_price_proxy")].drop_duplicates("series")
    market = origin_indices.set_index("series")["origin_price_eur_per_t"]
    legacy = workbook_rows(D.BOM_FILE)
    assumed_carbon = D.load_bom().loc[lambda d: d["material"].eq("Carbon steel")].groupby("product")["kg"].sum()
    audit = []
    for version, workbook in [("legacy", legacy), ("v1", bom)]:
        for product, rows in workbook.groupby("product"):
            rows = rows.copy()
            source_rows = rows["material_label_source"].ne("Mass not allocated in workbook")
            missing_mass = rows["mass_kg"].isna()
            missing_share = rows["share_pct"].isna()
            rows["mass_missing_in_source"] = source_rows & missing_mass
            rows["share_missing_in_source"] = source_rows & missing_share
            rows["effective_mass_kg"] = rows["mass_kg"]
            assumed = rows["material_family"].eq("Carbon steel (to confirm)") & missing_mass
            if assumed.any():
                rows.loc[assumed, "effective_mass_kg"] = assumed_carbon.loc[product]
            mappings = [material_price(r.material_label_source, r.material_family) for r in rows.itertuples()]
            rows["series"], anchors = zip(*mappings)
            rows["reference_eur_per_t"] = pd.Series(anchors, index=rows.index).fillna(rows["series"].map(market))
            unknown = rows["material_family"].eq("To confirm")
            rows["coverage_status"] = np.select(
                [assumed, unknown, rows["series"].notna()],
                ["assumed_carbon_steel", "unidentified", "priced"], default="identified_unpriced",
            )
            gap = total.loc[product] - rows["effective_mass_kg"].sum()
            if gap > 1e-6:
                missing = {"product": product, "material_label_source": "Mass not allocated in workbook",
                           "material_family": "To confirm", "effective_mass_kg": gap, "coverage_status": "unidentified",
                           "mass_missing_in_source": False, "share_missing_in_source": False}
                rows = pd.concat([rows, pd.DataFrame([missing])], ignore_index=True)
            rows["origin_cost_eur_per_unit"] = rows["effective_mass_kg"] * rows["reference_eur_per_t"] / 1000
            rows["total_mass_kg"] = total.loc[product]
            rows["mass_share_of_current_total"] = rows["effective_mass_kg"] / rows["total_mass_kg"]
            audit.append(rows.assign(bom_version=version, workbook_rows=source_rows.sum()))
    reconciliation = pd.concat(audit, ignore_index=True)
    coverage = (reconciliation.groupby(["product", "bom_version", "coverage_status"], as_index=False)
                .agg(effective_mass_kg=("effective_mass_kg", "sum"),
                     mass_share=("mass_share_of_current_total", "sum"),
                     origin_cost_eur_per_unit=("origin_cost_eur_per_unit", lambda s: s.sum(min_count=1)),
                     source_missing_mass_rows=("mass_missing_in_source", "sum"),
                     source_missing_share_rows=("share_missing_in_source", "sum"),
                     source_rows=("workbook_rows", "max"), total_mass_kg=("total_mass_kg", "first")))
    return reconciliation, coverage


def assumption_sensitivity(summary, detail):
    """One-way what-if tests for unpriced mass and the PET/engineering plastic EUR/t anchors."""
    keys = ["product", "target_month"]
    priced = summary[keys + ["h", "unpriced_mass_kg", f"central_{COST}"]].copy()
    polymer = (detail.loc[detail["price_basis"].eq("assumed_level_indexed")]
               .groupby(keys, as_index=False)[f"central_{COST}"].sum()
               .rename(columns={f"central_{COST}": "polymer_cost_eur_per_unit"}))
    priced = priced.merge(polymer, on=keys, how="left", validate="one_to_one").fillna({"polymer_cost_eur_per_unit": 0})
    missing = priced.merge(pd.DataFrame({"unpriced_assumed_eur_per_kg": [1.0, 3.0, 6.0]}), how="cross")
    missing["impact_eur_per_unit"] = missing["unpriced_mass_kg"] * missing["unpriced_assumed_eur_per_kg"]
    polymers = priced.merge(pd.DataFrame({"polymer_anchor_change": [-0.25, 0.25]}), how="cross")
    polymers["impact_eur_per_unit"] = polymers["polymer_cost_eur_per_unit"] * polymers["polymer_anchor_change"]
    scenarios = pd.concat([missing.assign(test="unpriced_mass"), polymers.assign(test="polymer_anchor")], ignore_index=True)
    scenarios["scenario_material_cost_eur_per_unit"] = scenarios[f"central_{COST}"] + scenarios["impact_eur_per_unit"]
    return scenarios


def build_costs(bom, forecasts):
    horizons = pd.DataFrame({"h": HORIZONS, "target_month": [
        str(D.BOM_BASE.to_period("M") + h) for h in HORIZONS
    ]})
    detail = bom.merge(horizons, how="cross").merge(forecasts, on=["series", "h"], how="left", validate="many_to_one")
    detail.insert(1, "origin_month", f"{D.BOM_BASE:%Y-%m}")
    scale = (detail["assumed_base_eur_per_t"] / detail["y_t"]).where(detail["assumed_base_eur_per_t"].notna(), 1.0)
    states = {"origin": "y_t", "low_stress": "p10", "central": "p50", "high_stress": "p90"}
    cost_cols = []
    for state, forecast_col in states.items():
        price_col = f"{state}_price_eur_per_t"
        cost_col = f"{state}_{COST}"
        detail[price_col] = detail[forecast_col] * scale
        detail[cost_col] = detail["mass_kg"] * detail[price_col] / 1000
        cost_cols.append(cost_col)

    priced = detail["series"].notna()
    detail["priced_mass_kg"] = detail["mass_kg"].where(priced, 0.0)
    group_keys = ["product", "origin_month", "target_month", "h"]
    groups = detail.groupby(group_keys, sort=False)
    summary = groups[["mass_kg", "priced_mass_kg"]].sum().rename(columns={"mass_kg": "total_mass_kg"})
    summary = summary.join(groups[cost_cols].sum(min_count=1)).reset_index()
    summary["unpriced_mass_kg"] = summary["total_mass_kg"] - summary["priced_mass_kg"]
    summary["priced_mass_share"] = summary["priced_mass_kg"] / summary["total_mass_kg"]
    summary["central_material_cost_change_eur_per_unit"] = summary[f"central_{COST}"] - summary[f"origin_{COST}"]
    detail["central_cost_share_of_priced_bom"] = detail[f"central_{COST}"] / groups[f"central_{COST}"].transform("sum")
    detail["central_material_cost_change_eur_per_unit"] = detail[f"central_{COST}"] - detail[f"origin_{COST}"]
    return detail.drop(columns=list(states.values()) + ["priced_mass_kg", "actual", "point", "interval"]), summary
