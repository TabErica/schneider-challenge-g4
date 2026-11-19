"""Load and align every data source into monthly, EUR-based pandas objects.

Raw files are read in place (zips are not extracted) so the Data/ folder stays untouched.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data" / "Data"
OUT = ROOT / "outputs"

# --------------------------------------------------------------------------------------
# Target catalogue: every series referenced by the BOM (eu_series_ids / ww_series_ids),
# minus the empty licensed L-series. These are the "priceable BOM material proxies".
# --------------------------------------------------------------------------------------
TARGETS = {
    # id: (short name, family, region, unit)
    "D01": ("Copper (EUR/t)", "Base metals", "EU", "EUR/t"),
    "S07": ("Copper PPI EU", "Base metals", "EU", "index"),
    "S01": ("Copper (USD/t)", "Base metals", "WW", "USD/t"),
    "D02": ("Aluminium (EUR/t)", "Base metals", "EU", "EUR/t"),
    "S08": ("Aluminium PPI EU", "Base metals", "EU", "index"),
    "S02": ("Aluminium (USD/t)", "Base metals", "WW", "USD/t"),
    "S03": ("Nickel (USD/t)", "Base metals", "WW", "USD/t"),
    "S05": ("Zinc (USD/t)", "Base metals", "WW", "USD/t"),
    "S04": ("Iron ore (USD/t)", "Steel", "WW", "USD/t"),
    "S09": ("Steel PPI EU", "Steel", "EU", "index"),
    "S10": ("Steel PPI Germany", "Steel", "EU", "index"),
    "S12": ("Steel PPI US", "Steel", "WW", "index"),
    "S11": ("Plastics PPI EU", "Plastics", "EU", "index"),
    "S13": ("Plastics PPI US", "Plastics", "WW", "index"),
}

# Energy / macro drivers used as leading indicators (README §9).
DRIVERS = {
    "ED01": "Brent (EUR/bbl)",
    "ED02": "EU gas TTF (EUR/MWh)",
    "E09": "Power Germany (EUR/MWh)",
    "E10": "Power France (EUR/MWh)",
    "E11": "Power Hungary (EUR/MWh)",
    "E22": "IMF energy index",
    "S06": "EUR/USD",
}

# BOM masses are Schneider-internal: they are READ from data.zip/bom_materials.csv, never hard-coded.
# Only our modelling choices live here: which price proxy and which EUR/t level each material uses.
PRODUCT_ID = {"RMAirSeT IIQ 24kV IEC": "RMAirSeT_IIQ_24kV", "RM6_NE_IQI_433520": "RM6_NE_IQI_433520"}
FAMILY_MAP = {  # bom material_family -> (our material name, price proxy series, EUR/t level source)
    "Carbon steel": ("Carbon steel", "S09", "cru_cr_de"),
    "Carbon steel (to confirm)": ("Carbon steel", "S09", "cru_cr_de"),
    "Stainless steel": ("Stainless steel", "S09", "ss304_de"),
    "Copper": ("Copper", "D01", "D01"),
    "Galvanised steel": ("Galvanised steel", "S10", "cru_hdg_de"),
    "Plastic": ("Plastics", "S11", "plastic_pet"),
    "Engineering plastic": ("Plastics", "S11", "plastic_etp"),
    "Aluminium": ("Aluminium", "D02", "D02"),
}
# ASSUMPTIONS (to confirm with Schneider) ------------------------------------------------
ASSUMED_PRICE_EUR_T = {"plastic_pet": 1300.0, "plastic_etp": 2700.0}  # no free EUR/t level for polymers
ASSUMED_UNLABELLED_SHARE = 0.45  # RM6 largest unlabelled slice (~45%) assumed carbon steel (BOM comment)
BOM_BASE = pd.Timestamp("2026-09-01")


def _read_zip_csv(zip_name: str, member: str, **kw) -> pd.DataFrame:
    with zipfile.ZipFile(DATA / zip_name) as z:
        return pd.read_csv(io.BytesIO(z.read(member)), **kw)


def load_rm() -> pd.DataFrame:
    df = _read_zip_csv("data.zip", "data/rm_prices_monthly_wide.csv", parse_dates=["month"])
    df = df.set_index("month")
    return df[[c for c in df.columns if c[0] in "SD"]].astype(float)


def load_energy() -> pd.DataFrame:
    df = _read_zip_csv("data.zip", "data/energy_prices_monthly_wide.csv", parse_dates=["month"])
    return df.set_index("month").astype(float)


def _sheet_series(wb, sheet: str, col: int, start_row: int) -> pd.Series:
    out = {}
    for r in wb[sheet].iter_rows(values_only=True, min_row=start_row):
        d, v = r[0], r[col] if col < len(r) else None
        if hasattr(d, "year") and isinstance(v, (int, float)):
            out[pd.Timestamp(d.year, d.month, 1)] = float(v)  # 15th-of-month -> month start
    return pd.Series(out).sort_index()


def load_steel_licensed() -> pd.DataFrame:
    """Real EUR/t steel prices from the internal CRU/MEPS workbook (Germany)."""
    wb = openpyxl.load_workbook(DATA / "Copia de Europe Steel INDEXES.xlsx", read_only=True, data_only=True)
    df = pd.DataFrame({
        "meps_cr_de": _sheet_series(wb, "MEPS", 2, 6),
        "meps_hdg_de": _sheet_series(wb, "MEPS", 3, 6),
        "cru_cr_de": _sheet_series(wb, "CRU Flat", 12, 7),   # monthly average column
        "cru_hdg_de": _sheet_series(wb, "CRU Flat", 18, 7),
        "ss304_base_de": _sheet_series(wb, "CRU Stainless", 1, 6),
        "ss304_alloy_de": _sheet_series(wb, "CRU Stainless", 2, 6),
    })
    df["ss304_de"] = df["ss304_base_de"] + df["ss304_alloy_de"]
    return df


def load_epoxy() -> tuple[pd.DataFrame, list[str]]:
    """Epoxy cost build-up. Returns (frame, list of data-quality notes)."""
    raw = pd.read_excel(DATA / "Epoxy for student.xlsx", header=0)
    raw.columns = ["date", "_", "benzene", "acetone", "ech", "other", "epoxy"]
    raw = raw.dropna(subset=["date"])
    raw["month"] = pd.to_datetime(raw["date"]).dt.to_period("M").dt.to_timestamp()  # dates drift 1st->5th
    df = raw.set_index("month")[["benzene", "acetone", "ech", "other", "epoxy"]].astype(float)
    notes = []
    med = df["ech"].rolling(5, center=True, min_periods=3).median()
    bad = df.index[(df["ech"] / med) < 0.3]
    for m in bad:
        notes.append(f"ECH {m:%Y-%m} = {df.at[m, 'ech']:.0f} (rolling median {med[m]:.0f}) -> likely missing digit")
    return df, notes


def load_price_benchmark() -> pd.DataFrame:
    df = pd.read_excel(DATA / "RMU_IQI_EU_Price_Benchmark_2017-2026_Synthetic.xlsx", sheet_name="ML_Trend", header=3)
    df = df.dropna(subset=["Brand"])
    df.columns = ["id", "brand", "year", "p10", "p50", "p90", "yoy"]
    return df[["id", "brand", "year", "p10", "p50", "p90"]].astype({"year": int})


def eur_levels(rm: pd.DataFrame, steel: pd.DataFrame) -> dict[str, float]:
    """Latest available EUR/t level for each level source used in BOM weights."""
    lv = {}
    for key in ["cru_cr_de", "cru_hdg_de", "ss304_de"]:
        s = steel[key].dropna()
        s = s[s.index <= BOM_BASE]
        lv[key] = float(s.iloc[-1])
    for key in ["D01", "D02"]:
        lv[key] = float(rm.at[BOM_BASE, key])
    return lv


def load_bom() -> pd.DataFrame:
    """Priceable BOM lines with kg per unit, from data.zip. Applies the unlabelled-slice assumption."""
    raw = _read_zip_csv("data.zip", "data/bom_materials.csv")
    raw = raw[raw["product_ref"].isin(PRODUCT_ID)].copy()
    raw["product"] = raw["product_ref"].map(PRODUCT_ID)
    out = []
    for prod, g in raw.groupby("product"):
        known = g.dropna(subset=["mass_kg", "share_pct"])
        implied_total = (known["mass_kg"] / known["share_pct"]).median()
        for r in g.itertuples():
            if r.material_family not in FAMILY_MAP:
                continue  # "Other" / "To confirm": not priceable
            kg = r.mass_kg
            if pd.isna(kg) and "to confirm" in r.material_family:
                kg = ASSUMED_UNLABELLED_SHARE * implied_total
            if pd.isna(kg):
                continue
            mat, proxy, src = FAMILY_MAP[r.material_family]
            out.append({"product": prod, "material": mat, "kg": kg, "proxy": proxy, "level_src": src,
                        "kg_assumed": pd.isna(r.mass_kg)})
    df = pd.DataFrame(out)
    agg = df.groupby(["product", "material", "proxy"], as_index=False).agg(
        kg=("kg", "sum"), level_src=("level_src", "first"), kg_assumed=("kg_assumed", "any"))
    # plastics with several grades: mass-weighted assumed price is resolved in bom_cost_table
    agg["level_src"] = agg.apply(lambda r: _plastic_src(df, r) if r.material == "Plastics" else r.level_src, axis=1)
    return agg


def _plastic_src(df, r):
    g = df[(df["product"] == r["product"]) & (df["material"] == "Plastics")]
    return tuple(zip(g["level_src"], g["kg"]))


def bom_cost_table(rm: pd.DataFrame, steel: pd.DataFrame) -> pd.DataFrame:
    lv = eur_levels(rm, steel) | ASSUMED_PRICE_EUR_T
    rows = []
    for r in load_bom().itertuples():
        if isinstance(r.level_src, tuple):  # several plastic grades -> mass-weighted price
            price = sum(lv[s] * kg for s, kg in r.level_src) / sum(kg for _, kg in r.level_src)
            assumed = True
        else:
            price, assumed = lv[r.level_src], r.level_src in ASSUMED_PRICE_EUR_T
        rows.append({"product": r.product, "material": r.material, "kg": r.kg, "kg_assumed": r.kg_assumed,
                     "proxy": r.proxy, "eur_per_t": price, "eur_per_unit": r.kg * price / 1000,
                     "price_assumed": assumed})
    df = pd.DataFrame(rows)
    df["cost_share"] = df["eur_per_unit"] / df.groupby("product")["eur_per_unit"].transform("sum")
    return df


def bom_index(rm: pd.DataFrame, steel: pd.DataFrame) -> pd.DataFrame:
    """Laspeyres BOM cost index per product, base 2026-09 = 100, fixed cost-share weights."""
    costs = bom_cost_table(rm, steel)
    out = {}
    for prod, g in costs.groupby("product"):
        idx = 0.0
        for _, r in g.iterrows():
            p = rm[r["proxy"]]
            idx = idx + r["cost_share"] * p / p[BOM_BASE]
        out[prod] = 100 * idx
    return pd.DataFrame(out)


def panel() -> pd.DataFrame:
    """Monthly panel: all targets + BOM indices + drivers, Nov-2016 .. Sep-2026."""
    rm, en, steel = load_rm(), load_energy(), load_steel_licensed()
    df = rm[list(TARGETS)].join(bom_index(rm, steel)).join(en[[d for d in DRIVERS if d != "S06"]])
    df["S06"] = rm["S06"]
    return df.loc[: BOM_BASE]
