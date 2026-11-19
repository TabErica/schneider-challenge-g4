"""Load every source as monthly series (raw files read in place, zips not extracted)."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data" / "Data"
OUT = ROOT / "outputs"

# Priceable BOM material proxies (BOM eu/ww series ids, licensed L-series excluded): id -> (name, family, region, unit)
TARGETS = {
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
DRIVERS = {"ED01": "Brent (EUR/bbl)", "ED02": "EU gas TTF (EUR/MWh)", "E09": "Power Germany (EUR/MWh)", "S06": "EUR/USD"}

# BOM masses come from data.zip (Schneider-internal, never hard-coded). Here only: proxy series and EUR/t source.
PRODUCT_ID = {"RMAirSeT IIQ 24kV IEC": "RMAirSeT_IIQ_24kV", "RM6_NE_IQI_433520": "RM6_NE_IQI_433520"}
FAMILY_MAP = {  # BOM material_family -> (material, price proxy, EUR/t level source)
    "Carbon steel": ("Carbon steel", "S09", "cru_cr_de"),
    "Carbon steel (to confirm)": ("Carbon steel", "S09", "cru_cr_de"),
    "Stainless steel": ("Stainless steel", "S09", "ss304_de"),
    "Copper": ("Copper", "D01", "D01"),
    "Galvanised steel": ("Galvanised steel", "S10", "cru_hdg_de"),
    "Plastic": ("Plastics", "S11", "plastic_pet"),
    "Engineering plastic": ("Plastics", "S11", "plastic_etp"),
    "Aluminium": ("Aluminium", "D02", "D02"),
}
# Assumptions to confirm with Schneider
ASSUMED_PRICE_EUR_T = {"plastic_pet": 1300.0, "plastic_etp": 2700.0}
ASSUMED_UNLABELLED_SHARE = 0.45  # RM6 largest unlabelled slice = carbon steel
BOM_BASE = pd.Timestamp("2026-09-01")


def _read_zip_csv(zip_name, member, **kw):
    """Read `member` from the zip if present, else from the already-extracted folder (DATA / member)."""
    if not (DATA / zip_name).exists() and (DATA / member).exists():
        return pd.read_csv(DATA / member, **kw)
    with zipfile.ZipFile(DATA / zip_name) as z:
        return pd.read_csv(io.BytesIO(z.read(member)), **kw)


def load_rm():
    df = _read_zip_csv("data.zip", "data/rm_prices_monthly_wide.csv", parse_dates=["month"]).set_index("month")
    return df[[c for c in df.columns if c[0] in "SD"]].astype(float)


def load_energy():
    df = _read_zip_csv("data.zip", "data/energy_prices_monthly_wide.csv", parse_dates=["month"])
    return df.set_index("month").astype(float)


def _sheet_series(wb, sheet, col, start_row):
    out = {}
    for r in wb[sheet].iter_rows(values_only=True, min_row=start_row):
        d, v = r[0], r[col] if col < len(r) else None
        if hasattr(d, "year") and isinstance(v, (int, float)):
            out[pd.Timestamp(d.year, d.month, 1)] = float(v)
    return pd.Series(out).sort_index()


def load_steel_licensed():
    """CRU / MEPS EUR/t prices, Germany (internal workbook)."""
    wb = openpyxl.load_workbook(DATA / "Copia de Europe Steel INDEXES.xlsx", read_only=True, data_only=True)
    df = pd.DataFrame({
        "meps_cr_de": _sheet_series(wb, "MEPS", 2, 6),
        "cru_cr_de": _sheet_series(wb, "CRU Flat", 12, 7),
        "cru_hdg_de": _sheet_series(wb, "CRU Flat", 18, 7),
        "ss304_base_de": _sheet_series(wb, "CRU Stainless", 1, 6),
        "ss304_alloy_de": _sheet_series(wb, "CRU Stainless", 2, 6),
    })
    df["ss304_de"] = df["ss304_base_de"] + df["ss304_alloy_de"]
    return df


def load_epoxy():
    """Epoxy cost build-up and data-quality notes."""
    raw = pd.read_excel(DATA / "Epoxy for student.xlsx", header=0)
    raw.columns = ["date", "_", "benzene", "acetone", "ech", "other", "epoxy"]
    raw = raw.dropna(subset=["date"])
    raw["month"] = pd.to_datetime(raw["date"]).dt.to_period("M").dt.to_timestamp()
    df = raw.set_index("month")[["benzene", "acetone", "ech", "other", "epoxy"]].astype(float)
    med = df["ech"].rolling(5, center=True, min_periods=3).median()
    notes = [f"ECH {m:%Y-%m} = {df.at[m, 'ech']:.0f} (rolling median {med[m]:.0f}): likely typo"
             for m in df.index[(df["ech"] / med) < 0.3]]
    return df, notes


def load_price_benchmark():
    df = pd.read_excel(DATA / "RMU_IQI_EU_Price_Benchmark_2017-2026_Synthetic.xlsx", sheet_name="ML_Trend", header=3)
    df = df.dropna(subset=["Brand"])
    df.columns = ["id", "brand", "year", "p10", "p50", "p90", "yoy"]
    return df[["id", "brand", "year", "p10", "p50", "p90"]].astype({"year": int})


def eur_levels(rm, steel):
    lv = {k: float(steel[k].dropna().loc[:BOM_BASE].iloc[-1]) for k in ["cru_cr_de", "cru_hdg_de", "ss304_de"]}
    lv |= {k: float(rm.at[BOM_BASE, k]) for k in ["D01", "D02"]}
    return lv | ASSUMED_PRICE_EUR_T


def load_bom():
    """Priceable BOM lines (kg per unit) from data.zip, with the unlabelled-slice assumption applied."""
    raw = _read_zip_csv("data.zip", "data/bom_materials.csv")
    raw = raw[raw["product_ref"].isin(PRODUCT_ID)]
    rows = []
    for ref, g in raw.groupby("product_ref"):
        known = g.dropna(subset=["mass_kg", "share_pct"])
        implied_total = (known["mass_kg"] / known["share_pct"]).median()
        for r in g.itertuples():
            if r.material_family not in FAMILY_MAP:
                continue
            kg = ASSUMED_UNLABELLED_SHARE * implied_total if "to confirm" in r.material_family else r.mass_kg
            mat, proxy, src = FAMILY_MAP[r.material_family]
            rows.append({"product": PRODUCT_ID[ref], "material": mat, "proxy": proxy, "level_src": src, "kg": kg})
    return pd.DataFrame(rows)


def bom_cost_table(rm, steel):
    lv = eur_levels(rm, steel)
    b = load_bom()
    b["eur_per_unit"] = b["kg"] * b["level_src"].map(lv) / 1000
    df = b.groupby(["product", "material", "proxy"], as_index=False)[["kg", "eur_per_unit"]].sum()
    df["eur_per_t"] = 1000 * df["eur_per_unit"] / df["kg"]
    df["cost_share"] = df["eur_per_unit"] / df.groupby("product")["eur_per_unit"].transform("sum")
    return df


def bom_index(rm, steel):
    """Laspeyres material-cost index per product, fixed Sep-2026 cost shares, Sep-2026 = 100."""
    out = {}
    for prod, g in bom_cost_table(rm, steel).groupby("product"):
        out[prod] = 100 * sum(r.cost_share * rm[r.proxy] / rm.at[BOM_BASE, r.proxy] for r in g.itertuples())
    return pd.DataFrame(out)


def panel():
    """Targets + BOM indices + drivers, Nov-2016 .. Sep-2026."""
    rm, en, steel = load_rm(), load_energy(), load_steel_licensed()
    df = rm[list(TARGETS)].join(bom_index(rm, steel)).join(en[[d for d in DRIVERS if d != "S06"]])
    df["S06"] = rm["S06"]
    return df.loc[:BOM_BASE]
