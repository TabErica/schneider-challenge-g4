"""Load every source as monthly series (raw files read in place, zips not extracted)."""
from __future__ import annotations

import functools
import io
import zipfile
from pathlib import Path

import openpyxl
import pandas as pd

import targets as T  # target registry: the single source of what is forecast

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data" / "Data"
OUT = ROOT / "outputs"

# The 14 public-data series (S / D ids), derived from the registry: id -> (name, family, region, unit).
# Kept for the EDA tables; forecasting code reads targets.REGISTRY.
TARGETS = {i: (T.BY_ID[i].name, T.BY_ID[i].family, T.BY_ID[i].region, T.BY_ID[i].unit) for i in T.PUBLIC_IDS}
DRIVERS = {"ED01": "Brent (EUR/bbl)", "ED02": "EU gas TTF (EUR/MWh)", "E09": "Power Germany (EUR/MWh)", "S06": "EUR/USD"}

# BOM masses come from the standalone BOM workbook (Schneider-internal, never hard-coded). It replaces the older
# BOM shipped in data.zip (bom_materials.csv and the BOM_Materials sheet of data/RM_Prices_XHEC_Challenge.xlsx).
BOM_FILE = DATA / "RM_Prices_XHEC_Challenge.xlsx"
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
ASSUMED_UNLABELLED_SHARE = 0.45  # RM6 largest unlabelled slice = carbon steel, ~45% of mass (older BOM label)
BOM_BASE = pd.Timestamp("2026-09-01")
# Market-price version of the BOM index (see bom_index): steel lines follow CRU EUR/t prices, not the Eurostat PPI
MARKET_PROXY = {"Carbon steel": "cru_cr_de", "Stainless steel": "ss304_de", "Galvanised steel": "cru_hdg_de"}
BOM_MKT = "_mkt"  # column suffix of the market version
TRAIL_FILL = 2  # EDA only: months a price may be carried forward to reach BOM_BASE (never for forecasting)


def _read_zip_csv(zip_name, member, **kw):
    """Read `member` from the zip if present, else from the already-extracted folder (DATA / member)."""
    if not (DATA / zip_name).exists() and (DATA / member).exists():
        return pd.read_csv(DATA / member, **kw)
    with zipfile.ZipFile(DATA / zip_name) as z:
        return pd.read_csv(io.BytesIO(z.read(member)), **kw)


def load_rm(prefixes="SD"):
    """Raw-material wide table. prefixes="SDL" also keeps the empty licensed L-series."""
    df = _read_zip_csv("data.zip", "data/rm_prices_monthly_wide.csv", parse_dates=["month"]).set_index("month")
    return df[[c for c in df.columns if c[0] in prefixes]].astype(float)


def load_rm_all():
    return load_rm("SDL")


def read_catalog(name):
    """source_catalog.csv / energy_source_catalog.csv from data.zip."""
    return _read_zip_csv("data.zip", f"data/{name}")


def load_energy():
    df = _read_zip_csv("data.zip", "data/energy_prices_monthly_wide.csv", parse_dates=["month"])
    return df.set_index("month").astype(float)


def _sheet_series(wb, sheet, col, start_row):
    """One sheet column as a monthly series. Dates go to the 1st of their month; how many moved is in attrs."""
    out, moved = {}, 0
    for r in wb[sheet].iter_rows(values_only=True, min_row=start_row):
        d, v = r[0], r[col] if col < len(r) else None
        if hasattr(d, "year") and isinstance(v, (int, float)):
            m = pd.Timestamp(d.year, d.month, 1)
            if m in out:
                raise ValueError(f"{sheet} col {col}: two rows in {m:%Y-%m}")
            out[m], moved = float(v), moved + (d.day != 1)
    s = pd.Series(out).sort_index()
    s.attrs["dates_moved"] = int(moved)
    return s


def load_steel_licensed():
    """CRU / MEPS EUR/t prices, Germany (internal workbook)."""
    wb = openpyxl.load_workbook(DATA / "Copia de Europe Steel INDEXES.xlsx", read_only=True, data_only=True)
    cols = {
        "meps_cr_de": _sheet_series(wb, "MEPS", 2, 6),
        "cru_cr_de": _sheet_series(wb, "CRU Flat", 12, 7),
        "cru_hdg_de": _sheet_series(wb, "CRU Flat", 18, 7),
        "ss304_base_de": _sheet_series(wb, "CRU Stainless", 1, 6),
        "ss304_alloy_de": _sheet_series(wb, "CRU Stainless", 2, 6),
    }
    df = pd.DataFrame(cols)
    df["ss304_de"] = df["ss304_base_de"] + df["ss304_alloy_de"]
    df.attrs["dates_moved"] = {k: s.attrs["dates_moved"] for k, s in cols.items()}
    return df


def load_epoxy():
    """Epoxy cost build-up (EUR/t). Dates go to the 1st of their month; how many moved is in attrs."""
    raw = pd.read_excel(DATA / "Epoxy for student.xlsx", header=0)
    raw.columns = ["date", "_", "benzene", "acetone", "ech", "other", "epoxy"]
    raw = raw.dropna(subset=["date"])
    date = pd.to_datetime(raw["date"])
    raw["month"] = date.dt.to_period("M").dt.to_timestamp()
    if raw["month"].duplicated().any():
        raise ValueError("epoxy file: two rows in one month")
    df = raw.set_index("month")[["benzene", "acetone", "ech", "other", "epoxy"]].astype(float)
    df.attrs["dates_moved"] = {"epoxy file": int((date.dt.day != 1).sum())}
    return df


def load_westmetall():
    """Westmetall LME copper / aluminium monthly averages (cash, 3-month, stocks, ECB FX), as published (USD).

    Keeps the 'Period status' column so the provisional month-to-date row can be dropped downstream."""
    df = pd.read_excel(DATA / "LME_Copper_Aluminium_EU_Monthly_2016_2026.xlsx", sheet_name="Monthly Prices", header=4)
    df = df.rename(columns={
        "Month": "month", "Period status": "status", "Cu cash USD/t": "WM_CU_CASH", "Cu 3m USD/t": "WM_CU_3M",
        "Al cash USD/t": "WM_AL_CASH", "Al 3m USD/t": "WM_AL_3M", "USD per EUR": "WM_FX",
        "Cu avg stocks t": "WM_CU_STOCK", "Al avg stocks t": "WM_AL_STOCK"})
    df = df[pd.to_datetime(df["month"], errors="coerce").notna()]
    df["month"] = pd.to_datetime(df["month"]).dt.to_period("M").dt.to_timestamp()
    cols = ["WM_CU_CASH", "WM_CU_3M", "WM_AL_CASH", "WM_AL_3M", "WM_FX", "WM_CU_STOCK", "WM_AL_STOCK"]
    return df.set_index("month")[["status"] + cols].astype({c: float for c in cols})


def load_price_benchmark():
    df = pd.read_excel(DATA / "RMU_IQI_EU_Price_Benchmark_2017-2026_Synthetic.xlsx", sheet_name="ML_Trend", header=3)
    df = df.dropna(subset=["Brand"])
    df.columns = ["id", "brand", "year", "p10", "p50", "p90", "yoy"]
    return df[["id", "brand", "year", "p10", "p50", "p90"]].astype({"year": int})


LEVEL_IDS = ["cru_cr_de", "cru_hdg_de", "ss304_de", "D01", "D02"]


def _latest(s, at):
    """Latest published value of s at or before `at` (a value, not a fill of the series)."""
    s = s.loc[:at].dropna()
    if s.empty:
        raise ValueError(f"{s.name}: no value at or before {at:%Y-%m}")
    return float(s.iloc[-1])


def eur_levels(rm, steel, at=BOM_BASE):
    """EUR/t level of every BOM line at `at`: latest published value at or before `at` (stainless 304 at Sep-2026:
    its Aug-2026 value), plus the assumed polymer levels."""
    lv = {k: _latest(steel[k], at) for k in LEVEL_IDS[:3]} | {k: _latest(rm[k], at) for k in LEVEL_IDS[3:]}
    return lv | ASSUMED_PRICE_EUR_T


@functools.lru_cache(maxsize=1)
def _load_bom_cached():
    raw = pd.read_excel(BOM_FILE, sheet_name="BOM_Materials")
    raw = raw[raw["product_ref"].isin(PRODUCT_ID)]
    if set(raw["product_ref"]) != set(PRODUCT_ID):
        raise ValueError(f"{BOM_FILE.name}: expected products {sorted(PRODUCT_ID)}, found {sorted(set(raw['product_ref']))}")
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


def load_bom():
    """Priceable BOM lines (kg per unit) from BOM_FILE, with the unlabelled-slice assumption applied."""
    return _load_bom_cached().copy()


def bom_cost_table(rm, steel, at=BOM_BASE):
    """Cost of every BOM line at `at` (kg x EUR/t level) and its share of the product cost. at=BOM_BASE gives the
    submission weights; at=<origin> gives the weights a buyer could compute at that origin."""
    lv = eur_levels(rm, steel, at)
    b = load_bom()
    b["eur_per_unit"] = b["kg"] * b["level_src"].map(lv) / 1000
    df = b.groupby(["product", "material", "proxy"], as_index=False)[["kg", "eur_per_unit"]].sum()
    df["eur_per_t"] = 1000 * df["eur_per_unit"] / df["kg"]
    df["cost_share"] = df["eur_per_unit"] / df.groupby("product")["eur_per_unit"].transform("sum")
    return df


def _mover(material, proxy, version):
    """Price series that moves a BOM line in each index version."""
    return MARKET_PROXY.get(material, proxy) if version == "market" else proxy


def bom_refs(prices, costs, version="proxy", at=BOM_BASE):
    """Reference value of every price that moves a BOM line: its latest published value at or before `at`."""
    return {c: _latest(prices[c], at) for c in {_mover(r.material, r.proxy, version) for r in costs.itertuples()}}


def _price_relative(prices, col, ref, trail_fill=0):
    """prices[col] / ref. trail_fill > 0 (EDA only) carries a series that ends before BOM_BASE (ragged edge:
    stainless ends Aug-2026) forward at most trail_fill months, after its last value only."""
    s = prices[col].loc[:BOM_BASE]
    if trail_fill:
        last = s.last_valid_index()
        s = s.where(s.index <= last, s.ffill(limit=trail_fill))
        if pd.isna(s.get(BOM_BASE)):
            raise ValueError(f"{col}: no value at {BOM_BASE:%Y-%m} even after carrying forward {trail_fill} months")
    return s / ref


def bom_components(prices, costs, version="proxy", refs=None, trail_fill=0):
    """{index column: DataFrame month x material} of 100 x cost share x price relative. Rows sum to bom_index.

    version="proxy":  every line follows its BOM proxy; steel lines follow the Eurostat PPI (S09, S10).
    version="market": steel lines follow the CRU EUR/t price that also sets their weight (MARKET_PROXY), which is
                      the BOM file's first choice (L01-L03). Copper, aluminium and plastics are the same in both.
    refs: reference value of each moving price (default: bom_refs at BOM_BASE). With costs and refs at BOM_BASE the
    index is 100 in Sep-2026 when every input is published; the market version is NaN in Sep-2026 (stainless ends
    Aug-2026) unless trail_fill (EDA only) carries it forward.
    Index column names: product id, plus BOM_MKT for the market version."""
    refs = bom_refs(prices, costs, version) if refs is None else refs
    out = {}
    for prod, g in costs.groupby("product"):
        name = prod + (BOM_MKT if version == "market" else "")
        out[name] = pd.DataFrame({
            r.material: 100 * r.cost_share * _price_relative(
                prices, col := _mover(r.material, r.proxy, version), refs[col], trail_fill)
            for r in g.itertuples()})
    return out


def bom_index(prices, costs, version="proxy", refs=None, trail_fill=0):
    """Laspeyres material-cost index per product: fixed cost shares (`costs`), reference prices `refs`.
    NaN in a month where any material price is missing. See bom_components for the two versions."""
    comp = bom_components(prices, costs, version, refs, trail_fill)
    return pd.DataFrame({k: v.sum(axis=1, min_count=v.shape[1]) for k, v in comp.items()})


def bom_label(sid):
    """Readable name of a BOM index column."""
    prod = "RM6" if sid.startswith("RM6") else "AirSeT"
    return f"BOM {prod} ({'market prices' if sid.endswith(BOM_MKT) else 'PPI proxy'})"


def panel(window="full", trail_fill=0):
    """Preprocessed series + both BOM indices (Sep-2026 weights).

    window="full":   every series of the preprocessed panel with its full history (CRU from 2012, stainless from
                     2008, Westmetall from 2016-01, epoxy from 2018-01), up to Sep-2026. For forecasting.
    window="common": the EDA panel: the 14 public targets, both BOM indices and the drivers, Nov-2016 .. Sep-2026.
    trail_fill: months a BOM input may be carried forward to Sep-2026. EDA only (data.TRAIL_FILL); 0 = no fill,
                so the market-price indices end Aug-2026 with stainless."""
    import preprocess as P  # local import: preprocess imports this module

    c = P.load()
    costs = bom_cost_table(c, c)
    bom = [bom_index(c, costs, v, trail_fill=trail_fill) for v in ("proxy", "market")]
    if window == "full":
        return c.join(bom).loc[:BOM_BASE]
    if window != "common":
        raise ValueError(f"window must be 'full' or 'common', not {window!r}")
    df = c[list(TARGETS)].join(bom + [c[[d for d in DRIVERS if d != "S06"] + ["S06"]]])
    return df.loc[c["D01"].first_valid_index():BOM_BASE]
