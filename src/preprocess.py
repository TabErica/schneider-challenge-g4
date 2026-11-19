"""Preprocessing: every source -> one monthly panel in EUR, cut at the forecast origin -> outputs/preprocessed/.

Steps, in this order (each one is written to transform_log.csv):
  1. calendar  dates -> first day of the month, one row per month
  2. cutoff    drop every month after CUTOFF (2026-09). Done before any fill, so no later value can leak in.
  3. drop      empty licensed series (L01-L08), E23 (copy of S06), Westmetall provisional month-to-date row
  4. correct   known data errors (CORRECTIONS) -> NaN. The evidence is in notebooks/01_data_quality.ipynb.
  5. fill      interior gaps of at most MAX_GAP months, log-linear. Leading / trailing gaps stay NaN.
  6. derive    ss304_de = base + alloy surcharge (after the fill, so it uses the filled parts)
  7. fx        USD prices without an EUR twin -> EUR with the ECB monthly mean (S06, USD per 1 EUR)

Outputs: panel_monthly.csv (month x series), series_catalog.csv (one row per series), transform_log.csv.
Not in the panel: the price benchmark (yearly, see data.load_price_benchmark) and the BOM (data.load_bom).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import data as D

OUT = D.OUT / "preprocessed"
CUTOFF = pd.Timestamp("2026-09-01")  # last month used: data for Sep-2026 is the latest known at the origin
DOWNLOAD = pd.Timestamp("2026-11-01")  # month the files were retrieved (data.zip README: 12 Nov 2026)
MAX_GAP = 2  # months

# (series, month, reason). Values are set to NaN, then the fill step interpolates them.
CORRECTIONS = [
    ("ech", "2026-05", "ECH 257 EUR/t vs 2 479 (Apr) and 2 607 (Jun): about 10x too low, decimal-shift typo"),
    ("epoxy", "2026-05", "epoxy is built from ECH, so it carries the same typo (3 593 vs ~4 950 around it)"),
]

# USD prices that have no EUR version in the source files. Existing twins: S01->D01, S02->D02, E01->ED01,
# E05->ED02, E06->ED04, E07->ED03 (made by the data provider with the same ECB rate).
TO_EUR = ["S03", "S04", "S05", "E02", "E03", "E08", "WM_CU_CASH", "WM_CU_3M", "WM_AL_CASH", "WM_AL_3M"]

STEEL_META = {
    "meps_cr_de": ("CR coil Germany, MEPS low value", "EUR/t"),
    "cru_cr_de": ("CR coil Germany, CRU monthly average", "EUR/t"),
    "cru_hdg_de": ("HDG coil Germany, CRU monthly average", "EUR/t"),
    "ss304_base_de": ("Stainless 304 CR 2mm Germany, CRU base price", "EUR/t"),
    "ss304_alloy_de": ("Stainless 304 CR 2mm Germany, CRU alloy surcharge", "EUR/t"),
}
EPOXY_META = {
    "benzene": "Benzene Europe", "acetone": "Acetone Europe", "ech": "Epichlorohydrin (ECH) Europe",
    "epoxy_other": "Epoxy other costs (calculated in file)", "epoxy": "Epoxy resin Europe",
}
WM_META = {
    "WM_CU_CASH": ("Copper LME cash, monthly avg", "USD/t"), "WM_CU_3M": ("Copper LME 3-month, monthly avg", "USD/t"),
    "WM_AL_CASH": ("Aluminium LME cash, monthly avg", "USD/t"),
    "WM_AL_3M": ("Aluminium LME 3-month, monthly avg", "USD/t"),
    "WM_FX": ("EUR/USD ECB fixing, monthly avg (USD per EUR)", "USD per EUR"),
    "WM_CU_STOCK": ("Copper LME stocks, monthly avg", "t"), "WM_AL_STOCK": ("Aluminium LME stocks, monthly avg", "t"),
}


class Log:
    def __init__(self):
        self.rows = []

    def __call__(self, step, series, detail, n=None):
        self.rows.append({"step": step, "series": series, "detail": detail, "n": n})

    def frame(self):
        return pd.DataFrame(self.rows)


def _to_month_start(df, src, log):
    """Step 1: every date -> first day of its month; refuse two rows in one month.
    Loaders that read row by row (steel, epoxy) already do this and report it in df.attrs["dates_moved"]."""
    idx = pd.DatetimeIndex(df.index)
    month = idx.to_period("M").to_timestamp()
    if month.duplicated().any():
        raise ValueError(f"{src}: several rows in one month: {sorted(set(month[month.duplicated()]))}")
    moved = dict(df.attrs.get("dates_moved", {}))
    if (idx.day != 1).any():
        moved[src] = moved.get(src, 0) + int((idx.day != 1).sum())
    for name, n in moved.items():
        if n:
            log("calendar", name, "dates not on the 1st moved to the 1st of their month", n)
    return df.set_axis(month)


def load_sources(log):
    """Every source as published, on a month-start index, with its metadata rows for the catalog."""
    rm, en = D.load_rm_all(), D.load_energy()
    steel = D.load_steel_licensed().drop(columns="ss304_de")  # re-derived after the fill
    epoxy = D.load_epoxy()
    wm = D.load_westmetall()

    cat = pd.concat([D.read_catalog("source_catalog.csv"), D.read_catalog("energy_source_catalog.csv")])
    meta = {r.series_id: {"name": r.series_name, "source": r.provider, "unit": r.unit} for r in cat.itertuples()}
    meta |= {k: {"name": n, "source": "Europe Steel INDEXES (internal)", "unit": u} for k, (n, u) in STEEL_META.items()}
    meta |= {k: {"name": n, "source": "Epoxy for student.xlsx", "unit": "EUR/t"} for k, n in EPOXY_META.items()}
    meta |= {k: {"name": n, "source": "Westmetall (LME)", "unit": u} for k, (n, u) in WM_META.items()}

    epoxy = epoxy.rename(columns={"other": "epoxy_other"})
    parts = {"data.zip RM": rm, "data.zip energy": en, "Steel INDEXES": steel, "Epoxy": epoxy}
    parts = {k: _to_month_start(v, k, log) for k, v in parts.items()}

    # Westmetall: drop the provisional month-to-date row before anything else
    provisional = wm.index[wm["status"] != "Completed month"]
    for m in provisional:
        log("drop", "Westmetall", f"{m:%Y-%m} is a provisional month-to-date average", 1)
    parts["Westmetall"] = _to_month_start(wm.drop(index=provisional).drop(columns="status"), "Westmetall", log)

    overlap = set(rm.columns) & set(en.columns)
    if overlap:
        raise ValueError(f"series id used twice: {overlap}")
    for name, df in parts.items():
        log("load", name, f"{df.shape[1]} series, {df.index.min():%Y-%m} to {df.index.max():%Y-%m}", int(df.count().sum()))
    return pd.concat(parts.values(), axis=1, sort=True), meta


def apply_cutoff(p, log):
    """Step 2: keep months <= CUTOFF."""
    after = p.loc[p.index > CUTOFF]
    for s, n in after.count().items():
        if n:
            log("cutoff", s, f"dropped {n} value(s) after {CUTOFF:%Y-%m} (to {after[s].last_valid_index():%Y-%m})", int(n))
    return p.loc[:CUTOFF]


def drop_series(p, log):
    """Step 3: empty licensed columns and the duplicate FX column."""
    empty = [c for c in p.columns if p[c].isna().all()]
    for c in empty:
        log("drop", c, "no value in any month (licensed series, empty on purpose)", 0)
    p = p.drop(columns=empty)
    diff = (p["E23"] - p["S06"]).abs().max()
    if diff > 1e-9:
        raise ValueError(f"E23 differs from S06 by up to {diff}; check before dropping")
    log("drop", "E23", "identical to S06 (ECB USD per EUR); S06 is kept", int(p["E23"].count()))
    return p.drop(columns="E23")


def apply_corrections(p, log):
    """Step 4: known errors -> NaN."""
    p = p.copy()
    for s, month, reason in CORRECTIONS:
        m = pd.Timestamp(month)
        if m > CUTOFF:
            continue
        log("correct", s, f"{month}: {p.at[m, s]:.2f} set to NaN. {reason}", 1)
        p.at[m, s] = np.nan
    return p


def interior_gaps(s):
    """(start, end, length) of each NaN run that has values on both sides."""
    s = s.loc[s.first_valid_index():s.last_valid_index()]
    run = s.isna().ne(s.isna().shift()).cumsum()[s.isna()]
    return [(g.index[0], g.index[-1], len(g)) for _, g in run.groupby(run)]


def fill_gaps(p, log):
    """Step 5: interior gaps up to MAX_GAP months -> log-linear interpolation (linear if a value is <= 0)."""
    p = p.copy()
    for c in p.columns:
        for a, b, n in interior_gaps(p[c]):
            if n > MAX_GAP:
                log("fill", c, f"gap {a:%Y-%m} to {b:%Y-%m} ({n} months) longer than {MAX_GAP}: left NaN", 0)
                continue
            seg = p[c].loc[a - pd.DateOffset(months=1):b + pd.DateOffset(months=1)]
            log_ok = (seg.dropna() > 0).all()
            filled = np.exp(np.log(seg).interpolate()) if log_ok else seg.interpolate()
            p.loc[a:b, c] = filled.loc[a:b]
            log("fill", c, f"{a:%Y-%m} to {b:%Y-%m} ({n} months), {'log-' if log_ok else ''}linear interpolation", n)
    return p


def derive(p, log):
    """Step 6: stainless 304 transaction price = base + alloy surcharge."""
    p["ss304_de"] = p["ss304_base_de"] + p["ss304_alloy_de"]
    log("derive", "ss304_de", "ss304_base_de + ss304_alloy_de", int(p["ss304_de"].count()))
    return p


def convert_fx(p, log):
    """Step 7: USD -> EUR with the ECB monthly mean. Before S06 starts (2016-11), Westmetall's ECB rate is used."""
    fx = p["S06"].combine_first(p["WM_FX"])
    backfilled = int((p["S06"].isna() & p["WM_FX"].notna() & p[TO_EUR].notna().any(axis=1)).sum())
    if backfilled:
        log("fx", "S06", f"S06 starts {p['S06'].first_valid_index():%Y-%m}; earlier months use WM_FX "
            "(same ECB fixings, published by Westmetall)", backfilled)
    for s in TO_EUR:
        p[f"{s}_EUR"] = p[s] / fx
        log("fx", f"{s}_EUR", f"{s} / USD-per-EUR", int(p[f"{s}_EUR"].count()))
    return p


def build_catalog(p, raw, meta, log):
    """One row per panel series: what it is, where it comes from, coverage, and what was done to it."""
    lg = log.frame()
    rows = []
    for c in p.columns:
        base = c[:-4] if c.endswith("_EUR") else c
        m = dict(meta.get(base, {"name": c, "source": "derived", "unit": ""}))
        if c.endswith("_EUR"):
            m = {"name": m["name"] + " in EUR", "source": f"derived ({base} / S06)",
                 "unit": m["unit"].replace("USD", "EUR")}
        if c == "ss304_de":
            m = {"name": "Stainless 304 CR 2mm Germany, base + alloy", "source": "derived", "unit": "EUR/t"}
        s, r = p[c], raw[base] if base in raw else None
        last_raw = r.last_valid_index() if r is not None else s.last_valid_index()
        steps = lg[lg["series"] == c]
        rows.append({
            "id": c, **m, "first": s.first_valid_index(), "last": s.last_valid_index(), "n_obs": int(s.count()),
            "n_corrected": int(steps.loc[steps.step == "correct", "n"].sum()),
            "n_filled": int(steps.loc[steps.step == "fill", "n"].sum()),
            "n_after_cutoff": int(steps.loc[steps.step == "cutoff", "n"].sum()),
            "last_in_file": last_raw,
            # Months between the last value in the file and the download month, minus one: a rough publication lag
            "lag_m": (DOWNLOAD.to_period("M") - last_raw.to_period("M")).n - 1 if last_raw is not None else None,
        })
    cat = pd.DataFrame(rows)
    for c in ["first", "last", "last_in_file"]:
        cat[c] = pd.to_datetime(cat[c]).dt.strftime("%Y-%m")
    return cat


def build(save=True):
    log = Log()
    raw, meta = load_sources(log)
    p = apply_cutoff(raw, log)
    p = drop_series(p, log)
    p = apply_corrections(p, log)
    p = fill_gaps(p, log)
    p = derive(p, log)
    p = convert_fx(p, log)
    p.index.name = "month"
    cat = build_catalog(p, raw, meta, log)
    if save:
        OUT.mkdir(parents=True, exist_ok=True)
        p.to_csv(OUT / "panel_monthly.csv", float_format="%.6g")
        cat.to_csv(OUT / "series_catalog.csv", index=False)
        log.frame().to_csv(OUT / "transform_log.csv", index=False)
    return p, cat, log.frame()


def load():
    """Preprocessed panel (built if missing)."""
    f = OUT / "panel_monthly.csv"
    if not f.exists():
        return build()[0]
    return pd.read_csv(f, index_col="month", parse_dates=["month"])


def as_of(p, cat, origin):
    """Panel as it could be seen at `origin`: each series cut `lag_m` months before it (lag from the catalog)."""
    origin = pd.Timestamp(origin)
    lag = cat.set_index("id")["lag_m"].fillna(0).clip(lower=0)
    out = p.loc[:origin].copy()
    for c in out.columns:
        out.loc[out.index > origin - pd.DateOffset(months=int(lag.get(c, 0))), c] = np.nan
    return out


if __name__ == "__main__":
    panel, catalog, steps = build()
    print(f"panel: {panel.shape[0]} months x {panel.shape[1]} series, "
          f"{panel.index.min():%Y-%m} to {panel.index.max():%Y-%m}")
    print(steps.groupby("step", sort=False)["n"].agg(["count", "sum"]).to_string())
    print(steps[steps.step.isin(["correct", "fill", "fx"]) & (steps.series != "S06")].head(20).to_string(index=False))
    print("wrote", OUT)
