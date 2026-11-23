"""Leading indicators of the family models (models/multi.py): what each target may use, when it is known, and proof
that nothing after the origin is used -> outputs/features/.

Every feature is a log change of one source series ending at month m (or m - lag), so its value at m only needs
data up to m. A model forecasting from origin t reads the row of the target's anchor a <= t.

    python features.py      # writes feature_availability.csv and epoxy_identity.csv, runs the leakage test

Outputs
  feature_availability.csv  feature, targets, source series, transformation, lag relative to the origin (months),
                            first month available, economic meaning
  epoxy_identity.csv        is the epoxy price an exact function of benzene, acetone, ECH and "other costs"?
Leakage test (check_no_leakage): at every origin from Jan-2018 to the cutoff, the design matrix of every family target
is built twice, from the panel as seen at the origin (preprocess.as_of + nowcast, as in the backtest) and from the
full panel; the rows at the origin must be identical. Raises AssertionError otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import data as D
import preprocess as P

OUT = D.OUT / "features"


@dataclass(frozen=True)
class Feature:
    name: str
    source: tuple  # one series, or (3-month, cash) for an LME basis
    transform: str  # dlog1 | dlog3 | basis
    lag: int  # months between the end of the change and the row month
    meaning: str


FEATURES = [
    Feature("cru_cr_r1", ("cru_cr_de",), "dlog1", 0,
            "CRU CR coil change in the anchor month: the market price leads the steel PPI by about one month "
            "(data-quality notebook, r ~ 0.7 with the next month's PPI change)"),
    Feature("cru_cr_r1_l1", ("cru_cr_de",), "dlog1", 1,
            "CRU CR coil change one month earlier: the rest of the 1-month lead (contract prices adjust with delay)"),
    Feature("iron_r3", ("S04_EUR",), "dlog3", 0, "Iron ore in EUR: main raw material of blast-furnace steel"),
    Feature("coal_r3", ("E08_EUR",), "dlog3", 0, "Coal in EUR: coking coal and energy cost of steelmaking"),
    Feature("ttf_r3", ("ED02",), "dlog3", 0,
            "EU gas (TTF): energy cost of steel reheating and of aluminium; 3-month change because monthly gas "
            "moves are noisy"),
    Feature("power_r3", ("E09",), "dlog3", 0, "German power price: electricity is the largest cost of aluminium "
            "smelting"),
    Feature("zinc_r3", ("S05_EUR",), "dlog3", 0, "Zinc in EUR: coating metal of hot-dip galvanised coil"),
    Feature("nickel_r1", ("S03_EUR",), "dlog1", 0,
            "Nickel in EUR: the stainless alloy surcharge is set from the nickel price of the previous 1-2 months"),
    Feature("nickel_r1_l1", ("S03_EUR",), "dlog1", 1, "Nickel change one month earlier (second month of the "
            "surcharge reference period)"),
    Feature("brent_r3", ("ED01",), "dlog3", 0,
            "Brent in EUR: polymer feedstock; oil moves of the last 3 months pass into the plastics PPI over the "
            "next 1-3 months"),
    Feature("alu_r3", ("D02",), "dlog3", 0, "Aluminium LME in EUR: leads the EU aluminium PPI (EDA, r = 0.59)"),
    Feature("cu_basis", ("WM_CU_3M", "WM_CU_CASH"), "basis", 0,
            "LME copper 3-month / cash - 1: contango (loose market) or backwardation (tight market)"),
    Feature("cu_stock_r3", ("WM_CU_STOCK",), "dlog3", 0, "LME copper stocks, 3-month change: inventory build = "
            "looser market"),
    Feature("al_basis", ("WM_AL_3M", "WM_AL_CASH"), "basis", 0, "LME aluminium 3-month / cash - 1"),
    Feature("al_stock_r3", ("WM_AL_STOCK",), "dlog3", 0, "LME aluminium stocks, 3-month change"),
]
BY_NAME = {f.name: f for f in FEATURES}

# Leading indicators per target (3-5 at most, own lags come on top). Targets not listed have no family model.
TARGET_FEATURES = {
    "S09": ["cru_cr_r1", "cru_cr_r1_l1", "iron_r3"],
    "S10": ["cru_cr_r1", "cru_cr_r1_l1", "iron_r3"],
    "cru_cr_de": ["iron_r3", "coal_r3", "ttf_r3"],
    "cru_hdg_de": ["iron_r3", "coal_r3", "ttf_r3", "zinc_r3"],
    "ss304_de": ["nickel_r1", "nickel_r1_l1"],
    "S11": ["brent_r3"],
    "S08": ["alu_r3", "ttf_r3", "power_r3"],
    "D02": ["al_basis", "al_stock_r3", "ttf_r3", "power_r3"],
    "S02": ["al_basis", "al_stock_r3", "ttf_r3", "power_r3"],
    "D01": ["cu_basis", "cu_stock_r3"],
    "S01": ["cu_basis", "cu_stock_r3"],
}
OWN = {"own_r1": "target's own 1-month log change (AR term)", "own_r3": "target's own 3-month log change (AR term)"}
EPOXY_INPUTS = ["benzene", "acetone", "ech", "epoxy_other"]


def _one(p, f):
    if f.transform == "basis":
        x = p[f.source[0]] / p[f.source[1]] - 1
    else:
        k = 1 if f.transform == "dlog1" else 3
        lg = np.log(p[f.source[0]])
        x = lg - lg.shift(k)
    return x.shift(f.lag)


def build(p):
    """Month x every feature, from panel p (values at m use data up to m only)."""
    return pd.DataFrame({f.name: _one(p, f) for f in FEATURES}, index=p.index)


def design(p, s, feats=None):
    """Month x columns of target s: own 1- and 3-month log changes + its leading indicators."""
    feats = build(p) if feats is None else feats
    lg = np.log(p[s])
    X = pd.DataFrame({"own_r1": lg.diff(), "own_r3": lg - lg.shift(3)}, index=p.index)
    return X.join(feats[TARGET_FEATURES[s]])


def availability(full):
    """One row per feature (and the own lags): sources, transformation, relative lag, first month, meaning."""
    lags = P.relative_lags(full)
    feats = build(full)
    rows = []
    for f in FEATURES:
        rows.append({"feature": f.name, "targets": " ".join(t for t, v in TARGET_FEATURES.items() if f.name in v),
                     "source": " / ".join(f.source),
                     "transform": {"dlog1": "1-month log change", "dlog3": "3-month log change",
                                   "basis": "3-month price / cash price - 1"}[f.transform],
                     "lag_vs_origin_m": int(max(lags.get(c, 0) for c in f.source)) + f.lag,
                     "first_month": f"{feats[f.name].first_valid_index():%Y-%m}", "meaning": f.meaning})
    for k, v in OWN.items():
        rows.append({"feature": k, "targets": " ".join(TARGET_FEATURES), "source": "target itself",
                     "transform": v.split(" (")[0].replace("target's own ", ""), "lag_vs_origin_m": 0,
                     "first_month": "history_start + 1-3 months", "meaning": v})
    return pd.DataFrame(rows)


def check_no_leakage(cut=None, start="2018-01-01"):
    """Design rows at each origin: built from the panel as seen at the origin == built from the full panel."""
    cut = P.load() if cut is None else cut
    lags = P.relative_lags(cut)
    full = P.nowcast(P.as_of(cut, P.CUTOFF, lags), P.CUTOFF)[0]
    ffull = build(full)
    n = 0
    for t in pd.date_range(start, P.CUTOFF, freq="MS"):
        pt = P.nowcast(P.as_of(cut, t, lags), t)[0]
        ft = build(pt)
        for s in TARGET_FEATURES:
            a = pt[s].last_valid_index()
            got, ref = design(pt, s, ft).loc[a], design(full, s, ffull).loc[a]
            same = np.isclose(got.values, ref.values, rtol=1e-12, atol=0, equal_nan=True)
            if s == "ss304_de" and t < P.CUTOFF:
                # stainless at its anchor is a nowcast at t and the published value in the full panel: only the own
                # changes may differ, never an exogenous feature
                same |= got.index.str.startswith("own_")
            assert same.all(), f"leakage at origin {t:%Y-%m}, target {s}: {list(got.index[~same])}"
            n += 1
    return n


def epoxy_identity(cut=None):
    """Fits of epoxy on its four cost columns (preprocessed file: ECH typo of May-2026 corrected)."""
    cut = P.load() if cut is None else cut
    e = cut[["epoxy"] + EPOXY_INPUTS].dropna()
    rows = []

    def fit(name, y, X, const=True):
        A = np.column_stack([np.ones(len(X))] * const + [X])
        b, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = y - A @ b
        rows.append({"form": name, "n": len(y), "r2": 1 - r.var() / y.var(), "max_abs_residual": np.abs(r).max(),
                     "coefficients": "; ".join(f"{c}={v:.4f}" for c, v in
                                               zip((["const"] if const else []) + EPOXY_INPUTS, b))})

    I = e[EPOXY_INPUTS].values
    fit("sum of the four columns (epoxy = benzene + acetone + ECH + other)", e["epoxy"].values, I.sum(1, keepdims=True),
        const=False)
    rows[-1]["coefficients"] = f"max |epoxy - sum| = {np.abs(e['epoxy'].values - I.sum(1)).max():.1f} EUR/t"
    fit("level, linear, no constant", e["epoxy"].values, I, const=False)
    fit("level, linear, constant", e["epoxy"].values, I)
    fit("log level, log-linear", np.log(e["epoxy"].values), np.log(I))
    d = np.log(e).diff().dropna()
    fit("monthly log change on input log changes", d["epoxy"].values, d[EPOXY_INPUTS].values)
    out = pd.DataFrame(rows)
    out["exact_identity"] = out["max_abs_residual"] < 1e-6
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cut = P.load()
    av = availability(cut)
    av.to_csv(OUT / "feature_availability.csv", index=False)
    ei = epoxy_identity(cut)
    ei.to_csv(OUT / "epoxy_identity.csv", index=False, float_format="%.6g")
    n = check_no_leakage(cut)
    print(f"leakage test passed: {n} (origin, target) design rows identical as seen at the origin and in the full panel")
    print(av[["feature", "targets", "source", "lag_vs_origin_m", "first_month"]].to_string(index=False))
    print("\nepoxy identity check:")
    print(ei[["form", "r2", "max_abs_residual", "exact_identity"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
