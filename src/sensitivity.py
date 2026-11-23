"""Sensitivity runs of the backtest -> outputs/backtest/<variant>/ and outputs/backtest/sensitivity.csv.

Each variant changes one choice against the baseline (backtest.py with no option); the baseline is never changed:
  no_nowcast           stainless 304's missing month not nowcast: stainless and the market-price BOM from t - 1 (h + 1)
  drop_provisional     last 2 months of the Eurostat PPIs removed at every origin. Their anchor moves back 2 months
                       (h_eff = h + 2), so this measures the cost of an older anchor, not revision risk: without
                       Eurostat vintages the size of revisions cannot be measured (assumption A06)
  bom_weights_origin   BOM cost shares from the prices known at each origin, not the Sep-2026 shares
sensitivity.csv: rel. MAE vs naive, MAPE, 80% coverage and h_eff of the series each variant touches, baseline vs variant.
Run after backtest.py (reads the baseline scores).

Cache: a variant is re-run only when its inputs change. The key hashes the preprocessed panel, the BOM workbook, the
source of the modules the backtest runs on and the variant's arguments; it is stored as cache_key.txt next to the
variant's scores. --force re-runs every variant.
"""
from __future__ import annotations

import argparse
import hashlib

import pandas as pd

import backtest as B
import data as D
import preprocess as P

VARIANTS = {  # name -> (backtest arguments, series the variant is about)
    "no_nowcast": (["--no-nowcast"], ["ss304_de", "RM6_NE_IQI_433520_mkt", "RMAirSeT_IIQ_24kV_mkt"]),
    "drop_provisional": (["--drop-provisional"], ["S07", "S08", "S09", "S10", "S11",
                                                  "RM6_NE_IQI_433520", "RMAirSeT_IIQ_24kV"]),
    "bom_weights_origin": (["--bom-weights", "origin"], B.BOM_IDS),
}
MODELS = ["naive", "ets_local", "ridge_local", "gbm_global"]
CODE = ["backtest.py", "preprocess.py", "data.py", "targets.py"]


def cache_key(args):
    h = hashlib.sha256()
    for f in [P.OUT / "panel_monthly.csv", D.BOM_FILE] + [D.ROOT / "src" / c for c in CODE]:
        h.update(f.name.encode() + f.read_bytes())
    h.update(" ".join(args).encode())
    return h.hexdigest()


def variant_scores(name, args, force=False):
    """Scores of one variant: from the cache when its key matches, else a fresh backtest run."""
    d = B.OUT / name
    key, kf, sf = cache_key(args), d / "cache_key.txt", d / "scores_per_series.csv"
    if not force and kf.exists() and sf.exists() and kf.read_text().strip() == key:
        print(f"=== variant {name}: cached ({kf.relative_to(D.ROOT)}) ===", flush=True)
        return pd.read_csv(sf)
    print(f"\n=== variant {name} ===", flush=True)
    B.main(args)
    kf.write_text(key + "\n")
    return pd.read_csv(sf)  # same rounding whether cached or fresh, so sensitivity.csv does not depend on the cache


def compare(base, var, series, name):
    keep = lambda d: d[d.series.isin(series) & d.model.isin(MODELS)].set_index(["series", "model", "h"])
    cols = ["mape", "rel_mae", "coverage80", "h_eff"]
    out = keep(base)[cols].join(keep(var)[cols], lsuffix="_base", rsuffix="_variant")
    out.insert(0, "variant", name)
    return out.reset_index()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-run every variant, ignore the cache")
    a = ap.parse_args(argv)
    base = pd.read_csv(B.OUT / "scores_per_series.csv")
    tab = pd.concat([compare(base, variant_scores(name, args, a.force), series, name)
                     for name, (args, series) in VARIANTS.items()])
    tab.to_csv(B.OUT / "sensitivity.csv", index=False, float_format="%.4f")
    show = tab[tab.model.isin(["naive", "ets_local"])].copy()
    print("\nMAPE, 80% coverage and h_eff, baseline -> variant (naive and ETS):")
    print(show.pivot_table(index=["variant", "series", "model"], columns="h",
                           values=["mape_base", "mape_variant", "coverage80_base", "coverage80_variant",
                                   "h_eff_variant"]).round(3).to_string())


if __name__ == "__main__":
    main()
