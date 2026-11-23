"""Sensitivity runs of the backtest -> outputs/backtest/<variant>/ and outputs/backtest/sensitivity.csv.

Each variant changes one choice against the baseline (backtest.py with no option); the baseline is never changed:
  nowcast              stainless 304's missing month filled at every origin (nickel + steel PPI), not forecast h + 1
  drop_provisional     last 2 months of the Eurostat PPIs removed at every origin
  bom_weights_origin   BOM cost shares from the prices known at each origin, not the Sep-2026 shares
sensitivity.csv: rel. MAE vs naive, MAPE and 80% coverage of the series each variant touches, baseline vs variant.
Run after backtest.py (reads the baseline scores).
"""
from __future__ import annotations

import pandas as pd

import backtest as B

VARIANTS = {  # name -> (backtest arguments, series the variant is about)
    "nowcast": (["--nowcast"], ["ss304_de", "RM6_NE_IQI_433520_mkt", "RMAirSeT_IIQ_24kV_mkt"]),
    "drop_provisional": (["--drop-provisional"], ["S07", "S08", "S09", "S10", "S11",
                                                  "RM6_NE_IQI_433520", "RMAirSeT_IIQ_24kV"]),
    "bom_weights_origin": (["--bom-weights", "origin"], B.BOM_IDS),
}
MODELS = ["naive", "ets_local", "ridge_local", "gbm_global"]


def compare(base, var, series, name):
    keep = lambda d: d[d.series.isin(series) & d.model.isin(MODELS)].set_index(["series", "model", "h"])
    cols = ["mape", "rel_mae", "coverage80", "h_eff"]
    out = keep(base)[cols].join(keep(var)[cols], lsuffix="_base", rsuffix="_variant")
    out.insert(0, "variant", name)
    return out.reset_index()


def main():
    base = pd.read_csv(B.OUT / "scores_per_series.csv")
    tabs = []
    for name, (args, series) in VARIANTS.items():
        print(f"\n=== variant {name} ===", flush=True)
        _, per, _ = B.main(args)
        tabs.append(compare(base, per, series, name))
    tab = pd.concat(tabs)
    tab.to_csv(B.OUT / "sensitivity.csv", index=False, float_format="%.4f")
    show = tab[tab.model.isin(["naive", "ets_local"])].copy()
    print("\nMAPE and 80% coverage, baseline -> variant (naive and ETS):")
    print(show.pivot_table(index=["variant", "series", "model"], columns="h",
                           values=["mape_base", "mape_variant", "coverage80_base", "coverage80_variant"])
          .round(3).to_string())


if __name__ == "__main__":
    main()
