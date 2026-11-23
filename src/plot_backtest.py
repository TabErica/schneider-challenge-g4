"""fig08: backtest error relative to naive, per series / model / horizon (run after backtest.py)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

import data as D
import targets as T
from eda import DECK, DIV_MID, DIV_NEG, DIV_POS, INK, MUTED, note, save, title

per = pd.read_csv(D.OUT / "backtest" / "scores_per_series.csv")

GROUPS = [  # submitted targets only (targets.py); the epoxy inputs are in scores_per_series.csv
    ("Market prices (exchange-traded)", ["D01", "S01", "D02", "S02", "S03", "S05", "S04"]),
    ("Market prices (CRU, epoxy)", ["cru_cr_de", "cru_hdg_de", "ss304_de", "epoxy"]),
    ("Producer price indices", ["S07", "S08", "S09", "S10", "S11", "S12", "S13"]),
    ("BOM cost index", T.ids("bom_index")),
]
if {s for _, ids in GROUPS for s in ids} != set(T.ids(in_submission=True)):
    raise ValueError("plot_backtest.GROUPS must list every submitted target of targets.py")
MODELS = [("ets_local+emp", "ETS (damped)"), ("auto_ets+emp", "AutoETS"), ("theta+emp", "Theta"),
          ("combo_mean+emp", "mean(naive,\nAutoETS)"), ("gbm_global+native", "GBM\nglobal")]
name = T.name

rows = [s for _, ids in GROUPS for s in ids]
cols = [(m, h) for m, _ in MODELS for h in (3, 6, 9)]
pv = per.pivot_table(index="series", columns=["model", "h"], values="rel_mae")
M = np.array([[pv.at[s, c] for c in cols] for s in rows])

cmap = LinearSegmentedColormap.from_list("div", [DIV_NEG, DIV_MID, DIV_POS])
fig, ax = plt.subplots(figsize=(13, 7.6) if DECK else (14, 9.6))
im = ax.imshow(M, cmap=cmap, norm=TwoSlopeNorm(vmin=0.7, vcenter=1.0, vmax=1.5), aspect="auto")
ax.grid(False)
ax.set_yticks(range(len(rows)), [name(s) for s in rows], fontsize=8.5)
ax.set_xticks(range(len(cols)), [f"{h}m" for _, h in cols], fontsize=8.5)
for j, (_, lab) in enumerate(MODELS):
    ax.text(j * 3 + 1, -1.2, lab, ha="center", va="bottom", fontsize=9, fontweight="bold", color=INK)
    if j:
        ax.axvline(j * 3 - 0.5, color="#fcfcfb", lw=4)
y0 = 0
for g, ids in GROUPS:
    if y0:
        ax.axhline(y0 - 0.5, color="#fcfcfb", lw=4)
    ax.text(len(cols) - 0.3, y0 + len(ids) / 2 - 0.5, g, rotation=-90, va="center", ha="left", fontsize=8, color=MUTED)
    y0 += len(ids)
for i in range(len(rows)):
    for j in range(len(cols)):
        v = M[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5 if DECK else 7.5,
                color="#ffffff" if (v < 0.8 or v > 1.35) else INK)
cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.08)
cb.set_label("error / naive error   (<1 = beats 'price stays the same')")
title(ax, "Backtest 2020-2026: who beats the naive forecast?", pad=62)
note(fig, 0.01, 0.0, "Rolling origins Jan-2020 to Sep-2026 minus h, data as of each origin; MAE of P50 relative to "
         "random walk. Stainless 304: missing month nowcast. Blue = better than naive, red = worse.",
     fontsize=7.5, color=MUTED)
save(fig, "fig08_backtest_vs_naive.png")
