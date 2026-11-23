"""LME forward as a point forecast, and the quote-basis table (cash vs 3-month price as the forecast of future cash).

Point model lme_fwd: for copper and aluminium (D01, D02 in EUR; S01, S02 in USD), P50 = the LME 3-month price at the
origin (Westmetall monthly average; EUR version for the D series), held flat for h = 6 and 9. Data: Westmetall
monthly averages, the same month as the origin (lag 0).

quote_basis(): for every origin and h, the h-month-ahead LME cash monthly average forecast by (a) the cash price at
the origin and (b) the 3-month price at the origin, both from Westmetall: bias and MAE of each.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models.base import HORIZONS, Model, point

FORWARD = {"D01": "WM_CU_3M_EUR", "D02": "WM_AL_3M_EUR", "S01": "WM_CU_3M", "S02": "WM_AL_3M"}
QUOTE = {("Copper", "USD"): ("WM_CU_CASH", "WM_CU_3M"), ("Copper", "EUR"): ("WM_CU_CASH_EUR", "WM_CU_3M_EUR"),
         ("Aluminium", "USD"): ("WM_AL_CASH", "WM_AL_3M"), ("Aluminium", "EUR"): ("WM_AL_CASH_EUR", "WM_AL_3M_EUR")}


class LmeForward(Model):
    name = "lme_fwd"
    series = set(FORWARD)

    def predict(self, ctx, s):
        a = ctx.anchor[s]
        f = ctx.pt[FORWARD[s]].get(a, np.nan)
        if pd.isna(f):
            return {}
        return {h: point(float(np.log(f) - np.log(ctx.y_a(s))), int(ctx.pt[FORWARD[s]].loc[:a].count()))
                for h in HORIZONS}


def models():
    return [LmeForward()]


def quote_basis(panel, origin, periods):
    """Cash vs 3-month price at the origin as the forecast of the cash average h months later.
    `panel`: data as of `origin`; `periods`: {label: (first origin, last origin)}. Errors are pred / actual - 1."""
    rows = []
    for (metal, ccy), (cash, fwd) in QUOTE.items():
        for h in HORIZONS:
            actual = panel[cash].shift(-h)
            for basis, col in (("cash", cash), ("3-month", fwd)):
                err = (panel[col] / actual - 1).dropna()
                for label, (a, b) in periods.items():
                    e = err.loc[a:b]
                    rows.append({"metal": metal, "currency": ccy, "h": h, "basis": basis, "period": label,
                                 "n": len(e), "bias": e.mean(), "mae": e.abs().mean(),
                                 "first_origin": f"{e.index.min():%Y-%m}", "last_origin": f"{e.index.max():%Y-%m}"})
    out = pd.DataFrame(rows)
    key = ["metal", "currency", "h", "period"]
    mae = out.pivot_table(index=key, columns="basis", values="mae")
    out = out.merge((mae["3-month"] / mae["cash"]).rename("mae_3m_over_cash").reset_index(), on=key)
    return out.sort_values(key + ["basis"])
