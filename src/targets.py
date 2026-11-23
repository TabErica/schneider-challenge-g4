"""Target registry: the one place that says what is forecast. backtest.py and every later model read it from here.

Fields
  id             column in the preprocessed panel (or BOM index column built by data.bom_index)
  name, family, region (EU | WW), unit
  target_type    "material" (a price or price index) or "bom_index" (material cost index of one product)
  in_submission  True = goes into the forecast file; False = input series, backtested but not submitted
  history_start  first month of the series' history (YYYY-MM). Univariate models use all of it; models with drivers
                 use the months where every input is known. Checked against the data at every backtest run.

This module imports nothing from the project, so data.py can derive its legacy TARGETS dict from it.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Target:
    id: str
    name: str
    family: str
    region: str
    unit: str
    target_type: str
    in_submission: bool
    history_start: str


def _m(id, name, family, region, unit, start, submit=True):
    return Target(id, name, family, region, unit, "material", submit, start)


def _b(id, name):
    return Target(id, name, "BOM", "EU", "index (origin = 100)", "bom_index", True, "2016-11")


REGISTRY: list[Target] = [
    # Public data (data.zip), the original TARGETS of data.py, same order
    _m("D01", "Copper (EUR/t)", "Base metals", "EU", "EUR/t", "2016-11"),
    _m("S07", "Copper PPI EU", "Base metals", "EU", "index", "2016-11"),
    _m("S01", "Copper (USD/t)", "Base metals", "WW", "USD/t", "2016-11"),
    _m("D02", "Aluminium (EUR/t)", "Base metals", "EU", "EUR/t", "2016-11"),
    _m("S08", "Aluminium PPI EU", "Base metals", "EU", "index", "2016-11"),
    _m("S02", "Aluminium (USD/t)", "Base metals", "WW", "USD/t", "2016-11"),
    _m("S03", "Nickel (USD/t)", "Base metals", "WW", "USD/t", "2016-11"),
    _m("S05", "Zinc (USD/t)", "Base metals", "WW", "USD/t", "2016-11"),
    _m("S04", "Iron ore (USD/t)", "Steel", "WW", "USD/t", "2016-11"),
    _m("S09", "Steel PPI EU", "Steel", "EU", "index", "2016-11"),
    _m("S10", "Steel PPI Germany", "Steel", "EU", "index", "2016-11"),
    _m("S12", "Steel PPI US", "Steel", "WW", "index", "2016-11"),
    _m("S11", "Plastics PPI EU", "Plastics", "EU", "index", "2016-11"),
    _m("S13", "Plastics PPI US", "Plastics", "WW", "index", "2016-11"),
    # Market prices (internal CRU workbook, epoxy file)
    _m("cru_cr_de", "CR coil Germany, CRU (EUR/t)", "Steel", "EU", "EUR/t", "2012-05"),
    _m("cru_hdg_de", "HDG coil Germany, CRU (EUR/t)", "Steel", "EU", "EUR/t", "2012-05"),
    _m("ss304_de", "Stainless 304 Germany, CRU (EUR/t)", "Steel", "EU", "EUR/t", "2008-01"),
    _m("epoxy", "Epoxy resin Europe (EUR/t)", "Chemicals", "EU", "EUR/t", "2018-01"),
    # Material cost index per product, PPI-proxy and market-price versions (data.bom_index)
    _b("RM6_NE_IQI_433520", "BOM RM6 (PPI proxy)"),
    _b("RMAirSeT_IIQ_24kV", "BOM AirSeT (PPI proxy)"),
    _b("RM6_NE_IQI_433520_mkt", "BOM RM6 (market prices)"),
    _b("RMAirSeT_IIQ_24kV_mkt", "BOM AirSeT (market prices)"),
    # Inputs of the epoxy cost build-up: backtested, not submitted
    _m("benzene", "Benzene Europe (EUR/t)", "Chemicals", "EU", "EUR/t", "2018-01", submit=False),
    _m("acetone", "Acetone Europe (EUR/t)", "Chemicals", "EU", "EUR/t", "2018-01", submit=False),
    _m("ech", "Epichlorohydrin Europe (EUR/t)", "Chemicals", "EU", "EUR/t", "2018-01", submit=False),
]
BY_ID = {t.id: t for t in REGISTRY}
if len(BY_ID) != len(REGISTRY):
    raise ValueError("target id used twice in REGISTRY")

# The 14 public-data series that were data.TARGETS before the registry (EDA tables keep using exactly these)
PUBLIC_IDS = [t.id for t in REGISTRY[:14]]


def ids(target_type=None, in_submission=None):
    """Target ids in registry order, optionally filtered."""
    return [t.id for t in REGISTRY if (target_type is None or t.target_type == target_type)
            and (in_submission is None or t.in_submission == in_submission)]


def name(sid):
    return BY_ID[sid].name if sid in BY_ID else sid


def frame():
    import pandas as pd

    return pd.DataFrame([t.__dict__ for t in REGISTRY])
