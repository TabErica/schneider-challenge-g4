"""Smoke test of the decision tool (app/): every page loads, and default inputs give the pipeline's numbers.

Run: uv run pytest tests/test_app_smoke.py. Tests that need outputs/ are skipped without them (CI has no data).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
sys.path.insert(0, str(APP))  # as `streamlit run app/Home.py` does for the main script's folder

import loaders as L  # noqa: E402

PAGES = ["Home.py", "pages/1_Material_forecasts.py", "pages/2_BOM_cost.py", "pages/3_Quote_simulator.py"]
needs_outputs = pytest.mark.skipif(bool(L.missing(*L.FILES)), reason="no pipeline outputs: run `uv run python main.py`")


def run(page):
    at = AppTest.from_file(str(APP / page), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_without_outputs_shows_message(page, tmp_path, monkeypatch):
    monkeypatch.setattr(L, "OUT", tmp_path)
    at = run(page)
    assert any("uv run python main.py" in e.value for e in at.error)


@needs_outputs
@pytest.mark.parametrize("page", PAGES)
def test_page_loads(page):
    run(page)


@needs_outputs
def test_every_target_product_and_scenario():
    at = run(PAGES[1])
    targets = at.selectbox[0].options
    assert len(targets) == 20
    for option in targets:
        at.selectbox[0].set_value(next(t for t in L.forecast_targets() if option.endswith(f"({t})"))).run()
        assert not at.exception, option
    at = run(PAGES[2])
    for product in L.PRODUCTS:
        for h in L.HORIZONS:
            at.radio[0].set_value(product).run()
            at.radio[1].set_value(h).run()
            assert not at.exception, (product, h)
            assert "drives €" in at.markdown[-1].value or any("drives €" in m.value for m in at.markdown)
    at = run(PAGES[3])
    for product in L.PRODUCTS:
        for h in L.HORIZONS:
            for scenario in ("upside", "baseline", "downside"):
                at.sidebar.radio[0].set_value(product).run()
                at.sidebar.radio[1].set_value(h).run()
                at.sidebar.radio[2].set_value(scenario).run()
                assert not at.exception, (product, h, scenario)


def same(old, new):
    """Column by column: numbers within rtol 1e-12, everything else as text."""
    assert list(old.columns) == list(new.columns)
    for col in old.columns:
        a, b = old[col].reset_index(drop=True), new[col].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            assert np.allclose(a.astype(float), b.astype(float), rtol=1e-12, atol=1e-9, equal_nan=True), col
        else:
            assert a.astype(str).replace("nan", "").equals(b.astype(str).replace({"nan": "", "NaT": ""})), col


@needs_outputs
def test_default_inputs_reproduce_pipeline_tables():
    from pricing_scenarios import decision_table, load_commercial_inputs

    pricing, validity, offers = L.quote_tables(load_commercial_inputs(None, L.read("product_costs")))
    same(L.read("pricing_revenue"), pricing)
    same(pd.read_csv(L.OUT / "cost_pricing/eto_validity_backtest.csv", dtype={"target_month": str}), validity)
    same(pd.read_csv(L.OUT / "cost_pricing/eto_quote_policy.csv", dtype={"target_month": str}), offers)
    same(pd.read_csv(L.OUT / "cost_pricing/pricing_decisions.csv", dtype={"target_month": str}),
         decision_table(pricing, L.read("holdout_scores")))


@needs_outputs
@pytest.mark.parametrize(("share", "policy"), [(0.0, "hold_price"), (0.5, "half_central_cost"),
                                               (1.0, "full_central_cost")])
def test_custom_pass_through_matches_pipeline_policies(share, policy):
    from pricing_scenarios import load_commercial_inputs

    pricing, _, _ = L.quote_tables(load_commercial_inputs(None, L.read("product_costs")), share)
    custom = pricing.loc[pricing["policy"].eq(L.CUSTOM_POLICY)].reset_index(drop=True)
    reference = L.read("pricing_revenue").query("policy == @policy").reset_index(drop=True)
    for col in ("selling_price_change_eur", "high_stress_cost_change_net_of_price_eur_per_unit",
                "break_even_volume_change_for_revenue"):
        assert np.allclose(custom[col], reference[col], rtol=1e-12, atol=1e-9), col


@needs_outputs
def test_quote_page_defaults_show_pipeline_offer():
    at = run(PAGES[3])
    expected = (pd.read_csv(L.OUT / "cost_pricing/eto_quote_policy.csv")
                .query("product == 'RM6_NE_IQI_433520' and h == 3 and scenario == 'baseline'").iloc[0])
    shown = {}
    for m in at.metric:  # "Margin at risk" appears twice: recommended offer first, then pass-through
        shown.setdefault(m.label, []).append(m.value)
    assert shown["Price uplift"] == [f"€{expected['price_adjustment_eur']:+,.2f}"]
    assert shown["Quotation validity"] == [f"{expected['selected_validity_calendar_months']:.0f} months"]
    assert shown["Indexation trigger"] == [f"index {expected['index_trigger_level']:.1f}"]
    recommended, pass_through = shown["Margin at risk"]
    assert recommended == f"€{expected['cost_stress_left_on_fixed_offer_eur_per_unit']:,.2f} per unit"
    half = (L.read("pricing_revenue")
            .query("product == 'RM6_NE_IQI_433520' and h == 3 and policy == 'half_central_cost'").iloc[0])
    assert pass_through == f"€{max(half['high_stress_cost_change_net_of_price_eur_per_unit'], 0):,.2f} per unit"


@needs_outputs
def test_forecast_page_inputs_are_consistent():
    forecast = L.read("forecast")
    models = L.read("models")
    for target in L.forecast_targets():
        if target in L.PRODUCTS:
            continue
        rows = forecast.query("target_id == @target").merge(models, left_on=["target_id", "horizon_m"],
                                                            right_on=["series", "h"], validate="one_to_one")
        assert rows["model_name"].eq(rows["label"]).all(), target
        _, _, anchor, _ = L.material_history(target)
        naive = rows.loc[rows["point_model"].eq("naive")]
        assert np.allclose(naive["p50"], anchor, rtol=1e-9), target  # a naive P50 is the anchor: units line up
