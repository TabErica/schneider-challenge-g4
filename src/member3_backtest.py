"""Rolling-origin validation of the current BOM, measured against realized price proxies."""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsforecast.models import AutoETS

import data as D
from member3_forecast_inputs import load_branch_forecasts
import targets as T
from models.base import HORIZONS

BACKTEST = D.OUT / "backtest"
MIN_MODEL_ERRORS = 12
MIN_BAND_ERRORS = 20
POINT_METHODS = ("bottom_up", "flat_cost", "direct_autoets")
COMPARISON_METHODS = (*POINT_METHODS, "legacy_bottom_up")
EVALUATION_CUTOFF = pd.Timestamp("2024-12-01")
SELECTION_REVISION = "ca5b97127a17a962f28d905f23e3a08794031a3b"
FORECAST_COLUMNS = ["model", "point", "interval", "series", "h", "origin", "anchor", "y_t", "actual", "p10", "p50", "p90"]


def release_lags(series):
    catalog = pd.read_csv(D.OUT / "preprocessed" / "series_catalog.csv").set_index("id")
    return catalog.loc[series, ["lag_m", "rel_lag_m"]].max(axis=1).astype(int).to_dict()


def model_choices(rows, lag):
    """At each origin, compare models on all the same previously published outcomes as naive."""
    h = int(rows["h"].iloc[0])
    forecasts = rows.pivot(index="origin", columns="point", values="p50").sort_index()
    actual = rows.loc[rows["point"].eq("naive")].set_index("origin")["actual"]
    error = forecasts.sub(actual.reindex(forecasts.index), axis=0).abs()
    counts = error.notna().cumsum()
    counts.index += pd.DateOffset(months=h + lag)
    counts = counts.reindex(forecasts.index, method="ffill").fillna(0).astype(int)
    total_error = error.fillna(0).cumsum()
    total_error.index += pd.DateOffset(months=h + lag)
    total_error = total_error.reindex(forecasts.index, method="ffill")
    eligible = counts.eq(counts["naive"], axis=0) & counts.ge(MIN_MODEL_ERRORS) & forecasts.notna()
    mae = total_error.div(counts.where(eligible))
    choice = pd.Series("naive", index=forecasts.index, name="point")
    available = mae.notna().any(axis=1)
    choice.loc[available] = mae.loc[available].idxmin(axis=1)
    return choice.to_frame().assign(selection_n=counts["naive"]).reset_index()


def select_material_forecasts(series, lags):
    historical = pd.read_csv(BACKTEST / "backtest_predictions.csv", usecols=FORECAST_COLUMNS,
                             parse_dates=["origin", "anchor"])
    current = pd.read_csv(BACKTEST / "forecasts_at_origin.csv", usecols=FORECAST_COLUMNS,
                          parse_dates=["origin", "anchor"])
    if current.empty or not current["origin"].eq(D.BOM_BASE).all():
        raise ValueError(f"Use the default backtest exports at {D.BOM_BASE:%Y-%m}")
    historical = historical.loc[historical["origin"].lt(D.BOM_BASE)]
    rows = pd.concat([historical, current], ignore_index=True)
    rows = rows.loc[rows["series"].isin(series) & rows["interval"].eq("emp") & rows["h"].isin(HORIZONS)]
    rows = rows.loc[rows["anchor"].eq(rows["origin"])].dropna(subset=["y_t", "p10", "p50", "p90"])
    expected = pd.MultiIndex.from_product([series, HORIZONS], names=["series", "h"])
    naive = rows.loc[rows["origin"].eq(D.BOM_BASE) & rows["point"].eq("naive")]
    if not expected.difference(pd.MultiIndex.from_frame(naive[["series", "h"]])).empty:
        raise ValueError("Backtest is missing a current material forecast or the default stainless nowcast")
    selected = []
    for (sid, _), group in rows.groupby(["series", "h"], sort=True):
        choice = model_choices(group, lags[sid])
        selected.append(group.merge(choice, on=["origin", "point"], validate="one_to_one"))
    result = pd.concat(selected, ignore_index=True)
    levels = result[["y_t", "p10", "p50", "p90"]]
    if (not np.isfinite(levels).all().all() or not levels.gt(0).all().all()
            or not (levels["p10"].le(levels["p50"]) & levels["p50"].le(levels["p90"])).all()):
        raise ValueError("Selected forecasts require positive, finite and ordered prices or indices")
    return result


def branch_material_forecasts(series, lags):
    """Retrieve Member 2's final forecast rows and a separately frozen validation strategy."""
    selected, heldout, plausibility = load_branch_forecasts(BACKTEST, EVALUATION_CUTOFF)
    if selected["origin"].max() != D.BOM_BASE or heldout["origin"].max() != D.BOM_BASE:
        raise ValueError("Member 2 forecasts must end at the submission origin")
    legacy = select_material_forecasts(series, lags)
    public = set(T.ids("material", in_submission=True))
    keys = ["series", "h", "origin"]
    selected = selected.loc[selected["series"].isin(public) & selected["anchor"].eq(selected["origin"])].copy()
    heldout = heldout.loc[heldout["series"].isin(public) & heldout["anchor"].eq(heldout["origin"])]
    expected = pd.MultiIndex.from_product([sorted(public), HORIZONS], names=["series", "h"])
    current = selected.loc[selected["origin"].eq(D.BOM_BASE)]
    if not expected.equals(pd.MultiIndex.from_frame(current[["series", "h"]]).sort_values()):
        raise ValueError("Missing anchored submission forecasts for one or more material/horizon pairs")
    public_keys = pd.MultiIndex.from_frame(legacy.loc[legacy["series"].isin(public), keys])
    selected_keys = pd.MultiIndex.from_frame(selected[keys])
    heldout_keys = pd.MultiIndex.from_frame(heldout[keys])
    if not public_keys.difference(selected_keys).empty or not selected_keys.difference(heldout_keys).empty:
        raise ValueError("Member 2 is missing forecasts needed for product validation")

    selected["forecast_source"] = "member2_interval_pooling"
    selected["selection_revision"] = SELECTION_REVISION
    selected["selection_n"] = np.nan
    precursors = legacy.loc[~legacy["series"].isin(public)].assign(
        forecast_source="legacy_precursor_only", selection_revision="")
    forecasts = pd.concat([selected, precursors], ignore_index=True)
    forecasts = forecasts.merge(legacy[keys + ["p50"]].rename(columns={"p50": "legacy_bottom_up"}),
                                on=keys, how="left", validate="one_to_one")
    forecasts = forecasts.merge(heldout[keys + ["p50"]].rename(columns={"p50": "heldout_bottom_up"}),
                                on=keys, how="left", validate="one_to_one")
    levels = forecasts[["y_t", "p10", "p50", "p90"]]
    if (not np.isfinite(levels).all().all() or not levels.gt(0).all().all()
            or not (levels["p10"].le(levels["p50"]) & levels["p50"].le(levels["p90"])).all()):
        raise ValueError("Selected material forecasts must be finite, positive and ordered")
    return forecasts, plausibility.loc[plausibility["series"].isin(public)].copy()


def product_rows(bom, forecasts, panel, lags):
    """Aggregate matched material point forecasts and fit Direct AutoETS on the same basket."""
    comparisons = [method for method in ("legacy_bottom_up", "heldout_bottom_up") if method in forecasts]
    rows = []
    priced = bom.dropna(subset=["series"])
    for product, lines in priced.groupby("product"):
        series = sorted(lines["series"].unique())
        release_lag = max(lags[s] for s in series)
        direct = lines.loc[lines["assumed_base_eur_per_t"].isna()].groupby("series")["mass_kg"].sum() / 1000
        indexed = (lines["mass_kg"] * lines["assumed_base_eur_per_t"] / 1000).groupby(lines["series"]).sum()
        for origin, block in forecasts.loc[forecasts["series"].isin(series)].groupby("origin"):
            anchor = block.drop_duplicates("series").set_index("series")["y_t"].reindex(series)
            if anchor.isna().any():
                continue
            # The assumed polymer EUR/t anchor is available at every origin. Its index scale uses that month's PPI.
            weights = direct.reindex(series, fill_value=0) + indexed.reindex(series, fill_value=0) / anchor
            history = panel.loc[:origin, series].copy()
            history.loc[origin] = anchor  # same as-of stainless nowcast as the upstream material forecasts
            history = history.dropna().dot(weights)
            model = AutoETS(season_length=1, model="ZZZ", damped=None)
            direct_path = np.exp(model.forecast(y=np.log(history.to_numpy()), h=max(HORIZONS))["mean"])
            for h, group in block.groupby("h"):
                levels = group.set_index("series").reindex(series)
                if levels["p50"].isna().any():
                    continue
                target = origin + pd.DateOffset(months=int(h))
                rows.append({"product": product, "origin": origin, "target_month": target, "h": h,
                             "outcome_available": target + pd.DateOffset(months=release_lag),
                             "actual_proxy_cost": levels["actual"].dot(weights),
                             "bottom_up": levels["p50"].dot(weights), "flat_cost": anchor.dot(weights),
                             "direct_autoets": direct_path[h - 1]}
                            | {method: levels[method].dot(weights) for method in comparisons})
    return pd.DataFrame(rows)


def add_product_bands(rows, method="bottom_up"):
    """Calibrate product bands on past joint basket errors available at each origin."""
    groups = []
    for _, block in rows.groupby(["product", "h"], sort=False):
        block = block.sort_values("origin").copy()
        scored = block.dropna(subset=["actual_proxy_cost"])
        residuals = np.log(scored["actual_proxy_cost"] / scored[method])
        bands = []
        for row in block.itertuples():
            errors = residuals.loc[scored["outcome_available"].le(row.origin)]
            if len(errors) >= MIN_BAND_ERRORS:
                lo, hi = errors.quantile([0.1, 0.9])
                point = getattr(row, method)
                bands.append((point * np.exp(min(lo, 0)), point * np.exp(max(hi, 0)), len(errors)))
            else:
                bands.append((np.nan, np.nan, len(errors)))
        block[[f"{method}_p10", f"{method}_p90", "band_n"]] = pd.DataFrame(bands, index=block.index)
        groups.append(block)
    return pd.concat(groups, ignore_index=True)


def score_products(rows, methods=POINT_METHODS):
    common = rows.loc[rows["outcome_available"].le(D.BOM_BASE)].dropna(subset=["actual_proxy_cost", *methods])
    records = []
    for (product, h), block in common.groupby(["product", "h"]):
        actual = block["actual_proxy_cost"]
        bands = block.dropna(subset=["bottom_up_p10", "bottom_up_p90"])
        covered = bands["actual_proxy_cost"].between(bands["bottom_up_p10"], bands["bottom_up_p90"])
        flat_mae = (block["flat_cost"] - actual).abs().mean()
        for method in methods:
            err = block[method] - actual
            mae = err.abs().mean()
            records.append({"product": product, "h": h, "method": method, "n": len(err),
                            "first_origin": block["origin"].min(), "last_origin": block["origin"].max(),
                            "mae_eur_per_unit": mae, "mae_relative_to_flat": mae / flat_mae if flat_mae else np.nan,
                            "mape": (err.abs() / actual).mean(), "bias_eur_per_unit": err.mean(),
                            "band_n": len(bands) if method == "bottom_up" else 0,
                            "band_coverage": covered.mean() if method == "bottom_up" else np.nan})
    return pd.DataFrame(records)
