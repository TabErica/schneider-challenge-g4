"""Read the foundation-model branch's existing forecast outputs for the BOM cost step."""
from __future__ import annotations

import pandas as pd

import selection as S

KEY = ["series", "h", "origin"]
REQUIRED = ("backtest_predictions.csv", "backtest_warmup.csv", "forecasts_at_origin.csv")


def forecasts_for_strategy(rows, submission):
    """Keep the actual model and the selected interval for each forecast."""
    strategy = submission[["series", "h", "point_strategy", "interval_method"]].rename(
        columns={"point_strategy": "point", "interval_method": "interval"})
    selected = rows.merge(strategy, on=["series", "h", "point", "interval"], validate="many_to_one")
    selected["point"] = selected["underlying"].fillna(selected["point"])
    selected["model"] = selected["point"] + "+" + selected["interval"]
    if selected.duplicated(KEY).any():
        raise ValueError("Expected one forecast per material, horizon and origin")
    return selected


def load_branch_forecasts(backtest_dir, holdout_cutoff):
    missing = [name for name in REQUIRED if not (backtest_dir / name).is_file()]
    if missing:
        missing_names = ", ".join(missing)
        raise ValueError(f"Missing foundation-model forecast inputs: {missing_names}. "
                         "Generate the branch's default backtest outputs first. See README (Cost and pricing).")
    predictions, warmup = S.load(backtest_dir)
    origins = sorted(predictions["origin"].unique())
    if not origins:
        raise ValueError("The branch backtest has no candidate forecasts")
    if "ewmap" not in set(predictions["interval"]):
        raise ValueError("The forecast inputs predate the foundation-model branch's ewmap outputs. See README (Cost and pricing).")

    available = set(predictions.loc[predictions["interval"].eq("emp"), ["point", *KEY]]
                    .itertuples(index=False, name=None))
    choices = S.rolling_selection(S.point_errors(predictions, warmup), origins, available)
    naive_keys = {tuple(key) for point, *key in available if point == "naive"}
    choices = choices.loc[[key in naive_keys for key in choices[KEY].itertuples(index=False, name=None)]]
    k = S.tune_asym(predictions)
    adjusted = S.add_up(predictions, k)
    rows = pd.concat([adjusted, S.selected_rows(adjusted, choices)], ignore_index=True)

    def submission_map(sample, origin):
        report = S.report(sample)
        strategy = S.point_strategy(report)
        return S.submission_map(strategy, S.interval_choice(report, strategy), choices, k, origin)

    last = pd.Timestamp(origins[-1])
    submission = submission_map(rows, last)
    known = rows.loc[rows["origin"].le(holdout_cutoff) & rows["reveal"].le(holdout_cutoff)]
    frozen = submission_map(known, holdout_cutoff)

    raw = pd.read_csv(backtest_dir / "backtest_predictions.csv",
                      usecols=[*KEY, "anchor", "point", "interval"], parse_dates=["origin", "anchor"])
    anchors = raw.loc[raw["point"].eq("naive") & raw["interval"].eq("emp"), [*KEY, "anchor"]]
    selected = forecasts_for_strategy(rows, submission).merge(anchors, on=KEY, validate="one_to_one")
    heldout = forecasts_for_strategy(rows, frozen).merge(anchors, on=KEY, validate="one_to_one")
    return selected, heldout, S.plausibility(rows, submission, last)
