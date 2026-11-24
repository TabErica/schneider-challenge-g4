"""Model selection on top of the backtest -> outputs/selection/. Rules and candidates: selection_config.py.

Reads outputs/backtest/backtest_predictions.csv and backtest_warmup.csv (run backtest.py first); runs in seconds.

  1. Rolling selection, meta-model "selected": at every origin t and horizon h, the family-level ranking of the
     candidates on the errors published in the last 24 months (nothing after t); each target takes its family's best
     candidate that applies to it, or naive when that candidate's gain over naive is below 3%. Its band at t is the
     band of the model it selected (same interval method). Log: selected_choices.csv.
  2. Asymmetric bands "<method>_up": the upper distance P50 -> P90 widened by k, k tuned on 2020-01..2023-06 only
     (asym_k.csv).
  3. Scores of every candidate and "selected", every band, by family x h x period (2020-01..2023-06, 2023-07..end,
     2022, all) -> selection_report.csv:
       rel_mae_gmean     geometric mean over the family's targets of MAE / naive's MAE (same rows)
       dm_*_family       Diebold-Mariano vs naive on the family's mean scaled absolute error per origin
                         (|error| / MASE scale, Newey-West lag h - 1); dm_q_family = BH over every candidate x
                         family x h test of the period (family "all" excluded from that BH family)
       n_better_q / n_worse_q  targets significantly better / worse than naive (per-target DM, BH over every
                         candidate x target x h of the period)
       coverage80, above, below (share above P90 / below P10), width, winkler (alpha 0.2, / y at the anchor), pinball
     selection_summary.csv: naive, combo_median and selected, all 22 targets, point scores by period.
  4. Interval choice per family x h (interval_choice.csv) and point strategy per family x h -> submission_models.csv:
     point model, interval method and the reason, per submitted target x h.
  5. band_plausibility.csv: the submitted P10 / P50 / P90 at the last origin against the target's largest and
     95th-percentile past rise of the same length (sanity check only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import data as D
import scoring as S
import selection_config as C
import targets as T

BT = D.OUT / "backtest"
OUT = D.OUT / "selection"
SUB_IDS = T.ids(in_submission=True)
FAMILY = {t.id: t.family for t in T.REGISTRY}
KEY = ["series", "h", "origin"]
COLS = ["model", "point", "interval", "series", "h", "origin", "h_eff", "y_t", "actual", "q10", "q50", "q90", "p10",
        "p50", "p90", "fallback", "n_train", "mase_scale", "reveal"]
PERIODS = ["all", S.P1, S.P2, S.Y2022]


def eligible(point, series):
    allowed = C.CANDIDATES[point][0]
    return allowed is None or series in allowed


def load(bt=BT):
    """Candidate rows (submitted targets where the candidate applies, the four base bands) and their warm-up P50s."""
    res = pd.read_csv(bt / "backtest_predictions.csv", usecols=COLS, parse_dates=["origin", "reveal"])
    res = res[res["point"].isin(C.CANDIDATES) & res["series"].isin(SUB_IDS) & res["interval"].isin(C.BASE_METHODS)]
    res = res[[eligible(p, s) for p, s in zip(res["point"], res["series"])]].reset_index(drop=True)
    warm = pd.read_csv(bt / "backtest_warmup.csv", parse_dates=["origin", "reveal"])
    warm = warm[warm["point"].isin(C.CANDIDATES) & warm["series"].isin(SUB_IDS)]
    warm = warm[[eligible(p, s) for p, s in zip(warm["point"], warm["series"])]]
    return res, warm


# ---------------------------------------------------------------------------------------------------------------------
# 1. Rolling selection


def point_errors(res, warm):
    """Wide table (series, h, origin, reveal) x candidate: absolute error of the P50, warm-up and scored origins."""
    pts = pd.concat([warm, res[res["interval"] == "emp"]], ignore_index=True)
    pts = pts[pts["actual"].notna()].assign(ae=lambda d: (d["p50"] - d["actual"]).abs())
    return pts.pivot_table(index=KEY + ["reveal"], columns="point", values="ae").reset_index()


def family_scores(win):
    """Window rows of one h -> (family x candidate: geometric mean of per-target MAE / naive MAE, n targets),
    per-target ratios."""
    ratios = []
    for s, g in win.groupby("series"):
        for c in C.CANDIDATES:
            if c == "naive" or c not in g or not eligible(c, s):
                continue
            ok = g[c].notna() & g["naive"].notna()
            if ok.sum() >= C.MIN_ERRORS:
                ratios.append({"series": s, "family": FAMILY[s], "point": c, "n": int(ok.sum()),
                               "ratio": g.loc[ok, c].mean() / g.loc[ok, "naive"].mean()})
    ratios = pd.DataFrame(ratios, columns=["series", "family", "point", "n", "ratio"])
    fam = ratios.groupby(["family", "point"]).agg(score=("ratio", lambda x: float(np.exp(np.log(x).mean()))),
                                                  n_targets=("series", "size")).reset_index()
    return fam, ratios


def rolling_selection(err, origins, available):
    """Choice of every (target, h, origin). `available`: set of (point, series, h, origin) with a forecast."""
    rows = []
    for t in origins:
        lo = t - pd.DateOffset(months=C.WINDOW_M)
        win_t = err[(err["reveal"] <= t) & (err["reveal"] > lo)]
        for h in sorted(err["h"].unique()):
            fam, _ = family_scores(win_t[win_t["h"] == h])
            score = {(f, p): v for f, p, v in fam[["family", "point", "score"]].itertuples(index=False)}
            for s in SUB_IDS:
                f = FAMILY[s]
                cands = sorted(((score[f, c], c) for c in C.CANDIDATES
                                if c != "naive" and eligible(c, s) and (f, c) in score))
                row = {"origin": t, "h": h, "series": s, "family": f}
                row |= {f"score_{c}": score.get((f, c), np.nan) for c in C.CANDIDATES if c != "naive"}
                if not cands:
                    row |= {"best": "", "best_score": np.nan, "chosen": "naive", "rule": "no candidate with enough errors"}
                else:
                    v, c = cands[0]
                    row |= {"best": c, "best_score": v}
                    if v > 1 - C.MIN_GAIN:
                        row |= {"chosen": "naive", "rule": f"gain {1 - v:.1%} < {C.MIN_GAIN:.0%}"}
                    elif (c, s, h, t) not in available:
                        row |= {"chosen": "naive", "rule": f"{c} has no forecast"}
                    else:
                        row |= {"chosen": c, "rule": f"gain {1 - v:.1%} >= {C.MIN_GAIN:.0%}"}
                rows.append(row)
    return pd.DataFrame(rows)


def selected_rows(res, choice):
    """Rows of the meta-model: at each (target, h, origin) the chosen candidate's P50 and bands (every method)."""
    m = res.merge(choice[KEY + ["chosen"]], on=KEY)
    sel = m[m["point"] == m["chosen"]].drop(columns="chosen")
    n_methods = res["interval"].nunique()
    if len(sel) != len(choice) * n_methods:
        raise ValueError(f"selected: {len(sel)} rows, expected {len(choice) * n_methods}")
    return sel.assign(underlying=sel["point"], point="selected", model="selected+" + sel["interval"])


# ---------------------------------------------------------------------------------------------------------------------
# 2. Asymmetric bands


def tune_asym(res):
    """k per (point, method, family, h) on 2020-01..2023-06 origins whose outcome is published by ASYM_TUNE_BY."""
    r = res[S.in_period(res["origin"], S.P1) & (res["reveal"] <= C.ASYM_TUNE_BY) & res["actual"].notna()]
    rows = []
    for (p, m, f, h), g in r.assign(family=r["series"].map(FAMILY)).groupby(["point", "interval", "family", "h"]):
        rlog = np.log(g["actual"] / g["y_t"]).to_numpy()
        q50, up = g["q50"].to_numpy(), (g["q90"] - g["q50"]).to_numpy()
        above = np.array([(rlog > q50 + k * up).mean() for k in C.ASYM_GRID])
        ok = np.flatnonzero(above <= C.ASYM_TARGET_ABOVE)
        i = ok[0] if len(ok) else len(C.ASYM_GRID) - 1
        rows.append({"point": p, "interval": m, "family": f, "h": h, "n_tune": len(g), "above_k1": above[0],
                     "k": C.ASYM_GRID[i], "above_tuned": above[i], "reached_target": bool(len(ok))})
    return pd.DataFrame(rows)


def add_up(res, k):
    """'<method>_up' rows: P90 = P50 + k (P90 - P50) in log, P10 unchanged."""
    up = res.assign(family=res["series"].map(FAMILY)).merge(k[["point", "interval", "family", "h", "k"]],
                                                           on=["point", "interval", "family", "h"])
    up["q90"] = up["q50"] + up["k"] * (up["q90"] - up["q50"])
    up["p90"] = up["y_t"] * np.exp(up["q90"])
    up["interval"] = up["interval"] + C.UP
    up["model"] = up["point"] + "+" + up["interval"]
    return pd.concat([res, up.drop(columns=["family", "k"])], ignore_index=True)


# ---------------------------------------------------------------------------------------------------------------------
# 3. Report


def _gmean(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.log(x).mean())) if len(x) else np.nan


def family_dm(rp):
    """Per (point, family, h): DM vs naive on the family's mean scaled absolute error per origin."""
    pts = rp[rp["interval"] == "emp"]
    rows = []
    for fam_name, fp in (("all", pts), *pts.groupby("family")):
        for (p, h), g in fp[fp["point"] != "naive"].groupby(["point", "h"]):
            d = (g["sae"] - g["naive_sae"]).groupby(g["origin"]).mean().sort_index()
            st, pv = S.dm_test(d.values, int(h) - 1)
            rows.append({"point": p, "family": fam_name, "h": h, "dm_stat_family": st, "dm_p_family": pv})
    out = pd.DataFrame(rows)
    out["dm_q_family"] = np.nan
    fam = out["family"] != "all"
    out.loc[fam, "dm_q_family"] = S.bh_qvalues(out.loc[fam, "dm_p_family"])
    out.loc[~fam, "dm_q_family"] = S.bh_qvalues(out.loc[~fam, "dm_p_family"])
    return out


def report(rows):
    r = S.scored_rows(rows)
    out = []
    for period in PERIODS:
        rp = r if period == "all" else r[S.in_period(r["origin"], period)]
        per = S.per_series(rp)  # per target; DM per target, BH over candidates x targets x h of the period
        per = per[per["interval"] == "emp"]
        tgt = per.assign(better=(per["dm_q"] < S.Q_LEVEL) & (per["dm_stat"] < 0),
                         worse=(per["dm_q"] < S.Q_LEVEL) & (per["dm_stat"] > 0))
        frames = []
        for fam_name, g in (("all", rp), *rp.groupby("family")):
            band = g.groupby(["point", "interval", "h"]).agg(
                n_obs=("ae", "size"), coverage80=("covered", "mean"), above=("above", "mean"),
                below=("below", "mean"), width=("width", "mean"), winkler=("winkler", "mean"),
                pinball=("pinball", "mean")).reset_index()
            tg = tgt if fam_name == "all" else tgt[tgt["family"] == fam_name]
            pt = tg.groupby(["point", "h"]).agg(n_targets=("series", "nunique"), rel_mae_gmean=("rel_mae", _gmean),
                                                n_better_q=("better", "sum"), n_worse_q=("worse", "sum")).reset_index()
            frames.append(band.merge(pt, on=["point", "h"], how="left").assign(family=fam_name))
        rep = pd.concat(frames, ignore_index=True).merge(family_dm(rp), on=["point", "family", "h"], how="left")
        out.append(rep.assign(period=period))
    rep = pd.concat(out, ignore_index=True)
    cols = ["period", "family", "h", "point", "interval", "n_targets", "n_obs", "rel_mae_gmean", "dm_stat_family",
            "dm_p_family", "dm_q_family", "n_better_q", "n_worse_q", "coverage80", "above", "below", "width",
            "winkler", "pinball"]
    order = {p: i for i, p in enumerate(PERIODS)}
    rep = rep.sort_values(["family", "h", "point", "interval", "period"], key=lambda c: c.map(order)
                          if c.name == "period" else c)
    return rep[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------------------------------------------------
# 4. Point strategy, interval choice, submission map


def point_strategy(rep):
    pts = rep[(rep["interval"] == "emp") & (rep["family"] != "all")]
    g = pts.pivot_table(index=["family", "h", "point"], columns="period", values="rel_mae_gmean")
    rows = []
    for (f, h), x in g.groupby(level=["family", "h"]):
        x = x.droplevel(["family", "h"])
        ok = [(max(x.at[s, S.P1], x.at[s, S.P2]), s) for s in C.POINT_STRATEGIES
              if s in x.index and x.at[s, S.P1] < 1 and x.at[s, S.P2] < 1]
        desc = "; ".join(f"{s} {x.at[s, S.P1]:.3f} / {x.at[s, S.P2]:.3f}" for s in C.POINT_STRATEGIES if s in x.index)
        if ok:
            worst, s = min(ok)
            why = (f"{s}: MAE / naive below 1 in both periods (2020-01..2023-06 / 2023-07..end: {desc}); "
                   f"lowest worse-period ratio {worst:.3f}")
        else:
            s = "naive"
            why = f"no strategy below 1 x naive's MAE in both periods ({desc})"
        rows.append({"family": f, "h": h, "strategy": s, "point_reason": why})
    return pd.DataFrame(rows)


def interval_choice(rep, strat):
    lo, hi = C.COVERAGE_RANGE
    rows = []
    for st in strat.itertuples(index=False):
        g = rep[(rep["family"] == st.family) & (rep["h"] == st.h) & (rep["point"] == st.strategy)]
        w = g.pivot_table(index="interval", columns="period", values=["coverage80", "winkler", "pinball"])
        t = pd.DataFrame({"cov_p1": w["coverage80"][S.P1], "cov_p2": w["coverage80"][S.P2],
                          "cov_2022": w["coverage80"][S.Y2022], "winkler": w["winkler"]["all"],
                          "pinball": w["pinball"]["all"]}).reindex(list(C.METHODS)).dropna(subset=["cov_p1"])
        t.index.name = "interval"
        t["admissible"] = t["cov_p1"].between(lo, hi) & t["cov_p2"].between(lo, hi)
        if t["admissible"].any():
            best = t[t["admissible"]].sort_values(["winkler", "pinball"]).index[0]
            why = (f"coverage {t.at[best, 'cov_p1']:.2f} / {t.at[best, 'cov_p2']:.2f} within [{lo:.2f}, {hi:.2f}] "
                   f"in both periods; lowest Winkler ({t.at[best, 'winkler']:.3f}) of the "
                   f"{int(t['admissible'].sum())} admissible bands")
        else:
            best = t.assign(c=-t["cov_p1"].clip(upper=hi)).sort_values(["c", "winkler"]).index[0]
            why = (f"no band within [{lo:.2f}, {hi:.2f}] in both periods; highest stress-period coverage "
                   f"({t.at[best, 'cov_p1']:.2f}, 2023-07..end {t.at[best, 'cov_p2']:.2f})")
        rows.append(t.reset_index()
                    .assign(family=st.family, h=st.h, point=st.strategy, chosen=lambda d: d["interval"] == best,
                            interval_reason=lambda d: np.where(d["chosen"], why, "")))
    cols = ["family", "h", "point", "interval", "cov_p1", "cov_p2", "cov_2022", "winkler", "pinball", "admissible",
            "chosen", "interval_reason"]
    return pd.concat(rows, ignore_index=True)[cols]


def submission_map(strat, itv, choice, k, last):
    fin = choice[choice["origin"] == last].set_index(["series", "h"])
    chosen_itv = itv[itv["chosen"]].set_index(["family", "h"])
    kk = k.set_index(["point", "interval", "family", "h"])["k"]
    rows = []
    for st in strat.itertuples(index=False):
        it = chosen_itv.loc[(st.family, st.h)]
        for s in [s for s in SUB_IDS if FAMILY[s] == st.family]:
            if st.strategy == "selected":
                c = fin.loc[(s, st.h)]
                model = c["chosen"]
                pr = (f"{st.point_reason}. Rolling rule at {last:%Y-%m} (errors published in the last {C.WINDOW_M} "
                      f"months): {model}" + (f" ({c['best']} {c['best_score']:.3f} x naive, {c['rule']})"
                                             if c["best"] else f" ({c['rule']})"))
            else:
                model, pr = st.strategy, st.point_reason
            base = it["interval"]
            k_up = kk.get((model, base.removesuffix(C.UP), st.family, st.h), np.nan) if base.endswith(C.UP) else np.nan
            rows.append({"series": s, "name": T.name(s), "family": st.family, "h": st.h, "point_strategy": st.strategy,
                         "point_model": model, "interval_method": base, "k_up": k_up, "label": f"{model}+{base}",
                         "cov_p1": it["cov_p1"], "cov_p2": it["cov_p2"], "winkler": it["winkler"],
                         "point_reason": pr, "interval_reason": it["interval_reason"]})
    order = {s: i for i, s in enumerate(SUB_IDS)}
    return pd.DataFrame(rows).sort_values(["series", "h"], key=lambda c: c.map(order) if c.name == "series" else c)


def plausibility(rows, sub, last):
    """Submitted bands at the last origin against the target's own history known at that origin (data.panel, no
    fill): largest, 95th-percentile h_eff-month rise and largest fall. p90_above_max_rise flags a P90 beyond every
    rise of that length in the data (a sanity check, it changes no choice)."""
    pan = D.panel()
    at = rows[rows["origin"] == last].drop_duplicates(["model", "series", "h"]).set_index(["model", "series", "h"])
    out = []
    for r in sub.itertuples(index=False):
        if (r.label, r.series, r.h) not in at.index:
            continue
        x = at.loc[(r.label, r.series, r.h)]
        he = int(x["h_eff"])
        lg = np.log(pan[r.series].loc[:last].dropna())
        ch = np.exp((lg.shift(-he) - lg).dropna()) - 1
        p90 = x["p90"] / x["y_t"] - 1
        out.append({"series": r.series, "h": r.h, "label": r.label, "p10_pct": x["p10"] / x["y_t"] - 1,
                    "p50_pct": x["p50"] / x["y_t"] - 1, "p90_pct": p90, "hist_p95_rise": ch.quantile(0.95),
                    "hist_max_rise": ch.max(), "hist_max_fall": ch.min(), "n_windows": len(ch),
                    "p90_above_max_rise": bool(p90 > ch.max())})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------------------------------------------


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res, warm = load()
    origins = sorted(res["origin"].unique())
    last = pd.Timestamp(origins[-1])
    err = point_errors(res, warm)
    avail = set(res.loc[res["interval"] == "emp", ["point", "series", "h", "origin"]].itertuples(index=False,
                                                                                                name=None))
    choice = rolling_selection(err, origins, avail)
    # keep the (target, h, origin) the backtest has a row for (no row once the outcome falls after the data cutoff)
    has = {tuple(k) for p, *k in avail if p == "naive"}
    choice = choice[[k in has for k in zip(choice["series"], choice["h"], choice["origin"])]].reset_index(drop=True)
    k = tune_asym(res)
    res = add_up(res, k)
    rows = pd.concat([res, selected_rows(res, choice)], ignore_index=True)
    rep = report(rows)
    strat = point_strategy(rep)
    itv = interval_choice(rep, strat)
    sub = submission_map(strat, itv, choice, k, last)
    plaus = plausibility(rows, sub, last)

    pd.DataFrame([{"point": p, "targets": "all submitted" if a is None else " ".join(a), "reason": why}
                  for p, (a, why) in C.CANDIDATES.items()]).to_csv(OUT / "candidates.csv", index=False)
    choice.to_csv(OUT / "selected_choices.csv", index=False, float_format="%.4f")
    k.to_csv(OUT / "asym_k.csv", index=False, float_format="%.4f")
    rep.to_csv(OUT / "selection_report.csv", index=False, float_format="%.4f")
    summ = rep[(rep["family"] == "all") & (rep["interval"] == "emp") & rep["point"].isin(["naive", "combo_median",
                                                                                          "selected"])]
    summ = summ[["period", "h", "point", "n_targets", "rel_mae_gmean", "dm_stat_family", "dm_q_family",
                 "n_better_q", "n_worse_q"]]
    summ.to_csv(OUT / "selection_summary.csv", index=False, float_format="%.4f")
    itv.to_csv(OUT / "interval_choice.csv", index=False, float_format="%.4f")
    sub.to_csv(OUT / "submission_models.csv", index=False, float_format="%.4f")
    plaus.to_csv(OUT / "band_plausibility.csv", index=False, float_format="%.4f")
    print_summary(rep, choice, k, strat, itv, sub)
    print(f"\nBand plausibility at {last:%Y-%m}: {int(plaus['p90_above_max_rise'].sum())} of {len(plaus)} submitted "
          f"P90s above the largest past rise of the same length (band_plausibility.csv)")
    return rep, choice, sub


def print_summary(rep, choice, k, strat, itv, sub):
    pd.set_option("display.width", 250)
    pts = rep[(rep["family"] == "all") & (rep["interval"] == "emp")]
    print("Point P50, 22 targets: geometric-mean MAE / naive (n targets better / worse than naive, BH q < 0.05):")
    cell = pts.assign(c=pts["rel_mae_gmean"].map("{:.3f}".format) + " (" + pts["n_better_q"].astype("Int64")
                      .astype(str) + "/" + pts["n_worse_q"].astype("Int64").astype(str) + ")")
    print(cell.pivot_table(index=["point"], columns=["period", "h"], values="c", aggfunc="first").to_string())
    sc = choice[choice["origin"] < choice["origin"].max()]
    print("\nRolling selection: share of (target, origin) per chosen model, by h:")
    print(pd.crosstab(sc["chosen"], sc["h"], normalize="columns").round(3).to_string())
    print("\nAsymmetric k (median over families) per point x method x h:")
    print(k.pivot_table(index=["point", "interval"], columns="h", values="k", aggfunc="median").to_string())
    band = rep[(rep["family"] == "all") & rep["point"].isin(["naive", "combo_median", "selected"])]
    print("\n80% coverage (above P90 / below P10), 22 targets:")
    band = band.assign(c=band["coverage80"].map("{:.2f}".format) + " (" + band["above"].map("{:.2f}".format) + "/"
                       + band["below"].map("{:.2f}".format) + ")")
    print(band[band["period"] != "all"].pivot_table(index=["point", "interval"], columns=["period", "h"], values="c",
                                                    aggfunc="first").to_string())
    print("\nPoint strategy and band per family x h:")
    print(strat.merge(itv[itv["chosen"]][["family", "h", "interval", "cov_p1", "cov_p2", "admissible"]],
                      on=["family", "h"]).drop(columns="point_reason").round(3).to_string(index=False))
    print("\nsubmission_models.csv:")
    print(sub[["series", "h", "point_strategy", "point_model", "interval_method", "k_up"]].to_string(index=False))


if __name__ == "__main__":
    main()
