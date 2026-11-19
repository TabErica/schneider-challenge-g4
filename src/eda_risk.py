"""Risk EDA -> outputs/eda/ (tables + fig11-fig15). Narrative: notebooks/02_eda_risk.ipynb.

  1. stationarity and tails   unit-root tests, volatility clustering, fat tails, why 80% bands come out too narrow
  2. cost-index rises         largest / P90 / P95 rise of each BOM cost index over 3, 6, 9 months, and its drivers
  3. regimes                  six periods, volatility per period
  4. co-movement              correlation of material prices, its stability, and summing P90s vs the true P90

Window: Nov-2016 to Sep-2026 (the preprocessed panel). A "change" is the log change of a monthly average price.
"""
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

import data as D
import preprocess as P
from eda import AXIS, DECK, DIV_MID, DIV_NEG, DIV_POS, GRID, INK, INK2, MUTED, OUT, PROD_COLOR, SLOT, SURF, note, save, title

HORIZONS = (3, 6, 9)
Z80 = stats.norm.ppf(0.9)  # half-width of a normal 80% band, in standard deviations
BOM_MKT = ["RM6_NE_IQI_433520_mkt", "RMAirSeT_IIQ_24kV_mkt"]
BOM_PROXY = ["RM6_NE_IQI_433520", "RMAirSeT_IIQ_24kV"]

SERIES = {  # id -> label, grouped: market prices, PPIs, energy / FX, BOM indices
    "D01": "Copper LME", "D02": "Aluminium LME", "S03_EUR": "Nickel LME", "S05_EUR": "Zinc LME",
    "S04_EUR": "Iron ore", "cru_cr_de": "CR coil DE (CRU)", "cru_hdg_de": "HDG coil DE (CRU)",
    "ss304_de": "Stainless 304 DE (CRU)", "epoxy": "Epoxy resin",
    "S07": "Copper PPI EU", "S08": "Aluminium PPI EU", "S09": "Steel PPI EU", "S10": "Steel PPI DE",
    "S11": "Plastics PPI EU",
    "ED01": "Brent", "ED02": "Gas TTF", "E09": "Power DE", "S06": "EUR/USD",
} | {b: D.bom_label(b) for b in BOM_MKT + BOM_PROXY}

# Materials of the market-version BOM index (what a joint price scenario has to simulate), plus S09 for comparison
MATERIALS = {"D01": "Copper", "D02": "Aluminium", "cru_cr_de": "CR coil", "cru_hdg_de": "HDG coil",
             "ss304_de": "Stainless 304", "S11": "Plastics PPI", "S03_EUR": "Nickel", "S09": "Steel PPI"}

REGIMES = [  # (label, first month, last month); boundaries set from the data (turning points of the main series)
    ("2017-19 calm", "2016-12", "2020-02"),
    ("2020 COVID", "2020-03", "2020-12"),
    ("2021 reopening rally", "2021-01", "2021-12"),
    ("2022 war & energy shock", "2022-01", "2022-12"),
    ("2023-25 normalisation", "2023-01", "2025-09"),
    ("2025-26 copper rally, oil shock", "2025-10", "2026-09"),
]
EVENTS = [("2020-04", "COVID low"), ("2022-03", "nickel squeeze"), ("2022-08", "energy price peak"),
          ("2026-03", "oil shock")]
STRESS = ["2020 COVID", "2021 reopening rally", "2022 war & energy shock"]
CALM = ["2017-19 calm", "2023-25 normalisation"]


def load():
    """Studied series (SERIES) on the common window, plus the cost tables needed for contributions."""
    c = P.load()
    p = D.panel()
    df = c.join(p[BOM_MKT + BOM_PROXY]).loc[p.index[0]:p.index[-1], list(SERIES)]
    return df, c, D.bom_cost_table(c, c)


def log_changes(df, h=1):
    return np.log(df).diff(h)


def regime_of(index):
    lab = pd.Series(pd.NA, index=index, dtype="object")
    for name, a, b in REGIMES:
        lab.loc[a:b] = name
    return lab


# ---------------------------------------------------------------- 1. stationarity, clustering, tails
def _adf(x):
    return adfuller(x, autolag="AIC", result_object=False)[1]


def _kpss(x):
    return kpss(x, nlags="auto", result_object=False)[1]


def _arch_p(e, nlags=6):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # statsmodels return-type notice, not about the data
        return het_arch(e, nlags=nlags)[1]


def stationarity(df):
    """ADF (H0: unit root) and KPSS (H0: stationary) p-values, on log levels and on monthly log changes."""
    rows = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)  # KPSS p-values are capped to [0.01, 0.10]
        for s in df.columns:
            lv = np.log(df[s]).dropna()
            r = lv.diff().dropna()
            rows[SERIES[s]] = {
                "ADF level p": _adf(lv), "KPSS level p": _kpss(lv), "ADF change p": _adf(r), "KPSS change p": _kpss(r)}
    t = pd.DataFrame(rows).T
    t["level"] = np.where((t["ADF level p"] > 0.05) & (t["KPSS level p"] < 0.05), "unit root",
                          np.where((t["ADF level p"] < 0.05) & (t["KPSS level p"] > 0.05), "stationary", "unclear"))
    t["change"] = np.where((t["ADF change p"] < 0.05) & (t["KPSS change p"] > 0.05), "stationary", "unclear")
    return t


def tails(df):
    """Shape of monthly log changes: fat tails (kurtosis, Jarque-Bera, share beyond 1.28 / 2.33 sd) and
    volatility clustering (ARCH-LM and Ljung-Box on squared changes, autocorrelation of squared changes)."""
    rows = {}
    for s in df.columns:
        r = log_changes(df[s]).dropna()
        z = (r - r.mean()) / r.std()
        e = r - r.mean()
        rows[SERIES[s]] = {
            "monthly sd": r.std(), "skew": stats.skew(r), "excess kurtosis": stats.kurtosis(r),
            "Jarque-Bera p": stats.jarque_bera(r).pvalue,
            "beyond 1.28 sd (normal 20%)": (z.abs() > Z80).mean(),
            "beyond 2.33 sd (normal 2%)": (z.abs() > stats.norm.ppf(0.99)).mean(),
            "ARCH-LM p (6 lags)": _arch_p(e),
            "Ljung-Box sq. p (6 lags)": acorr_ljungbox(e ** 2, lags=[6])["lb_pvalue"].iloc[0],
            "autocorr sq. lag 1": (e ** 2).autocorr(1)}
    return pd.DataFrame(rows).T


def band_coverage(df, h=3, window=24, min_obs=24, first_origin="2020-01-01"):
    """Why a naive 80% band (centred on today's price, sd from past h-month changes) misses, origins 2020-2026.

    covered (all past)   sd from all past changes, as in the backtest
    covered (last N m)   sd from the last `window` months only (reacts to volatility clustering)
    above / below band   where the misses go (10% / 10% if the band were right)
    sd 2020-26 / at 2020 realised sd of the evaluation period vs the sd known at the first origin
    covered (hindsight)  band with the realised 2020-26 sd: the coverage a correct width would have given
    Only data known at each origin is used, except in the hindsight column."""
    L = np.log(df)
    fwd, past = L.shift(-h) - L, L - L.shift(h)
    sd_all = past.expanding(min_obs).std()
    sd_win = past.rolling(window, min_periods=min_obs).std()
    rows = {}
    for s in df.columns:
        ok = fwd[s].notna() & sd_all[s].notna() & sd_win[s].notna() & (fwd.index >= first_origin)
        f, sd = fwd[s][ok], sd_all[s][ok]
        rows[SERIES[s]] = {
            "n": int(ok.sum()), "covered (all past)": (f.abs() <= Z80 * sd).mean(),
            f"covered (last {window} m)": (f.abs() <= Z80 * sd_win[s][ok]).mean(),
            "above band": (f > Z80 * sd).mean(), "below band": (f < -Z80 * sd).mean(),
            "sd 2020-26 / at 2020": f.std() / sd.iloc[0], "covered (hindsight sd)": (f.abs() <= Z80 * f.std()).mean(),
            "autocorr of monthly changes": L[s].diff().autocorr(1)}
    return pd.DataFrame(rows).T


def fig_qq(df, ids=("D01", "cru_cr_de", "ss304_de", "S09", "S11", "ED02", *BOM_MKT)):
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.2), sharex=True, sharey=True)
    for ax, s in zip(axes.flat, ids):
        r = log_changes(df[s]).dropna()
        z = np.sort((r - r.mean()) / r.std())
        q = stats.norm.ppf((np.arange(1, len(z) + 1) - 0.5) / len(z))
        ax.plot([-3.5, 3.5], [-3.5, 3.5], color=AXIS, lw=1)
        ax.scatter(q, z, s=14, color=SLOT[0], edgecolor=SURF, linewidth=0.6, zorder=3)
        ax.set_title(SERIES[s], fontsize=9)
        ax.set_xlim(-3.2, 3.2)
    for ax in axes[1]:
        ax.set_xlabel("normal quantile")
    for ax in axes[:, 0]:
        ax.set_ylabel("standardised change")
    if not DECK:
        fig.suptitle("Monthly price changes vs a normal distribution: points off the line = fat tails",
                     x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    note(fig, 0.01, -0.01, "Nov-2016 to Sep-2026, monthly log changes, standardised. BOM indices: market-price version.",
         fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.01, 1, 0.95))
    return fig


# ---------------------------------------------------------------- 2. cost-index rises
def cost_rises(df, cols=BOM_MKT + BOM_PROXY):
    """Change of each cost index over h months, over every start month: median, P90, P95, max (and when)."""
    rows = []
    for c in cols:
        L = np.log(df[c])
        for h in HORIZONS:
            ch = np.expm1(L.shift(-h) - L).dropna()
            rows.append({"index": D.bom_label(c), "h": h, "median": ch.median(), "P90": ch.quantile(0.9),
                         "P95": ch.quantile(0.95), "max": ch.max(), "max from": f"{ch.idxmax():%Y-%m}",
                         "share of windows > +10%": (ch > 0.10).mean(), "n windows": len(ch)})
    return pd.DataFrame(rows)


def rise_contributions(prices, costs, h=3, top=3):
    """For the `top` largest h-month rises of each market-version index: percentage points from each material."""
    rows = []
    for c, comp in D.bom_components(prices, costs, "market").items():
        comp = comp.loc["2016-11":D.BOM_BASE]
        idx = comp.sum(axis=1)
        ch = (idx.shift(-h) / idx - 1).dropna()
        chosen = []
        for t in ch.sort_values(ascending=False).index:  # skip windows overlapping one already chosen
            if all(abs((t - u).days) > 31 * h for u in chosen):
                chosen.append(t)
            if len(chosen) == top:
                break
        for t in chosen:
            end = t + pd.DateOffset(months=h)
            contrib = 100 * (comp.loc[end] - comp.loc[t]) / idx.loc[t]
            rows.append({"index": D.bom_label(c), "window": f"{t:%Y-%m} to {end:%Y-%m}", "rise %": 100 * ch[t]}
                        | contrib.round(1).to_dict())
    return pd.DataFrame(rows).fillna(0)


def fig_rises(df, h=3):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, (mkt, prx), prod in zip(axes, zip(BOM_MKT, BOM_PROXY), ["RM6", "AirSeT"]):
        shade_regimes(ax, labels=False)
        for c, ls, lw, lab in [(mkt, "-", 2, "market prices"), (prx, "--", 1.5, "PPI proxy")]:
            ch = 100 * np.expm1(np.log(df[c]).shift(-h) - np.log(df[c])).dropna()
            ax.plot(ch.index, ch, color=PROD_COLOR[prod], ls=ls, lw=lw, label=lab)
        p95 = 100 * np.expm1(np.log(df[mkt]).shift(-h) - np.log(df[mkt])).quantile(0.95)
        ax.axhline(p95, color=INK2, lw=1, ls=":")
        ax.text(df.index[0], p95, f" P95 market = +{p95:.0f}%", va="bottom", fontsize=8, color=INK2)
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_title(f"{'RM6' if prod == 'RM6' else 'RM AirSeT'}: cost change over the next {h} months")
        ax.legend(loc="lower left")
    axes[0].set_ylabel("% change of the material cost index")
    note(fig, 0.01, -0.02, "x = month the price is quoted; y = change of the cost index from that month to 3 months later. "
         "Grey bands: 2020 COVID, 2022 war & energy shock, 2025-26 copper rally.", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- 3. regimes
def shade_regimes(ax, labels=True):
    """Grey band on every second regime; labels centred in each regime, on two alternating rows."""
    for i, (name, a, b) in enumerate(REGIMES):
        a, b = pd.Timestamp(a), pd.Timestamp(b) + pd.offsets.MonthEnd(0)
        if i % 2:
            ax.axvspan(a, b, color=GRID, alpha=0.6, lw=0)
        if labels:
            ax.text(a + (b - a) / 2, 0.985 if i % 2 == 0 else 0.94, name, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=7.5, color=MUTED)


def regime_vol(df):
    """Annualised volatility (sd of monthly log changes x sqrt 12) per regime, and the ratio to the 2017-19 calm."""
    r = log_changes(df)
    lab = regime_of(r.index)
    vol = r.groupby(lab).std().reindex([n for n, *_ in REGIMES]).T * np.sqrt(12)
    vol.index = [SERIES[s] for s in vol.index]
    n = lab.value_counts().reindex([n for n, *_ in REGIMES])
    return vol, vol.div(vol[REGIMES[0][0]], axis=0), n


def fig_regimes(df, ids=("D01", "D02", "S03_EUR", "cru_cr_de", "ss304_de", "S11", "RMAirSeT_IIQ_24kV_mkt",
                         "RM6_NE_IQI_433520_mkt")):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    r = log_changes(df)
    for i, s in enumerate(ids):
        x = df[s].dropna()
        color = PROD_COLOR["AirSeT"] if "AirSeT" in s else PROD_COLOR["RM6"] if "RM6" in s else SLOT[i]
        lw = 2.6 if s in BOM_MKT else 1.6
        a1.plot(x.index, 100 * x / x.iloc[0], color=color, lw=lw, label=SERIES[s])
        a2.plot(r.index, r[s].rolling(12).std() * np.sqrt(12) * 100, color=color, lw=lw)
    for ax in (a1, a2):
        shade_regimes(ax, labels=ax is a1)
    for i, (m, lab) in enumerate(EVENTS):
        a1.annotate(lab, (pd.Timestamp(m), 0.88), xycoords=("data", "axes fraction"), xytext=(0, -14 - 13 * (i % 2)),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=7.5, color=INK2, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    a1.set_yscale("log")
    a1.set_ylim(60, 520)
    a1.set_yticks([75, 100, 150, 200, 300], ["75", "100", "150", "200", "300"])
    a1.minorticks_off()
    a1.axhline(100, color=AXIS, lw=1)
    a1.set_ylabel("index, Nov-2016 = 100 (log)")
    a1.legend(loc="lower right", ncol=4, fontsize=8)
    a2.set_ylabel("rolling 12-month\nvolatility (% a year)")
    title(a1, "Six regimes since 2016: prices (top) and their volatility (bottom)", pad=4)
    note(fig, 0.01, -0.01, "Energy prices: fig01. Volatility = sd of monthly log changes over the past 12 months "
         "x sqrt(12). Bold: BOM cost indices (market prices).", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    return fig


def fig_regime_vol(ratio):
    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#86b6ef", "#2a78d6", "#104281"])
    m = ratio.values.astype(float)
    ax.imshow(np.log2(m), cmap=cmap, vmin=-1, vmax=3, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(ratio.shape[1]), [c.replace(" ", "\n", 1) for c in ratio.columns], fontsize=8)
    ax.set_yticks(range(ratio.shape[0]), ratio.index, fontsize=8)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]:.1f}x", ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if m[i, j] >= 3 else INK)
    title(ax, "Volatility in each regime, relative to the 2017-19 calm (1.0x)")
    note(fig, 0.01, -0.01, "Annualised sd of monthly log changes per regime / the same in 2017-19. "
         "Colour on a log scale. 2020 COVID has 10 months only.", fontsize=7.5, color=MUTED)
    return fig


# ---------------------------------------------------------------- 4. co-movement
def correlations(df, h=1):
    r = log_changes(df[list(MATERIALS)], h).dropna(how="all")
    full = r.corr()
    lab = regime_of(r.index)
    calm, stress = r[lab.isin(CALM)].corr(), r[lab.isin(STRESS)].corr()
    for m in (full, calm, stress):
        m.index = m.columns = list(MATERIALS.values())
    return full, calm, stress


def mean_pairwise(corr, names=("Copper", "Aluminium", "CR coil", "HDG coil", "Stainless 304", "Plastics PPI")):
    c = corr.loc[list(names), list(names)].values
    return c[np.triu_indices_from(c, 1)].mean()


def rolling_mean_corr(df, window=24):
    r = log_changes(df[list(MATERIALS)[:6]]).dropna()
    out = {}
    for end in range(window, len(r) + 1):
        c = r.iloc[end - window:end].corr().values
        out[r.index[end - 1]] = c[np.triu_indices_from(c, 1)].mean()
    return pd.Series(out)


def sum_of_quantiles(df, prices, costs, h=3):
    """Weighted sum of each material's P10 / P90 h-month change vs the P10 / P90 of the cost index itself
    (market version). Equal only if all materials always move together; the gap is why scenarios are needed."""
    rows = []
    for c, comp in D.bom_components(prices, costs, "market").items():
        comp = comp.loc["2016-11":D.BOM_BASE]
        w = comp.loc[D.BOM_BASE] / 100  # cost shares at the base month
        mat = {m: _market_col(costs, c, m) for m in comp.columns}
        ch = {m: np.expm1(np.log(prices[col]).shift(-h) - np.log(prices[col])).loc[comp.index].dropna()
              for m, col in mat.items()}
        idx_ch = np.expm1(np.log(comp.sum(axis=1)).shift(-h) - np.log(comp.sum(axis=1))).dropna()
        for q in (0.1, 0.9):
            summed = sum(w[m] * ch[m].quantile(q) for m in comp.columns)
            rows.append({"index": D.bom_label(c), "quantile": f"P{int(q * 100)}", "sum of material quantiles": summed,
                         "quantile of the index": idx_ch.quantile(q)})
    t = pd.DataFrame(rows)
    t["sum / true"] = t["sum of material quantiles"] / t["quantile of the index"]
    return t


def _market_col(costs, prod_col, material):
    """Price column a material follows in the market-version index."""
    prod = prod_col.removesuffix(D.BOM_MKT)
    proxy = costs.loc[(costs["product"] == prod) & (costs["material"] == material), "proxy"].iloc[0]
    return D.MARKET_PROXY.get(material, proxy)


def fig_correlation(full, calm, stress, roll):
    fig = plt.figure(figsize=(13, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1])
    ax = fig.add_subplot(gs[0])
    cmap = LinearSegmentedColormap.from_list("div", [DIV_NEG, DIV_MID, DIV_POS])
    m = full.values
    ax.imshow(m, cmap=cmap, norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1))
    ax.grid(False)
    ax.set_xticks(range(len(full)), full.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(full)), full.index, fontsize=8)
    for i in range(len(full)):
        for j in range(len(full)):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if abs(m[i, j]) > 0.75 else INK)
    ax.set_title("Correlation of monthly price changes, Nov-2016 to Sep-2026", fontsize=10)
    ax2 = fig.add_subplot(gs[1])
    shade_regimes(ax2, labels=False)
    ax2.plot(roll.index, roll, color=SLOT[0], lw=2)
    ax2.axhline(mean_pairwise(calm), color=MUTED, ls="--", lw=1)
    ax2.axhline(mean_pairwise(stress), color=INK2, ls=":", lw=1)
    ax2.text(0.01, mean_pairwise(calm) - 0.005, "calm regimes (2017-19, 2023-25)", transform=ax2.get_yaxis_transform(),
             ha="left", va="top", fontsize=8, color=MUTED)
    ax2.text(1.0, mean_pairwise(stress), "stress regimes (2020-22)", transform=ax2.get_yaxis_transform(),
             ha="right", va="bottom", fontsize=8, color=INK2)
    ax2.set_ylabel("mean pairwise correlation")
    ax2.set_title("Average correlation of the six BOM materials, rolling 24 months", fontsize=10)
    note(fig, 0.01, -0.02, "Monthly log changes. BOM materials: copper, aluminium, CR coil, HDG coil, stainless 304, "
         "plastics PPI (market-price version of the cost index).", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- run
def main():
    df, prices, costs = load()
    tables = {
        "risk_stationarity.csv": stationarity(df),
        "risk_tails.csv": tails(df),
        "risk_band_coverage.csv": band_coverage(df),
        "risk_cost_rises.csv": cost_rises(df),
        "risk_rise_contributions.csv": rise_contributions(prices, costs),
        "risk_regime_vol.csv": regime_vol(df)[0],
        "risk_sum_of_quantiles.csv": sum_of_quantiles(df, prices, costs),
    }
    for name, t in tables.items():
        t.to_csv(OUT / name, float_format="%.4f")
    full, calm, stress = correlations(df)
    full.to_csv(OUT / "risk_correlation.csv", float_format="%.3f")
    save(fig_qq(df), "fig11_tails_qq.png")
    save(fig_rises(df), "fig12_cost_index_rises.png")
    save(fig_regimes(df), "fig13_regimes.png")
    save(fig_regime_vol(regime_vol(df)[1]), "fig14_regime_volatility.png")
    save(fig_correlation(full, calm, stress, rolling_mean_corr(df)), "fig15_correlation.png")


if __name__ == "__main__":
    main()
