#!/usr/bin/env python3
"""
Leverage-environment dashboard for SPY, EFA and EEM.

Builds a dashboard page and a fuller detail report (plus a JSON/CSV record) that track
momentum, volatility, serial correlation and breadth for each ETF, rank every
reading against that ETF's own history, and simulate how daily-reset 2x / 3x
exposure has actually compounded. The methodology and every known caveat are
written into both pages.

    python dashboard.py --out site               # live data from Yahoo Finance
    python dashboard.py --out site --synthetic   # made-up data, for offline testing

Designed to run unattended once per trading day (see .github/workflows/daily.yml).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration - everything you might want to tune lives here
# --------------------------------------------------------------------------- #

FETCH_START = "1995-01-01"     # earliest data requested (gives windows time to warm up)
HISTORY_START = "1998-01-01"   # earliest date used for percentile ranking and charts
TRADING_DAYS = 252

MOM_SHORT, MOM_LONG = 63, 252  # 3-month and 12-month momentum, in trading days (breadth uses the same)
MONTH = 21                     # trading days in a month, for the 'one month ago' markers
VOL_SHORT, VOL_LONG = 21, 63   # realized-volatility windows
MIN_RANK_OBS = 504             # observations required before a percentile is reported

LEVERAGE = (2, 3)

# Leverage from a volatility budget: effective volatility over the monthly holding
# horizon = volatility from daily returns x sqrt(variance ratio at that horizon), and
# implied leverage = budget / effective volatility, capped.
VR_Q = 21                      # holding horizon in trading days (sleeves rebalanced monthly)
VR_WINDOW = 504                # estimation window for the variance ratio (about 2 years)
VOL_BUDGETS = (0.15, 0.30, 0.45)
LEV_CAPS = (2.0, 2.5, 3.0)
FINANCING_SPREAD = 0.005       # added to the T-bill rate on the borrowed portion
EXPENSE_RATIO = 0.0095         # typical leveraged-ETF annual fee
FALLBACK_CASH_RATE = 0.04      # used only if the T-bill series cannot be fetched
CASH_TICKER = "^IRX"           # 13-week T-bill yield, in percent

BREADTH_MIN_MEMBERS = 5        # components needed before breadth is reported

# Favorability = 100 minus the point-in-time percentile of effective volatility against the
# ETF's own history: 100 is the calmest it has been, 0 the most turbulent. Five levels with
# equal shares of history (quintiles), best first.
LEVELS = (("very_favorable", "Very favorable"), ("favorable", "Favorable"), ("neutral", "Neutral"),
          ("unfavorable", "Unfavorable"), ("very_unfavorable", "Very unfavorable"))
LEVEL_KEYS = tuple(k for k, _ in LEVELS)
SCORE_REF_LINES = (20, 50, 80)          # red reference lines drawn on the favorability charts
SCORE_CUTS = (20.0, 40.0, 60.0, 80.0)   # very unfavorable <= 20, unfavorable <= 40, favorable >= 60, very favorable >= 80
BACKTEST_HORIZON = 63

# Detail report
REPORT_HORIZONS = (21, 63, 126, 252)                      # forward windows, trading days
LEV_LOOKBACKS = ((63, "3 months"), (126, "6 months"), (252, "1 year"),
                 (756, "3 years"), (1260, "5 years"))
SCORE_BANDS = tuple(range(0, 101, 10))

DEFAULT_EUROPE_WEIGHT = 0.70   # used only if the blend weight cannot be fitted

# Breadth components and APPROXIMATE, STATIC index weights (percent). The weights
# are rough present-day figures applied to all of history; update them from the
# fund providers' fact sheets if you rely on the weighted breadth reading.
ASSETS = {
    "SPY": {
        "name": "US large cap",
        "index": "S&P 500",
        "proxy": None,
        "breadth_kind": "sector ETFs",
        "breadth": {"XLK": 33, "XLF": 13.5, "XLY": 10.5, "XLC": 10, "XLV": 9.5,
                    "XLI": 8.5, "XLP": 5.5, "XLE": 3, "XLU": 2.5, "XLRE": 2, "XLB": 2},
    },
    "EFA": {
        "name": "Developed ex-US",
        "index": "MSCI EAFE",
        "proxy": ("VEURX", "VPACX"),      # Europe + Pacific blend, weight fitted
        "breadth_kind": "country ETFs",
        "breadth": {"EWJ": 22, "EWU": 15, "EWQ": 11, "EWG": 10, "EWL": 9.5, "EWA": 7,
                    "EWN": 5, "EWD": 3.5, "EWP": 3.5, "EWI": 3, "EDEN": 2.5, "EWH": 2,
                    "EWS": 1.7, "EFNL": 1, "EWK": 1, "EIS": 1, "ENOR": 0.7,
                    "EIRL": 0.3, "EWO": 0.2, "ENZL": 0.2},
    },
    "EEM": {
        "name": "Emerging markets",
        "index": "MSCI Emerging Markets",
        "proxy": ("VEIEX",),
        "breadth_kind": "country ETFs",
        "breadth": {"FXI": 28, "EWT": 19, "INDA": 17, "EWY": 12, "EWZ": 4.5, "KSA": 3.5,
                    "EZA": 3.5, "EWW": 2, "UAE": 1.5, "EWM": 1.3, "EIDO": 1.2,
                    "EPOL": 1, "THD": 1, "KWT": 0.7, "QAT": 0.7, "TUR": 0.5,
                    "GREK": 0.5, "ECH": 0.5, "EPHE": 0.4, "EPU": 0.3},
    },
}
CORE = list(ASSETS)

COMPONENT_NAMES = {
    "XLK": "Technology", "XLF": "Financials", "XLY": "Consumer discretionary",
    "XLC": "Communication services", "XLV": "Health care", "XLI": "Industrials",
    "XLP": "Consumer staples", "XLE": "Energy", "XLU": "Utilities", "XLRE": "Real estate",
    "XLB": "Materials",
    "EWJ": "Japan", "EWU": "United Kingdom", "EWQ": "France", "EWG": "Germany",
    "EWL": "Switzerland", "EWA": "Australia", "EWN": "Netherlands", "EWD": "Sweden",
    "EWP": "Spain", "EWI": "Italy", "EDEN": "Denmark", "EWH": "Hong Kong",
    "EWS": "Singapore", "EFNL": "Finland", "EWK": "Belgium", "EIS": "Israel",
    "ENOR": "Norway", "EIRL": "Ireland", "EWO": "Austria", "ENZL": "New Zealand",
    "FXI": "China (large cap)", "EWT": "Taiwan", "INDA": "India", "EWY": "South Korea",
    "EWZ": "Brazil", "KSA": "Saudi Arabia", "EZA": "South Africa", "EWW": "Mexico",
    "UAE": "United Arab Emirates", "EWM": "Malaysia", "EIDO": "Indonesia",
    "EPOL": "Poland", "THD": "Thailand", "KWT": "Kuwait", "QAT": "Qatar",
    "TUR": "Turkey", "GREK": "Greece", "ECH": "Chile", "EPHE": "Philippines", "EPU": "Peru",
}

# (key, label, display format, direction for leverage, core?)
# direction: True = higher is better, False = lower is better, None = not scored either way.
# Core metrics are on the dashboard; the rest (momentum, breadth) are context in the report.
METRICS = [
    ("eff_vol", "Effective volatility", "pct", False, True),
    ("base_vol", "Volatility, blended 21- and 63-day", "pct", False, True),
    ("vol_21", "Volatility, 21-day", "pct", False, True),
    ("vol_63", "Volatility, 63-day", "pct", False, True),
    ("vr21", "Variance ratio, 21-day (trend persistence)", "num", None, True),
    ("mom_3m", "3-month momentum", "pct", True, False),
    ("mom_12m", "12-month momentum", "pct", True, False),
    ("b3", "Breadth: components up over 3 months", "pct", True, False),
    ("b12", "Breadth: components up over 12 months", "pct", True, False),
]


# --------------------------------------------------------------------------- #
# Metric functions (pure; covered by tests/test_metrics.py)
# --------------------------------------------------------------------------- #

def momentum(price: pd.Series, days: int) -> pd.Series:
    """Total return over the last `days` trading days of an adjusted price series."""
    return price / price.shift(days) - 1.0


def realized_vol(ret: pd.Series, days: int) -> pd.Series:
    """Annualized standard deviation of daily returns over a rolling window."""
    return ret.rolling(days).std() * math.sqrt(TRADING_DAYS)


def variance_ratio_lm(r: np.ndarray, q: int) -> float:
    """Lo-MacKinlay bias-corrected variance ratio at horizon q from one window of daily log returns.

    Uses overlapping q-day sums, demeaned, with the denominator m = q(n-q+1)(1-q/n) that
    makes the estimate unbiased (equal to 1 on average) when returns are a random walk.
    """
    r = np.asarray(r, dtype=float)
    n = len(r)
    mu = r.mean()
    var1 = ((r - mu) ** 2).sum() / (n - 1)
    sums = np.convolve(r, np.ones(q), "valid")
    m = q * (n - q + 1) * (1 - q / n)
    return float(((sums - q * mu) ** 2).sum() / m / var1)


def rolling_variance_ratio_lm(log_ret: pd.Series, q: int, window: int) -> pd.Series:
    return log_ret.rolling(window).apply(lambda x: variance_ratio_lm(x, q), raw=True)


def vr_standard_error(q: int, n: int) -> float:
    """Standard error of the overlapping variance ratio when returns are independent (Lo-MacKinlay)."""
    return math.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * n))


def implied_leverage(budget: float, eff_vol, cap: float | None = None):
    """Leverage that keeps the position's effective volatility at the budget, optionally capped."""
    lev = budget / eff_vol
    return lev if cap is None else np.minimum(lev, cap)


def expanding_percentile(s: pd.Series, min_obs: int = MIN_RANK_OBS) -> pd.Series:
    """Percentile (0-100) of each value against all values up to and including it.

    Point-in-time by construction: no later data is used, so the same function
    serves both today's reading and the backtest.
    """
    return s.dropna().expanding(min_periods=min_obs).rank(pct=True).reindex(s.index) * 100.0


def compound(ret: pd.Series, days: int) -> pd.Series:
    """Compounded return over a rolling window of daily returns."""
    return np.expm1(np.log1p(ret.clip(lower=-0.999)).rolling(days).sum())


def leveraged_returns(ret: pd.Series, lev: float, cash_rate: pd.Series | float = 0.0,
                      spread: float = 0.0, fee: float = 0.0) -> pd.Series:
    """Daily returns of a fund that resets to `lev` times exposure every day.

    The borrowed (lev - 1) portion pays the cash rate plus a spread; `fee` is
    the annual expense ratio. Rates are annual decimals.
    """
    cost = ((lev - 1) * (cash_rate + spread) + fee) / TRADING_DAYS
    return lev * ret - cost


def vol_drag(vol: float, lev: float) -> float:
    """Approximate annual return lost to volatility versus lev x the unlevered return."""
    return lev * (lev - 1) / 2.0 * vol ** 2


def breadth(component_px: pd.DataFrame, weights: dict, days: int,
            min_members: int = BREADTH_MIN_MEMBERS) -> pd.DataFrame:
    """Share of components with a positive `days`-day return: equal- and index-weighted."""
    mom = component_px / component_px.shift(days) - 1.0
    have = mom.notna()
    up = (mom > 0) & have
    n = have.sum(axis=1)
    w = pd.Series(weights, dtype=float).reindex(mom.columns).fillna(0.0)
    eq = (up.sum(axis=1) / n).where(n >= min_members)
    wt = (up.mul(w, axis=1).sum(axis=1) / have.mul(w, axis=1).sum(axis=1)).where(n >= min_members)
    return pd.DataFrame({"equal": eq, "weighted": wt, "n": n})


def fit_blend_weight(target: pd.Series, a: pd.Series, b: pd.Series, years: int = 5):
    """Weight on `a` (rest on `b`) whose monthly returns best match `target`.

    Fitted on monthly returns so that stale daily pricing in the proxies does not
    distort it. Returns (weight, correlation of the fitted blend with the target).
    """
    m = pd.concat({"t": target, "a": a, "b": b}, axis=1).dropna()
    m = m.resample("ME").last().pct_change(fill_method=None).dropna().iloc[: years * 12]
    if len(m) < 24:
        return DEFAULT_EUROPE_WEIGHT, float("nan")
    grid = np.linspace(0, 1, 101)
    err = [float(((w * m["a"] + (1 - w) * m["b"] - m["t"]) ** 2).sum()) for w in grid]
    w = float(grid[int(np.argmin(err))])
    return w, float(np.corrcoef(w * m["a"] + (1 - w) * m["b"], m["t"])[0, 1])


def splice(etf_px: pd.Series, proxy_ret: pd.Series | None) -> pd.Series:
    """Price series that follows the ETF once it exists and the proxy's returns before."""
    start = etf_px.first_valid_index()
    r = etf_px.pct_change(fill_method=None)
    if proxy_ret is not None:
        r = r.where(r.index > start, proxy_ret)
    first = r.first_valid_index()
    if first is None:
        return etf_px
    level = (1.0 + r.fillna(0.0)).cumprod()
    level.iloc[: max(r.index.get_loc(first) - 1, 0)] = np.nan   # nothing before the first base price
    return level * (etf_px.dropna().iloc[-1] / level.iloc[-1])


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def all_tickers() -> list[str]:
    t = list(CORE)
    for a in ASSETS.values():
        t += list(a["proxy"] or ()) + list(a["breadth"])
    return list(dict.fromkeys(t)) + [CASH_TICKER]


def _closes(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        close = raw["Close"] if "Close" in level0 else raw.xs("Close", axis=1, level=1)
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    return close.dropna(how="all", axis=1)


def fetch_live(tickers: list[str], notes: list[str]) -> pd.DataFrame:
    """Dividend- and split-adjusted daily closes from Yahoo Finance, with retries."""
    import yfinance as yf

    out = pd.DataFrame()
    missing = list(tickers)
    for attempt in range(4):
        if not missing:
            break
        if attempt:
            time.sleep(15 * attempt)
        try:
            raw = yf.download(missing, start=FETCH_START, auto_adjust=True,
                              progress=False, threads=True, group_by="column")
            got = _closes(raw, missing)
        except Exception as exc:  # network / rate-limit errors: retry
            print(f"fetch attempt {attempt + 1} failed: {exc}", file=sys.stderr)
            continue
        got = got[[c for c in got.columns if got[c].notna().sum() > 30]]
        out = got if out.empty else out.join(got, how="outer")
        missing = [t for t in missing if t not in out.columns]
    if missing:
        notes.append("No data returned for: " + ", ".join(sorted(missing)) + ".")

    # Drop today's bar if the US market has not closed yet (it would be partial).
    try:
        from zoneinfo import ZoneInfo
        now = dt.datetime.now(ZoneInfo("America/New_York"))
        if len(out) and out.index[-1].date() == now.date() and now.time() < dt.time(16, 20):
            out = out.iloc[:-1]
            notes.append("Today's bar was dropped because the US market had not closed.")
    except Exception:
        pass
    return out.sort_index()


# Approximate first-trade dates, used ONLY to shape the synthetic test data.
_SYN_START = {"SPY": "1993-01-29", "EFA": "2001-08-27", "EEM": "2003-04-14",
              "VEIEX": "1994-05-04", "VEURX": "1990-06-18", "VPACX": "1990-06-18",
              "XLRE": "2015-10-08", "XLC": "2018-06-19", "EWT": "2000-06-23",
              "EWY": "2000-05-12", "EWZ": "2000-07-14", "EZA": "2003-02-07",
              "FXI": "2004-10-08", "INDA": "2012-02-03", "^IRX": "1990-01-02"}


def fetch_synthetic(tickers: list[str], seed: int = 7) -> pd.DataFrame:
    """Made-up prices with the same shape as the real data. Never real readings."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(FETCH_START, dt.date.today() - dt.timedelta(days=1))
    n = len(idx)
    vol = np.empty(n)
    v = 0.007
    for i in range(n):  # slow-moving market volatility with occasional spikes
        v = max(0.003, 0.97 * v + 0.03 * 0.007 + 0.0005 * rng.standard_normal()
                + (0.012 if rng.random() < 0.0015 else 0))
        vol[i] = v
    market = 0.0004 + vol * rng.standard_normal(n)
    out = {}
    for t in tickers:
        if t == CASH_TICKER:
            s = pd.Series(np.clip(3 + np.cumsum(0.02 * rng.standard_normal(n)), 0.05, 7), idx)
        else:
            beta = rng.uniform(0.7, 1.3)
            own = rng.uniform(0.4, 0.9) * vol * rng.standard_normal(n)
            r = beta * market + own + rng.uniform(-0.0001, 0.0002)
            if t.startswith("V") and len(t) == 5:  # mutual fund: mimic stale pricing
                r = 0.65 * r + 0.35 * np.roll(r, 1)
            s = pd.Series(50 * np.exp(np.cumsum(r)), idx)
        start = _SYN_START.get(t)
        if start is None:
            start = "1998-12-22" if t.startswith("XL") else (
                "1996-03-18" if t.startswith("EW") else "2010-01-04")
        out[t] = s[s.index >= start]
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def build(px: pd.DataFrame, notes: list[str], synthetic: bool) -> dict:
    for t in CORE:
        if t not in px.columns:
            raise SystemExit(f"Core ticker {t} is missing; refusing to publish.")
    cal = px["SPY"].dropna().loc[FETCH_START:].index       # master trading calendar
    px = px.reindex(cal)
    asof = cal[-1]
    if not synthetic and (pd.Timestamp.today().normalize() - asof).days > 7:
        raise SystemExit(f"Latest data is {asof.date()}, more than a week old; refusing to publish.")
    for t in CORE:
        if pd.isna(px[t].iloc[-1]):
            raise SystemExit(f"{t} has no price for {asof.date()}; refusing to publish.")

    if CASH_TICKER in px.columns and px[CASH_TICKER].notna().sum() > 250:
        cash = (px[CASH_TICKER] / 100.0).ffill().bfill()
        cash_note = "13-week T-bill yield (^IRX)"
    else:
        cash = pd.Series(FALLBACK_CASH_RATE, index=cal)
        cash_note = f"fixed {FALLBACK_CASH_RATE:.1%} (T-bill series unavailable)"
        notes.append("T-bill series unavailable; financing cost uses a fixed rate.")

    frames, info = {}, {}
    for a, cfg in ASSETS.items():
        etf = px[a]
        etf_start = etf.first_valid_index()
        meta = {"etf_start": etf_start, "proxy": None, "blend_weight": None,
                "blend_corr": None, "stale_ac_backfill": None, "stale_ac_etf": None}
        proxy_ret = None
        prox = [p for p in (cfg["proxy"] or ()) if p in px.columns]
        if cfg["proxy"] and len(prox) < len(cfg["proxy"]):
            notes.append(f"{a}: backfill fund data missing, so history starts at the ETF's "
                         f"own inception ({etf_start.date()}).")
        elif len(prox) == 1:
            proxy_ret = px[prox[0]].pct_change(fill_method=None)
            meta["proxy"] = prox[0]
        elif len(prox) == 2:
            w, corr = fit_blend_weight(etf, px[prox[0]], px[prox[1]])
            proxy_ret = (w * px[prox[0]].pct_change(fill_method=None)
                         + (1 - w) * px[prox[1]].pct_change(fill_method=None))
            meta.update(proxy=f"{w:.0%} {prox[0]} + {1 - w:.0%} {prox[1]}",
                        blend_weight=w, blend_corr=corr)
        price = splice(etf, proxy_ret)
        ret = price.pct_change(fill_method=None)
        log_ret = np.log(price).diff()

        f = pd.DataFrame(index=cal)
        f["price"] = price
        f["mom_3m"] = momentum(price, MOM_SHORT)
        f["mom_12m"] = momentum(price, MOM_LONG)
        f["vol_21"] = realized_vol(ret, VOL_SHORT)
        f["vol_63"] = realized_vol(ret, VOL_LONG)

        # Core metric. Effective volatility = volatility from daily returns x sqrt(VR_21): the
        # risk of a position held for a month. The variance ratio is left blank wherever its
        # window touches backfilled prices, because stale fund pricing inflates serial correlation.
        f["vr21"] = rolling_variance_ratio_lm(log_ret, VR_Q, VR_WINDOW)
        if proxy_ret is not None:
            f.loc[f.index < cal[min(cal.get_loc(etf_start) + VR_WINDOW, len(cal) - 1)], "vr21"] = np.nan
        f["base_vol"] = np.sqrt((f["vol_21"] ** 2 + f["vol_63"] ** 2) / 2.0)
        f["eff_vol"] = f["base_vol"] * np.sqrt(f["vr21"])
        for b in VOL_BUDGETS:
            f[f"lev_{round(b * 100)}"] = implied_leverage(b, f["eff_vol"])     # uncapped

        if proxy_ret is not None:  # evidence for the stale-pricing caveat
            pre = ret[(ret.index >= HISTORY_START) & (ret.index <= etf_start)].dropna()
            post = ret[ret.index > etf_start].dropna()
            if len(pre) > 250:
                meta["stale_ac_backfill"] = float(pre.autocorr(1))
                meta["stale_ac_etf"] = float(post.autocorr(1))

        comps = px[[c for c in cfg["breadth"] if c in px.columns]].ffill(limit=5)
        lost = [c for c in cfg["breadth"] if c not in px.columns]
        if lost:
            notes.append(f"{a} breadth is missing components: {', '.join(lost)}.")
        b12 = breadth(comps, cfg["breadth"], MOM_LONG)
        b3 = breadth(comps, cfg["breadth"], MOM_SHORT)
        f["b12"], f["b12w"], f["b12n"] = b12["equal"], b12["weighted"], b12["n"]
        f["b3"], f["b3w"] = b3["equal"], b3["weighted"]

        # Daily-reset leverage: trailing 1-year simulation and the forward window for the backtest.
        f["ret"] = ret
        f["dd"] = price / price.cummax() - 1.0         # drawdown from the highest prior close
        f["und_1y"] = compound(ret, TRADING_DAYS)
        f["fwd_1x"] = compound(ret, BACKTEST_HORIZON).shift(-BACKTEST_HORIZON)
        for lev in LEVERAGE:
            gross = leveraged_returns(ret, lev)
            net = leveraged_returns(ret, lev, cash, FINANCING_SPREAD, EXPENSE_RATIO)
            f[f"dg_{lev}x"], f[f"dn_{lev}x"] = gross, net
            f[f"gross_{lev}x"] = compound(gross, TRADING_DAYS)
            f[f"net_{lev}x"] = compound(net, TRADING_DAYS)
            f[f"gap_{lev}x"] = f[f"gross_{lev}x"] - lev * f["und_1y"]
            f[f"fwd_net_{lev}x"] = compound(net, BACKTEST_HORIZON).shift(-BACKTEST_HORIZON)
            f[f"fwd_gap_{lev}x"] = (compound(gross, BACKTEST_HORIZON).shift(-BACKTEST_HORIZON)
                                    - lev * f["fwd_1x"])
        frames[a], info[a] = f, meta

    # One common ranking window so no ETF is judged against an easier history.
    firsts = [frames[a][["mom_12m", "vol_63"]].dropna().index[0] for a in CORE]
    rank_start = max(max(firsts), pd.Timestamp(HISTORY_START))
    if rank_start > pd.Timestamp(HISTORY_START) + pd.Timedelta(days=45):
        notes.append(f"Common history starts {rank_start.date()}, later than the "
                     f"{HISTORY_START} target, because earlier data was unavailable.")

    for a in CORE:
        f, meta = frames[a], info[a]
        for key, *_ in METRICS:
            f["p_" + key] = expanding_percentile(f[key].where(f.index >= rank_start))
        f["score"] = 100.0 - f["p_eff_vol"]          # favorability: low effective volatility = favorable
        label = pd.Series("neutral", index=f.index, dtype=object)
        label[f["score"] >= SCORE_CUTS[2]] = "favorable"
        label[f["score"] >= SCORE_CUTS[3]] = "very_favorable"
        label[f["score"] <= SCORE_CUTS[1]] = "unfavorable"
        label[f["score"] <= SCORE_CUTS[0]] = "very_unfavorable"
        f["label"] = label.where(f["score"].notna())

    return {"frames": frames, "info": info, "asof": asof, "rank_start": rank_start,
            "cash": cash, "cash_note": cash_note, "px": px}


def _num(x, digits=4):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(x) else round(x, digits)


def make_payload(res: dict, notes: list[str], synthetic: bool) -> dict:
    frames, info, asof, rank_start = res["frames"], res["info"], res["asof"], res["rank_start"]

    assets = {}
    for a in CORE:
        f, meta, cfg = frames[a], info[a], ASSETS[a]
        last = f.iloc[-1]
        cells = {}
        for key, _, _, higher_better, _ in METRICS:
            hist = f[key][f.index >= rank_start].dropna()
            pct = _num(last["p_" + key], 1)
            cells[key] = {
                "value": _num(last[key]), "pct": pct,
                "fav": None if pct is None or higher_better is None else _num(pct if higher_better else 100 - pct, 1),
                "prev": _num(f[key].iloc[-1 - MONTH]),
                "q": [_num(hist.quantile(p)) for p in (0.01, 0.25, 0.5, 0.75, 0.99)] if len(hist) else None,
                "n": int(len(hist)), "since": str(hist.index[0].date()) if len(hist) else None,
            }
        lev = {}
        for L in LEVERAGE:
            lev[str(L)] = {"naive": _num(L * last["und_1y"]), "gross": _num(last[f"gross_{L}x"]),
                           "gap": _num(last[f"gap_{L}x"]), "net": _num(last[f"net_{L}x"]),
                           "drag": _num(vol_drag(last["vol_21"], L))}
        ef = f[(f.index > meta["etf_start"]) & f["label"].notna() & f[f"fwd_net_{LEVERAGE[-1]}x"].notna()]
        backtest = []
        for lab in LEVEL_KEYS:
            g = ef[ef["label"] == lab]
            row = {"label": lab, "days": int(len(g)), "share": _num(len(g) / max(len(ef), 1), 3),
                   "windows": int(len(g) // BACKTEST_HORIZON)}
            if len(g):
                row.update(fwd_1x=_num(g["fwd_1x"].median()))
                for L in LEVERAGE:
                    row[f"net_{L}x"] = _num(g[f"fwd_net_{L}x"].median())
                    row[f"gap_{L}x"] = _num(g[f"fwd_gap_{L}x"].median())
                top = LEVERAGE[-1]
                row["win"] = _num((g[f"fwd_net_{top}x"] > 0).mean(), 3)
                row["p5"] = _num(g[f"fwd_net_{top}x"].quantile(0.05))
            backtest.append(row)
        assets[a] = {
            "name": cfg["name"], "index": cfg["index"], "breadth_kind": cfg["breadth_kind"],
            "etf_start": str(meta["etf_start"].date()), "proxy": meta["proxy"],
            "blend_weight": _num(meta["blend_weight"], 2), "blend_corr": _num(meta["blend_corr"], 3),
            "stale_ac_backfill": _num(meta["stale_ac_backfill"], 3),
            "stale_ac_etf": _num(meta["stale_ac_etf"], 3),
            "score": _num(last["score"], 1), "label": last["label"] if isinstance(last["label"], str) else None,
            "score_prev": _num(f["score"].iloc[-1 - MONTH], 1), "dd": _num(last["dd"]),
            "score_from": str(f["score"].first_valid_index().date()) if f["score"].notna().any() else None,
            "b12w": _num(last["b12w"]), "b3w": _num(last["b3w"]),
            "b_members": int(last["b12n"]) if pd.notna(last["b12n"]) else 0,
            "b_total": len(cfg["breadth"]), "und_1y": _num(last["und_1y"]),
            "cells": cells, "lev": lev, "backtest": backtest,
            "vb": {k: _num(last[k]) for k in ("vol_21", "vol_63", "base_vol", "vr21", "eff_vol")}
                  | {k + "_prev": _num(f[k].iloc[-1 - MONTH]) for k in ("vr21", "eff_vol", "base_vol")}
                  | {"lev": {str(round(b * 100)): _num(last[f"lev_{round(b * 100)}"]) for b in VOL_BUDGETS},
                     "lev_prev": {str(round(b * 100)): _num(f[f"lev_{round(b * 100)}"].iloc[-1 - MONTH])
                                  for b in VOL_BUDGETS},
                     "vr_from": str(f["vr21"].first_valid_index().date()) if f["vr21"].notna().any() else None},
            "backtest_from": str(ef.index[0].date()) if len(ef) else None,
        }

    # Weekly samples (last trading day of each week) keep the page small.
    keys = ["score", "eff_vol", "base_vol", "vol_21", "vol_63", "vr21"] + [f"lev_{round(b * 100)}" for b in VOL_BUDGETS]
    weekly = None
    series = {}
    for a in CORE:
        f = frames[a][frames[a].index >= rank_start]
        wk = f.groupby(f.index.to_period("W-FRI")).tail(1)
        weekly = wk.index
        series[a] = {k: [_num(v) for v in wk[k]] for k in keys}
        series[a]["dd"] = _weekly_low(f, "dd")
    series["dates"] = [str(d.date()) for d in weekly]

    return {
        "asof": str(asof.date()),
        "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "synthetic": synthetic, "rank_start": str(rank_start.date()),
        "cash_now": _num(res["cash"].iloc[-1]), "cash_note": res["cash_note"],
        "notes": notes, "assets": assets, "series": series, "order": CORE,
        "metrics": [{"key": k, "label": lab, "fmt": fmt, "higher_better": hb, "core": core}
                    for k, lab, fmt, hb, core in METRICS],
        "config": {"mom": [MOM_SHORT, MOM_LONG], "vol": [VOL_SHORT, VOL_LONG],
                   "vr_q": VR_Q, "vr_window": VR_WINDOW, "vr_se": _num(vr_standard_error(VR_Q, VR_WINDOW), 3),
                   "budgets": [round(b * 100) for b in VOL_BUDGETS], "caps": list(LEV_CAPS),
                   "min_rank": MIN_RANK_OBS,
                   "leverage": list(LEVERAGE), "spread": FINANCING_SPREAD, "fee": EXPENSE_RATIO,
                   "cuts": list(SCORE_CUTS), "ref_lines": list(SCORE_REF_LINES),
                   "levels": [{"key": k, "name": n} for k, n in LEVELS],
                   "horizon": BACKTEST_HORIZON,
                   "breadth_min": BREADTH_MIN_MEMBERS,
                   "breadth": {a: ASSETS[a]["breadth"] for a in CORE}},
    }


def _weekly_low(f: pd.DataFrame, key: str) -> list:
    """Lowest value in each week, so a trough between weekly samples is not missed."""
    return [_num(v) for v in f[key].groupby(f.index.to_period("W-FRI")).min()]


def _weekly(f: pd.DataFrame, keys: list[str], digits: int = 4) -> tuple[pd.DatetimeIndex, dict]:
    wk = f.groupby(f.index.to_period("W-FRI")).tail(1)
    return wk.index, {k: [_num(v, digits) for v in wk[k]] for k in keys}


def _fwd_stats(g: pd.DataFrame, top: int) -> dict:
    """Median outcomes over the rows of `g`, which holds forward-return columns."""
    if not len(g):
        return {"days": 0}
    return {"days": int(len(g)), "fwd_1x": _num(g["f1"].median()), "net": _num(g["fn"].median()),
            "gap": _num(g["fg"].median()), "win": _num((g["fn"] > 0).mean(), 3),
            "p5": _num(g["fn"].quantile(0.05))}


def make_report(res: dict, payload: dict) -> dict:
    """Everything the dashboard summarizes, in full: the payload for report.html."""
    frames, info, px, asof = res["frames"], res["info"], res["px"], res["asof"]
    rank_start, top = res["rank_start"], LEVERAGE[-1]
    metric_keys = [m[0] for m in METRICS]
    extra = ["b12w", "b3w", "und_1y", "score"] + [f"lev_{round(b * 100)}" for b in VOL_BUDGETS] \
        + [f"{k}_{L}x" for L in LEVERAGE for k in ("gap", "net")]
    series, assets = {}, {}
    for a in CORE:
        f, meta, cfg = frames[a], info[a], ASSETS[a]
        shown = f[f.index >= rank_start]
        dates, raw = _weekly(shown, metric_keys + extra)
        _, pcts = _weekly(shown, ["p_" + k for k in metric_keys], digits=1)
        series[a] = {**raw, **pcts, "dd": _weekly_low(shown, "dd")}
        series["dates"] = [str(d.date()) for d in dates]

        lookback = {k: [_num(f[k].iloc[-1 - n]) for n in (21, 63, 252)] for k in metric_keys}

        comps = []
        for t, w in cfg["breadth"].items():
            row = {"ticker": t, "name": COMPONENT_NAMES.get(t, ""), "weight": w}
            if t in px.columns:
                c = px[t].ffill(limit=5)
                row.update(m3=_num(momentum(c, MOM_SHORT).iloc[-1]), m12=_num(momentum(c, MOM_LONG).iloc[-1]),
                           since=str(px[t].first_valid_index().date()))
            comps.append(row)

        lev = []
        for n, label in LEV_LOOKBACKS:
            und = compound(f["ret"], n).iloc[-1]
            row = {"label": label, "und": _num(und)}
            for L in LEVERAGE:
                row[f"naive_{L}"] = _num(L * und)
                row[f"gross_{L}"] = _num(compound(f[f"dg_{L}x"], n).iloc[-1])
                row[f"net_{L}"] = _num(compound(f[f"dn_{L}x"], n).iloc[-1])
            lev.append(row)

        # Forward outcomes, using only dates after the ETF itself began trading.
        era = (f.index > meta["etf_start"]) & f["label"].notna()
        by_horizon = {}
        fwd63 = None
        for h in REPORT_HORIZONS:
            f1 = compound(f["ret"], h).shift(-h)
            fw = pd.DataFrame({"f1": f1, "fn": compound(f[f"dn_{top}x"], h).shift(-h),
                               "fg": compound(f[f"dg_{top}x"], h).shift(-h) - top * f1})
            fw = fw[era & fw["fn"].notna()]
            by_horizon[str(h)] = {lab: _fwd_stats(fw[f["label"].reindex(fw.index) == lab], top)
                                  for lab in LEVEL_KEYS}
            if h == BACKTEST_HORIZON:
                fwd63 = fw
        score = f["score"].reindex(fwd63.index)
        bands = []
        for lo, hi in zip(SCORE_BANDS[:-1], SCORE_BANDS[1:]):
            g = fwd63[(score >= lo) & ((score < hi) | (hi == SCORE_BANDS[-1]))]
            bands.append({"band": f"{lo} to {hi}", "share": _num(len(g) / max(len(fwd63), 1), 3),
                          **_fwd_stats(g, top)})
        alone = []
        for key, label, _, higher_better, _ in METRICS:
            if higher_better is None:
                continue
            p = f["p_" + key].reindex(fwd63.index)
            fav = p if higher_better else 100 - p
            thirds = {"fav": fwd63[fav >= 200 / 3], "mid": fwd63[(fav > 100 / 3) & (fav < 200 / 3)],
                      "unfav": fwd63[fav <= 100 / 3]}
            alone.append({"key": key, "label": label,
                          **{k: _fwd_stats(v, top) for k, v in thirds.items()}})

        lab = f["label"].dropna()
        changes = lab[lab != lab.shift()]
        recent = changes[changes.index >= asof - pd.Timedelta(days=365)]
        prev = lab.shift().reindex(recent.index)
        assets[a] = {
            "lookback": lookback, "components": comps, "lev": lev, "by_horizon": by_horizon,
            "bands": bands, "alone": alone,
            "streak": int(len(lab.loc[changes.index[-1]:])) if len(changes) else 0,
            "changes": [{"date": str(d.date()), "from": prev[d] if isinstance(prev[d], str) else None,
                         "to": recent[d]} for d in recent.index][::-1],
        }

    roles = {t: "Core ETF" for t in CORE}
    for a, cfg in ASSETS.items():
        for t in cfg["proxy"] or ():
            roles[t] = f"Backfill for {a}"
        for t in cfg["breadth"]:
            roles[t] = f"Breadth for {a}: {COMPONENT_NAMES.get(t, t)}"
    roles[CASH_TICKER] = "Financing rate (13-week T-bill yield)"
    coverage = []
    for t, role in roles.items():
        c = px[t].dropna() if t in px.columns else pd.Series(dtype=float)
        coverage.append({"ticker": t, "role": role, "first": str(c.index[0].date()) if len(c) else None,
                         "last": str(c.index[-1].date()) if len(c) else None, "n": int(len(c))})

    out = dict(payload)
    out["series"] = series
    out["report"] = {"assets": assets, "coverage": coverage, "horizons": list(REPORT_HORIZONS),
                     "fetch_start": FETCH_START}
    return out


def render_page(title: str, body: str, script: str, payload: dict) -> str:
    blob = json.dumps(payload, allow_nan=False, separators=(",", ":")).replace("</", "<\\/")
    return (PAGE.replace("__TITLE__", title).replace("__CSS__", CSS).replace("__BODY__", body)
            .replace("__SCRIPT__", JS_COMMON + JS_DOC + script).replace("/*__PAYLOAD__*/null", blob))


def write_outputs(payload: dict, out_dir: Path, history_path: Path | None,
                  report: dict | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(
        render_page("Leverage Environment", DASH_BODY, DASH_JS, payload), encoding="utf-8")
    if report is not None:
        (out_dir / "report.html").write_text(
            render_page("Leverage Environment: Detail", REPORT_BODY, REPORT_JS, report), encoding="utf-8")
    slim = {k: v for k, v in payload.items() if k != "series"}
    (out_dir / "latest.json").write_text(json.dumps(slim, indent=1), encoding="utf-8")

    if history_path is None or payload["synthetic"]:
        return
    # One row per ETF per day: a genuine point-in-time record that builds up over time.
    fields = ["date", "etf", "label", "score"] + [m["key"] for m in payload["metrics"]] \
        + ["p_" + m["key"] for m in payload["metrics"]] \
        + ["base_vol", "vr21", "eff_vol"] + [f"lev_{b}" for b in payload["config"]["budgets"]]
    rows = []
    if history_path.exists():
        with history_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            rows = [r for r in reader if r.get("date") != payload["asof"]]
            fields += [c for c in (reader.fieldnames or []) if c not in fields]   # keep older columns
    for a in payload["order"]:
        d = payload["assets"][a]
        row = {"date": payload["asof"], "etf": a, "label": d["label"], "score": d["score"]}
        for m in payload["metrics"]:
            row[m["key"]] = d["cells"][m["key"]]["value"]
            row["p_" + m["key"]] = d["cells"][m["key"]]["pct"]
        for k in ("base_vol", "vr21", "eff_vol"):
            row[k] = d["vb"][k]
        for b, v in d["vb"]["lev"].items():
            row["lev_" + b] = v                       # uncapped
        rows.append(row)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(sorted(rows, key=lambda r: (r["date"], r["etf"])))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="site", help="output folder for index.html, report.html, latest.json")
    ap.add_argument("--history", default="data/history.csv", help="daily log to append to ('' to skip)")
    ap.add_argument("--synthetic", action="store_true", help="use made-up data (offline test)")
    args = ap.parse_args()

    notes: list[str] = []
    px = fetch_synthetic(all_tickers()) if args.synthetic else fetch_live(all_tickers(), notes)
    if px.empty:
        raise SystemExit("No price data could be fetched; refusing to publish.")
    res = build(px, notes, args.synthetic)
    payload = make_payload(res, notes, args.synthetic)
    write_outputs(payload, Path(args.out), Path(args.history) if args.history else None,
                  report=make_report(res, payload))
    for a in CORE:
        d = payload["assets"][a]
        print(f"{payload['asof']} {a}: {d['label']} (score {d['score']})")
    for n in notes:
        print("note:", n)


# --------------------------------------------------------------------------- #
# Page templates (data is injected as JSON; everything renders in the browser)
# --------------------------------------------------------------------------- #

PAGE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@3.6.0/plotly.min.js"></script>
<style>
__CSS__</style>
</head>
<body>
<main>
__BODY__</main>
<script>
const D = /*__PAYLOAD__*/null;
__SCRIPT__</script>
</body>
</html>
'''

CSS = r''':root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;
--good:#0ca30c;--warn:#fab219;--crit:#d03b3b;--band:rgba(11,11,11,.05)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;
--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--band:rgba(255,255,255,.06)}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--band:rgba(255,255,255,.06)}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:18px;margin:44px 0 6px}
h3{font-size:15px;margin:22px 0 4px}
p{margin:6px 0}.sub{color:var(--ink2);max-width:78ch}.small{font-size:13px;color:var(--ink2)}
.banner{background:var(--warn);color:#0b0b0b;padding:10px 14px;border-radius:8px;font-weight:600;margin:0 0 16px}
.notes{border:1px solid var(--border);border-left:4px solid var(--warn);border-radius:8px;padding:10px 14px;margin:14px 0;background:var(--surface)}
.notes ul{margin:4px 0 0;padding-left:18px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:14px;margin-top:18px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px}
.card .tk{display:flex;align-items:center;gap:8px;font-weight:650;font-size:17px}
.sw{width:12px;height:12px;border-radius:3px;display:inline-block;flex:none}
.card .nm{color:var(--ink2);font-size:13px;margin-bottom:10px}
.verdict{display:flex;flex-wrap:wrap;align-items:baseline;gap:2px 10px;margin:6px 0 2px}
.verdict .lab{font-size:22px;font-weight:650;display:flex;align-items:center;gap:7px;white-space:nowrap}
.verdict .sc{color:var(--ink2);font-size:13px}
.ico{font-size:13px;letter-spacing:-2px;margin-right:2px}.ico.favorable,.ico.very_favorable{color:var(--good)}.ico.unfavorable,.ico.very_unfavorable{color:var(--crit)}.ico.neutral{color:var(--muted)}
.flag{display:inline-block;font-size:12px;border:1px solid var(--border);border-radius:999px;padding:1px 8px;margin-top:4px;color:var(--ink2)}
dl{display:grid;grid-template-columns:1fr auto;gap:3px 12px;margin:12px 0 0;font-size:13.5px}
dt{color:var(--ink2)}dd{margin:0;text-align:right;font-variant-numeric:tabular-nums}
dt.h{grid-column:1/-1;color:var(--muted);font-size:11.5px;text-transform:uppercase;letter-spacing:.05em;margin-top:8px}
.scroll{overflow-x:auto;border:1px solid var(--border);border-radius:12px;background:var(--surface);margin-top:12px}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{padding:9px 12px;text-align:left;border-bottom:1px solid var(--grid);vertical-align:top}
tr:last-child td{border-bottom:0}
th{font-weight:600;color:var(--ink2);font-size:12.5px;white-space:nowrap}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
td.nw{white-space:nowrap}#bt th{white-space:normal;vertical-align:bottom}td.grp{color:var(--muted);font-size:11.5px;text-transform:uppercase;letter-spacing:.05em;padding-top:14px;border-bottom:0}
.cell{min-width:170px}.cell .v{font-weight:600;font-variant-numeric:tabular-nums}
.cell .p{color:var(--ink2);font-size:12.5px;margin-left:6px}
.cell .f{font-size:12px;color:var(--ink2);display:flex;align-items:center;gap:5px;margin-top:1px}
.dot{width:7px;height:7px;border-radius:50%;display:inline-block}
.rb{display:block;width:100%;height:16px;margin-top:3px}
.legend{display:flex;flex-wrap:wrap;gap:16px;align-items:center;margin:12px 0 8px;font-size:13.5px}
.legend span{display:flex;align-items:center;gap:6px}
.legend .line{width:16px;height:3px;border-radius:2px;display:inline-block}
.btns{margin-left:auto;display:flex;gap:4px}
button{font:inherit;font-size:13px;padding:4px 11px;border-radius:7px;border:1px solid var(--border);background:var(--surface);color:var(--ink);cursor:pointer}
button[aria-pressed="true"]{background:var(--ink);color:var(--surface)}
.charts{display:grid;grid-template-columns:1fr;gap:14px}
@media (max-width:520px){.charts{grid-template-columns:1fr}}
.chart{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:12px 8px 4px 8px;min-width:0}
.chart h3{margin:0 8px 0;font-size:14px}.chart p{margin:0 8px;font-size:12.5px;color:var(--ink2)}
.plot{height:230px}
.doc{max-width:82ch}.doc li{margin:5px 0}.doc ul{padding-left:20px;margin:6px 0}
code{font-size:.92em;background:var(--band);padding:1px 4px;border-radius:4px}
footer{margin-top:40px;color:var(--muted);font-size:12.5px}
.plot.tall{height:360px}
#vbcharts{display:grid;gap:14px;margin:10px 0 6px}
.vbleg{display:inline-flex;gap:14px;margin-left:8px;font-weight:400}
.vbleg span{display:inline-flex;align-items:center;gap:5px}
td.cap{color:var(--ink2)}
#ddcharts{display:grid;gap:14px;margin:10px 0 6px}
.ctl{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:12px 0 4px;font-size:13.5px}
select{font:inherit;font-size:13.5px;padding:5px 8px;border-radius:7px;border:1px solid var(--border);background:var(--surface);color:var(--ink);max-width:100%}
.seg{display:flex;gap:4px}
th.w{white-space:normal;vertical-align:bottom}
.toc{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:13.5px;margin:14px 0 0}
h3.etf{display:flex;align-items:center;gap:8px;margin-top:26px}
.up{color:var(--ink)}.muted{color:var(--muted)}
'''

JS_COMMON = r'''const $ = (id) => document.getElementById(id);
const C = D.config, A = D.order;
const pct = (x, d = 1) => x == null ? "n/a" : (x * 100).toFixed(d) + "%";
const spct = (x, d = 1) => x == null ? "n/a" : (x > 0 ? "+" : "") + (x * 100).toFixed(d) + "%";
const num = (x, d = 2) => x == null ? "n/a" : x.toFixed(d);
const ord = (n) => { n = Math.round(n); const s = ["th", "st", "nd", "rd"], v = n % 100; return n + (s[(v - 20) % 10] || s[v] || s[0]); };
const fmtv = (m, x, signed) => m.fmt === "pct" ? (signed ? spct(x) : pct(x)) : num(x);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const COLOR = { SPY: "--s1", EFA: "--s2", EEM: "--s3" };
const ICON = { very_favorable: "\u25B2\u25B2", favorable: "\u25B2", neutral: "\u25A0", unfavorable: "\u25BC", very_unfavorable: "\u25BC\u25BC" };
const LEVELS = C.levels.map((l) => l.key);
const lname = (k) => (C.levels.find((l) => l.key === k) || { name: "n/a" }).name;
const cap = (s) => s ? s[0].toUpperCase() + s.slice(1) : "n/a";
const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
const favWord = (f) => f == null ? "" : f >= 66.7 ? "favorable" : f <= 33.3 ? "unfavorable" : "middling";
const favColor = (f) => f >= 66.7 ? "var(--good)" : f <= 33.3 ? "var(--crit)" : "var(--muted)";
const quint = (p) => p == null ? null : Math.min(5, Math.max(1, Math.ceil(p / 20)));
const pctText = (key, p) => p == null ? "not enough history" : (key !== "mom_12m" && p < 1) ? "lowest 1%" :
  key === "mom_12m" ? "quintile " + quint(p) + " of 5" : ord(p) + " percentile";

if (D.synthetic) $("banner").innerHTML = '<div class="banner">SYNTHETIC TEST DATA. Every number on this page is made up to test the layout and is not a market reading.</div>';
$("asof").textContent = "Data through " + D.asof + "; ranked against history since " + D.rank_start + ". Page built " + D.generated + ".";
if (D.notes.length) $("notes").innerHTML = '<div class="notes"><strong>Data notes for this run</strong><ul>' +
  D.notes.map((n) => "<li>" + esc(n) + "</li>").join("") + "</ul></div>";

const TOP = C.leverage[C.leverage.length - 1];
const backfilled = A.filter((a) => D.assets[a].proxy);
const shadeEnd = backfilled.map((a) => D.assets[a].etf_start).sort().pop();
const ZOOM_HINT = matchMedia("(pointer: coarse)").matches ? "" : " Drag across a chart to zoom in on a period or range; double-click it to reset.";
const rangeBtns = () => '<div class="btns" role="group" aria-label="Time range">' + [["1Y", 1], ["5Y", 5], ["10Y", 10], ["All", 0]].map(([t, y]) =>
    '<button data-y="' + y + '" aria-pressed="' + (y === 0) + '">' + t + "</button>").join("") + "</div>";
const legendHtml = () => A.map((a) => '<span><i class="line" style="background:var(' + COLOR[a] + ')"></i>' + a + "</span>").join("") + rangeBtns();
const shadeNote = () => backfilled.length ? "The shaded era is backfilled from mutual funds (" +
  backfilled.map((a) => a + " before " + D.assets[a].etf_start).join(", ") + ") and is approximate; see the caveats below. Weekly samples." + ZOOM_HINT : "Weekly samples." + ZOOM_HINT;
// Drag-to-zoom needs a mouse or trackpad; on touch screens a drag has to scroll the page.
const CAN_ZOOM = !matchMedia("(pointer: coarse)").matches;
let years = 0;
function xRange() {
  const d = D.series.dates, end = d[d.length - 1];
  if (!years) return [d[0], end];
  const s = new Date(end); s.setFullYear(s.getFullYear() - years);
  const iso = s.toISOString().slice(0, 10);
  return [iso < d[0] ? d[0] : iso, end];
}
// One line per ETF. c: {pct, zero, band, lo, hi}; get(a) returns that ETF's values.
function plotLines(id, c, get) {
  const [x0, x1] = xRange(), ink2 = css("--ink2"), grid = css("--grid"), axis = css("--axis"), band = css("--band");
  const k = c.pct ? 100 : 1; let lo = Infinity, hi = -Infinity;
  const traces = A.map((a) => {
    const y = get(a).map((v) => v == null ? null : v * k);
    D.series.dates.forEach((d, j) => { if (d >= x0 && d <= x1 && y[j] != null) { lo = Math.min(lo, y[j]); hi = Math.max(hi, y[j]); } });
    return { x: D.series.dates, y, name: a, mode: "lines", line: { color: css(COLOR[a]), width: 1.6 },
      hovertemplate: "%{y:." + (c.pct ? 1 : 2) + "f}" + (c.pct ? "%" : "") + "<extra>" + a + "</extra>" };
  });
  if (c.band) { lo = Math.min(lo, -c.band); hi = Math.max(hi, c.band); }
  if (!isFinite(lo)) { lo = 0; hi = 1; }
  const pad = (hi - lo) * 0.07 || 1, shapes = [];
  if (shadeEnd && shadeEnd > x0) shapes.push({ type: "rect", xref: "x", yref: "paper", x0: x0, x1: shadeEnd, y0: 0, y1: 1, fillcolor: band, line: { width: 0 }, layer: "below" });
  if (c.band) shapes.push({ type: "rect", xref: "paper", yref: "y", x0: 0, x1: 1, y0: -c.band, y1: c.band, fillcolor: band, line: { width: 0 }, layer: "below" });
  (c.red || []).forEach((v) => shapes.push({ type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: v * k, y1: v * k, line: { color: css("--crit"), width: 1.2 }, layer: "above" }));
  if (c.zero != null) shapes.push({ type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: c.zero * k, y1: c.zero * k, line: { color: axis, width: 1 }, layer: "below" });
  Plotly.react(id, traces, {
    margin: { l: 46, r: 12, t: 8, b: 26 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", showlegend: false,
    hovermode: "x unified", hoverlabel: { bgcolor: css("--surface"), bordercolor: axis, font: { color: css("--ink"), size: 12 } },
    font: { family: 'system-ui,-apple-system,"Segoe UI",sans-serif', size: 11, color: ink2 },
    xaxis: { range: [x0, x1], hoverformat: "%b %-d, %Y", showgrid: false, linecolor: axis, tickcolor: axis, fixedrange: !CAN_ZOOM },
    yaxis: { range: [lo - pad, hi + pad], gridcolor: grid, zeroline: false, ticksuffix: c.pct ? "%" : "", fixedrange: !CAN_ZOOM },
    dragmode: CAN_ZOOM ? "zoom" : false, shapes,
  }, { displayModeBar: false, responsive: true, doubleClick: "reset" });
}
function bindRange(draw) {
  document.querySelectorAll(".btns button").forEach((b) => b.addEventListener("click", () => {
    years = +b.dataset.y;
    document.querySelectorAll(".btns button").forEach((o) => o.setAttribute("aria-pressed", String(o.dataset.y === b.dataset.y)));
    draw();
  }));
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
}
const rgba = (hex, al) => { const h = hex.replace("#", ""); return "rgba(" + [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)).join(",") + "," + al + ")"; };
// One ETF: favorability on top, drawdown beneath, on a shared time axis (two panels, not two scales on one).
function plotScoreDrawdown(id, a) {
  const [x0, x1] = xRange(), ink2 = css("--ink2"), grid = css("--grid"), axis = css("--axis"), band = css("--band"), col = css(COLOR[a]);
  const dates = D.series.dates, dd = D.series[a].dd.map((v) => v == null ? null : v * 100);
  let lo = -1; dates.forEach((d, j) => { if (d >= x0 && d <= x1 && dd[j] != null) lo = Math.min(lo, dd[j]); });
  const shapes = [];
  C.ref_lines.forEach((v) => shapes.push({ type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: v, y1: v, line: { color: css("--crit"), width: 1.2 }, layer: "above" }));
  const start = D.assets[a].proxy ? D.assets[a].etf_start : null;
  if (start && start > x0) shapes.push({ type: "rect", xref: "x", yref: "paper", x0: x0, x1: start, y0: 0, y1: 1, fillcolor: band, line: { width: 0 }, layer: "below" });
  const yax = { gridcolor: grid, zeroline: false, fixedrange: !CAN_ZOOM, title: { font: { size: 11, color: ink2 }, standoff: 6 } };
  Plotly.react(id, [
    { x: dates, y: D.series[a].score, name: "Score", mode: "lines", line: { color: col, width: 1.6 }, yaxis: "y", hovertemplate: "%{y:.0f}<extra>Favorability</extra>" },
    { x: dates, y: dd, name: "Drawdown", mode: "lines", line: { color: col, width: 1.2 }, fill: "tozeroy", fillcolor: rgba(col, 0.22), yaxis: "y2", hovertemplate: "%{y:.1f}%<extra>Drawdown</extra>" },
  ], {
    margin: { l: 58, r: 12, t: 8, b: 26 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", showlegend: false,
    hovermode: "x unified", hoversubplots: "axis", hoverlabel: { bgcolor: css("--surface"), bordercolor: axis, font: { color: css("--ink"), size: 12 } },
    font: { family: 'system-ui,-apple-system,"Segoe UI",sans-serif', size: 11, color: ink2 },
    xaxis: { range: [x0, x1], anchor: "y2", hoverformat: "%b %-d, %Y", showgrid: false, linecolor: axis, tickcolor: axis, fixedrange: !CAN_ZOOM },
    yaxis: Object.assign({}, yax, { domain: [0.5, 1], range: [0, 100], title: Object.assign({ text: "Favorability" }, yax.title) }),
    yaxis2: Object.assign({}, yax, { domain: [0, 0.43], range: [lo * 1.08, 0], ticksuffix: "%", title: Object.assign({ text: "Drawdown" }, yax.title) }),
    dragmode: CAN_ZOOM ? "zoom" : false, shapes,
  }, { displayModeBar: false, responsive: true, doubleClick: "reset" });
}
const DASHES = ["solid", "dash", "dot"];
// One ETF: implied leverage at each volatility budget under one cap (top), drawdown beneath, shared time axis.
function plotLevDrawdown(id, a, cap) {
  const [x0, x1] = xRange(), ink2 = css("--ink2"), grid = css("--grid"), axis = css("--axis"), band = css("--band"), col = css(COLOR[a]);
  const dates = D.series.dates, dd = D.series[a].dd.map((v) => v == null ? null : v * 100);
  let lo = -1; dates.forEach((d, j) => { if (d >= x0 && d <= x1 && dd[j] != null) lo = Math.min(lo, dd[j]); });
  const shapes = [{ type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: 1, y1: 1, line: { color: axis, width: 1 }, layer: "below" }];
  const start = D.assets[a].proxy ? D.assets[a].etf_start : null;
  if (start && start > x0) shapes.push({ type: "rect", xref: "x", yref: "paper", x0: x0, x1: start, y0: 0, y1: 1, fillcolor: band, line: { width: 0 }, layer: "below" });
  const yax = { gridcolor: grid, zeroline: false, fixedrange: !CAN_ZOOM, title: { font: { size: 11, color: ink2 }, standoff: 6 } };
  const traces = C.budgets.map((b, i) => ({ x: dates, y: D.series[a]["lev_" + b].map((v) => v == null ? null : Math.min(cap, v)),
    name: b + "% budget", mode: "lines", line: { color: col, width: 1.6, dash: DASHES[i] }, yaxis: "y", hovertemplate: "%{y:.2f}x<extra>" + b + "% budget</extra>" }));
  traces.push({ x: dates, y: dd, name: "Drawdown", mode: "lines", line: { color: col, width: 1.2 }, fill: "tozeroy", fillcolor: rgba(col, 0.22), yaxis: "y2", hovertemplate: "%{y:.1f}%<extra>Drawdown</extra>" });
  Plotly.react(id, traces, {
    margin: { l: 58, r: 12, t: 8, b: 26 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", showlegend: false,
    hovermode: "x unified", hoversubplots: "axis", hoverlabel: { bgcolor: css("--surface"), bordercolor: axis, font: { color: css("--ink"), size: 12 } },
    font: { family: 'system-ui,-apple-system,"Segoe UI",sans-serif', size: 11, color: ink2 },
    xaxis: { range: [x0, x1], anchor: "y2", hoverformat: "%b %-d, %Y", showgrid: false, linecolor: axis, tickcolor: axis, fixedrange: !CAN_ZOOM },
    yaxis: Object.assign({}, yax, { domain: [0.5, 1], range: [0, cap * 1.08], ticksuffix: "x", title: Object.assign({ text: "Leverage" }, yax.title) }),
    yaxis2: Object.assign({}, yax, { domain: [0, 0.43], range: [lo * 1.08, 0], ticksuffix: "%", title: Object.assign({ text: "Drawdown" }, yax.title) }),
    dragmode: CAN_ZOOM ? "zoom" : false, shapes,
  }, { displayModeBar: false, responsive: true, doubleClick: "reset" });
}
const NO_PLOTLY = '<p class="small">Charts need the Plotly library, which could not be loaded. The tables are unaffected.</p>';
'''

JS_DOC = r'''
// ---- documentation (shared by both pages) ----------------------------------
function renderDoc(extra) {
const efa = D.assets.EFA, eem = D.assets.EEM;
const stale = (d) => d.stale_ac_backfill == null ? "" : " In this data the lag-1 autocorrelation of daily returns is " +
  num(d.stale_ac_backfill) + " in the backfilled era against " + num(d.stale_ac_etf) + " once the ETF trades.";
const members = (a) => Object.entries(C.breadth[a]).map(([t, w]) => t + " " + w + "%").join(", ");
const from = (a) => D.assets[a].score_from || "n/a";
$("doc").innerHTML = `
<h3>What this page is and is not</h3>
<ul>
<li>It shows how favorable conditions are for holding leverage in each ETF, relative to that ETF's own history. It does not say which ETF to hold (that is left to your momentum system), it does not predict returns, and it is not investment advice. Leveraged positions can lose most of their value quickly in any environment.</li>
<li>All leveraged figures are simulations of an idealized fund. Real funds differ through swap pricing, tracking error, taxes, premiums and discounts, and their own fee schedules.</li>
</ul>

<h3>The core measure: effective volatility</h3>
<ul>
<li>The variance of a return over a holding period of H days is H &times; daily variance &times; VR_H, where VR_H is the variance ratio at that horizon. In volatility terms: <strong>effective volatility = volatility measured from daily returns &times; &radic;VR_H</strong>. H is ${C.vr_q} trading days, matching sleeves rebalanced monthly. This is the risk a position held for a month actually carries; leverage multiplies it.</li>
<li>Volatility from daily returns is the root-mean-square of the ${C.vol[0]}-day and ${C.vol[1]}-day annualized readings, so it reacts within weeks. It is backward-looking realized volatility, not a forecast, and it can jump faster than any window registers.</li>
<li>The variance ratio is the variance of overlapping ${C.vr_q}-day returns divided by ${C.vr_q} times the variance of daily returns, using the Lo&ndash;MacKinlay bias-corrected, demeaned estimator over ${C.vr_window} trading days, so it moves slowly. Even so it is noisy: its standard error is about &plusmn;${num(C.vr_se)} when returns are independent, and larger when volatility clusters, which moves effective volatility by roughly &plusmn;${Math.round(C.vr_se * 50)}%. Volatility itself cancels out of the ratio; it measures only how daily moves add up.</li>
<li>Annualized effective volatility is monthly risk expressed on an annual scale (monthly volatility &times; &radic;12). It is not a statement about the variability of a full year's return.</li>
<li>No momentum or return forecast enters the measure. Volatility and serial correlation are each counted once.</li>
</ul>

<h3>Favorability and the five levels</h3>
<ul>
<li>Favorability = 100 minus the point-in-time percentile of effective volatility against the ETF's own history: 100 is the calmest the ETF has been, 0 the most turbulent. Each past day is ranked only against data that existed on that day, with at least ${C.min_rank} observations required, so favorability starts on ${from("SPY")} for SPY, ${from("EFA")} for EFA and ${from("EEM")} for EEM.</li>
<li>Five levels with equal shares of history: very favorable at ${C.cuts[3]} or above, favorable from ${C.cuts[2]}, neutral between ${C.cuts[1]} and ${C.cuts[2]}, unfavorable at ${C.cuts[1]} or below, very unfavorable at ${C.cuts[0]} or below. Because the levels are quintiles of the ETF's own history, they make no claim of equal risk and involve no fitting.</li>
<li>Because the measure is relative to each ETF's own history, the same level on two ETFs does not mean the same absolute risk; EEM at a favorable level can still be more volatile than SPY at an unfavorable one. Absolute risk is in the effective-volatility figure and the implied-leverage grid.</li>
<li>Red lines on the favorability charts at ${C.ref_lines.join(", ")} are visual reference levels only.</li>
</ul>

<h3>The variance ratio as trend persistence</h3>
<ul>
<li>Below 1, daily moves have tended to reverse within the month, so a month's swings are smaller than daily volatility implies. Above 1, they have tended to continue, so swings are larger. For these three ETFs it has been below 1 most of the time, reflecting the short-term reversals in index returns that Bessembinder (2026) discusses.</li>
<li>It enters favorability only through effective volatility, as a risk adjustment. Above 1 also means an existing trend is more likely to persist, which favors a trend-following position and a daily-reset fund (Bessembinder's rebalancing effect); that is a statement about returns, so it is shown but not scored either way.</li>
<li>The variance ratio is left blank wherever its window includes backfilled mutual-fund prices, whose stale pricing inflates serial correlation, so effective volatility and favorability start about two years after EFA and EEM began trading.</li>
</ul>

<h3>Leverage from a volatility budget</h3>
<ul>
<li>Implied leverage = budget &divide; effective volatility, then capped. Budgets are annualized on the same convention as effective volatility. Values below 1x mean holding part of the position in cash.</li>
<li>Not captured: the drag and skew inside a daily-reset 3x fund (Bessembinder's median effect), financing costs, and sudden gaps. Realized volatility is backward-looking and understates risk just before a shock; the cap is the protection there.</li>
</ul>

<h3>Data</h3>
<ul>
<li>Prices are daily closes from Yahoo Finance, adjusted for dividends and splits, so every return is a total return. Yahoo revises adjusted history whenever a dividend is paid, so past readings can shift slightly between runs. If the three core ETFs cannot be fetched or the data is more than a week old, the job fails instead of publishing.</li>
<li>The ETFs are SPY (${D.assets.SPY.index}), EFA (${efa.index}) and EEM (${eem.index}). Percentiles use one common window starting ${D.rank_start}.</li>
<li>The page is built overnight, at about 1am Pacific time, after each trading day. A reading dated ${D.asof} uses that day's close.</li>
</ul>

<h3>Backfilled history before the ETFs existed</h3>
<ul>
<li>EFA began trading on ${efa.etf_start}. ${efa.proxy ? "Earlier returns come from a fixed blend of two Vanguard index funds, " + esc(efa.proxy) + " (Europe and Pacific), fitted to EFA's monthly returns over the first five years of overlap" + (efa.blend_corr == null ? "" : " (correlation " + num(efa.blend_corr, 3) + ")") + "." : "No backfill was available in this run."}</li>
<li>EEM began trading on ${eem.etf_start}. ${eem.proxy ? "Earlier returns come from Vanguard's emerging markets index fund (" + esc(eem.proxy) + "), which then tracked a select MSCI emerging markets index, close to but not the same as EEM's." : "No backfill was available in this run."}</li>
<li><strong>Stale pricing.</strong> In that era the funds were priced from foreign closing prices set hours before the US close, which manufactures positive day-to-day correlation and understates daily volatility.${stale(efa)}${stale(eem)} Backfilled prices are used for volatility, momentum and drawdowns, but never for the variance ratio.</li>
</ul>

<h3>Percentiles and range bars</h3>
<ul>
<li>A percentile says how high a reading is, not whether that is good; the direction for leverage is stated separately. Range bars span the 1st to 99th percentile of history, so single crisis days do not squash everything else to one side.</li>
</ul>

<h3>Drawdown charts</h3>
<ul>
<li>Drawdown is measured on total-return prices from the highest prior close in the data, plotted as the worst close of each week.</li>
<li>Favorability falls as a drawdown deepens largely because volatility rises with the decline itself. That is measurement, not early warning: what matters is where favorability stood before a decline began.</li>
</ul>

<h3>Momentum and breadth (detail report only)</h3>
<ul>
<li>Shown for context and not part of favorability. Momentum is total return over ${C.mom[0]} and ${C.mom[1]} trading days. Breadth is the share of sector ETFs (SPY) or single-country ETFs (EFA, EEM) with a positive return over the same windows; these are coarse proxies, not the underlying stocks, and the index-weighted figure uses approximate fixed weights set by hand. Current sets: SPY: ${members("SPY")}. EFA: ${members("EFA")}. EEM: ${members("EEM")}.</li>
</ul>

<h3>Leverage simulation and point-in-time check</h3>
<ul>
<li>Each day the simulated fund earns L times the ETF's total return. After costs it also pays the cash rate plus ${(C.spread * 100).toFixed(2)}% on the borrowed (L-1) portion and a ${(C.fee * 100).toFixed(2)}% annual fee; the cash rate is the ${esc(D.cash_note)}, currently ${pct(D.cash_now, 2)}. Not modelled: intraday moves, tracking error, widening swap spreads, fund closures, and the fact that a 3x fund is wiped out by a single-day fall of about 33%.</li>
<li>The check uses only dates when favorability exists, with levels as they stood on each date, and looks ${C.horizon} trading days ahead. Adjacent days share almost all of their forward window, so the independent-period count is the honest sample size. The sample contains only a handful of major bear markets, and those few episodes dominate the unfavorable rows. Past simulated results do not indicate future results.</li>
</ul>` + (extra || "");
$("foot").textContent = "Generated " + D.generated + " · data through " + D.asof + " · source: Yahoo Finance via yfinance";
}
'''

DASH_BODY = r'''<div id="banner"></div>
<h1>Leverage Environment</h1>
<p class="sub">How favorable current conditions are for holding leverage in SPY, EFA and EEM, relative to each ETF's own history.
The core measure is effective volatility: the risk a position held for a month actually carries, combining daily volatility with how
daily moves add up over the month. Which ETF to hold is left to your momentum system. <span id="asof"></span></p>
<p class="small">This is a monitoring tool built on approximations, not investment advice. Read
<a href="#method">Methodology and caveats</a> before relying on any number here.
Momentum, breadth and every result in full are in the <a href="report.html">detail report</a>.</p>
<div id="notes"></div>

<section class="cards" id="cards"></section>

<h2>Favorability against drawdowns</h2>
<p class="sub">Favorability, 0 to 100 (top), above each ETF's drawdown from its highest prior close (bottom), on the same time axis.
100 is the calmest the ETF has been; red lines mark 20, 50 and 80.</p>
<div class="legend" id="legend"></div>
<p class="small" id="shade-note"></p>
<div id="ddcharts"></div>

<h2 id="vb">Leverage from a volatility budget</h2>
<p class="sub">The leverage that would hold a position's risk at a chosen budget:
<strong>implied leverage = budget &divide; effective volatility</strong>, where
<strong>effective volatility = volatility &times; &radic;VR&#8322;&#8321;</strong>, then capped. Values below 1x mean holding part of the position in cash.</p>
<div class="scroll"><table id="vbin"></table></div>
<div class="scroll"><table id="vbgrid"></table></div>
<div class="ctl"><span>Leverage cap on the charts</span><div class="seg" id="capbtns" role="group" aria-label="Leverage cap"></div><span id="vbrange" style="margin-left:auto"></span></div>
<p class="small" id="vbnote"></p>
<div id="vbcharts"></div>

<h2>Readings against each ETF's own history</h2>
<p class="sub">Each cell shows today's value, its percentile, and a bar spanning the 1st to 99th percentile of history:
the shaded block is the middle half, the tick is the median, the solid dot is today and the hollow dot is one month ago.</p>
<div class="scroll"><table id="grid"></table></div>

<h2>History</h2>
<div class="charts" id="charts"></div>

<h2>Does favorability line up with later outcomes? A point-in-time check</h2>
<p class="sub" id="bt-intro"></p>
<div class="scroll"><table id="bt"></table></div>

<h2 id="method">Methodology and caveats</h2>
<div class="doc" id="doc"></div>
<footer id="foot"></footer>
'''

DASH_JS = r'''// ---- cards ---------------------------------------------------------------
const xf = (v) => v == null ? "n/a" : v.toFixed(2) + "x";
const vrNote = (v) => v == null ? "" : v < 1 ? "moves have tended to reverse within the month" : "moves have tended to continue within the month";
$("cards").innerHTML = A.map((a) => {
  const d = D.assets[a], c = d.cells, v = d.vb;
  const row = (k, val) => "<dt>" + k + "</dt><dd>" + val + "</dd>";
  return '<article class="card"><div class="tk"><i class="sw" style="background:var(' + COLOR[a] + ')"></i>' + a + "</div>" +
    '<div class="nm">' + d.name + " · " + d.index + "</div>" +
    '<div class="verdict"><span class="lab"><span class="ico ' + d.label + '">' + (ICON[d.label] || "") + "</span>" + lname(d.label) + "</span>" +
    '<span class="sc">favorability ' + (d.score == null ? "n/a" : d.score.toFixed(0)) + " of 100" +
    (d.score_prev == null ? "" : " (" + d.score_prev.toFixed(0) + " a month ago)") + "</span></div>" +
    "<dl>" +
    '<dt class="h">Effective volatility (annualized)</dt>' +
    row("Now", "<strong>" + pct(v.eff_vol) + "</strong> · " + pctText("eff_vol", c.eff_vol.pct)) +
    row("A month ago", pct(v.eff_vol_prev)) +
    '<dt class="h">Volatility from daily returns</dt>' +
    row(C.vol[0] + "-day / " + C.vol[1] + "-day", pct(v.vol_21) + " / " + pct(v.vol_63)) +
    row("Blended", pct(v.base_vol) + " · " + pctText("base_vol", c.base_vol.pct)) +
    '<dt class="h">Trend persistence</dt>' +
    row("Variance ratio, " + C.vr_q + "-day", num(v.vr21) + " ± " + num(C.vr_se)) +
    '<dd style="grid-column:1/-1;text-align:left;color:var(--ink2);font-size:12.5px">' + vrNote(v.vr21) + "</dd>" +
    '<dt class="h">Implied leverage, uncapped</dt>' +
    row(C.budgets.map((b) => b + "%").join(" / ") + " budget", C.budgets.map((b) => xf(v.lev[b])).join(" / ")) +
    '<dt class="h">Position</dt>' +
    row("Drawdown from high", pct(d.dd)) +
    "</dl></article>";
}).join("");

// ---- grid with range bars -------------------------------------------------
function rangeBar(m, c) {
  if (!c.q || c.value == null) return "";
  const lo = c.q[0], hi = c.q[4], x = (v) => 4 + 192 * Math.min(1, Math.max(0, (v - lo) / ((hi - lo) || 1)));
  const tip = "1st pct " + fmtv(m, lo) + ", median " + fmtv(m, c.q[2]) + ", 99th pct " + fmtv(m, hi) +
    "; one month ago " + fmtv(m, c.prev) + "; history since " + c.since;
  return '<svg class="rb" viewBox="0 0 200 16" preserveAspectRatio="none" role="img" aria-label="' + tip + '"><title>' + tip + "</title>" +
    '<line x1="4" x2="196" y1="8" y2="8" stroke="var(--axis)" stroke-width="2" vector-effect="non-scaling-stroke"/>' +
    '<rect x="' + x(c.q[1]) + '" y="4" width="' + Math.max(1, x(c.q[3]) - x(c.q[1])) + '" height="8" rx="2" fill="var(--axis)"/>' +
    '<line x1="' + x(c.q[2]) + '" x2="' + x(c.q[2]) + '" y1="2" y2="14" stroke="var(--muted)" stroke-width="1.5" vector-effect="non-scaling-stroke"/>' +
    (c.prev == null ? "" : '<ellipse cx="' + x(c.prev) + '" cy="8" rx="2.6" ry="4" fill="var(--surface)" stroke="var(--ink2)" stroke-width="1.2" vector-effect="non-scaling-stroke"/>') +
    '<ellipse cx="' + x(c.value) + '" cy="8" rx="2.9" ry="4.6" fill="var(--ink)" stroke="var(--surface)" stroke-width="1.5" vector-effect="non-scaling-stroke"/></svg>';
}
const HINT = {
  eff_vol: "the core measure; lower is better for leverage",
  vr21: "not scored: below 1 lowers monthly risk; above 1 raises it, but means trends persist",
};
$("grid").innerHTML = "<thead><tr><th>Metric</th>" + A.map((a) => "<th>" + a + "</th>").join("") + "</tr></thead><tbody>" +
  D.metrics.filter((m) => m.core).map((m) =>
    "<tr><td>" + m.label + '<div class="small">' + (HINT[m.key] || (m.higher_better ? "higher is better for leverage" : "lower is better for leverage")) + "</div></td>" +
    A.map((a) => {
      const c = D.assets[a].cells[m.key];
      if (c.value == null) return '<td class="cell"><span class="p">n/a</span></td>';
      return '<td class="cell"><span class="v">' + fmtv(m, c.value) + '</span><span class="p">' + pctText(m.key, c.pct) + "</span>" +
        (c.fav == null ? "" : '<div class="f"><i class="dot" style="background:' + favColor(c.fav) + '"></i>' + favWord(c.fav) + "</div>") +
        rangeBar(m, c) + "</td>";
    }).join("") + "</tr>").join("") + "</tbody>";

// ---- point-in-time check --------------------------------------------------
$("bt-intro").textContent = "For every past day, the level was computed using only data available on that day, then compared with what a simulated " +
  "daily-reset fund did over the following " + C.horizon + " trading days. If favorability is useful, risk should fall step by step from very unfavorable to very favorable: " +
  "watch the worst-5% and compounding columns more than the medians. Windows overlap heavily, so the last column gives the rough count of independent periods.";
$("bt").innerHTML = '<thead><tr><th>ETF</th><th>Level</th><th class="n">Share of days</th><th class="n">Median 1x</th>' +
  C.leverage.map((L) => '<th class="n">Median ' + L + "x after costs</th>").join("") +
  '<th class="n">Median ' + TOP + 'x compounding effect</th><th class="n">' + TOP + 'x positive</th><th class="n">' + TOP + 'x worst 5%</th><th class="n">Independent periods</th></tr></thead><tbody>' +
  A.map((a) => D.assets[a].backtest.map((r, i) => "<tr>" +
    (i ? "" : '<td class="nw" rowspan="' + LEVELS.length + '"><strong>' + a + '</strong><div class="small">from ' + (D.assets[a].backtest_from || "n/a") + "</div></td>") +
    '<td class="nw"><span class="ico ' + r.label + '">' + ICON[r.label] + "</span> " + lname(r.label) + '</td><td class="n">' + pct(r.share, 0) + '</td><td class="n">' + spct(r.fwd_1x) + "</td>" +
    C.leverage.map((L) => '<td class="n">' + spct(r["net_" + L + "x"]) + "</td>").join("") +
    '<td class="n">' + spct(r["gap_" + TOP + "x"]) + '</td><td class="n">' + pct(r.win, 0) + '</td><td class="n">' + spct(r.p5) + '</td><td class="n">' + r.windows + "</td></tr>").join("")).join("") + "</tbody>";

// ---- history charts -------------------------------------------------------
const CHARTS = [
  { key: "score", title: "Favorability", note: "100 minus the effective-volatility percentile; red lines at " + C.ref_lines.join(", "), red: C.ref_lines },
  { key: "eff_vol", title: "Effective volatility", note: "Blended volatility × √VR" + C.vr_q + ", annualized", pct: true },
  { key: "vr21", title: "Variance ratio, " + C.vr_q + "-day", note: "Bias-corrected, " + C.vr_window + "-day window; red line at 1 (no serial correlation); below 1 = reversals within the month", red: [1] },
  { key: "base_vol", title: "Volatility, blended", note: "Root-mean-square of the " + C.vol[0] + "- and " + C.vol[1] + "-day readings", pct: true },
  { key: "vol_21", title: "Volatility, " + C.vol[0] + "-day", note: "Annualized standard deviation of daily returns", pct: true },
  { key: "vol_63", title: "Volatility, " + C.vol[1] + "-day", note: "Annualized standard deviation of daily returns", pct: true },
];
$("legend").innerHTML = legendHtml();
$("shade-note").textContent = shadeNote();
$("charts").innerHTML = CHARTS.map((c, i) => '<div class="chart"><h3>' + c.title + "</h3><p>" + c.note + '</p><div class="plot" id="pl' + i + '"></div></div>').join("");

// ---- volatility-budget leverage ------------------------------------------
$("vbin").innerHTML = '<thead><tr><th>ETF</th><th class="n">21-day volatility</th><th class="n">63-day volatility</th><th class="n">Blended volatility</th>' +
  '<th class="n">Variance ratio, 21-day</th><th class="n">Effective volatility</th><th class="n">Effective volatility a month ago</th></tr></thead><tbody>' +
  A.map((a) => { const v = D.assets[a].vb; return "<tr><td><strong>" + a + '</strong></td><td class="n">' + pct(v.vol_21) + '</td><td class="n">' + pct(v.vol_63) +
    '</td><td class="n">' + pct(v.base_vol) + '</td><td class="n">' + num(v.vr21) + " ± " + num(C.vr_se) + '</td><td class="n"><strong>' + pct(v.eff_vol) +
    '</strong></td><td class="n">' + pct(v.eff_vol_prev) + "</td></tr>"; }).join("") + "</tbody>";
$("vbgrid").innerHTML = '<thead><tr><th>ETF</th><th>Volatility budget</th><th class="n">Uncapped</th>' + C.caps.map((c) => '<th class="n">Cap ' + c + "x</th>").join("") +
  '<th class="n">Uncapped a month ago</th></tr></thead><tbody>' +
  A.flatMap((a) => C.budgets.map((b, i) => { const L = D.assets[a].vb.lev[b], P = D.assets[a].vb.lev_prev[b];
    return "<tr>" + (i ? "<td></td>" : "<td><strong>" + a + "</strong></td>") + "<td>" + b + '%</td><td class="n">' + xf(L) + "</td>" +
      C.caps.map((c) => L == null ? '<td class="n">n/a</td>' : (L >= c ? '<td class="n cap">' + xf(c) + " (cap)</td>" : '<td class="n"><strong>' + xf(L) + "</strong></td>")).join("") +
      '<td class="n">' + xf(P) + "</td></tr>"; })).join("") + "</tbody>";
let levCap = C.caps[C.caps.length - 1];
$("capbtns").innerHTML = C.caps.map((c) => '<button data-cap="' + c + '" aria-pressed="' + (c === levCap) + '">' + c + "x</button>").join("");
$("vbrange").innerHTML = rangeBtns();
$("vbnote").textContent = "Lines: solid " + C.budgets[0] + "%, dashed " + C.budgets[1] + "%, dotted " + C.budgets[2] + "% volatility budget; the grey line marks 1x. " +
  "The variance ratio needs " + C.vr_window + " trading days of the ETF's own prices, so the lines start later for the backfilled ETFs. Weekly samples.";
$("vbcharts").innerHTML = A.map((a) => '<div class="chart"><h3><i class="sw" style="background:var(' + COLOR[a] + ');margin-right:6px"></i>' + a +
  ': implied leverage and drawdown</h3><p>Implied leverage at each budget under the selected cap (top); decline from the highest prior close (bottom)</p><div class="plot tall" id="vb' + a + '"></div></div>').join("");
document.querySelectorAll("#capbtns button").forEach((btn) => btn.addEventListener("click", () => {
  levCap = +btn.dataset.cap;
  document.querySelectorAll("#capbtns button").forEach((o) => o.setAttribute("aria-pressed", String(o === btn)));
  if (window.Plotly) A.forEach((a) => plotLevDrawdown("vb" + a, a, levCap));
}));
$("ddcharts").innerHTML = A.map((a) => '<div class="chart"><h3><i class="sw" style="background:var(' + COLOR[a] + ');margin-right:6px"></i>' + a +
  ': favorability and drawdown</h3><p>Favorability, 0 to 100 (top); decline from the highest prior close, worst close each week (bottom)</p><div class="plot tall" id="dd' + a + '"></div></div>').join("");
function draw() {
  if (!window.Plotly) { $("charts").innerHTML = NO_PLOTLY; $("ddcharts").innerHTML = ""; $("vbcharts").innerHTML = ""; return; }
  A.forEach((a) => plotScoreDrawdown("dd" + a, a));
  A.forEach((a) => plotLevDrawdown("vb" + a, a, levCap));
  CHARTS.forEach((c, i) => plotLines("pl" + i, c, (a) => D.series[a][c.key]));
}
bindRange(draw);
draw();
renderDoc("");
'''

REPORT_BODY = r'''<div id="banner"></div>
<h1>Leverage Environment: full detail</h1>
<p class="sub">Every result behind the <a href="index.html">dashboard</a>: all readings with their history, momentum and breadth
for context, the leverage simulation over several lookbacks, and how the favorability levels and each metric have lined up
with later leveraged returns. <span id="asof"></span></p>
<p class="small">This is a monitoring tool built on approximations, not investment advice. Read
<a href="#method">Methodology and caveats</a> before relying on any number here.</p>
<div id="notes"></div>
<nav class="toc" aria-label="Sections"><a href="#s-sum">Summary</a><a href="#s-full">All readings</a><a href="#s-ex">History explorer</a>
<a href="#s-comps">Breadth by component</a><a href="#s-lev">Leverage simulation</a><a href="#s-bth">Levels by horizon</a>
<a href="#s-btb">Favorability bands</a><a href="#s-btm">Each metric alone</a><a href="#s-chg">Level changes</a><a href="#s-cov">Data coverage</a>
<a href="#method">Methodology and caveats</a></nav>

<h2 id="s-sum">Summary</h2>
<p class="sub">Favorability is 100 minus the percentile of effective volatility against the ETF's own history. Momentum and breadth are shown for context only and do not enter it.</p>
<div class="scroll"><table id="sum"></table></div>

<h2 id="s-full">All readings</h2>
<p class="sub">Today's value with its percentile, where it stood one month, three months and one year ago, and the
spread of its own history. The first five rows are the core measure and its inputs; momentum and breadth below them are context.
Twelve-month momentum is reported as a quintile because its history holds few independent observations.</p>
<div id="full"></div>

<h2 id="s-ex">History explorer</h2>
<p class="sub">Any series over time. For the readings in the table above you can switch between the raw value and its point-in-time percentile.</p>
<div class="ctl"><label for="sel">Series</label><select id="sel"></select>
<div class="seg" role="group" aria-label="Show"><button id="bv" aria-pressed="true">Value</button><button id="bp" aria-pressed="false">Percentile</button></div></div>
<div class="legend" id="legend"></div>
<p class="small" id="shade-note"></p>
<div class="chart"><div class="plot tall" id="ex"></div></div>

<h2 id="s-comps">Breadth by component</h2>
<p class="sub">The individual sector and country ETFs behind each breadth reading. Weights are approximate and fixed (see caveats).</p>
<div id="comps"></div>

<h2 id="s-lev">Leverage simulation across lookbacks</h2>
<p class="sub">Simulated daily-reset funds over several trailing periods, as cumulative returns. "Simple multiple" is the
unlevered return times the leverage; the gap between it and the simulated figure is the compounding effect.</p>
<div class="scroll"><table id="levh"></table></div>

<h2 id="s-bth">Favorability levels against later returns, by horizon</h2>
<p class="sub" id="bth-intro"></p>
<div class="scroll"><table id="bth"></table></div>

<h2 id="s-btb">Favorability bands against later returns</h2>
<p class="sub" id="btb-intro"></p>
<div class="scroll"><table id="btb"></table></div>

<h2 id="s-btm">Each metric by itself</h2>
<p class="sub" id="btm-intro"></p>
<div class="scroll"><table id="btm"></table></div>

<h2 id="s-chg">Level changes in the past year</h2>
<div id="chg"></div>

<h2 id="s-cov">Data coverage</h2>
<p class="sub">Every series fetched for this run and how far back it goes. Check here that the backfill funds and breadth components reach as far as expected.</p>
<div class="scroll"><table id="cov"></table></div>

<h2 id="method">Methodology and caveats</h2>
<div class="doc" id="doc"></div>
<footer id="foot"></footer>
'''

REPORT_JS = r'''
// ---- detail report ----------------------------------------------------------
const R = D.report, M = D.metrics;
const lab = (l) => l ? '<span class="ico ' + l + '">' + ICON[l] + "</span> " + lname(l) : "n/a";
// heads: [text, numeric?]; rows: arrays of cell html
function tbl(id, heads, rows, el) {
  const html = "<thead><tr>" + heads.map((h) => '<th class="w' + (h[1] ? " n" : "") + '">' + h[0] + "</th>").join("") + "</tr></thead><tbody>" +
    rows.map((r) => "<tr>" + r.map((c, i) => "<td" + (heads[i][1] ? ' class="n"' : ' class="nw"') + ">" + (c == null ? "" : c) + "</td>").join("") + "</tr>").join("") + "</tbody>";
  if (el) return '<div class="scroll"><table>' + html + "</table></div>";
  $(id).innerHTML = html;
}
const etfHead = (a) => '<h3 class="etf"><i class="sw" style="background:var(' + COLOR[a] + ')"></i>' + a + ' <span class="small">' + D.assets[a].name + " · " + D.assets[a].index + "</span></h3>";
const first = (a, i) => i ? "" : "<strong>" + a + "</strong>";
const f0 = (x) => x == null ? "n/a" : x.toFixed(0);

tbl("sum", [["ETF"], ["Level"], ["Favorability", 1], ["Effective volatility", 1], ["Variance ratio, 21-day", 1], ["Trading days at this level", 1],
  ["3-month momentum", 1], ["12-month momentum", 1], ["Breadth: up over 3 months", 1], ["Breadth: up over 12 months", 1]],
  A.map((a) => { const d = D.assets[a], c = d.cells; return ["<strong>" + a + "</strong>", lab(d.label), f0(d.score), pct(c.eff_vol.value), num(c.vr21.value),
    R.assets[a].streak, spct(c.mom_3m.value), spct(c.mom_12m.value), pct(c.b3.value, 0), pct(c.b12.value, 0)]; }));

$("full").innerHTML = A.map((a) => etfHead(a) + tbl(null,
  [["Metric"], ["Today", 1], ["Percentile"], ["For leverage"], ["1 month ago", 1], ["3 months ago", 1], ["1 year ago", 1],
   ["1st pct", 1], ["25th", 1], ["Median", 1], ["75th", 1], ["99th pct", 1], ["Ranked since"], ["Observations", 1]],
  M.map((m) => { const c = D.assets[a].cells[m.key], lb = R.assets[a].lookback[m.key], s = m.key.startsWith("mom"), q = c.q || [];
    return [m.label, fmtv(m, c.value, s), pctText(m.key, c.pct), !m.core ? "context only" : c.fav == null ? "not scored" : '<i class="dot" style="background:' + favColor(c.fav) + '"></i> ' + favWord(c.fav),
      fmtv(m, lb[0], s), fmtv(m, lb[1], s), fmtv(m, lb[2], s), fmtv(m, q[0], s), fmtv(m, q[1], s), fmtv(m, q[2], s), fmtv(m, q[3], s), fmtv(m, q[4], s),
      c.since || "n/a", c.n.toLocaleString()]; }), true)).join("");

// explorer
const OPTS = [{ key: "score", label: "Favorability", red: C.ref_lines }]
  .concat(M.map((m) => ({ key: m.key, label: m.label, pct: m.fmt === "pct", hasP: true,
    zero: m.key.startsWith("mom") ? 0 : null, red: m.key === "vr21" ? [1] : null })))
  .concat([
    { key: "b12w", label: "Breadth: up over 12 months, index weight", pct: true },
    { key: "b3w", label: "Breadth: up over 3 months, index weight", pct: true },
    { key: "und_1y", label: "Unlevered return, trailing year", pct: true, zero: 0 },
    { key: "dd", label: "Drawdown from prior high (worst close each week)", pct: true, zero: 0 },
  ], C.budgets.map((b) => ({ key: "lev_" + b, label: "Implied leverage at a " + b + "% volatility budget (uncapped)", zero: 1 })), C.leverage.flatMap((L) => [
    { key: "gap_" + L + "x", label: L + "x compounding effect, trailing year (before costs)", pct: true, zero: 0 },
    { key: "net_" + L + "x", label: L + "x simulated return after costs, trailing year", pct: true, zero: 0 },
  ]));
$("sel").innerHTML = OPTS.map((o, i) => '<option value="' + i + '">' + o.label + "</option>").join("");
$("sel").value = String(OPTS.findIndex((o) => o.key === "score"));
$("legend").innerHTML = legendHtml();
$("shade-note").textContent = shadeNote();
let showP = false;
function draw() {
  if (!window.Plotly) { $("ex").outerHTML = NO_PLOTLY; return; }
  const o = OPTS[+$("sel").value], p = showP && o.hasP;
  $("bp").disabled = !o.hasP; $("bv").setAttribute("aria-pressed", String(!p)); $("bp").setAttribute("aria-pressed", String(p));
  plotLines("ex", p ? { zero: 50 } : o, (a) => D.series[a][(p ? "p_" : "") + o.key]);
}
$("sel").addEventListener("change", draw);
$("bv").addEventListener("click", () => { showP = false; draw(); });
$("bp").addEventListener("click", () => { showP = true; draw(); });
bindRange(draw);
draw();

$("comps").innerHTML = A.map((a) => { const d = D.assets[a], cs = R.assets[a].components;
  return etfHead(a) + '<p class="small">' + cs.filter((c) => c.m12 > 0).length + " of " + cs.filter((c) => c.m12 != null).length + " up over 12 months; " +
    cs.filter((c) => c.m3 > 0).length + " of " + cs.filter((c) => c.m3 != null).length + " up over 3 months.</p>" +
    tbl(null, [["Component"], ["Ticker"], ["Approx. weight", 1], ["3-month return", 1], ["12-month return", 1], ["Data since"]],
      cs.map((c) => [esc(c.name), c.ticker, c.weight + "%", c.since ? spct(c.m3) : "no data", c.since ? spct(c.m12) : "no data", c.since || "n/a"]), true); }).join("");

tbl("levh", [["ETF"], ["Lookback"], ["Unlevered", 1]].concat(C.leverage.flatMap((L) => [[L + "x simple multiple", 1], [L + "x simulated, before costs", 1], [L + "x simulated, after costs", 1]])),
  A.flatMap((a) => R.assets[a].lev.map((r, i) => [first(a, i), r.label, spct(r.und)].concat(
    C.leverage.flatMap((L) => [spct(r["naive_" + L]), spct(r["gross_" + L]), spct(r["net_" + L])])))));

const HS = R.horizons;
$("bth-intro").textContent = "For every past day with a favorability reading, the level as it stood that day against the simulated " + TOP +
  "x fund over the following " + HS.join(", ") + " trading days. Each horizon shows the median return after costs and the median compounding effect " +
  "(simulated return before costs minus " + TOP + " times the unlevered return). If favorability is useful, risk should fall step by step from very unfavorable to very favorable at every horizon.";
tbl("bth", [["ETF"], ["Level"], ["Days", 1]].concat(HS.flatMap((h) => [[h + " days: " + TOP + "x after costs", 1], [h + " days: effect", 1]])),
  A.flatMap((a) => LEVELS.map((l, i) => [first(a, i), lab(l), (R.assets[a].by_horizon[String(C.horizon)][l].days || 0).toLocaleString()].concat(
    HS.flatMap((h) => { const r = R.assets[a].by_horizon[String(h)][l]; return [spct(r.net), spct(r.gap)]; })))));

$("btb-intro").textContent = "The same check by favorability band (deciles) over the following " + C.horizon + " trading days, to show whether outcomes improve steadily with favorability " +
  "or only at the extremes. Independent periods is the number of days divided by the horizon: the honest sample size.";
tbl("btb", [["ETF"], ["Favorability"], ["Share of days", 1], ["Median 1x", 1], ["Median " + TOP + "x after costs", 1], ["Median " + TOP + "x compounding effect", 1],
  [TOP + "x positive", 1], [TOP + "x worst 5%", 1], ["Independent periods", 1]],
  A.flatMap((a) => R.assets[a].bands.map((r, i) => [first(a, i), r.band, pct(r.share, 0), spct(r.fwd_1x), spct(r.net), spct(r.gap), pct(r.win, 0), spct(r.p5), Math.floor(r.days / C.horizon)])));

$("btm-intro").textContent = "Each metric taken alone, including the context metrics: days are split into thirds by that metric's percentile as it stood at the time (oriented so the favorable third is the one better for leverage), and compared on the following " +
  C.horizon + " trading days. The spread is the favorable third minus the unfavorable third; a metric that carries information should show a positive spread on all three ETFs.";
tbl("btm", [["ETF"], ["Metric"], [TOP + "x after costs: favorable third", 1], ["Middle third", 1], ["Unfavorable third", 1], ["Spread", 1],
  ["Compounding effect: favorable third", 1], ["Middle third", 1], ["Unfavorable third", 1], ["Spread", 1]],
  A.flatMap((a) => R.assets[a].alone.map((r, i) => { const d = (k) => r.fav[k] == null || r.unfav[k] == null ? null : r.fav[k] - r.unfav[k];
    return [first(a, i), r.label, spct(r.fav.net), spct(r.mid.net), spct(r.unfav.net), spct(d("net")), spct(r.fav.gap), spct(r.mid.gap), spct(r.unfav.gap), spct(d("gap"))]; })));

$("chg").innerHTML = A.map((a) => { const r = R.assets[a], d = D.assets[a];
  return etfHead(a) + '<p class="small">' + lname(d.label) + " for the last " + r.streak + " trading days.</p>" +
    (r.changes.length ? tbl(null, [["Date"], ["From"], ["To"]], r.changes.map((c) => [c.date, lab(c.from), lab(c.to)]), true) : '<p class="small">No level changes in the past year.</p>'); }).join("");

tbl("cov", [["Ticker"], ["Used for"], ["First date"], ["Last date"], ["Trading days", 1]],
  R.coverage.map((c) => ["<strong>" + c.ticker + "</strong>", esc(c.role), c.first || "no data", c.last || "no data", c.n.toLocaleString()]));

renderDoc(`
<h3>Reading the detail report</h3>
<ul>
<li>The tables that compare a level, a favorability band or a single metric with later returns slice one history many ways. With this many slices, some will look meaningful by chance. Give weight only to patterns that hold across all three ETFs and across horizons.</li>
<li>Longer horizons have fewer independent periods: divide the day count by the horizon. A ${HS[HS.length - 1]}-day result over a 20-year history rests on roughly 20 independent periods split across the levels.</li>
<li>Forward-return tables for favorability use only dates when it exists, which for EFA and EEM is well after each ETF began trading. The single-metric table uses dates after each ETF began trading; volatility percentiles there still rank against the full history, including the backfilled era.</li>
<li>In the multi-year leverage table, the simple multiple is a reference point, not something obtainable: over long periods compounding makes a daily-reset fund's return diverge widely from it in either direction.</li>
<li>Data requests start at ${R.fetch_start}, so a first date equal to the first trading day of that year means the series is older than shown.</li>
<li>History charts and the explorer use weekly samples (the last trading day of each week), so brief spikes within a week may not appear. Tables use daily data.</li>
</ul>`);
'''

if __name__ == "__main__":
    main()
