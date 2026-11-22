#!/usr/bin/env python3
"""
Leverage-environment dashboard: equities (SPY, EFA, EEM) on one page, and gold (GLD),
Treasuries (IEF, TLT) and bitcoin (BTC-USD) on a second.

For each asset it measures effective volatility (volatility from daily returns adjusted by
the variance ratio to a one-month holding period), ranks it against the asset's own
history to give a favorability score, and shows the leverage a volatility budget would
support, volatility drag, and how simulated daily-reset 2x / 3x exposure has compounded.
Each group gets a dashboard page and a fuller detail report; the methodology and every
known caveat are written into the pages.

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
CALENDAR_DAYS = 365            # assets that trade every day (bitcoin) count calendar days

# All windows below are in TRADING days. Assets on the calendar-day clock convert them with
# to_days(): 21 -> 30, 63 -> 91, 252 -> 365, 504 -> 730.
MOM_SHORT, MOM_LONG = 63, 252  # 3-month and 12-month momentum (breadth uses the same)
MONTH = 21                     # one month, for the 'one month ago' markers
VOL_SHORT, VOL_LONG = 21, 63   # realized-volatility windows
MIN_RANK_OBS = 504             # observations required before a percentile is reported

LEVERAGE = (2, 3)

# Leverage from a volatility budget: effective volatility over the monthly holding
# horizon = volatility from daily returns x sqrt(variance ratio at that horizon), and
# implied leverage = budget / effective volatility, capped.
VR_Q = 21                      # holding horizon (sleeves rebalanced monthly)
VR_Q_LONG = 63                 # longer horizon, shown for context only (not scored)
VR_WINDOW = 504                # estimation window for both variance ratios (about 2 years)
VOL_BUDGETS = (0.15, 0.30, 0.45)
LEV_CAPS = (2.0, 2.5, 3.0)
FINANCING_SPREAD = 0.005       # added to the T-bill rate on the borrowed portion
EXPENSE_RATIO = 0.0095         # typical leveraged-ETF annual fee
FALLBACK_CASH_RATE = 0.04      # used only if the T-bill series cannot be fetched
CASH_TICKER = "^IRX"           # 13-week T-bill yield, in percent

BREADTH_MIN_MEMBERS = 5        # components needed before breadth is reported

# Favorability = 100 minus the point-in-time percentile of effective volatility against the
# asset's own history: 100 is the calmest it has been, 0 the most turbulent. Five levels with
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

# proxy_kind: "blend" = two funds, weight fitted; "single" = one fund as is;
# "scaled" = one fund with its returns scaled to match the ETF (duration differences).
# stale: the proxy was priced from foreign closes hours before the US close, so its
# serial correlation is unusable and the variance ratio is left blank over that era.
# Breadth components carry APPROXIMATE, STATIC index weights (percent): rough present-day
# figures applied to all of history; update them from the fund providers' fact sheets.
ASSETS = {
    "SPY": {
        "name": "US large cap", "index": "S&P 500", "group": "equity", "calendar": "trading",
        "proxy": None, "proxy_kind": None, "stale": False,
        "breadth_kind": "sector ETFs",
        "breadth": {"XLK": 33, "XLF": 13.5, "XLY": 10.5, "XLC": 10, "XLV": 9.5,
                    "XLI": 8.5, "XLP": 5.5, "XLE": 3, "XLU": 2.5, "XLRE": 2, "XLB": 2},
    },
    "EFA": {
        "name": "Developed ex-US", "index": "MSCI EAFE", "group": "equity", "calendar": "trading",
        "proxy": ("VEURX", "VPACX"), "proxy_kind": "blend", "stale": True,   # Europe + Pacific
        "breadth_kind": "country ETFs",
        "breadth": {"EWJ": 22, "EWU": 15, "EWQ": 11, "EWG": 10, "EWL": 9.5, "EWA": 7,
                    "EWN": 5, "EWD": 3.5, "EWP": 3.5, "EWI": 3, "EDEN": 2.5, "EWH": 2,
                    "EWS": 1.7, "EFNL": 1, "EWK": 1, "EIS": 1, "ENOR": 0.7,
                    "EIRL": 0.3, "EWO": 0.2, "ENZL": 0.2},
    },
    "EEM": {
        "name": "Emerging markets", "index": "MSCI Emerging Markets", "group": "equity", "calendar": "trading",
        "proxy": ("VEIEX",), "proxy_kind": "single", "stale": True,
        "breadth_kind": "country ETFs",
        "breadth": {"FXI": 28, "EWT": 19, "INDA": 17, "EWY": 12, "EWZ": 4.5, "KSA": 3.5,
                    "EZA": 3.5, "EWW": 2, "UAE": 1.5, "EWM": 1.3, "EIDO": 1.2,
                    "EPOL": 1, "THD": 1, "KWT": 0.7, "QAT": 0.7, "TUR": 0.5,
                    "GREK": 0.5, "ECH": 0.5, "EPHE": 0.4, "EPU": 0.3},
    },
    "GLD": {
        "name": "Gold", "index": "Gold bullion (SPDR Gold Shares)", "group": "other", "calendar": "trading",
        "proxy": None, "proxy_kind": None, "stale": False, "breadth_kind": None, "breadth": {},
    },
    "IEF": {
        "name": "Intermediate Treasuries", "index": "US Treasuries, 7-10 years", "group": "other", "calendar": "trading",
        "proxy": ("VFITX",), "proxy_kind": "scaled", "stale": False, "breadth_kind": None, "breadth": {},
    },
    "TLT": {
        "name": "Long-term Treasuries", "index": "US Treasuries, 20+ years", "group": "other", "calendar": "trading",
        "proxy": ("VUSTX",), "proxy_kind": "scaled", "stale": False, "breadth_kind": None, "breadth": {},
    },
    "BTC-USD": {
        "name": "Bitcoin", "index": "Bitcoin in US dollars", "group": "other", "calendar": "calendar",
        "proxy": None, "proxy_kind": None, "stale": False, "breadth_kind": None, "breadth": {},
    },
}

# One dashboard page (and one detail report) per group. Equities are required: if they
# cannot be fetched the run fails instead of publishing. Other assets are left off with a
# note if their data is missing or stale.
GROUPS = {
    "equity": {"title": "Equities", "page": "index.html", "report": "report.html",
               "assets": ["SPY", "EFA", "EEM"]},
    "other": {"title": "Gold, Treasuries and Bitcoin", "page": "other.html", "report": "other-report.html",
              "assets": ["GLD", "IEF", "TLT", "BTC-USD"]},
}
CORE = GROUPS["equity"]["assets"]       # required for the run to publish
ALL = list(ASSETS)

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
# Core metrics are on the dashboard; the rest are context. "1-month" means 21 trading days,
# or 30 calendar days for assets on the calendar-day clock; "3-month" is 63 or 91.
METRICS = [
    ("eff_vol", "Effective volatility", "pct", False, True),
    ("base_vol", "Volatility, blended 1- and 3-month", "pct", False, True),
    ("vol_21", "Volatility, 1-month", "pct", False, True),
    ("vol_63", "Volatility, 3-month", "pct", False, True),
    ("vr21", "Variance ratio, 1-month (trend persistence)", "num", None, True),
    ("vr63", "Variance ratio, 3-month (context only)", "num", None, False),
    ("mom_3m", "3-month momentum", "pct", True, False),
    ("mom_12m", "12-month momentum", "pct", True, False),
    ("b3", "Breadth: components up over 3 months", "pct", True, False),
    ("b12", "Breadth: components up over 12 months", "pct", True, False),
]
BREADTH_KEYS = ("b3", "b12")


def to_days(n: int, calendar: str) -> int:
    """A window given in trading days, in the asset's own day count."""
    return n if calendar == "trading" else int(math.floor(n * CALENDAR_DAYS / TRADING_DAYS + 0.5))


def days_per_year(calendar: str) -> int:
    return TRADING_DAYS if calendar == "trading" else CALENDAR_DAYS


# --------------------------------------------------------------------------- #
# Metric functions (pure; covered by tests/test_metrics.py)
# --------------------------------------------------------------------------- #

def momentum(price: pd.Series, days: int) -> pd.Series:
    """Total return over the last `days` observations of an adjusted price series."""
    return price / price.shift(days) - 1.0


def realized_vol(ret: pd.Series, days: int, per_year: int = TRADING_DAYS) -> pd.Series:
    """Annualized standard deviation of daily returns over a rolling window."""
    return ret.rolling(days).std() * math.sqrt(per_year)


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


def expanding_median(s: pd.Series, min_obs: int = MIN_RANK_OBS) -> pd.Series:
    """Median of all values up to and including each date (point-in-time)."""
    return s.dropna().expanding(min_periods=min_obs).median().reindex(s.index)


def compound(ret: pd.Series, days: int) -> pd.Series:
    """Compounded return over a rolling window of daily returns."""
    return np.expm1(np.log1p(ret.clip(lower=-0.999)).rolling(days).sum())


def leveraged_returns(ret: pd.Series, lev: float, cash_rate: pd.Series | float = 0.0,
                      spread: float = 0.0, fee: float = 0.0, per_year: int = TRADING_DAYS) -> pd.Series:
    """Daily returns of a fund that resets to `lev` times exposure every day.

    The borrowed (lev - 1) portion pays the cash rate plus a spread; `fee` is
    the annual expense ratio. Rates are annual decimals, accrued over `per_year` days.
    """
    cost = ((lev - 1) * (cash_rate + spread) + fee) / per_year
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


def _monthly(frame: pd.DataFrame, years: int) -> pd.DataFrame:
    m = frame.dropna().resample("ME").last().pct_change(fill_method=None).dropna()
    return m.iloc[: years * 12]


def fit_blend_weight(target: pd.Series, a: pd.Series, b: pd.Series, years: int = 5):
    """Weight on `a` (rest on `b`) whose monthly returns best match `target`.

    Fitted on monthly returns so that stale daily pricing in the proxies does not
    distort it. Returns (weight, correlation of the fitted blend with the target).
    """
    m = _monthly(pd.concat({"t": target, "a": a, "b": b}, axis=1), years)
    if len(m) < 24:
        return DEFAULT_EUROPE_WEIGHT, float("nan")
    grid = np.linspace(0, 1, 101)
    err = [float(((w * m["a"] + (1 - w) * m["b"] - m["t"]) ** 2).sum()) for w in grid]
    w = float(grid[int(np.argmin(err))])
    return w, float(np.corrcoef(w * m["a"] + (1 - w) * m["b"], m["t"])[0, 1])


def fit_scale(target: pd.Series, proxy: pd.Series, years: int = 5):
    """Scale on the proxy's returns that best matches `target`'s monthly returns (least squares
    through the origin), e.g. to bridge a difference in bond duration. Returns (scale, correlation)."""
    m = _monthly(pd.concat({"t": target, "p": proxy}, axis=1), years)
    if len(m) < 24:
        return 1.0, float("nan")
    return float((m["t"] * m["p"]).sum() / (m["p"] ** 2).sum()), float(np.corrcoef(m["t"], m["p"])[0, 1])


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
    t = list(ALL)
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
    close = close.groupby(level=0).last()          # one row per date
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

    # Drop today's bar: before the US close it is partial for every asset, and bitcoin's
    # bar for the current day (in UTC) is never complete when this runs.
    try:
        from zoneinfo import ZoneInfo
        now = dt.datetime.now(ZoneInfo("America/New_York"))
        if len(out) and now.time() < dt.time(16, 20):
            out = out[out.index.date < now.date()]
        utc_today = dt.datetime.now(dt.timezone.utc).date()
        for t in out.columns:
            if ASSETS.get(t, {}).get("calendar") == "calendar":
                out.loc[out.index.date >= utc_today, t] = np.nan
    except Exception:
        pass
    return out.sort_index()


# Approximate first-trade dates, used ONLY to shape the synthetic test data.
_SYN_START = {"SPY": "1993-01-29", "EFA": "2001-08-27", "EEM": "2003-04-14",
              "VEIEX": "1994-05-04", "VEURX": "1990-06-18", "VPACX": "1990-06-18",
              "GLD": "2004-11-18", "IEF": "2002-07-26", "TLT": "2002-07-26",
              "VFITX": "1991-10-28", "VUSTX": "1986-05-19", "BTC-USD": "2014-09-17",
              "XLRE": "2015-10-08", "XLC": "2018-06-19", "EWT": "2000-06-23",
              "EWY": "2000-05-12", "EWZ": "2000-07-14", "EZA": "2003-02-07",
              "FXI": "2004-10-08", "INDA": "2012-02-03", "^IRX": "1990-01-02"}
_SYN_STALE = {"VEURX", "VPACX", "VEIEX"}


def _syn_vol(rng, n: int, base: float) -> np.ndarray:
    vol, v = np.empty(n), base
    for i in range(n):  # slow-moving volatility with occasional spikes
        v = max(0.43 * base, 0.97 * v + 0.03 * base + 0.07 * base * rng.standard_normal()
                + (1.7 * base if rng.random() < 0.0015 else 0))
        vol[i] = v
    return vol


def fetch_synthetic(tickers: list[str], seed: int = 7) -> pd.DataFrame:
    """Made-up prices with the same shape as the real data. Never real readings."""
    rng = np.random.default_rng(seed)
    end = dt.date.today() - dt.timedelta(days=1)
    idx = pd.bdate_range(FETCH_START, end)
    n = len(idx)
    market = 0.0004 + _syn_vol(rng, n, 0.007) * rng.standard_normal(n)
    rates = 0.0001 + _syn_vol(rng, n, 0.004) * rng.standard_normal(n)
    out = {}
    for t in tickers:
        if t == CASH_TICKER:
            s = pd.Series(np.clip(3 + np.cumsum(0.02 * rng.standard_normal(n)), 0.05, 7), idx)
        elif t == "BTC-USD":
            didx = pd.date_range(FETCH_START, end, freq="D")
            r = 0.001 + _syn_vol(rng, len(didx), 0.035) * rng.standard_normal(len(didx))
            s = pd.Series(100 * np.exp(np.cumsum(r)), didx)
        else:
            treas = t in ("IEF", "TLT", "VFITX", "VUSTX")
            src = rates if treas else market
            beta = (2.0 if t in ("TLT", "VUSTX") else 0.9) if treas else rng.uniform(0.7, 1.3)
            if t == "GLD":
                src, beta = 0.0003 + _syn_vol(rng, n, 0.009) * rng.standard_normal(n), 1.0
            own = rng.uniform(0.1, 0.3 if treas else 0.9) * np.abs(src).mean() * rng.standard_normal(n)
            r = beta * src + own + rng.uniform(-0.0001, 0.0002)
            if t in _SYN_STALE:  # mutual fund priced off foreign closes: mimic stale pricing
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

def calendar_info(calendar: str) -> dict:
    """Every window for one asset, in its own day count."""
    d = lambda n: to_days(n, calendar)    # noqa: E731
    return {"kind": calendar, "dpy": days_per_year(calendar), "month": d(MONTH),
            "vol": [d(VOL_SHORT), d(VOL_LONG)], "mom": [d(MOM_SHORT), d(MOM_LONG)],
            "vr_q": d(VR_Q), "vr_q_long": d(VR_Q_LONG), "vr_window": d(VR_WINDOW),
            "vr_se": vr_standard_error(d(VR_Q), d(VR_WINDOW)),
            "vr_se_long": vr_standard_error(d(VR_Q_LONG), d(VR_WINDOW)),
            "min_rank": d(MIN_RANK_OBS), "horizon": d(BACKTEST_HORIZON), "year": d(TRADING_DAYS)}


def build(px: pd.DataFrame, notes: list[str], synthetic: bool) -> dict:
    for t in CORE:
        if t not in px.columns:
            raise SystemExit(f"Core ticker {t} is missing; refusing to publish.")
    raw = px.sort_index()
    cal = raw["SPY"].dropna().loc[FETCH_START:].index      # trading calendar (US market days)
    px = raw.reindex(cal)
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
        if a not in raw.columns or raw[a].notna().sum() < 300:
            notes.append(f"{a}: no usable data this run, so it is left off the page.")
            continue
        ci = calendar_info(cfg["calendar"])
        dpy = ci["dpy"]
        if cfg["calendar"] == "trading":
            idx, src = cal, px
        else:   # trades every day: its own daily calendar
            idx = raw[a].dropna().loc[FETCH_START:].index
            src = raw.reindex(idx)
        etf = src[a]
        a_asof = etf.last_valid_index()
        if not synthetic and (asof - a_asof).days > 7:
            notes.append(f"{a}: latest price is {a_asof.date()}, more than a week old, so it is left off the page.")
            continue
        if a_asof < idx[-1]:
            notes.append(f"{a}: no price for {idx[-1].date()}; its readings are as of {a_asof.date()}.")
        idx = idx[idx <= a_asof]
        src, etf = src.loc[:a_asof], etf.loc[:a_asof]
        cash_a = cash.reindex(cash.index.union(idx)).ffill().bfill().reindex(idx)

        etf_start = etf.first_valid_index()
        meta = {"etf_start": etf_start, "asof": a_asof, "proxy": None, "blend_weight": None, "scale": None,
                "blend_corr": None, "stale_ac_backfill": None, "stale_ac_etf": None, "cal": ci}
        proxy_ret = None
        prox = [p for p in (cfg["proxy"] or ()) if p in src.columns and src[p].notna().sum() > 300]
        kind = cfg["proxy_kind"]
        if cfg["proxy"] and len(prox) < len(cfg["proxy"]):
            notes.append(f"{a}: backfill fund data missing, so history starts at the ETF's "
                         f"own inception ({etf_start.date()}).")
        elif kind == "single":
            proxy_ret = src[prox[0]].pct_change(fill_method=None)
            meta["proxy"] = prox[0]
        elif kind == "scaled":
            k, corr = fit_scale(etf, src[prox[0]])
            proxy_ret = k * src[prox[0]].pct_change(fill_method=None)
            meta.update(proxy=f"{prox[0]} returns × {k:.2f}", scale=k, blend_corr=corr)
        elif kind == "blend":
            w, corr = fit_blend_weight(etf, src[prox[0]], src[prox[1]])
            proxy_ret = (w * src[prox[0]].pct_change(fill_method=None)
                         + (1 - w) * src[prox[1]].pct_change(fill_method=None))
            meta.update(proxy=f"{w:.0%} {prox[0]} + {1 - w:.0%} {prox[1]}",
                        blend_weight=w, blend_corr=corr)
        price = splice(etf, proxy_ret)
        ret = price.pct_change(fill_method=None)
        log_ret = np.log(price).diff()

        f = pd.DataFrame(index=idx)
        f["price"] = price
        f["mom_3m"] = momentum(price, ci["mom"][0])
        f["mom_12m"] = momentum(price, ci["mom"][1])
        f["vol_21"] = realized_vol(ret, ci["vol"][0], dpy)
        f["vol_63"] = realized_vol(ret, ci["vol"][1], dpy)

        # Core metric. Effective volatility = volatility from daily returns x sqrt(VR at one
        # month): the risk of a position held for a month. The variance ratios are left blank
        # wherever their window touches stale-priced backfill, which inflates serial correlation.
        f["vr21"] = rolling_variance_ratio_lm(log_ret, ci["vr_q"], ci["vr_window"])
        f["vr63"] = rolling_variance_ratio_lm(log_ret, ci["vr_q_long"], ci["vr_window"])
        if proxy_ret is not None and cfg["stale"]:
            cut = idx[min(idx.get_loc(etf_start) + ci["vr_window"], len(idx) - 1)]
            f.loc[f.index < cut, ["vr21", "vr63"]] = np.nan
        f["base_vol"] = np.sqrt((f["vol_21"] ** 2 + f["vol_63"] ** 2) / 2.0)
        f["eff_vol"] = f["base_vol"] * np.sqrt(f["vr21"])
        for b in VOL_BUDGETS:
            f[f"lev_{round(b * 100)}"] = implied_leverage(b, f["eff_vol"])     # uncapped

        if proxy_ret is not None:  # lag-1 autocorrelation before and after the ETF existed
            pre = ret[(ret.index >= HISTORY_START) & (ret.index <= etf_start)].dropna()
            post = ret[ret.index > etf_start].dropna()
            if len(pre) > 250:
                meta["stale_ac_backfill"] = float(pre.autocorr(1))
                meta["stale_ac_etf"] = float(post.autocorr(1))

        if cfg["breadth"]:
            comps = src[[c for c in cfg["breadth"] if c in src.columns]].ffill(limit=5)
            lost = [c for c in cfg["breadth"] if c not in src.columns]
            if lost:
                notes.append(f"{a} breadth is missing components: {', '.join(lost)}.")
            b12 = breadth(comps, cfg["breadth"], ci["mom"][1])
            b3 = breadth(comps, cfg["breadth"], ci["mom"][0])
            f["b12"], f["b12w"], f["b12n"] = b12["equal"], b12["weighted"], b12["n"]
            f["b3"], f["b3w"] = b3["equal"], b3["weighted"]
        else:
            f[["b12", "b12w", "b12n", "b3", "b3w"]] = np.nan

        # Daily-reset leverage: trailing 1-year simulation and the forward window for the backtest.
        H, Y = ci["horizon"], ci["year"]
        f["ret"] = ret
        f["dd"] = price / price.cummax() - 1.0         # drawdown from the highest prior close
        f["und_1y"] = compound(ret, Y)
        f["fwd_1x"] = compound(ret, H).shift(-H)
        for lev in LEVERAGE:
            gross = leveraged_returns(ret, lev, per_year=dpy)
            net = leveraged_returns(ret, lev, cash_a, FINANCING_SPREAD, EXPENSE_RATIO, per_year=dpy)
            f[f"dg_{lev}x"], f[f"dn_{lev}x"] = gross, net
            f[f"gross_{lev}x"] = compound(gross, Y)
            f[f"net_{lev}x"] = compound(net, Y)
            f[f"gap_{lev}x"] = f[f"gross_{lev}x"] - lev * f["und_1y"]
            f[f"fwd_net_{lev}x"] = compound(net, H).shift(-H)
            f[f"fwd_gap_{lev}x"] = compound(gross, H).shift(-H) - lev * f["fwd_1x"]

        # Each asset is ranked against its own history from HISTORY_START, or from when its
        # data begins if later.
        rank_start = max(f["vol_63"].first_valid_index(), pd.Timestamp(HISTORY_START))
        for key, *_ in METRICS:
            f["p_" + key] = expanding_percentile(f[key].where(f.index >= rank_start), ci["min_rank"])
        f["score"] = 100.0 - f["p_eff_vol"]          # favorability: low effective volatility = favorable
        label = pd.Series("neutral", index=f.index, dtype=object)
        label[f["score"] >= SCORE_CUTS[2]] = "favorable"
        label[f["score"] >= SCORE_CUTS[3]] = "very_favorable"
        label[f["score"] <= SCORE_CUTS[1]] = "unfavorable"
        label[f["score"] <= SCORE_CUTS[0]] = "very_unfavorable"
        f["label"] = label.where(f["score"].notna())
        # Leverage that would put effective volatility at its own point-in-time median.
        f["eff_vol_med"] = expanding_median(f["eff_vol"].where(f.index >= rank_start), ci["min_rank"])
        f["lev_med"] = f["eff_vol_med"] / f["eff_vol"]
        meta["rank_start"] = rank_start
        frames[a], info[a] = f, meta

    return {"frames": frames, "info": info, "asof": asof, "cash": cash, "cash_note": cash_note,
            "px": px, "raw": raw}


def _num(x, digits=4):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(x) else round(x, digits)


def _ago(s: pd.Series, n: int):
    return s.iloc[-1 - n] if len(s) > n else None


def group_assets(res: dict, group: str) -> list[str]:
    return [a for a in GROUPS[group]["assets"] if a in res["frames"]]


def group_metrics(assets: list[str]) -> list[tuple]:
    has_breadth = any(ASSETS[a]["breadth"] for a in assets)
    return [m for m in METRICS if has_breadth or m[0] not in BREADTH_KEYS]


def _weekly_group(frames: dict, assets: list[str], cols: list[tuple]) -> tuple[list, dict]:
    """Weekly samples on one shared axis for assets on different calendars.

    cols: (key, digits, how) with how "last" (last observation of the week) or "min"
    (lowest value in the week, so a trough between samples is not missed). Weeks end on
    Friday; each week is dated by the latest observation any asset has in it.
    """
    start = pd.Timestamp(HISTORY_START)
    parts, label = {}, {}
    for a in assets:
        f = frames[a][frames[a].index >= start]
        per = f.index.to_period("W-FRI")
        last = f.groupby(per).tail(1)
        last.index = last.index.to_period("W-FRI")
        mins = f[[k for k, _, how in cols if how == "min"]].groupby(per).min()
        for p, d in pd.Series(f.index, index=per).groupby(level=0).max().items():
            label[p] = max(label.get(p, d), d)
        parts[a] = (last, mins)
    periods = sorted(label)
    out = {}
    for a in assets:
        last, mins = parts[a]
        out[a] = {k: [_num(v, digits) for v in (mins if how == "min" else last)[k].reindex(periods)]
                  for k, digits, how in cols}
    return [str(label[p].date()) for p in periods], out


def make_payload(res: dict, notes: list[str], synthetic: bool, group: str) -> dict:
    frames, info, asof = res["frames"], res["info"], res["asof"]
    order = group_assets(res, group)
    metrics = group_metrics(order)

    assets = {}
    for a in order:
        f, meta, cfg = frames[a], info[a], ASSETS[a]
        ci, rank_start = meta["cal"], meta["rank_start"]
        M = ci["month"]
        last = f.iloc[-1]
        cells = {}
        for key, _, _, higher_better, _ in metrics:
            hist = f[key][f.index >= rank_start].dropna()
            pct = _num(last["p_" + key], 1)
            cells[key] = {
                "value": _num(last[key]), "pct": pct,
                "fav": None if pct is None or higher_better is None else _num(pct if higher_better else 100 - pct, 1),
                "prev": _num(_ago(f[key], M)),
                "q": [_num(hist.quantile(p)) for p in (0.01, 0.25, 0.5, 0.75, 0.99)] if len(hist) else None,
                "n": int(len(hist)), "since": str(hist.index[0].date()) if len(hist) else None,
            }
        lev = {}
        for L in LEVERAGE:
            lev[str(L)] = {"naive": _num(L * last["und_1y"]), "gross": _num(last[f"gross_{L}x"]),
                           "gap": _num(last[f"gap_{L}x"]), "net": _num(last[f"net_{L}x"]),
                           "drag": _num(vol_drag(last["vol_21"], L))}
        H = ci["horizon"]
        ef = f[(f.index > meta["etf_start"]) & f["label"].notna() & f[f"fwd_net_{LEVERAGE[-1]}x"].notna()]
        backtest = []
        for lab in LEVEL_KEYS:
            g = ef[ef["label"] == lab]
            row = {"label": lab, "days": int(len(g)), "share": _num(len(g) / max(len(ef), 1), 3),
                   "windows": int(len(g) // H)}
            if len(g):
                row.update(fwd_1x=_num(g["fwd_1x"].median()))
                for L in LEVERAGE:
                    row[f"net_{L}x"] = _num(g[f"fwd_net_{L}x"].median())
                    row[f"gap_{L}x"] = _num(g[f"fwd_gap_{L}x"].median())
                top = LEVERAGE[-1]
                row["win"] = _num((g[f"fwd_net_{top}x"] > 0).mean(), 3)
                row["p5"] = _num(g[f"fwd_net_{top}x"].quantile(0.05))
            backtest.append(row)
        budgets = [round(b * 100) for b in VOL_BUDGETS]
        assets[a] = {
            "name": cfg["name"], "index": cfg["index"], "breadth_kind": cfg["breadth_kind"],
            "etf_start": str(meta["etf_start"].date()), "asof": str(meta["asof"].date()),
            "rank_start": str(rank_start.date()), "proxy": meta["proxy"], "proxy_kind": cfg["proxy_kind"],
            "stale": bool(cfg["stale"]), "cal": {k: (_num(v, 3) if isinstance(v, float) else v) for k, v in ci.items()},
            "blend_weight": _num(meta["blend_weight"], 2), "scale": _num(meta["scale"], 2),
            "blend_corr": _num(meta["blend_corr"], 3),
            "stale_ac_backfill": _num(meta["stale_ac_backfill"], 3),
            "stale_ac_etf": _num(meta["stale_ac_etf"], 3),
            "score": _num(last["score"], 1), "label": last["label"] if isinstance(last["label"], str) else None,
            "score_prev": _num(_ago(f["score"], M), 1), "dd": _num(last["dd"]),
            "score_from": str(f["score"].first_valid_index().date()) if f["score"].notna().any() else None,
            "b12w": _num(last["b12w"]), "b3w": _num(last["b3w"]),
            "b_members": int(last["b12n"]) if pd.notna(last["b12n"]) else 0,
            "b_total": len(cfg["breadth"]), "und_1y": _num(last["und_1y"]),
            "cells": cells, "lev": lev, "backtest": backtest,
            "vb": {k: _num(last[k]) for k in ("vol_21", "vol_63", "base_vol", "vr21", "vr63", "eff_vol", "eff_vol_med", "lev_med")}
                  | {k + "_prev": _num(_ago(f[k], M)) for k in ("vr21", "eff_vol", "base_vol", "lev_med")}
                  | {"lev": {str(b): _num(last[f"lev_{b}"]) for b in budgets},
                     "lev_prev": {str(b): _num(_ago(f[f"lev_{b}"], M)) for b in budgets},
                     "vr_from": str(f["vr21"].first_valid_index().date()) if f["vr21"].notna().any() else None},
            "backtest_from": str(ef.index[0].date()) if len(ef) else None,
        }

    # Weekly samples (last observation of each week) keep the page small.
    keys = ["score", "eff_vol", "base_vol", "vol_21", "vol_63", "vr21", "vr63", "lev_med"] \
        + [f"lev_{round(b * 100)}" for b in VOL_BUDGETS]
    dates, series = _weekly_group(frames, order, [(k, 4, "last") for k in keys] + [("dd", 4, "min")])
    series["dates"] = dates

    g = GROUPS[group]
    return {
        "asof": str(asof.date()),
        "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "synthetic": synthetic, "group": group, "title": g["title"],
        "pages": [{"key": k, "title": v["title"], "page": v["page"], "report": v["report"]} for k, v in GROUPS.items()],
        "cash_now": _num(res["cash"].iloc[-1]), "cash_note": res["cash_note"],
        "notes": [n for n in notes if any(n.startswith(x) for x in order) or not any(n.startswith(x) for x in ALL)],
        "assets": assets, "series": series, "order": order,
        "metrics": [{"key": k, "label": lab, "fmt": fmt, "higher_better": hb, "core": core}
                    for k, lab, fmt, hb, core in metrics],
        "config": {"mom": [MOM_SHORT, MOM_LONG], "vol": [VOL_SHORT, VOL_LONG],
                   "vr_q": VR_Q, "vr_q_long": VR_Q_LONG, "vr_window": VR_WINDOW,
                   "vr_se": _num(vr_standard_error(VR_Q, VR_WINDOW), 3),
                   "vr_se_long": _num(vr_standard_error(VR_Q_LONG, VR_WINDOW), 3),
                   "budgets": [round(b * 100) for b in VOL_BUDGETS], "caps": list(LEV_CAPS),
                   "min_rank": MIN_RANK_OBS, "history_start": HISTORY_START,
                   "leverage": list(LEVERAGE), "spread": FINANCING_SPREAD, "fee": EXPENSE_RATIO,
                   "cuts": list(SCORE_CUTS), "ref_lines": list(SCORE_REF_LINES),
                   "levels": [{"key": k, "name": n} for k, n in LEVELS],
                   "horizon": BACKTEST_HORIZON,
                   "breadth_min": BREADTH_MIN_MEMBERS,
                   "breadth": {a: ASSETS[a]["breadth"] for a in order if ASSETS[a]["breadth"]}},
    }


def _fwd_stats(g: pd.DataFrame, top: int) -> dict:
    """Median outcomes over the rows of `g`, which holds forward-return columns."""
    if not len(g):
        return {"days": 0}
    return {"days": int(len(g)), "fwd_1x": _num(g["f1"].median()), "net": _num(g["fn"].median()),
            "gap": _num(g["fg"].median()), "win": _num((g["fn"] > 0).mean(), 3),
            "p5": _num(g["fn"].quantile(0.05))}


def make_report(res: dict, payload: dict) -> dict:
    """Everything the dashboard summarizes, in full: the payload for the detail report."""
    frames, info, raw, asof = res["frames"], res["info"], res["raw"], res["asof"]
    top = LEVERAGE[-1]
    order = payload["order"]
    metric_keys = [m["key"] for m in payload["metrics"]]
    has_breadth = any(ASSETS[a]["breadth"] for a in order)
    extra = ["und_1y", "score", "lev_med"] + (["b12w", "b3w"] if has_breadth else []) \
        + [f"lev_{round(b * 100)}" for b in VOL_BUDGETS] + [f"{k}_{L}x" for L in LEVERAGE for k in ("gap", "net")]
    cols = [(k, 4, "last") for k in metric_keys + extra] + [("p_" + k, 1, "last") for k in metric_keys] + [("dd", 4, "min")]
    dates, series = _weekly_group(frames, order, cols)
    series["dates"] = dates

    assets = {}
    for a in order:
        f, meta, cfg = frames[a], info[a], ASSETS[a]
        ci = meta["cal"]
        d = lambda n: to_days(n, cfg["calendar"])    # noqa: E731
        lookback = {k: [_num(_ago(f[k], d(n))) for n in (21, 63, 252)] for k in metric_keys}

        comps = []
        for t, w in cfg["breadth"].items():
            row = {"ticker": t, "name": COMPONENT_NAMES.get(t, ""), "weight": w}
            if t in raw.columns and raw[t].notna().any():
                c = raw[t].reindex(f.index).ffill(limit=5)
                row.update(m3=_num(momentum(c, ci["mom"][0]).iloc[-1]), m12=_num(momentum(c, ci["mom"][1]).iloc[-1]),
                           since=str(raw[t].first_valid_index().date()))
            comps.append(row)

        lev = []
        for n, label in LEV_LOOKBACKS:
            und = compound(f["ret"], d(n)).iloc[-1]
            row = {"label": label, "und": _num(und)}
            for L in LEVERAGE:
                row[f"naive_{L}"] = _num(L * und)
                row[f"gross_{L}"] = _num(compound(f[f"dg_{L}x"], d(n)).iloc[-1])
                row[f"net_{L}"] = _num(compound(f[f"dn_{L}x"], d(n)).iloc[-1])
            lev.append(row)

        # Forward outcomes, using only dates after the ETF itself began trading. Horizons are
        # keyed in trading days and converted to the asset's own day count.
        era = (f.index > meta["etf_start"]) & f["label"].notna()
        by_horizon = {}
        fwd = None
        for h in REPORT_HORIZONS:
            hd = d(h)
            f1 = compound(f["ret"], hd).shift(-hd)
            fw = pd.DataFrame({"f1": f1, "fn": compound(f[f"dn_{top}x"], hd).shift(-hd),
                               "fg": compound(f[f"dg_{top}x"], hd).shift(-hd) - top * f1})
            fw = fw[era & fw["fn"].notna()]
            by_horizon[str(h)] = {lab: _fwd_stats(fw[f["label"].reindex(fw.index) == lab], top)
                                  for lab in LEVEL_KEYS}
            if h == BACKTEST_HORIZON:
                fwd = fw
        score = f["score"].reindex(fwd.index)
        bands = []
        for lo, hi in zip(SCORE_BANDS[:-1], SCORE_BANDS[1:]):
            g = fwd[(score >= lo) & ((score < hi) | (hi == SCORE_BANDS[-1]))]
            bands.append({"band": f"{lo} to {hi}", "share": _num(len(g) / max(len(fwd), 1), 3),
                          "windows": int(len(g) // ci["horizon"]), **_fwd_stats(g, top)})
        alone = []
        for m in payload["metrics"]:
            if m["higher_better"] is None:
                continue
            p = f["p_" + m["key"]].reindex(fwd.index)
            fav = p if m["higher_better"] else 100 - p
            thirds = {"fav": fwd[fav >= 200 / 3], "mid": fwd[(fav > 100 / 3) & (fav < 200 / 3)],
                      "unfav": fwd[fav <= 100 / 3]}
            alone.append({"key": m["key"], "label": m["label"],
                          **{k: _fwd_stats(v, top) for k, v in thirds.items()}})

        lab = f["label"].dropna()
        changes = lab[lab != lab.shift()]
        recent = changes[changes.index >= meta["asof"] - pd.Timedelta(days=365)]
        prev = lab.shift().reindex(recent.index)
        assets[a] = {
            "lookback": lookback, "components": comps, "lev": lev, "by_horizon": by_horizon,
            "bands": bands, "alone": alone,
            "streak": int(len(lab.loc[changes.index[-1]:])) if len(changes) else 0,
            "changes": [{"date": str(x.date()), "from": prev[x] if isinstance(prev[x], str) else None,
                         "to": recent[x]} for x in recent.index][::-1],
        }

    roles = {t: "Core asset" for t in order}
    for a in order:
        cfg = ASSETS[a]
        for t in cfg["proxy"] or ():
            roles[t] = f"Backfill for {a}"
        for t in cfg["breadth"]:
            roles[t] = f"Breadth for {a}: {COMPONENT_NAMES.get(t, t)}"
    roles[CASH_TICKER] = "Financing rate (13-week T-bill yield)"
    coverage = []
    for t, role in roles.items():
        c = raw[t].dropna() if t in raw.columns else pd.Series(dtype=float)
        coverage.append({"ticker": t, "role": role, "first": str(c.index[0].date()) if len(c) else None,
                         "last": str(c.index[-1].date()) if len(c) else None, "n": int(len(c))})

    out = dict(payload)
    out["series"] = series
    out["report"] = {"assets": assets, "coverage": coverage, "horizons": list(REPORT_HORIZONS),
                     "fetch_start": FETCH_START, "has_breadth": has_breadth}
    return out


def render_page(title: str, body: str, script: str, payload: dict) -> str:
    blob = json.dumps(payload, allow_nan=False, separators=(",", ":")).replace("</", "<\\/")
    return (PAGE.replace("__TITLE__", title).replace("__CSS__", CSS).replace("__BODY__", body)
            .replace("__SCRIPT__", JS_COMMON + JS_DOC + script).replace("/*__PAYLOAD__*/null", blob))


def write_outputs(payloads: dict, out_dir: Path, history_path: Path | None,
                  reports: dict | None = None) -> None:
    """payloads / reports: one per group, keyed like GROUPS."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for g, payload in payloads.items():
        page = GROUPS[g]
        (out_dir / page["page"]).write_text(
            render_page("Leverage Environment: " + page["title"], DASH_BODY, DASH_JS, payload), encoding="utf-8")
        if reports and g in reports:
            (out_dir / page["report"]).write_text(
                render_page("Leverage Environment: " + page["title"] + ", detail", REPORT_BODY, REPORT_JS, reports[g]),
                encoding="utf-8")
    first = next(iter(payloads.values()))
    slim = {"asof": first["asof"], "generated": first["generated"], "synthetic": first["synthetic"],
            "groups": {g: {k: v for k, v in p.items() if k != "series"} for g, p in payloads.items()}}
    (out_dir / "latest.json").write_text(json.dumps(slim, indent=1), encoding="utf-8")

    if history_path is None or first["synthetic"]:
        return
    # One row per asset per reading date: a genuine point-in-time record that builds up over time.
    keys = list(dict.fromkeys(m["key"] for p in payloads.values() for m in p["metrics"]))
    budgets = first["config"]["budgets"]
    fields = ["date", "etf", "label", "score"] + keys + ["p_" + k for k in keys] \
        + ["base_vol", "vr21", "eff_vol", "lev_med"] + [f"lev_{b}" for b in budgets]
    new = []
    for p in payloads.values():
        for a in p["order"]:
            d = p["assets"][a]
            row = {"date": d["asof"], "etf": a, "label": d["label"], "score": d["score"]}
            for k, c in d["cells"].items():
                row[k], row["p_" + k] = c["value"], c["pct"]
            for k in ("base_vol", "vr21", "eff_vol", "lev_med"):
                row[k] = d["vb"][k]
            for b, v in d["vb"]["lev"].items():
                row["lev_" + b] = v                       # uncapped
            new.append(row)
    seen = {(r["date"], r["etf"]) for r in new}
    rows = []
    if history_path.exists():
        with history_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            rows = [r for r in reader if (r.get("date"), r.get("etf")) not in seen]
            fields += [c for c in (reader.fieldnames or []) if c not in fields]   # keep older columns
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(sorted(rows + new, key=lambda r: (r["date"], r["etf"])))


def run(px: pd.DataFrame, notes: list[str], synthetic: bool):
    res = build(px, notes, synthetic)
    payloads, reports = {}, {}
    for g in GROUPS:
        if not group_assets(res, g):
            notes.append(f"No assets available for the {GROUPS[g]['title']} page; it was not rebuilt.")
            continue
        payloads[g] = make_payload(res, notes, synthetic, g)
        reports[g] = make_report(res, payloads[g])
    return res, payloads, reports


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="site", help="output folder for the pages and latest.json")
    ap.add_argument("--history", default="data/history.csv", help="daily log to append to ('' to skip)")
    ap.add_argument("--synthetic", action="store_true", help="use made-up data (offline test)")
    args = ap.parse_args()

    notes: list[str] = []
    px = fetch_synthetic(all_tickers()) if args.synthetic else fetch_live(all_tickers(), notes)
    if px.empty:
        raise SystemExit("No price data could be fetched; refusing to publish.")
    res, payloads, reports = run(px, notes, args.synthetic)
    write_outputs(payloads, Path(args.out), Path(args.history) if args.history else None, reports=reports)
    for p in payloads.values():
        for a in p["order"]:
            d = p["assets"][a]
            print(f"{d['asof']} {a}: {d['label']} (score {d['score']})")
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
--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#8a5cd6;
--good:#0ca30c;--warn:#fab219;--crit:#d03b3b;--band:rgba(11,11,11,.05)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;
--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#9b74e8;--band:rgba(255,255,255,.06)}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#9b74e8;--band:rgba(255,255,255,.06)}
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
#vbcharts,#dragcharts{display:grid;gap:14px;margin:10px 0 6px}
.tog{display:flex;flex-wrap:wrap;gap:4px}
.tog button{display:inline-flex;align-items:center;gap:6px}
.tog button[aria-pressed="true"]{background:var(--surface);color:var(--ink)}
.tog button[aria-pressed="false"]{opacity:.45;text-decoration:line-through}
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
nav.pages{display:flex;flex-wrap:wrap;gap:4px;margin:0 0 14px;font-size:13.5px}
nav.pages a{padding:4px 11px;border-radius:7px;border:1px solid var(--border);color:var(--ink);text-decoration:none;background:var(--surface)}
nav.pages a[aria-current="page"]{background:var(--ink);color:var(--surface);border-color:var(--ink)}
#sumtab{margin-top:18px}#sumtab th{vertical-align:bottom;white-space:nowrap;padding:8px 7px}.nw{white-space:nowrap}#sumtab td{padding:9px 7px;vertical-align:middle}
th button.sort{all:unset;cursor:pointer;display:inline-flex;align-items:flex-end;gap:3px;font-weight:600;color:var(--ink2)}
th button.sort:focus-visible{outline:2px solid var(--s1);outline-offset:2px;border-radius:3px}
th.n button.sort{justify-content:flex-end;text-align:right}
th button.sort .ar{font-size:10px;color:var(--muted);min-width:8px}th[aria-sort] button.sort{color:var(--ink)}
td .tkr{display:flex;align-items:center;gap:7px;font-weight:650;white-space:nowrap}td .nm2{color:var(--ink2);font-size:12px;white-space:nowrap}
.callout{border:1px solid var(--border);border-left:4px solid var(--s1);border-radius:8px;padding:10px 14px;margin:14px 0;background:var(--surface);max-width:90ch}
'''

JS_COMMON = r'''const $ = (id) => document.getElementById(id);
const C = D.config, A = D.order;
const pct = (x, d = 1) => x == null ? "n/a" : (x * 100).toFixed(d) + "%";
const spct = (x, d = 1) => x == null ? "n/a" : (x > 0 ? "+" : "") + (x * 100).toFixed(d) + "%";
const num = (x, d = 2) => x == null ? "n/a" : x.toFixed(d);
const ord = (n) => { n = Math.round(n); const s = ["th", "st", "nd", "rd"], v = n % 100; return n + (s[(v - 20) % 10] || s[v] || s[0]); };
const fmtv = (m, x, signed) => m.fmt === "pct" ? (signed ? spct(x) : pct(x)) : num(x);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const COLOR = Object.fromEntries(A.map((a, i) => [a, "--s" + (i % 4 + 1)]));
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
const LATER = A.filter((a) => D.assets[a].asof !== D.asof);
$("asof").textContent = "Data through " + D.asof + (LATER.length ? " (" + LATER.map((a) => a + " through " + D.assets[a].asof).join(", ") + ")" : "") +
  "; each asset is ranked against its own history. Page built " + D.generated + ".";
const IS_REPORT = !!D.report;
const PG = D.pages.find((p) => p.key === D.group);
if ($("nav")) $("nav").innerHTML = D.pages.map((p) => '<a href="' + p.page + '"' + (!IS_REPORT && p.key === D.group ? ' aria-current="page"' : "") + ">" + esc(p.title) + "</a>").join("") +
  '<a href="' + PG.report + '"' + (IS_REPORT ? ' aria-current="page"' : "") + ">" + esc(PG.title) + ": detail report</a>";
if ($("h1")) $("h1").textContent = (IS_REPORT ? "Leverage Environment: " + PG.title + ", full detail" : "Leverage Environment: " + PG.title);
const listAnd = (xs) => xs.length < 2 ? xs.join("") : xs.slice(0, -1).join(", ") + " and " + xs[xs.length - 1];
if ($("alist")) $("alist").textContent = listAnd(A);
document.querySelectorAll("a.dash").forEach((el) => el.setAttribute("href", PG.page));
document.querySelectorAll("a.rep").forEach((el) => el.setAttribute("href", PG.report));
// Windows are set in trading days; assets that trade every day (bitcoin) use calendar days.
const CALA = A.filter((a) => D.assets[a].cal.kind === "calendar");
const calDays = (n) => Math.floor(n * 365 / 252 + 0.5);
const tdays = (n) => n + " trading days" + (CALA.length ? " (" + calDays(n) + " calendar days for " + listAnd(CALA) + ")" : "");
const nWord = (n) => ["no", "one", "two", "three", "four", "five", "six", "seven", "eight"][n] || String(n);
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
const DASHES = ["solid", "dash", "dot", "dashdot"];
// One ETF: implied leverage at each volatility budget under one cap (top), drawdown beneath, shared time axis.
function plotLevDrawdown(id, a, cap, on) {
  const [x0, x1] = xRange(), ink2 = css("--ink2"), grid = css("--grid"), axis = css("--axis"), band = css("--band"), col = css(COLOR[a]);
  const dates = D.series.dates, dd = D.series[a].dd.map((v) => v == null ? null : v * 100);
  let lo = -1; dates.forEach((d, j) => { if (d >= x0 && d <= x1 && dd[j] != null) lo = Math.min(lo, dd[j]); });
  const shapes = [{ type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: 1, y1: 1, line: { color: axis, width: 1 }, layer: "below" }];
  const start = D.assets[a].proxy ? D.assets[a].etf_start : null;
  if (start && start > x0) shapes.push({ type: "rect", xref: "x", yref: "paper", x0: x0, x1: start, y0: 0, y1: 1, fillcolor: band, line: { width: 0 }, layer: "below" });
  const yax = { gridcolor: grid, zeroline: false, fixedrange: !CAN_ZOOM, title: { font: { size: 11, color: ink2 }, standoff: 6 } };
  const traces = C.budgets.map((b, i) => ({ x: dates, y: D.series[a]["lev_" + b].map((v) => v == null ? null : Math.min(cap, v)),
    name: b + "% budget", mode: "lines", line: { color: col, width: 1.6, dash: DASHES[i] }, yaxis: "y", hovertemplate: "%{y:.2f}x<extra>" + b + "% budget</extra>",
    visible: !on || on[i] }));
  const nb = C.budgets.length;
  traces.push({ x: dates, y: D.series[a].lev_med.map((v) => v == null ? null : Math.min(cap, v)), name: "Own median", mode: "lines",
    line: { color: col, width: 1.4, dash: DASHES[nb] }, opacity: 0.85, yaxis: "y", hovertemplate: "%{y:.2f}x<extra>Own median</extra>", visible: !on || on[nb] });
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
// ---- documentation (shared by both pages of each group) -------------------------
const PROXY_NAME = { VEURX: "Vanguard European Stock Index", VPACX: "Vanguard Pacific Stock Index", VEIEX: "Vanguard Emerging Markets Stock Index",
  VFITX: "Vanguard Intermediate-Term Treasury", VUSTX: "Vanguard Long-Term Treasury" };
function renderDoc(extra) {
const EQ = D.group === "equity";
const bf = A.filter((a) => D.assets[a].proxy), staleA = bf.filter((a) => D.assets[a].stale);
const stale = (a) => { const d = D.assets[a]; return d.stale_ac_backfill == null ? "" : " For " + a + " the lag-1 autocorrelation of daily returns is " +
  num(d.stale_ac_backfill) + " in the backfilled era against " + num(d.stale_ac_etf) + " once the ETF trades."; };
const members = (a) => Object.entries(C.breadth[a]).map(([t, w]) => t + " " + w + "%").join(", ");
const starts = listAnd(A.map((a) => a + " on " + (D.assets[a].score_from || "n/a")));
const backfill = (a) => { const d = D.assets[a];
  if (!d.proxy) return "<li>" + a + " has no backfill: its history starts on " + d.etf_start + ".</li>";
  const tk = d.proxy.match(/[A-Z]{4,5}/g) || [], nm = tk.map((t) => (PROXY_NAME[t] || t) + " (" + t + ")");
  const corr = d.blend_corr == null ? "" : " (correlation " + num(d.blend_corr, 3) + ")";
  if (d.proxy_kind === "blend") return "<li>" + a + " began trading on " + d.etf_start + ". Earlier returns come from a fixed blend of " + listAnd(nm) + ", " + esc(d.proxy) +
    ", fitted to " + a + "'s monthly returns over the first five years of overlap" + corr + ".</li>";
  if (d.proxy_kind === "scaled") return "<li>" + a + " began trading on " + d.etf_start + ". Earlier returns come from the " + nm[0] + " fund, with its daily returns multiplied by " +
    num(d.scale) + " so that its monthly returns best match " + a + "'s over the first five years of overlap" + corr + ". The scale bridges the difference in interest-rate sensitivity " +
    "(duration) between the fund and the ETF; the fund's maturity mix drifted over time, so the match is approximate. Treasury fund prices are struck at the US close, so there is no stale-pricing " +
    "problem and the variance ratio uses the backfilled era.</li>";
  return "<li>" + a + " began trading on " + d.etf_start + ". Earlier returns come from the " + nm[0] + " fund, which then tracked a related index, close to but not the same as " + a + "'s.</li>"; };
const shiftEx = EQ ? "A market whose normal volatility has changed for good, for example as a market matured or its make-up changed, is judged partly against an era that no longer applies."
  : "Bitcoin's volatility has fallen a long way since its early years, so against its full history it can read as favorable for long stretches while still being far more volatile than any other asset here. " +
    "Treasury volatility was unusually low for much of 2010 to 2021 and has been higher since 2022, so recent Treasury readings are judged partly against that calmer era.";
$("doc").innerHTML = `
<h3>What this page is and is not</h3>
<ul>
<li>It shows how favorable conditions are for holding leverage in each asset, relative to that asset's own history. It does not say which asset to hold (that is left to your momentum system), it does not predict returns, and it is not investment advice. Leveraged positions can lose most of their value quickly in any environment.</li>
<li>All leveraged figures are simulations of an idealized fund. Real funds differ through swap pricing, tracking error, taxes, premiums and discounts, and their own fee schedules.</li>
</ul>
` + (CALA.length ? `
<h3>Bitcoin's calendar: calendar days, not trading days</h3>
<ul>
<li><strong>Bitcoin trades every day, including weekends and holidays, so its windows count calendar days and annualize with 365 days.</strong> Its volatility windows are ${calDays(C.vol[0])} and ${calDays(C.vol[1])} days, its variance ratios are at ${calDays(C.vr_q)} and ${calDays(C.vr_q_long)} days estimated over ${calDays(C.vr_window)} days, percentiles need ${calDays(C.min_rank)} observations, and its forward windows are ${calDays(C.horizon)} days. Each covers the same span of time as the trading-day window used for the other assets (${C.vol[0]}, ${C.vol[1]}, ${C.vr_window} and so on, annualized with 252), so the readings are comparable.</li>
<li>Its readings can be dated later than the other assets' because it keeps trading when the US market is closed. A day's bar is taken as complete only once that day has ended in UTC.</li>
<li>Leveraged bitcoin funds listed in the US reset only on US trading days, so a weekend move reaches them as one return. The simulations here reset every calendar day, which differs from such a fund.</li>
</ul>` : "") + `
<h3>The core measure: effective volatility</h3>
<ul>
<li>The variance of a return over a holding period of H days is H &times; daily variance &times; VR_H, where VR_H is the variance ratio at that horizon. In volatility terms: <strong>effective volatility = volatility measured from daily returns &times; &radic;VR_H</strong>. H is one month (${tdays(C.vr_q)}), matching sleeves rebalanced monthly. This is the risk a position held for a month actually carries; leverage multiplies it.</li>
<li>Volatility from daily returns is the root-mean-square of the 1-month and 3-month annualized readings (${C.vol[0]} and ${C.vol[1]} trading days${CALA.length ? "; " + calDays(C.vol[0]) + " and " + calDays(C.vol[1]) + " calendar days for " + listAnd(CALA) : ""}), so it reacts within weeks. It is backward-looking realized volatility, not a forecast, and it can jump faster than any window registers.</li>
<li>The variance ratio is the variance of overlapping one-month returns divided by the number of days in the month times the variance of daily returns, using the Lo&ndash;MacKinlay bias-corrected, demeaned estimator over about two years (${tdays(C.vr_window)}), so it moves slowly. Even so it is noisy: its standard error is about &plusmn;${num(C.vr_se)} when returns are independent, and larger when volatility clusters, which moves effective volatility by roughly &plusmn;${Math.round(C.vr_se * 50)}%. Volatility itself cancels out of the ratio; it measures only how daily moves add up.</li>
<li>Annualized effective volatility is monthly risk expressed on an annual scale (monthly volatility &times; &radic;12). It is not a statement about the variability of a full year's return.</li>
<li>No momentum or return forecast enters the measure. Volatility and serial correlation are each counted once.</li>
</ul>

<h3>Favorability and the five levels</h3>
<ul>
<li>Favorability = 100 minus the point-in-time percentile of effective volatility against the asset's own history: 100 is the calmest the asset has been, 0 the most turbulent. Each past day is ranked only against data that existed on that day, with at least two years of observations required, so favorability starts ${starts}.</li>
<li>Five levels with equal shares of history: very favorable at ${C.cuts[3]} or above, favorable from ${C.cuts[2]}, neutral between ${C.cuts[1]} and ${C.cuts[2]}, unfavorable at ${C.cuts[1]} or below, very unfavorable at ${C.cuts[0]} or below. Because the levels are quintiles of the asset's own history, they make no claim of equal risk and involve no fitting.</li>
<li>Because the measure is relative to each asset's own history, the same level on two assets does not mean the same absolute risk: ${EQ ? "EEM at a favorable level can still be more volatile than SPY at an unfavorable one" : "bitcoin at a favorable level is still several times as volatile as intermediate Treasuries at an unfavorable one"}. Absolute risk is in the effective-volatility figure and the implied-leverage grid.</li>
<li>Red lines on the favorability charts at ${C.ref_lines.join(", ")} are visual reference levels only.</li>
</ul>

<h3>Caution: "own history" grows over time</h3>
<ul>
<li><strong>Each asset is ranked against all of its own history up to each date (an expanding window), not a rolling one.</strong> That keeps every asset on the same footing and involves no choice of window length, but a lasting shift in what is normal for an asset is not absorbed: it shows up as long runs at one end of the scale. ${shiftEx}</li>
<li>Assets with short histories are judged against fewer episodes, so their percentiles are less settled and can shift as new extremes arrive. Ranking starts on ${listAnd(A.map((a) => a + " " + D.assets[a].rank_start))}. A rolling-window version may be added later for comparison.</li>
</ul>

<h3>The variance ratio as trend persistence</h3>
<ul>
<li>Below 1, daily moves have tended to reverse within the month, so a month's swings are smaller than daily volatility implies. Above 1, they have tended to continue, so swings are larger.${EQ ? " For these three ETFs it has been below 1 most of the time, reflecting the short-term reversals in index returns that Bessembinder (2026) discusses." : ""}</li>
<li>The one-month ratio enters favorability only through effective volatility, as a risk adjustment. Above 1 also means an existing trend is more likely to persist, which favors a trend-following position and a daily-reset fund (Bessembinder's rebalancing effect); that is a statement about returns, so it is shown but not scored either way.</li>
<li><strong>The three-month variance ratio is context only</strong> and does not enter favorability. It shows whether moves have tended to build or reverse over a quarter, a slower kind of trendiness than the one-month ratio captures. It uses the same two-year window, so it is noisier: its standard error is about &plusmn;${num(C.vr_se_long)}, so only readings well away from 1 say much.</li>
` + (staleA.length ? `<li>The variance ratios are left blank wherever their window includes stale-priced mutual-fund backfill, which inflates serial correlation, so effective volatility and favorability start about two years after ${listAnd(staleA)} began trading.</li>` : "") + `
</ul>

<h3>Leverage from a volatility budget</h3>
<ul>
<li>Implied leverage = budget &divide; effective volatility, then capped. Budgets are annualized on the same convention as effective volatility. Values below 1x mean holding part of the position in cash.</li>
<li>The same budgets and caps apply to every asset, so the grid compares absolute risk across assets. For low-volatility assets such as Treasuries most cells hit the caps; for bitcoin most fall below 1x.</li>
<li><strong>Own median</strong> is the leverage that would bring effective volatility to the asset's own median since ranking began (point-in-time): above 1x when the asset is calmer than usual, below 1x when it is more turbulent. It answers "how levered would I be if I held this asset's usual risk", independent of any budget.</li>
<li>Not captured: the drag and skew inside a daily-reset 3x fund (Bessembinder's median effect), financing costs, and sudden gaps. Realized volatility is backward-looking and understates risk just before a shock; the cap is the protection there.</li>
</ul>

<h3>Volatility drag</h3>
<ul>
<li>Drag &asymp; L(L&minus;1)/2 &times; effective volatility&sup2; per year: how much a position held at leverage L for a month gives up in its typical (compound, or median) outcome compared with L times the asset's own compound return. It does not lower the average outcome; it widens the gap between the average and the typical one, which is what compounding over many months experiences.</li>
<li>It uses effective volatility because each sleeve is held for a month between rebalances. Inside a daily-reset fund the drag depends on daily volatility instead, so it is larger by a factor of 1 &divide; VR (about 40% larger at a ratio of 0.7), which is the same rebalancing cost seen from another angle. With a 1x/3x mix only the 3x portion bears that.</li>
<li>Not included: financing costs and fees, trend (which can offset or outweigh the drag), and higher-order effects that matter in extreme volatility, when the approximation understates the damage.</li>
</ul>

<h3>Data</h3>
<ul>
<li>Prices are daily closes from Yahoo Finance, adjusted for dividends and splits, so every return is a total return. Yahoo revises adjusted history whenever a dividend is paid, so past readings can shift slightly between runs. ${EQ ? "If the three ETFs cannot be fetched or the data is more than a week old, the job fails instead of publishing." : "If an asset's data is missing or more than a week old, it is left off this page with a note; the equities page is unaffected."}</li>
<li>The assets are ${listAnd(A.map((a) => a + " (" + esc(D.assets[a].index) + ")"))}.</li>
<li>The pages are built overnight, at about 1am Pacific time, Tuesday to Saturday. A reading dated ${D.asof} uses that day's close.</li>
</ul>

<h3>Backfilled history</h3>
<ul>
${A.map(backfill).join("\n")}
` + (staleA.length ? `<li><strong>Stale pricing.</strong> In the backfilled era the equity funds were priced from foreign closing prices set hours before the US close, which manufactures positive day-to-day correlation and understates daily volatility.${staleA.map(stale).join("")} Those backfilled prices are used for volatility, momentum and drawdowns, but never for the variance ratio.</li>` : "") + `
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

<h3>Momentum${C.breadth && Object.keys(C.breadth).length ? " and breadth" : ""} (detail report only)</h3>
<ul>
<li>Shown for context and not part of favorability. Momentum is total return over 3 and 12 months (${C.mom[0]} and ${C.mom[1]} trading days${CALA.length ? "; " + calDays(C.mom[0]) + " and " + calDays(C.mom[1]) + " calendar days for " + listAnd(CALA) : ""}).` +
(C.breadth && Object.keys(C.breadth).length ? ` Breadth is the share of sector ETFs (SPY) or single-country ETFs (EFA, EEM) with a positive return over the same windows; these are coarse proxies, not the underlying stocks, and the index-weighted figure uses approximate fixed weights set by hand. Current sets: ${Object.keys(C.breadth).map((a) => a + ": " + members(a)).join(". ")}.` : " Breadth does not apply to these assets and is not shown.") + `</li>
</ul>

<h3>Leverage simulation and point-in-time check</h3>
<ul>
<li>Each day the simulated fund earns L times the asset's total return. After costs it also pays the cash rate plus ${(C.spread * 100).toFixed(2)}% on the borrowed (L-1) portion and a ${(C.fee * 100).toFixed(2)}% annual fee; the cash rate is the ${esc(D.cash_note)}, currently ${pct(D.cash_now, 2)}. Not modelled: intraday moves, tracking error, widening swap spreads, fund closures, and the fact that a 3x fund is wiped out by a single-day fall of about 33%.${EQ ? "" : " For Treasuries the financing cost matters most: a levered Treasury position earns roughly L times the bond's return minus the cash rate and spread on the borrowed part, so when short rates are at or above bond yields the carry is negative."}</li>
<li>The check uses only dates when favorability exists, with levels as they stood on each date, and looks ${tdays(C.horizon)} ahead. Adjacent days share almost all of their forward window, so the independent-period count is the honest sample size. The sample contains only a handful of major bear markets, and those few episodes dominate the unfavorable rows. Past simulated results do not indicate future results.</li>
</ul>` + (extra || "");
$("foot").textContent = "Generated " + D.generated + " · data through " + D.asof + " · source: Yahoo Finance via yfinance";
}
'''

DASH_BODY = r'''<div id="banner"></div>
<nav class="pages" id="nav" aria-label="Pages"></nav>
<h1 id="h1">Leverage Environment</h1>
<p class="sub">How favorable current conditions are for holding leverage in <span id="alist"></span>, relative to each asset's own history.
The core measure is effective volatility: the risk a position held for a month actually carries, combining daily volatility with how
daily moves add up over the month. Which asset to hold is left to your momentum system. <span id="asof"></span></p>
<p class="small">This is a monitoring tool built on approximations, not investment advice. Read
<a href="#method">Methodology and caveats</a> before relying on any number here.
Momentum and every result in full are in the <a class="rep" href="report.html">detail report</a>.</p>
<div id="notes"></div>
<div id="calnote"></div>

<div class="scroll" id="sumwrap"><table id="sumtab"></table></div>
<p class="small" id="sumnote"></p>

<h2>Favorability against drawdowns</h2>
<p class="sub">Favorability, 0 to 100 (top), above each asset's drawdown from its highest prior close (bottom), on the same time axis.
100 is the calmest the asset has been; red lines mark 20, 50 and 80.</p>
<div class="legend" id="legend"></div>
<p class="small" id="shade-note"></p>
<div id="ddcharts"></div>

<h2 id="vb">Leverage from a volatility budget</h2>
<p class="sub">The leverage that would hold a position's risk at a chosen budget:
<strong>implied leverage = budget &divide; effective volatility</strong>, where
<strong>effective volatility = volatility &times; &radic;VR (one month)</strong>, then capped. Values below 1x mean holding part of the position in cash.
"Own median" is the leverage that would bring the asset to its usual effective volatility.</p>
<div class="scroll"><table id="vbin"></table></div>
<div class="scroll"><table id="vbgrid"></table></div>
<div class="ctl"><span>Leverage cap on the charts</span><div class="seg" id="capbtns" role="group" aria-label="Leverage cap"></div>
<span>Show</span><div class="tog" id="vbtog" role="group" aria-label="Lines shown"></div><span id="vbrange" style="margin-left:auto"></span></div>
<p class="small" id="vbnote"></p>
<div id="vbcharts"></div>

<h2 id="drag">Volatility drag</h2>
<p class="sub">How much a leveraged position held for a month gives up each year, in its typical (compound) outcome, compared with
simply multiplying the asset's own compound return by the leverage: <strong>drag &asymp; L(L&minus;1)/2 &times; effective volatility&sup2;</strong>.
It grows with the square of volatility, so it is small in calm markets and large in turbulent ones.</p>
<div class="scroll"><table id="dragtab"></table></div>
<div class="ctl"><span>Show</span><div class="tog" id="dragtog" role="group" aria-label="Leverage levels shown"></div><span id="dragrange" style="margin-left:auto"></span></div>
<p class="small" id="dragnote"></p>
<div id="dragcharts"></div>

<h2>Readings against each asset's own history</h2>
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

DASH_JS = r'''// ---- summary table (sortable) ------------------------------------------------
const xf = (v) => v == null ? "n/a" : v.toFixed(2) + "x";
if (CALA.length) $("calnote").innerHTML = '<div class="callout"><strong>Bitcoin runs on calendar days.</strong> ' + listAnd(CALA) +
  " trades every day, so its windows count calendar days (" + calDays(C.vol[0]) + " and " + calDays(C.vol[1]) + "-day volatility, a " + calDays(C.vr_q) +
  "-day variance ratio over " + calDays(C.vr_window) + " days) and annualize with 365 days. The other assets use trading days (" + C.vol[0] + ", " + C.vol[1] + " and " +
  C.vr_window + ") and 252. Both cover the same spans of time. See the <a href=\"#method\">caveats</a>.</div>";
const LRANK = (l) => l ? LEVELS.indexOf(l) : null;
const SUMCOLS = [
  { h: "Asset", sort: (a) => A.indexOf(a), cell: (a, d) => '<div class="tkr"><i class="sw" style="background:var(' + COLOR[a] + ')"></i>' + a + '</div><div class="nm2">' + esc(d.name) + "</div>" },
  { h: "Level", sort: (a, d) => LRANK(d.label), asc: true, cell: (a, d) => d.label ? '<span class="nw"><span class="ico ' + d.label + '">' + ICON[d.label] + "</span> " + lname(d.label) + "</span>" : "n/a" },
  { h: "Favorability", n: 1, sort: (a, d) => d.score, cell: (a, d) => d.score == null ? "n/a" : "<strong>" + d.score.toFixed(0) + "</strong>" },
  { h: "Month ago", n: 1, sort: (a, d) => d.score_prev, cell: (a, d) => d.score_prev == null ? "n/a" : d.score_prev.toFixed(0) },
  { h: "Change", n: 1, sort: (a, d) => d.score == null || d.score_prev == null ? null : d.score - d.score_prev,
    cell: (a, d) => d.score == null || d.score_prev == null ? "n/a" : ((d.score - d.score_prev) > 0 ? "+" : "") + (d.score - d.score_prev).toFixed(0) },
  { h: "Effective vol", t: "Effective volatility: blended volatility × √(1-month variance ratio), annualized", n: 1, sort: (a, d) => d.vb.eff_vol, cell: (a, d) => "<strong>" + pct(d.vb.eff_vol) + "</strong>" },
  { h: "Blended vol", t: "Root-mean-square of 1- and 3-month volatility", n: 1, sort: (a, d) => d.vb.base_vol, cell: (a, d) => pct(d.vb.base_vol) },
  { h: "Vol 1m", t: "Volatility over 1 month (21 trading days; 30 calendar days for bitcoin)", n: 1, sort: (a, d) => d.vb.vol_21, cell: (a, d) => pct(d.vb.vol_21) },
  { h: "Vol 3m", t: "Volatility over 3 months (63 trading days; 91 calendar days for bitcoin)", n: 1, sort: (a, d) => d.vb.vol_63, cell: (a, d) => pct(d.vb.vol_63) },
  { h: "VR 1m", t: "Variance ratio at a 1-month horizon (enters effective volatility)", n: 1, sort: (a, d) => d.vb.vr21, cell: (a, d) => num(d.vb.vr21) },
  { h: "VR 3m", t: "Variance ratio at a 3-month horizon (context only, not scored)", n: 1, sort: (a, d) => d.vb.vr63, cell: (a, d) => num(d.vb.vr63) },
].concat(C.budgets.map((b) => ({ h: "Lev " + b + "%", t: "Implied leverage at a " + b + "% volatility budget, uncapped", n: 1, sort: (a, d) => d.vb.lev[b], cell: (a, d) => xf(d.vb.lev[b]) })), [
  { h: "Lev median", t: "Leverage that would bring effective volatility to the asset's own median, uncapped", n: 1, sort: (a, d) => d.vb.lev_med, cell: (a, d) => xf(d.vb.lev_med) },
  { h: "Drawdown", t: "Decline from the highest prior close", n: 1, sort: (a, d) => d.dd, cell: (a, d) => pct(d.dd) },
].concat(LATER.length ? [{ h: "Data through", sort: (a, d) => d.asof, cell: (a, d) => '<span class="nw">' + d.asof + "</span>" }] : []));
let sumSort = { col: 0, dir: 1 };
function drawSummary() {
  const rows = A.slice().sort((x, y) => {
    const c = SUMCOLS[sumSort.col], vx = c.sort(x, D.assets[x]), vy = c.sort(y, D.assets[y]);
    if (vx == null && vy == null) return A.indexOf(x) - A.indexOf(y);
    if (vx == null) return 1;
    if (vy == null) return -1;
    return (vx < vy ? -1 : vx > vy ? 1 : A.indexOf(x) - A.indexOf(y)) * sumSort.dir;
  });
  $("sumtab").innerHTML = "<thead><tr>" + SUMCOLS.map((c, i) => {
    const on = i === sumSort.col, aria = on ? ' aria-sort="' + (sumSort.dir > 0 ? "ascending" : "descending") + '"' : "";
    return "<th" + (c.n ? ' class="n"' : "") + aria + '><button class="sort" data-i="' + i + '"' + (c.t ? ' title="' + esc(c.t) + '"' : "") + ">" + c.h + '<span class="ar" aria-hidden="true">' +
      (on ? (sumSort.dir > 0 ? "▲" : "▼") : "") + "</span></button></th>"; }).join("") + "</tr></thead><tbody>" +
    rows.map((a) => "<tr>" + SUMCOLS.map((c) => "<td" + (c.n ? ' class="n"' : "") + ">" + c.cell(a, D.assets[a]) + "</td>").join("") + "</tr>").join("") + "</tbody>";
  $("sumtab").querySelectorAll("button.sort").forEach((b) => b.addEventListener("click", () => {
    const i = +b.dataset.i, c = SUMCOLS[i];
    // First click: the natural order for that column (best level first, highest number first).
    sumSort = i === sumSort.col ? { col: i, dir: -sumSort.dir } : { col: i, dir: c.n ? -1 : 1 };
    drawSummary();
    $("sumtab").querySelector('button.sort[data-i="' + i + '"]').focus();
  }));
}
drawSummary();
$("sumnote").textContent = "Select a column heading to sort; select it again to reverse. Favorability is 0 to 100 against the asset's own history. Vol = volatility, annualized; " +
  "VR = variance ratio; Lev = implied leverage at a volatility budget or at the asset's own median effective volatility. Variance ratios have standard errors of about ±" + num(C.vr_se) + " (1-month) and ±" + num(C.vr_se_long) + " (3-month); the 3-month ratio is context only. " +
  "Leverage figures are uncapped; the caps are applied in the grid below.";

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
  vr63: "context only: whether moves have built (above 1) or reversed (below 1) over a quarter",
};
$("grid").innerHTML = "<thead><tr><th>Metric</th>" + A.map((a) => "<th>" + a + "</th>").join("") + "</tr></thead><tbody>" +
  D.metrics.filter((m) => m.core || m.key === "vr63").map((m) =>
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
  "daily-reset fund did over the following " + tdays(C.horizon) + ". If favorability is useful, risk should fall step by step from very unfavorable to very favorable: " +
  "watch the worst-5% and compounding columns more than the medians. Windows overlap heavily, so the last column gives the rough count of independent periods.";
$("bt").innerHTML = '<thead><tr><th>Asset</th><th>Level</th><th class="n">Share of days</th><th class="n">Median 1x</th>' +
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
  { key: "eff_vol", title: "Effective volatility", note: "Blended volatility × √(1-month variance ratio), annualized", pct: true },
  { key: "vr21", title: "Variance ratio, 1-month", note: "Bias-corrected, two-year window; red line at 1 (no serial correlation); below 1 = reversals within the month", red: [1] },
  { key: "vr63", title: "Variance ratio, 3-month (context only)", note: "Same estimator and window at a 3-month horizon; not part of favorability; red line at 1; standard error about ±" + num(C.vr_se_long), red: [1] },
  { key: "base_vol", title: "Volatility, blended", note: "Root-mean-square of the 1- and 3-month readings", pct: true },
  { key: "vol_21", title: "Volatility, 1-month", note: "Annualized standard deviation of daily returns over " + tdays(C.vol[0]), pct: true },
  { key: "vol_63", title: "Volatility, 3-month", note: "Annualized standard deviation of daily returns over " + tdays(C.vol[1]), pct: true },
];
$("legend").innerHTML = legendHtml();
$("shade-note").textContent = shadeNote();
$("charts").innerHTML = CHARTS.map((c, i) => '<div class="chart"><h3>' + c.title + "</h3><p>" + c.note + '</p><div class="plot" id="pl' + i + '"></div></div>').join("");

// ---- volatility-budget leverage ------------------------------------------
$("vbin").innerHTML = '<thead><tr><th>Asset</th><th class="n">1-month volatility</th><th class="n">3-month volatility</th><th class="n">Blended volatility</th>' +
  '<th class="n">Variance ratio, 1-month</th><th class="n">Effective volatility</th><th class="n">Effective volatility a month ago</th><th class="n">Own median effective volatility</th></tr></thead><tbody>' +
  A.map((a) => { const v = D.assets[a].vb; return "<tr><td><strong>" + a + '</strong></td><td class="n">' + pct(v.vol_21) + '</td><td class="n">' + pct(v.vol_63) +
    '</td><td class="n">' + pct(v.base_vol) + '</td><td class="n">' + num(v.vr21) + " ± " + num(D.assets[a].cal.vr_se) + '</td><td class="n"><strong>' + pct(v.eff_vol) +
    '</strong></td><td class="n">' + pct(v.eff_vol_prev) + '</td><td class="n">' + pct(v.eff_vol_med) + "</td></tr>"; }).join("") + "</tbody>";
$("vbgrid").innerHTML = '<thead><tr><th>Asset</th><th>Volatility budget</th><th class="n">Uncapped</th>' + C.caps.map((c) => '<th class="n">Cap ' + c + "x</th>").join("") +
  '<th class="n">Uncapped a month ago</th></tr></thead><tbody>' +
  A.flatMap((a) => C.budgets.concat(["med"]).map((b, i) => { const v = D.assets[a].vb, med = b === "med";
    const L = med ? v.lev_med : v.lev[b], P = med ? v.lev_med_prev : v.lev_prev[b];
    return "<tr>" + (i ? "<td></td>" : "<td><strong>" + a + "</strong></td>") + "<td>" + (med ? "Own median (" + pct(v.eff_vol_med) + ")" : b + "%") + '</td><td class="n">' + xf(L) + "</td>" +
      C.caps.map((c) => L == null ? '<td class="n">n/a</td>' : (L >= c ? '<td class="n cap">' + xf(c) + " (cap)</td>" : '<td class="n"><strong>' + xf(L) + "</strong></td>")).join("") +
      '<td class="n">' + xf(P) + "</td></tr>"; })).join("") + "</tbody>";
let levCap = C.caps[C.caps.length - 1];
$("capbtns").innerHTML = C.caps.map((c) => '<button data-cap="' + c + '" aria-pressed="' + (c === levCap) + '">' + c + "x</button>").join("");
$("vbrange").innerHTML = rangeBtns();
$("vbnote").textContent = "Use the Show buttons to turn lines on or off on all " + nWord(A.length) + " charts; the grey line marks 1x. " +
  "The variance ratio needs about two years of prices (" + tdays(C.vr_window) + ")" + (A.some((a) => D.assets[a].stale) ? ", of the ETF's own prices where the backfill is stale-priced," : "") +
  " so the lines start later than the price history. Weekly samples.";
$("vbcharts").innerHTML = A.map((a) => '<div class="chart"><h3><i class="sw" style="background:var(' + COLOR[a] + ');margin-right:6px"></i>' + a +
  ': implied leverage and drawdown</h3><p>Implied leverage at each budget under the selected cap (top); decline from the highest prior close (bottom)</p><div class="plot tall" id="vb' + a + '"></div></div>').join("");
document.querySelectorAll("#capbtns button").forEach((btn) => btn.addEventListener("click", () => {
  levCap = +btn.dataset.cap;
  document.querySelectorAll("#capbtns button").forEach((o) => o.setAttribute("aria-pressed", String(o === btn)));
  if (window.Plotly) A.forEach((a) => plotLevDrawdown("vb" + a, a, levCap, vbOn));
}));
// ---- legend toggles (shared across the three ETF charts) -------------------
function toggles(el, labels, state, redraw) {
  const dash = ["", "5 3", "1.5 2.5", "7 2 1.5 2"];
  $(el).innerHTML = labels.map((t, i) => '<button data-i="' + i + '" aria-pressed="' + state[i] + '"><svg width="22" height="8" aria-hidden="true">' +
    '<line x1="1" x2="21" y1="4" y2="4" stroke="currentColor" stroke-width="2" stroke-dasharray="' + dash[i] + '"/></svg>' + t + "</button>").join("");
  $(el).querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    const i = +b.dataset.i; state[i] = !state[i]; b.setAttribute("aria-pressed", String(state[i])); if (window.Plotly) redraw();
  }));
}
const vbOn = C.budgets.map(() => true).concat([true]);
toggles("vbtog", C.budgets.map((b) => b + "% budget").concat(["Own median"]), vbOn, () => A.forEach((a) => plotLevDrawdown("vb" + a, a, levCap, vbOn)));

// ---- volatility drag ------------------------------------------------------
const DRAG_L = [1.5, 2, 3];
const dragOn = DRAG_L.map(() => true);
toggles("dragtog", DRAG_L.map((L) => L + "x"), dragOn, () => A.forEach((a) => plotDrag("dg" + a, a)));
$("dragrange").innerHTML = rangeBtns();
const drag = (ev, L) => ev == null ? null : L * (L - 1) / 2 * ev * ev;
$("dragtab").innerHTML = '<thead><tr><th>Asset</th><th class="n">Effective volatility</th>' + DRAG_L.map((L) => '<th class="n">Drag at ' + L + "x</th>").join("") +
  DRAG_L.map((L) => '<th class="n">' + L + "x a month ago</th>").join("") + "</tr></thead><tbody>" +
  A.map((a) => { const v = D.assets[a].vb; return "<tr><td><strong>" + a + '</strong></td><td class="n">' + pct(v.eff_vol) + "</td>" +
    DRAG_L.map((L) => '<td class="n"><strong>' + pct(drag(v.eff_vol, L)) + "</strong></td>").join("") +
    DRAG_L.map((L) => '<td class="n">' + pct(drag(v.eff_vol_prev, L)) + "</td>").join("") + "</tr>"; }).join("") + "</tbody>";
$("dragnote").textContent = "Per year. Use the Show buttons to turn leverage levels on or off on all " + nWord(A.length) + " charts; the scale adjusts to the lines shown. " +
  "Based on effective volatility, so it applies to a position held for a month between rebalances; see the caveats for how it differs inside a daily-reset fund. Weekly samples.";
$("dragcharts").innerHTML = A.map((a) => '<div class="chart"><h3><i class="sw" style="background:var(' + COLOR[a] + ');margin-right:6px"></i>' + a +
  ': volatility drag</h3><p>Yearly drag on the typical outcome at ' + DRAG_L.join("x, ").replace(/, ([^,]*)$/, " and $1") + 'x leverage</p><div class="plot" id="dg' + a + '"></div></div>').join("");
function plotDrag(id, a) {
  const [x0, x1] = xRange(), ink2 = css("--ink2"), grid = css("--grid"), axis = css("--axis"), band = css("--band"), col = css(COLOR[a]);
  const dates = D.series.dates, ev = D.series[a].eff_vol;
  let hi = 0;
  const traces = DRAG_L.map((L, i) => {
    const y = ev.map((v) => v == null ? null : drag(v, L) * 100);
    if (dragOn[i]) dates.forEach((d, j) => { if (d >= x0 && d <= x1 && y[j] != null) hi = Math.max(hi, y[j]); });
    return { x: dates, y, name: L + "x", mode: "lines", line: { color: col, width: 1.6, dash: DASHES[i] }, hovertemplate: "%{y:.1f}%<extra>" + L + "x</extra>", visible: dragOn[i] };
  });
  const shapes = [];
  const start = D.assets[a].proxy ? D.assets[a].etf_start : null;
  if (start && start > x0) shapes.push({ type: "rect", xref: "x", yref: "paper", x0: x0, x1: start, y0: 0, y1: 1, fillcolor: band, line: { width: 0 }, layer: "below" });
  Plotly.react(id, traces, {
    margin: { l: 52, r: 12, t: 8, b: 26 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", showlegend: false,
    hovermode: "x unified", hoverlabel: { bgcolor: css("--surface"), bordercolor: axis, font: { color: css("--ink"), size: 12 } },
    font: { family: 'system-ui,-apple-system,"Segoe UI",sans-serif', size: 11, color: ink2 },
    xaxis: { range: [x0, x1], hoverformat: "%b %-d, %Y", showgrid: false, linecolor: axis, tickcolor: axis, fixedrange: !CAN_ZOOM },
    yaxis: { range: [0, (hi || 1) * 1.07], gridcolor: grid, zeroline: false, ticksuffix: "%", fixedrange: !CAN_ZOOM },
    dragmode: CAN_ZOOM ? "zoom" : false, shapes,
  }, { displayModeBar: false, responsive: true, doubleClick: "reset" });
}
$("ddcharts").innerHTML = A.map((a) => '<div class="chart"><h3><i class="sw" style="background:var(' + COLOR[a] + ');margin-right:6px"></i>' + a +
  ': favorability and drawdown</h3><p>Favorability, 0 to 100 (top); decline from the highest prior close, worst close each week (bottom)</p><div class="plot tall" id="dd' + a + '"></div></div>').join("");
function draw() {
  if (!window.Plotly) { $("charts").innerHTML = NO_PLOTLY; $("ddcharts").innerHTML = ""; $("vbcharts").innerHTML = ""; $("dragcharts").innerHTML = ""; return; }
  A.forEach((a) => plotScoreDrawdown("dd" + a, a));
  A.forEach((a) => plotDrag("dg" + a, a));
  A.forEach((a) => plotLevDrawdown("vb" + a, a, levCap, vbOn));
  CHARTS.forEach((c, i) => plotLines("pl" + i, c, (a) => D.series[a][c.key]));
}
bindRange(draw);
draw();
renderDoc("");
'''

REPORT_BODY = r'''<div id="banner"></div>
<nav class="pages" id="nav" aria-label="Pages"></nav>
<h1 id="h1">Leverage Environment: full detail</h1>
<p class="sub">Every result behind the <a class="dash" href="index.html">dashboard</a> for <span id="alist"></span>: all readings with their history,
momentum for context, the leverage simulation over several lookbacks, and how the favorability levels and each metric have lined up
with later leveraged returns. <span id="asof"></span></p>
<p class="small">This is a monitoring tool built on approximations, not investment advice. Read
<a href="#method">Methodology and caveats</a> before relying on any number here.</p>
<div id="notes"></div>
<nav class="toc" aria-label="Sections"><a href="#s-sum">Summary</a><a href="#s-full">All readings</a><a href="#s-ex">History explorer</a>
<a href="#s-comps" class="br">Breadth by component</a><a href="#s-lev">Leverage simulation</a><a href="#s-bth">Levels by horizon</a>
<a href="#s-btb">Favorability bands</a><a href="#s-btm">Each metric alone</a><a href="#s-chg">Level changes</a><a href="#s-cov">Data coverage</a>
<a href="#method">Methodology and caveats</a></nav>

<h2 id="s-sum">Summary</h2>
<p class="sub" id="sum-intro"></p>
<div class="scroll"><table id="sum"></table></div>

<h2 id="s-full">All readings</h2>
<p class="sub">Today's value with its percentile, where it stood one month, three months and one year ago, and the
spread of its own history. The first five rows are the core measure and its inputs; the rows below them are context and do not enter favorability.
Twelve-month momentum is reported as a quintile because its history holds few independent observations.</p>
<div id="full"></div>

<h2 id="s-ex">History explorer</h2>
<p class="sub">Any series over time. For the readings in the table above you can switch between the raw value and its point-in-time percentile.</p>
<div class="ctl"><label for="sel">Series</label><select id="sel"></select>
<div class="seg" role="group" aria-label="Show"><button id="bv" aria-pressed="true">Value</button><button id="bp" aria-pressed="false">Percentile</button></div></div>
<div class="legend" id="legend"></div>
<p class="small" id="shade-note"></p>
<div class="chart"><div class="plot tall" id="ex"></div></div>

<section id="comps-sec">
<h2 id="s-comps">Breadth by component</h2>
<p class="sub">The individual sector and country ETFs behind each breadth reading. Weights are approximate and fixed (see caveats).</p>
<div id="comps"></div>
</section>

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
<p class="sub">Every series fetched for this page and how far back it goes. Check here that the backfill funds reach as far as expected.</p>
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
const BR = R.has_breadth;
const dayWord = (a) => D.assets[a].cal.kind === "calendar" ? "days" : "trading days";
const etfHead = (a) => '<h3 class="etf"><i class="sw" style="background:var(' + COLOR[a] + ')"></i>' + a + ' <span class="small">' + D.assets[a].name + " · " + D.assets[a].index + "</span></h3>";
const first = (a, i) => i ? "" : "<strong>" + a + "</strong>";
const f0 = (x) => x == null ? "n/a" : x.toFixed(0);

$("sum-intro").textContent = "Favorability is 100 minus the percentile of effective volatility against the asset's own history. Momentum" + (BR ? " and breadth are" : " is") +
  " shown for context only and does not enter it.";
tbl("sum", [["Asset"], ["Level"], ["Favorability", 1], ["Effective volatility", 1], ["Variance ratio, 1-month", 1], ["Variance ratio, 3-month (context)", 1], ["Days at this level", 1],
  ["3-month momentum", 1], ["12-month momentum", 1]].concat(BR ? [["Breadth: up over 3 months", 1], ["Breadth: up over 12 months", 1]] : []),
  A.map((a) => { const d = D.assets[a], c = d.cells; return ["<strong>" + a + "</strong>", lab(d.label), f0(d.score), pct(c.eff_vol.value), num(c.vr21.value), num(c.vr63.value),
    R.assets[a].streak, spct(c.mom_3m.value), spct(c.mom_12m.value)].concat(BR ? [pct(c.b3.value, 0), pct(c.b12.value, 0)] : []); }));
if (!BR) { $("comps-sec").remove(); document.querySelector("nav.toc a.br").remove(); }

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
    zero: m.key.startsWith("mom") ? 0 : null, red: m.key.startsWith("vr") ? [1] : null })))
  .concat(BR ? [
    { key: "b12w", label: "Breadth: up over 12 months, index weight", pct: true },
    { key: "b3w", label: "Breadth: up over 3 months, index weight", pct: true }] : [], [
    { key: "und_1y", label: "Unlevered return, trailing year", pct: true, zero: 0 },
    { key: "dd", label: "Drawdown from prior high (worst close each week)", pct: true, zero: 0 },
  ], C.budgets.map((b) => ({ key: "lev_" + b, label: "Implied leverage at a " + b + "% volatility budget (uncapped)", zero: 1 })),
  [{ key: "lev_med", label: "Leverage at own median effective volatility (uncapped)", zero: 1 }], C.leverage.flatMap((L) => [
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

if (BR) $("comps").innerHTML = A.map((a) => { const d = D.assets[a], cs = R.assets[a].components;
  return etfHead(a) + '<p class="small">' + cs.filter((c) => c.m12 > 0).length + " of " + cs.filter((c) => c.m12 != null).length + " up over 12 months; " +
    cs.filter((c) => c.m3 > 0).length + " of " + cs.filter((c) => c.m3 != null).length + " up over 3 months.</p>" +
    tbl(null, [["Component"], ["Ticker"], ["Approx. weight", 1], ["3-month return", 1], ["12-month return", 1], ["Data since"]],
      cs.map((c) => [esc(c.name), c.ticker, c.weight + "%", c.since ? spct(c.m3) : "no data", c.since ? spct(c.m12) : "no data", c.since || "n/a"]), true); }).join("");

tbl("levh", [["Asset"], ["Lookback"], ["Unlevered", 1]].concat(C.leverage.flatMap((L) => [[L + "x simple multiple", 1], [L + "x simulated, before costs", 1], [L + "x simulated, after costs", 1]])),
  A.flatMap((a) => R.assets[a].lev.map((r, i) => [first(a, i), r.label, spct(r.und)].concat(
    C.leverage.flatMap((L) => [spct(r["naive_" + L]), spct(r["gross_" + L]), spct(r["net_" + L])])))));

const HS = R.horizons;
$("bth-intro").textContent = "For every past day with a favorability reading, the level as it stood that day against the simulated " + TOP +
  "x fund over the following " + HS.join(", ") + " trading days" +
  (CALA.length ? " (" + HS.map(calDays).join(", ") + " calendar days for " + listAnd(CALA) + ")" : "") + ". Each horizon shows the median return after costs and the median compounding effect " +
  "(simulated return before costs minus " + TOP + " times the unlevered return). If favorability is useful, risk should fall step by step from very unfavorable to very favorable at every horizon.";
tbl("bth", [["Asset"], ["Level"], ["Days", 1]].concat(HS.flatMap((h) => [[h + " days: " + TOP + "x after costs", 1], [h + " days: effect", 1]])),
  A.flatMap((a) => LEVELS.map((l, i) => [first(a, i), lab(l), (R.assets[a].by_horizon[String(C.horizon)][l].days || 0).toLocaleString()].concat(
    HS.flatMap((h) => { const r = R.assets[a].by_horizon[String(h)][l]; return [spct(r.net), spct(r.gap)]; })))));

$("btb-intro").textContent = "The same check by favorability band (deciles) over the following " + tdays(C.horizon) + ", to show whether outcomes improve steadily with favorability " +
  "or only at the extremes. Independent periods is the number of days divided by the horizon: the honest sample size.";
tbl("btb", [["Asset"], ["Favorability"], ["Share of days", 1], ["Median 1x", 1], ["Median " + TOP + "x after costs", 1], ["Median " + TOP + "x compounding effect", 1],
  [TOP + "x positive", 1], [TOP + "x worst 5%", 1], ["Independent periods", 1]],
  A.flatMap((a) => R.assets[a].bands.map((r, i) => [first(a, i), r.band, pct(r.share, 0), spct(r.fwd_1x), spct(r.net), spct(r.gap), pct(r.win, 0), spct(r.p5), r.windows])));

$("btm-intro").textContent = "Each metric taken alone, including the context metrics: days are split into thirds by that metric's percentile as it stood at the time (oriented so the favorable third is the one better for leverage), and compared on the following " +
  tdays(C.horizon) + ". The spread is the favorable third minus the unfavorable third; a metric that carries information should show a positive spread on every asset.";
tbl("btm", [["Asset"], ["Metric"], [TOP + "x after costs: favorable third", 1], ["Middle third", 1], ["Unfavorable third", 1], ["Spread", 1],
  ["Compounding effect: favorable third", 1], ["Middle third", 1], ["Unfavorable third", 1], ["Spread", 1]],
  A.flatMap((a) => R.assets[a].alone.map((r, i) => { const d = (k) => r.fav[k] == null || r.unfav[k] == null ? null : r.fav[k] - r.unfav[k];
    return [first(a, i), r.label, spct(r.fav.net), spct(r.mid.net), spct(r.unfav.net), spct(d("net")), spct(r.fav.gap), spct(r.mid.gap), spct(r.unfav.gap), spct(d("gap"))]; })));

$("chg").innerHTML = A.map((a) => { const r = R.assets[a], d = D.assets[a];
  return etfHead(a) + '<p class="small">' + lname(d.label) + " for the last " + r.streak + " " + dayWord(a) + ".</p>" +
    (r.changes.length ? tbl(null, [["Date"], ["From"], ["To"]], r.changes.map((c) => [c.date, lab(c.from), lab(c.to)]), true) : '<p class="small">No level changes in the past year.</p>'); }).join("");

tbl("cov", [["Ticker"], ["Used for"], ["First date"], ["Last date"], ["Days with data", 1]],
  R.coverage.map((c) => ["<strong>" + c.ticker + "</strong>", esc(c.role), c.first || "no data", c.last || "no data", c.n.toLocaleString()]));

renderDoc(`
<h3>Reading the detail report</h3>
<ul>
<li>The tables that compare a level, a favorability band or a single metric with later returns slice one history many ways. With this many slices, some will look meaningful by chance. Give weight only to patterns that hold across assets and across horizons.</li>
<li>Longer horizons have fewer independent periods: divide the day count by the horizon. A ${HS[HS.length - 1]}-day result over a 20-year history rests on roughly 20 independent periods split across the levels.</li>
<li>Forward-return tables for favorability use only dates when it exists, which for assets with short histories or stale-priced backfill is well after they began trading. The single-metric table uses dates after each ETF began trading; volatility percentiles there still rank against the full history, including any backfilled era.</li>
${D.group === "equity" ? "" : "<li>Bitcoin's and gold's histories are short: a one-year result rests on only a handful of independent periods per level.</li>"}
<li>In the multi-year leverage table, the simple multiple is a reference point, not something obtainable: over long periods compounding makes a daily-reset fund's return diverge widely from it in either direction.</li>
<li>Data requests start at ${R.fetch_start}, so a first date equal to the first trading day of that year means the series is older than shown.</li>
<li>History charts and the explorer use weekly samples (the last observation of each week ending Friday), so brief spikes within a week may not appear. Tables use daily data.</li>
</ul>`);
'''

if __name__ == "__main__":
    main()
