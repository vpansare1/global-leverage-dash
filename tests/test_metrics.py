"""Checks each metric against a case whose answer is known in advance."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dashboard as d  # noqa: E402

IDX = pd.bdate_range("2000-01-03", periods=6000)


def test_momentum_of_steady_growth():
    price = pd.Series(100 * 1.001 ** np.arange(600), IDX[:600])
    assert d.momentum(price, 21).iloc[-1] == pytest.approx(1.001 ** 21 - 1)
    assert d.momentum(price, 252).iloc[-1] == pytest.approx(1.001 ** 252 - 1)
    assert math.isnan(d.momentum(price, 252).iloc[251])


def test_realized_vol_recovers_known_volatility():
    rng = np.random.default_rng(0)
    ret = pd.Series(rng.normal(0, 0.20 / math.sqrt(252), 6000), IDX)
    assert d.realized_vol(ret, 6000).iloc[-1] == pytest.approx(0.20, rel=0.03)


def test_expanding_percentile_is_point_in_time():
    s = pd.Series(np.arange(1000.0), IDX[:1000])          # always a new high
    p = d.expanding_percentile(s, min_obs=100)
    assert p.iloc[-1] == 100.0 and p.iloc[500] == 100.0
    assert p.iloc[:99].isna().all()
    changed = s.copy()
    changed.iloc[600:] = -1.0                              # altering the future...
    assert d.expanding_percentile(changed, 100).iloc[500] == p.iloc[500]   # ...changes nothing earlier


def test_leverage_compounding_against_hand_calculation():
    up = pd.Series([0.01] * 252, IDX[:252])                # steady trend helps
    assert d.compound(d.leveraged_returns(up, 2), 252).iloc[-1] == pytest.approx(1.02 ** 252 - 1)
    assert 1.02 ** 252 - 1 > 2 * (1.01 ** 252 - 1)
    chop = pd.Series([0.02, -0.02] * 126, IDX[:252])       # chop hurts
    und = d.compound(chop, 252).iloc[-1]
    lev = d.compound(d.leveraged_returns(chop, 3), 252).iloc[-1]
    assert lev == pytest.approx((1.06 * 0.94) ** 126 - 1)
    assert lev < 3 * und


def test_wipeout_is_permanent():
    ret = pd.Series([0.0, -0.37, 0.5, 0.5, 0.5, 0.0], IDX[:6])        # a 37% fall, then a rally
    lev = d.leveraged_returns(ret, 3)                                   # -111% on the bad day
    c = d.compound(lev, 3)
    assert c.iloc[2] == -1.0 and c.iloc[3] == -1.0                      # every window holding that day
    assert c.iloc[5] == pytest.approx(2.5 ** 2 - 1)                      # windows after it are unaffected
    assert d.compound(d.leveraged_returns(ret, 2), 3).iloc[3] == pytest.approx(0.26 * 2 * 2 - 1)


def test_financing_cost_and_fee_are_charged():
    flat = pd.Series([0.0] * 252, IDX[:252])
    net = d.compound(d.leveraged_returns(flat, 3, cash_rate=0.04, spread=0.005, fee=0.0095), 252).iloc[-1]
    assert net == pytest.approx((1 - (2 * 0.045 + 0.0095) / 252) ** 252 - 1)


def test_vol_drag_matches_formula_and_simulation():
    assert d.vol_drag(0.15, 2) == pytest.approx(0.0225)
    assert d.vol_drag(0.30, 3) == pytest.approx(0.27)
    rng = np.random.default_rng(3)                         # zero-drift returns, 20% volatility
    r = pd.Series(rng.normal(0, 0.20 / math.sqrt(252), 6000), IDX)
    years = 6000 / 252
    growth = lambda x: np.log1p(x).sum() / years           # noqa: E731
    simulated = 3 * growth(r) - growth(3 * r)
    assert simulated == pytest.approx(d.vol_drag(r.std() * math.sqrt(252), 3), rel=0.15)


def test_breadth_counts_and_weights():
    px = pd.DataFrame({"A": [1, 2.0], "B": [1, 2.0], "C": [1, 0.5], "D": [1, 0.5]}, index=IDX[:2])
    b = d.breadth(px, {"A": 70, "B": 10, "C": 10, "D": 10}, days=1, min_members=2).iloc[-1]
    assert b["equal"] == pytest.approx(0.5) and b["weighted"] == pytest.approx(0.8)
    assert math.isnan(d.breadth(px, {}, days=1, min_members=5).iloc[-1]["equal"])


def test_splice_follows_proxy_then_etf():
    idx = IDX[:6]
    etf = pd.Series([np.nan, np.nan, 100, 110, 121, 133.1], idx)
    proxy_ret = pd.Series([np.nan, 0.05, 0.05, 0.99, 0.99, 0.99], idx)   # ignored after ETF start
    s = d.splice(etf, proxy_ret)
    assert s.iloc[2:].tolist() == pytest.approx([100, 110, 121, 133.1])
    assert s.iloc[1] == pytest.approx(100 / 1.05) and s.iloc[0] == pytest.approx(100 / 1.05 ** 2)


def test_blend_weight_is_recovered():
    rng = np.random.default_rng(4)
    n = 2000
    a = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), IDX[:n])
    b = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), IDX[:n])
    target = (1 + 0.65 * a.pct_change().fillna(0) + 0.35 * b.pct_change().fillna(0)).cumprod()
    w, corr = d.fit_blend_weight(target, a, b)
    assert w == pytest.approx(0.65, abs=0.03) and corr > 0.99


def test_variance_ratio_lm_unbiased_for_random_walk():
    rng = np.random.default_rng(5)
    est = [d.variance_ratio_lm(x, 21) for x in rng.normal(0, 0.01, (400, 504))]
    assert np.mean(est) == pytest.approx(1.0, abs=0.04)
    assert np.std(est) == pytest.approx(d.vr_standard_error(21, 504), rel=0.15)


def test_variance_ratio_lm_recovers_ar1_and_ignores_scale():
    rng = np.random.default_rng(6)
    phi, q = -0.05, 21
    theory = 1 + 2 * sum((1 - k / q) * phi ** k for k in range(1, q))
    e = rng.normal(0, 0.01, (400, 504))
    x = np.zeros_like(e)
    for t in range(1, e.shape[1]):
        x[:, t] = phi * x[:, t - 1] + e[:, t]
    assert np.mean([d.variance_ratio_lm(s, q) for s in x]) == pytest.approx(theory, abs=0.04)
    assert d.variance_ratio_lm(5 * x[0], q) == pytest.approx(d.variance_ratio_lm(x[0], q))   # volatility cancels


def test_effective_volatility_and_implied_leverage():
    sd, h, vr = 0.11 / math.sqrt(252), 21, 0.8
    monthly = math.sqrt(h * sd ** 2 * vr)                       # horizon variance form
    assert monthly == pytest.approx(sd * math.sqrt(h) * math.sqrt(vr))
    assert monthly * math.sqrt(12) == pytest.approx(0.11 * math.sqrt(vr))
    assert d.implied_leverage(0.15, 0.10) == pytest.approx(1.5)
    assert d.implied_leverage(0.45, 0.10, cap=2.5) == pytest.approx(2.5)


def test_calendar_day_windows_cover_the_same_time():
    assert [d.to_days(n, "calendar") for n in (21, 63, 126, 252, 504)] == [30, 91, 183, 365, 730]
    assert d.to_days(63, "trading") == 63
    rng = np.random.default_rng(8)                         # 60% annual volatility on a 365-day clock
    ret = pd.Series(rng.normal(0, 0.60 / math.sqrt(365), 8000), pd.date_range("2000-01-01", periods=8000))
    assert d.realized_vol(ret, 8000, 365).iloc[-1] == pytest.approx(0.60, rel=0.03)
    flat = pd.Series([0.0] * 365, pd.date_range("2000-01-01", periods=365))
    cost = d.compound(d.leveraged_returns(flat, 2, cash_rate=0.04, per_year=365), 365).iloc[-1]
    assert cost == pytest.approx((1 - 0.04 / 365) ** 365 - 1)  # a year of financing, accrued daily


def test_scale_is_recovered():
    rng = np.random.default_rng(9)
    n = 2000
    p = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.004, n))), IDX[:n])
    target = (1 + 1.6 * p.pct_change().fillna(0)).cumprod()
    k, corr = d.fit_scale(target, p)
    assert k == pytest.approx(1.6, abs=0.03) and corr > 0.99


def test_weekly_samples_align_different_calendars():
    t = pd.DataFrame({"x": np.arange(10.0), "dd": -np.arange(10.0)}, index=pd.bdate_range("2020-01-06", periods=10))
    c = pd.DataFrame({"x": np.arange(15.0), "dd": -np.arange(15.0)}, index=pd.date_range("2020-01-04", periods=15))
    dates, out = d._weekly_group({"T": t, "C": c}, ["T", "C"], [("x", 4, "last"), ("dd", 4, "min")])
    assert dates == ["2020-01-10", "2020-01-17", "2020-01-18"]   # weeks end Friday; dated by the latest observation
    assert out["T"]["x"] == [4.0, 9.0, None] and out["C"]["x"] == [6.0, 13.0, 14.0]
    assert out["T"]["dd"] == [-4.0, -9.0, None] and out["C"]["dd"] == [-6.0, -13.0, -14.0]


def test_full_pipeline_on_synthetic_data(tmp_path):
    notes = []
    res, payloads, reports = d.run(d.fetch_synthetic(d.all_tickers()), notes, synthetic=True)
    d.write_outputs(payloads, tmp_path, tmp_path / "history.csv", reports=reports)
    for g, cfg in d.GROUPS.items():
        assert (tmp_path / cfg["page"]).stat().st_size > 50_000
        assert (tmp_path / cfg["report"]).stat().st_size > 50_000
        assert set(reports[g]["report"]["assets"]) == set(cfg["assets"])
    assert not (tmp_path / "history.csv").exists()          # synthetic runs never touch the log
    assert "b3" not in [m["key"] for m in payloads["other"]["metrics"]]   # no breadth outside equities
    btc = payloads["other"]["assets"]["BTC-USD"]["cal"]
    assert (btc["dpy"], btc["vr_q"], btc["vr_window"], btc["vol"]) == (365, 30, 730, [30, 91])
    assert res["frames"]["BTC-USD"].index.dayofweek.max() == 6        # bitcoin keeps its weekends
    assert res["info"]["TLT"]["proxy"] and not d.ASSETS["TLT"]["stale"]
    assert res["frames"]["TLT"]["vr21"].first_valid_index() < res["info"]["TLT"]["etf_start"]   # Treasury backfill keeps its VR
    for a in d.ALL:
        p = payloads[d.ASSETS[a]["group"]]["assets"][a]
        assert p["label"] in d.LEVEL_KEYS
        assert 0 <= p["score"] <= 100
        # Favorability is 100 minus the effective-volatility percentile, and every level follows from it.
        assert np.allclose(res["frames"][a]["score"].dropna(), 100 - res["frames"][a]["p_eff_vol"].dropna())
        f = res["frames"][a].dropna(subset=["score"])
        c = d.SCORE_CUTS
        base = pd.cut(f["score"], [-np.inf, c[0], c[1], np.nextafter(c[2], 0), np.nextafter(c[3], 0), np.inf],
                      labels=list(d.LEVEL_KEYS[::-1])).astype(str)
        assert (f["label"] == base).all()
        assert set(f["label"]) == set(d.LEVEL_KEYS)
        # Effective volatility and implied leverage follow from their definitions.
        g = f.dropna(subset=["eff_vol"])
        assert np.allclose(g["eff_vol"], g["base_vol"] * np.sqrt(g["vr21"]))
        assert np.allclose(g["lev_30"], 0.30 / g["eff_vol"])
        h = g.dropna(subset=["lev_med"])
        assert np.allclose(h["lev_med"], h["eff_vol_med"] / h["eff_vol"])
        vb = p["vb"]
        assert vb["eff_vol"] is not None and set(vb["lev"]) == {"15", "30", "45"} and vb["vr63"] is not None


def test_history_log_keeps_one_row_per_asset_and_date(tmp_path, monkeypatch):
    res, payloads, reports = d.run(d.fetch_synthetic(d.all_tickers()), [], synthetic=True)
    for p in payloads.values():
        p["synthetic"] = False
    log = tmp_path / "h.csv"
    d.write_outputs(payloads, tmp_path, log)
    d.write_outputs(payloads, tmp_path, log)                 # a rerun replaces, never duplicates
    rows = pd.read_csv(log)
    assert sorted(rows["etf"]) == sorted(d.ALL) and not rows.duplicated(["date", "etf"]).any()
