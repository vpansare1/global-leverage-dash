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


def test_serial_correlation_of_ar1_process():
    rng = np.random.default_rng(1)
    phi, e = 0.2, rng.normal(0, 0.01, 6000)
    r = np.zeros(6000)
    for i in range(1, 6000):
        r[i] = phi * r[i - 1] + e[i]
    r = pd.Series(r, IDX)
    assert d.rolling_autocorr(r, 5999).iloc[-1] == pytest.approx(phi, abs=0.03)
    # Theoretical VR(q) for AR(1): 1 + 2 * sum_{k=1}^{q-1} (1 - k/q) * phi^k
    for q in (5, 10):
        theory = 1 + 2 * sum((1 - k / q) * phi ** k for k in range(1, q))
        assert d.variance_ratio(r, q, 5900).iloc[-1] == pytest.approx(theory, abs=0.06)


def test_variance_ratio_of_random_walk_is_one():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0, 0.01, 6000), IDX)
    assert d.variance_ratio(r, 10, 5900).iloc[-1] == pytest.approx(1.0, abs=0.07)


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


def test_full_pipeline_on_synthetic_data(tmp_path):
    notes = []
    res = d.build(d.fetch_synthetic(d.all_tickers()), notes, synthetic=True)
    payload = d.make_payload(res, notes, synthetic=True)
    report = d.make_report(res, payload)
    d.write_outputs(payload, tmp_path, tmp_path / "history.csv", report=report)
    assert (tmp_path / "index.html").stat().st_size > 50_000
    assert (tmp_path / "report.html").stat().st_size > 50_000
    assert {"SPY", "EFA", "EEM"} <= set(report["report"]["assets"])
    assert len(report["report"]["coverage"]) == len(d.all_tickers())
    assert not (tmp_path / "history.csv").exists()          # synthetic runs never touch the log
    for a in d.CORE:
        assert payload["assets"][a]["label"] in d.LEVEL_KEYS
        assert 0 <= payload["assets"][a]["score"] <= 100
        # Every day's label must follow from that day's score and narrow-advance flag.
        f = res["frames"][a].dropna(subset=["score"])
        c = d.SCORE_CUTS
        base = pd.cut(f["score"], [-np.inf, c[0], c[1], np.nextafter(c[2], 0), np.nextafter(c[3], 0), np.inf],
                      labels=list(d.LEVEL_KEYS[::-1])).astype(str)
        step_down = {"very_favorable": "favorable", "favorable": "neutral"}
        expected = base.where(f["narrow"] != 1, base.map(lambda k: step_down.get(k, k)))
        assert (f["label"] == expected).all()
        assert set(f["label"]) == set(d.LEVEL_KEYS)
