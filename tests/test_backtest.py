"""Delta-hedged straddle backtest, checked on simulated paths where the right answer is known."""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from engine.backtest import ATTRIBUTION, COST_PARTS, Execution, cost_breakdown, run_backtest, straddle, summarize


def gbm_market(realised, implied, years=20, seed=0, s0=100.0, r=0.0, q=0.0):
    """Business-day closes from a GBM whose variance accrues in calendar time (weekends included),
    matching how the option is priced, with a constant implied vol."""
    dates = pd.bdate_range("2000-01-03", periods=int(252 * years))
    dt = np.r_[0, dates.to_series().diff().dt.days.to_numpy()[1:]] / 365
    z = np.random.default_rng(seed).standard_normal(len(dates))
    logret = (r - q - 0.5 * realised**2) * dt + realised * np.sqrt(dt) * z
    return pd.DataFrame({"spot": s0 * np.exp(np.cumsum(logret)), "iv": implied}, index=dates)


NO_COST = dict(r=0.0, q=0.0)


def test_fair_vol_has_zero_expected_pnl():
    # realised == implied: hedged P&L is only discrete-hedging noise around zero
    res = run_backtest(gbm_market(0.20, 0.20), side=-1, **NO_COST)
    p = res.trades["pnl"]
    assert abs(p.mean()) < 3 * p.std() / np.sqrt(len(p))
    assert abs(p.mean()) < 0.05 * res.trades["premium"].mean()


def test_short_wins_when_implied_above_realised_and_long_loses():
    m = gbm_market(0.15, 0.25)
    short, long_ = run_backtest(m, side=-1, **NO_COST), run_backtest(m, side=1, **NO_COST)
    assert short.trades["pnl"].mean() > 0
    assert (short.trades["pnl"] > 0).mean() > 0.9
    # with no costs the long book is the exact mirror image of the short book
    np.testing.assert_allclose(long_.daily["pnl"], -short.daily["pnl"], atol=1e-15)


def test_long_wins_when_realised_above_implied_and_matches_theory():
    res = run_backtest(gbm_market(0.30, 0.20, seed=3), side=1, **NO_COST)
    t = res.trades
    assert t["pnl"].mean() > 0
    # P&L per trade ~ vega * (realised^2 - implied^2) / (2 * implied): check within 25% on average
    vega = straddle(1.0, 1.0, 30 / 365, 0, 0.20, 0)["vega"]  # per $1 notional
    theory = vega * (t["realised_vol"] ** 2 - 0.20**2) / (2 * 0.20)
    assert t["pnl"].mean() == pytest.approx(theory.mean(), rel=0.25)
    # imperfect: P&L weights each move by gamma (path-dependent), realised vol weights all days equally
    assert np.corrcoef(t["pnl"], t["realised_vol"])[0, 1] > 0.6


def test_flat_spot_long_straddle_loses_exactly_the_premium():
    dates = pd.bdate_range("2020-01-01", periods=200)
    m = pd.DataFrame({"spot": 100.0, "iv": 0.2}, index=dates)
    res = run_backtest(m, side=1, **NO_COST)
    np.testing.assert_allclose(res.trades["pnl"], -res.trades["premium"], rtol=1e-12)


def test_attribution_adds_up_and_costs_are_charged():
    m = gbm_market(0.18, 0.20, years=5, seed=7).assign(iv=lambda d: 0.2 + 0.03 * np.sin(np.arange(len(d)) / 15))
    ex = Execution(notional=5e6, option_half_spread_vol=0.5, option_fee=1.5, fut_half_spread_pts=0.125, fut_fee=2)
    res = run_backtest(m, side=-1, r=0.03, q=0.01, execution=ex)
    np.testing.assert_allclose(res.daily[ATTRIBUTION].sum(axis=1), res.daily["pnl"], atol=1e-14)
    t = res.trades
    np.testing.assert_allclose(t[COST_PARTS].sum(axis=1), -t["costs"], rtol=1e-12)
    np.testing.assert_allclose(t["pnl_before_costs"] + t["costs"], t["pnl"], rtol=1e-12)
    np.testing.assert_allclose(-res.daily["costs"].sum(), t[COST_PARTS].sum().sum(), rtol=1e-12)
    # option spread = half-spread (vol pts) x straddle vega, per $ of notional
    vegas = np.array([straddle(1.0, 1.0, 30 / 365, 0.03, iv, 0.01)["vega"] for iv in m["iv"].loc[t["entry"]]])
    np.testing.assert_allclose(t["option_spread"], 0.005 * vegas, rtol=1e-12)
    # fees: 2 option contracts per straddle; each futures contract pays spread + fee
    np.testing.assert_allclose(t["option_fees"] * t["notional_usd"], 2 * 1.5 * t["straddles"], rtol=1e-12)
    np.testing.assert_allclose(t["futures_fees"] * t["notional_usd"], 2.0 * t["futures_traded"], rtol=1e-12)
    np.testing.assert_allclose(t["futures_spread"] * t["notional_usd"], 0.125 * 50 * t["futures_traded"], rtol=1e-12)
    assert (cost_breakdown(res).loc["TOTAL", "% of gross P&L"] > 0)


def test_whole_contracts_track_the_ideal_hedge():
    m = gbm_market(0.20, 0.20, years=5, seed=5, s0=4000)
    ideal = run_backtest(m, side=-1, **NO_COST)
    zero_cost = Execution(option_half_spread_vol=0, option_fee=0, fut_half_spread_pts=0, fut_fee=0)
    big = run_backtest(m, side=-1, execution=zero_cost, **NO_COST)  # $10M: ~25 straddles, ~50 ES per unit delta
    small = run_backtest(m, side=-1, execution=dataclasses.replace(zero_cost, notional=1e6), **NO_COST)
    expected = (10e6 / (100 * m["spot"].loc[big.trades["entry"]].to_numpy())).round()
    np.testing.assert_array_equal(big.trades["straddles"].to_numpy(), expected)
    assert big.trades["costs"].abs().max() == 0
    # rounding error is P&L noise, not drift; it shrinks as the book gets bigger relative to one contract
    err_big = (big.trades["pnl"] - ideal.trades["pnl"]).std()
    err_small = (small.trades["pnl"] - ideal.trades["pnl"]).std()
    assert err_big < err_small
    assert err_big < 0.1 * ideal.trades["pnl"].std()


def test_unhedged_straddle_carries_delta_risk():
    m = gbm_market(0.20, 0.20, years=10, seed=11)
    hedged = run_backtest(m, side=-1, **NO_COST)
    naked = run_backtest(m, side=-1, hedge=False, **NO_COST)
    assert naked.daily["delta"].abs().sum() > 0 and hedged.daily["delta"].abs().sum() == 0
    assert naked.trades["pnl"].std() > 1.5 * hedged.trades["pnl"].std()


def test_summary_and_validation():
    res = run_backtest(gbm_market(0.15, 0.20, years=3), side=-1)
    s = summarize(res)
    assert s["trades"] == len(res.trades) and s["sharpe"] > 0
    with pytest.raises(ValueError):
        run_backtest(gbm_market(0.2, 0.2), side=0)
    with pytest.raises(ValueError):
        run_backtest(gbm_market(0.2, 0.2).iloc[:10])
