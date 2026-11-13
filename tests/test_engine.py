import numpy as np
import pandas as pd
import pytest

from engine import VolSurface, bs_greeks, bs_price, implied_vol, put_call_parity_residual
from engine.data import prepare_quotes, synthetic_chain
from engine.implied_vol import BRENT, NEWTON

RNG = np.random.default_rng(42)
N = 20_000
S = RNG.uniform(50, 150, N)
K = RNG.uniform(50, 150, N)
T = RNG.uniform(0.02, 3.0, N)
R = RNG.uniform(-0.01, 0.08, N)
Q = RNG.uniform(0.0, 0.05, N)
V = RNG.uniform(0.05, 1.0, N)


# ----------------------------------------------------------------------------- pricer
def test_textbook_values():
    # Hull, Options, Futures and Other Derivatives, Example 15.6
    assert bs_price(42, 40, 0.5, 0.10, 0.20, 0, True) == pytest.approx(4.7594, abs=1e-4)
    assert bs_price(42, 40, 0.5, 0.10, 0.20, 0, False) == pytest.approx(0.8086, abs=1e-4)


def test_put_call_parity():
    assert np.abs(put_call_parity_residual(S, K, T, R, V, Q)).max() < 1e-10


def test_price_bounds_and_limits():
    c = bs_price(S, K, T, R, V, Q, True)
    assert np.all(c >= np.maximum(S * np.exp(-Q * T) - K * np.exp(-R * T), 0) - 1e-12)
    assert np.all(c <= S * np.exp(-Q * T) + 1e-12)
    # Vol -> 0: call -> discounted intrinsic on the forward
    assert bs_price(100, 90, 1, 0.05, 1e-6, 0, True) == pytest.approx(100 - 90 * np.exp(-0.05), abs=1e-8)


@pytest.mark.parametrize("is_call", [True, False])
def test_greeks_match_finite_differences(is_call):
    s, k, t, r, v, q = 100.0, np.array([80, 95, 100, 105, 130.0]), 0.75, 0.03, 0.25, 0.01
    g = bs_greeks(s, k, t, r, v, q, is_call)
    P = lambda **kw: bs_price(kw.get("s", s), k, kw.get("t", t), kw.get("r", r), kw.get("v", v), q, is_call)
    h = 1e-4
    fd = {
        "delta": (P(s=s + h) - P(s=s - h)) / (2 * h),
        "gamma": (P(s=s + h) - 2 * P() + P(s=s - h)) / h**2,
        "vega": (P(v=v + h) - P(v=v - h)) / (2 * h),
        "theta": -(P(t=t + h) - P(t=t - h)) / (2 * h),  # theta = dV/dt = -dV/dT
        "rho": (P(r=r + h) - P(r=r - h)) / (2 * h),
        "volga": (P(v=v + h) - 2 * P() + P(v=v - h)) / h**2,
        "vanna": (P(s=s + h, v=v + h) - P(s=s + h, v=v - h) - P(s=s - h, v=v + h) + P(s=s - h, v=v - h)) / (4 * h * h),
    }
    for name, approx in fd.items():
        np.testing.assert_allclose(g[name], approx, rtol=1e-4, atol=1e-5, err_msg=name)


# ----------------------------------------------------------------------------- implied vol
@pytest.mark.parametrize("is_call", [True, False])
def test_iv_round_trip(is_call):
    p = bs_price(S, K, T, R, V, Q, is_call)
    iv = implied_vol(p, S, K, T, R, Q, is_call)
    # Prices with essentially no time value carry no vol information; skip those.
    vega = bs_greeks(S, K, T, R, V, Q, is_call)["vega"]
    ok = vega > 1e-4
    assert np.isfinite(iv[ok]).all()
    np.testing.assert_allclose(iv[ok], V[ok], atol=1e-6)


def test_iv_returns_nan_for_arbitrage_prices():
    # call below intrinsic, call above spot, negative price
    iv = implied_vol(np.array([5.0, 101.0, -1.0]), 100, 90, 1.0, 0.0, 0.0, True)
    assert np.isnan(iv).all()


def test_brent_fallback_is_exercised_and_correct():
    p = bs_price(100, 160, 0.05, 0.02, 0.9, 0, True)  # far OTM, short-dated, tiny vega
    iv, method = implied_vol(p, 100, 160, 0.05, 0.02, 0, True, max_iter=1, return_method=True)
    assert method == BRENT
    assert iv == pytest.approx(0.9, abs=1e-8)
    iv, method = implied_vol(bs_price(100, 100, 1, 0.02, 0.2), 100, 100, 1, 0.02, return_method=True)
    assert method == NEWTON


# ----------------------------------------------------------------------------- surface
@pytest.fixture(scope="module")
def synthetic():
    spot, asof, chain, truth = synthetic_chain()
    quotes, _ = prepare_quotes(chain, spot, asof, rate=0.04)
    return spot, quotes, truth


def test_forward_and_dividend_recovered(synthetic):
    spot, quotes, _ = synthetic
    q_by_expiry = quotes.groupby("T")["q"].first()
    np.testing.assert_allclose(q_by_expiry, 0.012, atol=0.004)  # true q = 1.2%


def test_surface_recovers_true_vols(synthetic):
    spot, quotes, truth = synthetic
    s = VolSurface(quotes, spot)
    Tg = np.linspace(s.T[0], s.T[-1], 50)  # includes expiries that were never quoted
    KK, TT = np.meshgrid(np.linspace(0.85, 1.10, 40) * spot, Tg)
    inside = np.abs(np.log(KK / s.forward(TT))) <= 2 * 0.2 * np.sqrt(TT)
    err = (s.iv(KK, TT) - truth(KK, TT))[inside]
    assert np.sqrt(np.mean(err**2)) < 0.002  # < 0.2 vol points RMSE


def test_interpolating_spline_hits_quotes(synthetic):
    spot, quotes, _ = synthetic
    s = VolSurface(quotes, spot, smooth=False)
    np.testing.assert_allclose(s.iv(quotes["strike"], quotes["T"]), quotes["iv"], atol=1e-8)


def test_no_static_arbitrage_on_synthetic(synthetic):
    spot, quotes, _ = synthetic
    rep = VolSurface(quotes, spot).arbitrage_report()
    assert rep[["butterfly_viol", "monotonic_viol", "calendar_viol_raw"]].to_numpy().sum() == 0


def test_calendar_floor_makes_total_variance_monotone():
    # Second expiry deliberately has LOWER total variance than the first: calendar arbitrage.
    rows = []
    for T, vol in ((0.25, 0.30), (0.50, 0.15), (1.0, 0.25)):
        for k in np.linspace(-0.2, 0.2, 9):
            rows.append({"T": T, "k": k, "iv": vol, "iv_bid": vol - 0.005, "iv_ask": vol + 0.005,
                         "strike": 100 * np.exp(k), "forward": 100.0, "df": 1.0})
    q = pd.DataFrame(rows)
    s = VolSurface(q, 100.0)
    Ts = np.linspace(0.05, 1.5, 200)
    w = s.total_variance(np.full_like(Ts, 100.0), Ts)
    assert np.all(np.diff(w) >= -1e-12)
    assert VolSurface(q, 100.0).arbitrage_report()["calendar_viol_raw"].sum() > 0  # detected on raw slices
