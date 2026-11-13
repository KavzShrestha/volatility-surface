"""Vectorised Black-Scholes(-Merton) pricer and analytic Greeks for European options.

Every function accepts scalars or NumPy arrays and broadcasts them against each other,
so a whole option chain can be priced in one call with no Python loops.

Conventions
-----------
S      spot price
K      strike
T      time to expiry in years (must be > 0)
r      continuously-compounded risk-free rate
sigma  volatility (annualised, e.g. 0.20 = 20%)
q      continuous dividend yield / carry
is_call  bool or bool array; True = call, False = put

Greek units: vega/vanna/volga are per 1.00 change in vol (divide by 100 for "per vol point"),
theta is per year of calendar time (divide by 365 for per-day), rho is per 1.00 change in r.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr  # standard normal CDF, fast and vectorised

_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _pdf(x):
    return _INV_SQRT_2PI * np.exp(-0.5 * x * x)


def _prep(S, K, T, r, sigma, q, is_call):
    S, K, T, r, sigma, q = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q))
    )
    is_call = np.broadcast_to(np.asarray(is_call, dtype=bool), S.shape)
    if np.any(T <= 0) or np.any(sigma <= 0):
        raise ValueError("T and sigma must be strictly positive")
    sqrtT = np.sqrt(T)
    vol_sqrtT = sigma * sqrtT
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrtT
    d2 = d1 - vol_sqrtT
    return S, K, T, r, sigma, q, is_call, sqrtT, d1, d2


def bs_price(S, K, T, r, sigma, q=0.0, is_call=True):
    """European option price under Black-Scholes-Merton."""
    S, K, T, r, sigma, q, is_call, _, d1, d2 = _prep(S, K, T, r, sigma, q, is_call)
    dS = S * np.exp(-q * T)  # discounted spot (PV of the asset delivered at T)
    dK = K * np.exp(-r * T)  # PV of the strike
    call = dS * ndtr(d1) - dK * ndtr(d2)
    put = dK * ndtr(-d2) - dS * ndtr(-d1)
    return np.where(is_call, call, put)


def bs_greeks(S, K, T, r, sigma, q=0.0, is_call=True) -> dict[str, np.ndarray]:
    """Analytic first- and second-order Greeks. Returns a dict of arrays."""
    S, K, T, r, sigma, q, is_call, sqrtT, d1, d2 = _prep(S, K, T, r, sigma, q, is_call)
    eq = np.exp(-q * T)
    er = np.exp(-r * T)
    n_d1 = _pdf(d1)
    N_d1, N_d2 = ndtr(d1), ndtr(d2)

    delta = np.where(is_call, eq * N_d1, eq * (N_d1 - 1.0))
    gamma = eq * n_d1 / (S * sigma * sqrtT)  # same for calls and puts
    vega = S * eq * n_d1 * sqrtT  # same for calls and puts
    decay = -S * eq * n_d1 * sigma / (2.0 * sqrtT)
    theta = np.where(
        is_call,
        decay - r * K * er * N_d2 + q * S * eq * N_d1,
        decay + r * K * er * ndtr(-d2) - q * S * eq * ndtr(-d1),
    )
    rho = np.where(is_call, K * T * er * N_d2, -K * T * er * ndtr(-d2))
    vanna = -eq * n_d1 * d2 / sigma  # d(delta)/d(sigma) = d(vega)/dS
    volga = vega * d1 * d2 / sigma  # d(vega)/d(sigma), a.k.a. vomma

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
        "vanna": vanna,
        "volga": volga,
    }


def put_call_parity_residual(S, K, T, r, sigma, q=0.0):
    """C - P - (S e^{-qT} - K e^{-rT}). Should be ~0 (machine precision) for a correct pricer."""
    c = bs_price(S, K, T, r, sigma, q, True)
    p = bs_price(S, K, T, r, sigma, q, False)
    return c - p - (np.asarray(S) * np.exp(-q * np.asarray(T)) - np.asarray(K) * np.exp(-r * np.asarray(T)))
