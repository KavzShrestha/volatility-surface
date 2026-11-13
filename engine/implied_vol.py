"""Implied volatility solver: vectorised Newton-Raphson with a Brent fallback.

Newton converges quadratically and runs on the whole array at once, so it handles
almost every quote in a few iterations. It can fail where vega is tiny (deep ITM/OTM,
very short expiries) because the step price_error / vega blows up. Any element that
does not converge, or tries to step outside the volatility bounds, is passed to
scipy's Brent root finder, which is guaranteed to converge given a bracket.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .black_scholes import bs_greeks, bs_price

NEWTON, BRENT, FAILED = 0, 1, -1


def price_bounds(S, K, T, r, q, is_call):
    """No-arbitrage bounds for a European option. Any price outside has no implied vol."""
    dS = S * np.exp(-q * T)
    dK = K * np.exp(-r * T)
    lower = np.where(is_call, np.maximum(dS - dK, 0.0), np.maximum(dK - dS, 0.0))
    upper = np.where(is_call, dS, dK)
    return lower, upper


def implied_vol(
    price,
    S,
    K,
    T,
    r,
    q=0.0,
    is_call=True,
    tol=1e-10,
    max_iter=50,
    vol_lo=1e-4,
    vol_hi=5.0,
    return_method=False,
):
    """Invert Black-Scholes for sigma. Returns NaN where the price violates no-arbitrage bounds.

    If return_method is True, also returns an int array: 0 = Newton, 1 = Brent, -1 = failed.
    """
    price, S, K, T, r, q = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (price, S, K, T, r, q))
    )
    is_call = np.broadcast_to(np.asarray(is_call, dtype=bool), price.shape)
    shape = price.shape
    price, S, K, T, r, q, is_call = (a.ravel() for a in (price, S, K, T, r, q, is_call))

    sigma_out = np.full(price.shape, np.nan)
    method = np.full(price.shape, FAILED, dtype=int)

    lower, upper = price_bounds(S, K, T, r, q, is_call)
    # Strictly inside the bounds; a price exactly at intrinsic implies zero vol.
    valid = (T > 0) & np.isfinite(price) & (price > lower + 1e-12) & (price < upper - 1e-12)

    # ---- Newton-Raphson, vectorised over all still-active quotes ----
    # Initial guess (Manaster-Koehler) puts the start near the vol where vega is largest.
    F = S * np.exp((r - q) * T)
    sigma = np.sqrt(np.abs(2.0 * np.log(F / K)) / np.where(T > 0, T, 1.0))
    sigma = np.clip(np.where(sigma > 1e-3, sigma, 0.2), 0.05, 2.0)

    active = valid.copy()
    for _ in range(max_iter):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        s = sigma[idx]
        model = bs_price(S[idx], K[idx], T[idx], r[idx], s, q[idx], is_call[idx])
        vega = bs_greeks(S[idx], K[idx], T[idx], r[idx], s, q[idx], is_call[idx])["vega"]
        err = model - price[idx]

        # Converged when the Newton step in *vol* is below tol. A test on price error alone is
        # not enough: when vega is small, a tiny price error still means a large vol error.
        # The second condition stops at floating-point precision of the price itself.
        with np.errstate(divide="ignore", invalid="ignore"):
            step = np.abs(err) / vega
        done = (step < tol) | (np.abs(err) < 1e-14 * np.maximum(1.0, price[idx]))
        sigma_out[idx[done]] = s[done]
        method[idx[done]] = NEWTON
        active[idx[done]] = False

        rest = ~done
        with np.errstate(divide="ignore", invalid="ignore"):
            new_s = s[rest] - err[rest] / vega[rest]
        # Newton is unsafe here: vega is ~0 or the step leaves the bracket. Hand to Brent.
        bad = (vega[rest] < 1e-8) | ~np.isfinite(new_s) | (new_s <= vol_lo) | (new_s >= vol_hi)
        active[idx[rest][bad]] = False
        sigma[idx[rest][~bad]] = new_s[~bad]

    # ---- Brent fallback for anything Newton did not solve ----
    for i in np.flatnonzero(valid & (method == FAILED)):
        f = lambda v: bs_price(S[i], K[i], T[i], r[i], v, q[i], is_call[i]) - price[i]
        try:
            if f(vol_lo) * f(vol_hi) < 0:
                sigma_out[i] = brentq(f, vol_lo, vol_hi, xtol=1e-12, maxiter=200)
                method[i] = BRENT
        except (ValueError, RuntimeError):
            pass

    sigma_out = sigma_out.reshape(shape)
    if return_method:
        return sigma_out, method.reshape(shape)
    return sigma_out
