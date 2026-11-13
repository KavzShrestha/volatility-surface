"""Implied volatility surface built from cleaned quotes.

Strike direction: for every expiry, a cubic smoothing spline of implied vol against
log-forward-moneyness k = ln(K / F). Working in k rather than raw K makes slices at different
expiries comparable (the ATM point is always k = 0). The smoothing is weighted by the inverse
bid-ask spread in vol terms, so tight, liquid quotes pull harder than wide ones, and the curve
does not chase noise (an exact interpolating spline wiggles, and wiggles mean butterfly arbitrage).

Time direction: linear interpolation in total variance w = sigma^2 * T at fixed k. Total variance
is what adds up over time, so interpolating it (rather than vol) is natural. If w is non-decreasing
in T at every k, interpolated w stays non-decreasing too, meaning no calendar arbitrage.

Arbitrage awareness:
  * calendar: optionally enforce w_i(k) = max_{j<=i} w_j(k) across slices (on by default)
  * butterfly: diagnose call prices that are not convex in strike (negative implied density)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, make_smoothing_spline

from .black_scholes import bs_greeks, bs_price


def _fit_smile(x, iv, spread):
    """Cubic smoothing spline whose smoothness is chosen so the curve stays inside the market.

    The penalty lam trades fit against curvature. Rather than generic cross-validation, pick the
    smoothest curve whose residuals are, in RMS, no bigger than half the bid-ask spread: the
    market cannot tell those curves apart, and smoother is less likely to create arbitrage.
    """
    w = 1.0 / spread**2
    w = w / w.mean()
    half = 0.5 * spread
    best = None
    for lam in np.logspace(2, -10, 25):  # smoothest first
        try:
            spl = make_smoothing_spline(x, iv, w=w, lam=lam)
        except (ValueError, np.linalg.LinAlgError):
            continue
        best = spl
        if np.sqrt(np.mean(((spl(x) - iv) / half) ** 2)) <= 1.0:
            return spl
    return best if best is not None else CubicSpline(x, iv, bc_type="natural")


class VolSurface:
    def __init__(self, quotes: pd.DataFrame, spot: float, smooth: bool = True, enforce_calendar: bool = True):
        self.spot = float(spot)
        self.enforce_calendar = enforce_calendar
        T, carry, rate, splines, kmin, kmax = [], [], [], [], [], []
        for t, g in quotes.groupby("T"):
            g = g.sort_values("k").drop_duplicates("k")
            k, iv = g["k"].to_numpy(), g["iv"].to_numpy()
            x = k / np.sqrt(t)  # standardised moneyness: smiles of all expiries have similar width in x
            if smooth and len(k) >= 5:
                spread = (g["iv_ask"] - g["iv_bid"]).to_numpy()
                spread = np.where(np.isfinite(spread) & (spread > 0), spread, np.nanmedian(spread))
                spread = np.maximum(spread, np.nanpercentile(spread, 10))
                spl = _fit_smile(x, iv, spread)
            else:
                spl = CubicSpline(x, iv, bc_type="natural")
            T.append(t)
            carry.append(np.log(g["forward"].iloc[0] / spot) / t)  # (r - q) implied by the forward
            rate.append(-np.log(g["df"].iloc[0]) / t)
            splines.append(spl)
            kmin.append(k.min())
            kmax.append(k.max())
        self.T = np.array(T)
        self.carry = np.array(carry)
        self.rate = np.array(rate)
        self.splines = splines
        self.kmin, self.kmax = np.array(kmin), np.array(kmax)

    # ------------------------------------------------------------------ term structure of carry
    def r(self, T):
        return np.interp(T, self.T, self.rate)

    def q(self, T):
        return self.r(T) - np.interp(T, self.T, self.carry)

    def forward(self, T):
        return self.spot * np.exp(np.interp(T, self.T, self.carry) * np.asarray(T))

    # ------------------------------------------------------------------ volatility
    def slice_vol(self, i: int, k):
        """Vol on expiry slice i. Flat extrapolation beyond the quoted moneyness range."""
        x = np.clip(k, self.kmin[i], self.kmax[i]) / np.sqrt(self.T[i])
        return np.maximum(self.splines[i](x), 0.01)

    def _slice_w(self, k):
        """Total variance of every slice at the given k values: shape (n_slices, n_points)."""
        W = np.vstack([self.slice_vol(i, k) ** 2 * self.T[i] for i in range(len(self.T))])
        if self.enforce_calendar:
            W = np.maximum.accumulate(W, axis=0)
        return W

    def total_variance(self, K, T):
        K, T = np.broadcast_arrays(np.asarray(K, float), np.asarray(T, float))
        shape = K.shape
        K, T = K.ravel(), T.ravel()
        k = np.log(K / self.forward(T))
        W = self._slice_w(k)
        j = np.searchsorted(self.T, T)  # T lies between slice j-1 and slice j
        cols = np.arange(K.size)
        lo = np.clip(j - 1, 0, len(self.T) - 1)
        hi = np.clip(j, 0, len(self.T) - 1)
        T_lo, T_hi = self.T[lo], self.T[hi]
        w_lo, w_hi = W[lo, cols], W[hi, cols]
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(T_hi > T_lo, (T - T_lo) / (T_hi - T_lo), 0.0)
        w = w_lo + frac * (w_hi - w_lo)
        # Outside the quoted expiries, hold implied vol constant (scale variance with T).
        w = np.where(T < self.T[0], W[0, cols] * T / self.T[0], w)
        w = np.where(T > self.T[-1], W[-1, cols] * T / self.T[-1], w)
        return w.reshape(shape)

    def iv(self, K, T):
        return np.sqrt(self.total_variance(K, T) / np.asarray(T, float))

    def in_quoted_range(self, K, T):
        """True where (K, T) lies inside the moneyness range actually quoted (no extrapolation)."""
        k = np.log(np.asarray(K, float) / self.forward(T))
        return (k >= np.interp(T, self.T, self.kmin)) & (k <= np.interp(T, self.T, self.kmax))

    # ------------------------------------------------------------------ pricing off the surface
    def price(self, K, T, is_call=True):
        return bs_price(self.spot, K, T, self.r(T), self.iv(K, T), self.q(T), is_call)

    def greeks(self, K, T, is_call=True):
        """Sticky-strike Greeks: Black-Scholes Greeks evaluated at the surface vol for (K, T)."""
        return bs_greeks(self.spot, K, T, self.r(T), self.iv(K, T), self.q(T), is_call)

    # ------------------------------------------------------------------ diagnostics
    def arbitrage_report(self, n_grid: int = 200) -> pd.DataFrame:
        """Per expiry: butterfly violations (non-convex call prices in K) and calendar violations
        (total variance below the previous expiry's), measured on the raw fitted slices."""
        rows = []
        prev_w = None
        k_common = np.linspace(max(self.kmin.max(), -0.1), min(self.kmax.min(), 0.05), n_grid)
        for i, T in enumerate(self.T):
            k = np.linspace(self.kmin[i], self.kmax[i], n_grid)
            F = self.spot * np.exp(self.carry[i] * T)
            K = F * np.exp(k)
            C = bs_price(self.spot, K, T, self.rate[i], self.slice_vol(i, k), self.rate[i] - self.carry[i], True)
            # Second difference of call price on a uniform-in-k (non-uniform in K) grid.
            dC = np.diff(C) / np.diff(K)
            convexity = np.diff(dC)
            butterfly = int((convexity < -1e-9 * self.spot).sum())
            monotone = int((np.diff(C) > 1e-9 * self.spot).sum())  # calls must fall as K rises
            w = self.slice_vol(i, k_common) ** 2 * T
            calendar = 0 if prev_w is None else int((w < prev_w - 1e-8).sum())
            prev_w = w
            rows.append({"T": T, "days": round(T * 365, 1), "butterfly_viol": butterfly,
                         "monotonic_viol": monotone, "calendar_viol_raw": calendar})
        return pd.DataFrame(rows)

    def grid(self, moneyness=np.linspace(0.8, 1.15, 36), T=None) -> pd.DataFrame:
        """IV grid indexed by expiry (years) with strikes as a fraction of spot as columns."""
        T = np.linspace(self.T[0], self.T[-1], 30) if T is None else np.asarray(T)
        KK, TT = np.meshgrid(moneyness * self.spot, T)
        return pd.DataFrame(self.iv(KK, TT), index=pd.Index(np.round(T, 4), name="T"),
                            columns=pd.Index(np.round(moneyness, 3), name="K/S"))
