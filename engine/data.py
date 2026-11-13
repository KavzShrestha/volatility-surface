"""Option chain ingestion (CBOE delayed-quotes REST API, or a synthetic chain) and cleaning.

Raw chain schema (one row per listed option):
    expiry (Timestamp), strike, is_call, bid, ask, volume, open_interest, vendor_iv
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import requests

from .black_scholes import bs_price
from .implied_vol import implied_vol

CBOE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
OCC_RE = re.compile(r"^(?P<root>[A-Z0-9]+?)(?P<date>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$")
YEAR_SECONDS = 365.0 * 24 * 3600


# --------------------------------------------------------------------------- ingestion
def fetch_cboe_chain(symbol: str = "_SPX", roots: tuple[str, ...] | None = None, timeout: float = 20.0):
    """Download the delayed (~15 min) option chain from CBOE's public REST endpoint.

    Index symbols take a leading underscore (``_SPX``). Returns (spot, asof, chain DataFrame).
    ``roots`` keeps only those OCC roots (e.g. SPXW = PM-settled weeklies; SPX = AM-settled monthlies).
    """
    resp = requests.get(CBOE_URL.format(symbol=symbol), timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    data = payload["data"]
    spot = float(data["current_price"])
    asof = pd.Timestamp(payload["timestamp"])  # US/Eastern wall-clock time

    rows = []
    for o in data["options"]:
        m = OCC_RE.match(o["option"])
        if not m:
            continue
        rows.append(
            {
                "root": m["root"],
                "expiry": pd.Timestamp("20" + m["date"]),
                "is_call": m["cp"] == "C",
                "strike": int(m["strike"]) / 1000.0,
                "bid": o.get("bid"),
                "ask": o.get("ask"),
                "volume": o.get("volume"),
                "open_interest": o.get("open_interest"),
                "vendor_iv": o.get("iv"),
            }
        )
    chain = pd.DataFrame(rows)
    if roots is None:
        roots = (symbol.lstrip("_"),) if symbol.lstrip("_") != "SPX" else ("SPXW",)
    chain = chain[chain["root"].isin(roots)].drop(columns="root").reset_index(drop=True)
    return spot, asof, chain


def ssvi_total_variance(k, T, atm_vol=0.18, rho=-0.6, eta=1.2, gamma=0.45, term_slope=0.03):
    """Gatheral's SSVI surface w(k, T). With eta * (1 + |rho|) <= 2 it is free of static arbitrage.

    Used as the known "true" surface for the synthetic chain, so the engine can be checked
    against an answer we know exactly.
    """
    theta = (atm_vol**2 + term_slope * np.sqrt(T) * atm_vol**2) * T  # ATM total variance, increasing in T
    phi = eta / (theta**gamma * (1.0 + theta) ** (1.0 - gamma))
    return 0.5 * theta * (1.0 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + 1.0 - rho**2))


def synthetic_chain(spot=100.0, r=0.04, q=0.012, seed=7, asof="2026-11-12 16:00"):
    """A realistic fake chain priced off a known SSVI surface, with bid/ask spreads and noise."""
    rng = np.random.default_rng(seed)
    asof = pd.Timestamp(asof)
    rows = []
    for days in (9, 23, 37, 65, 93, 128, 184, 275, 365, 548):
        expiry = (asof + pd.Timedelta(days=days)).normalize()
        T = ((expiry + pd.Timedelta(hours=16)) - asof).total_seconds() / YEAR_SECONDS
        F = spot * np.exp((r - q) * T)
        width = 0.10 + 0.35 * np.sqrt(T)
        strikes = np.unique(np.round(spot * np.exp(np.linspace(-width * 1.5, width, 41)), 0))
        k = np.log(strikes / F)
        vol = np.sqrt(ssvi_total_variance(k, T) / T)
        for is_call in (True, False):
            px = bs_price(spot, strikes, T, r, vol, q, is_call)
            half_spread = np.maximum(0.01, 0.015 * px) + 0.005 * rng.random(px.size)
            mid = px + rng.normal(0.0, 0.25, px.size) * half_spread  # mid is a noisy estimate of fair value
            bid = np.round(np.maximum(mid - half_spread, 0.0), 2)
            ask = np.round(mid + half_spread, 2)
            for K, b, a in zip(strikes, bid, ask):
                rows.append({"expiry": expiry, "is_call": is_call, "strike": K, "bid": b, "ask": a,
                             "volume": np.nan, "open_interest": np.nan, "vendor_iv": np.nan})
    truth = lambda K, T: np.sqrt(ssvi_total_variance(np.log(K / (spot * np.exp((r - q) * T))), T) / T)
    return spot, asof, pd.DataFrame(rows), truth


# --------------------------------------------------------------------------- cleaning
def implied_forward(calls: pd.DataFrame, puts: pd.DataFrame, spot: float, D: float, n_strikes: int = 6):
    """Back out the forward F for one expiry from put-call parity: C - P = D * (F - K).

    Each strike gives an estimate F = K + (C - P) / D; take the median over the strikes nearest
    the money (most liquid, tightest spreads). This captures the dividends / carry the market
    is actually pricing in, instead of assuming a dividend yield. The discount factor D comes
    from the rate curve: regressing it out of parity too is possible but noisy.
    """
    m = calls.merge(puts, on="strike", suffixes=("_c", "_p"))
    if len(m) < 3:
        return None
    m = m.iloc[np.argsort(np.abs(m["strike"].to_numpy() - spot))[:n_strikes]]
    F = float(np.median(m["strike"] + (m["mid_c"] - m["mid_p"]) / D))
    return F if 0.5 * spot < F < 1.5 * spot else None


def prepare_quotes(
    chain: pd.DataFrame,
    spot: float,
    asof: pd.Timestamp,
    rate: float = 0.04,
    q_fallback: float = 0.0,
    min_days: float = 5,
    max_days: float = 800,
    min_bid: float = 0.05,
    max_rel_spread: float = 0.30,
    max_abs_k: float = 0.6,
    min_points: int = 6,
) -> tuple[pd.DataFrame, dict]:
    """Clean a raw chain and convert it into implied vols (one OTM quote per strike and expiry)."""
    c = chain.copy()
    c = c[(c["bid"] > 0) & (c["ask"] > c["bid"])]
    c["mid"] = 0.5 * (c["bid"] + c["ask"])
    c["T"] = ((c["expiry"] + pd.Timedelta(hours=16)) - asof).dt.total_seconds() / YEAR_SECONDS
    c = c[(c["T"] * 365 >= min_days) & (c["T"] * 365 <= max_days)]

    out, stats = [], {"raw": len(chain), "expiries_used": 0, "forward_fallbacks": 0}
    for expiry, g in c.groupby("expiry"):
        T = g["T"].iloc[0]
        D = np.exp(-rate * T)
        F = implied_forward(g[g["is_call"]], g[~g["is_call"]], spot, D)
        if F is None:
            stats["forward_fallbacks"] += 1
            F = spot * np.exp((rate - q_fallback) * T)
        # Use OTM options only: puts below the forward, calls above. They are the most liquid,
        # carry no intrinsic value to swamp the vol signal, and have little early-exercise premium.
        g = g[np.where(g["strike"] < F, ~g["is_call"], g["is_call"])].copy()
        g = g[(g["bid"] >= min_bid) & ((g["ask"] - g["bid"]) / g["mid"] <= max_rel_spread)]
        g["k"] = np.log(g["strike"] / F)
        g = g[g["k"].abs() <= max_abs_k]
        if len(g) < min_points:
            continue
        r_eff = -np.log(D) / T
        q_eff = r_eff - np.log(F / spot) / T
        g["forward"], g["df"], g["r"], g["q"] = F, D, r_eff, q_eff
        K, cp = g["strike"].to_numpy(), g["is_call"].to_numpy()
        g["iv"], g["iv_method"] = implied_vol(g["mid"].to_numpy(), spot, K, T, r_eff, q_eff, cp, return_method=True)
        g["iv_bid"] = implied_vol(g["bid"].to_numpy(), spot, K, T, r_eff, q_eff, cp)
        g["iv_ask"] = implied_vol(g["ask"].to_numpy(), spot, K, T, r_eff, q_eff, cp)
        g = g[np.isfinite(g["iv"])]
        if len(g) >= min_points:
            out.append(g)
            stats["expiries_used"] += 1

    quotes = pd.concat(out, ignore_index=True).sort_values(["T", "strike"]).reset_index(drop=True)
    stats["clean"] = len(quotes)
    stats["newton"] = int((quotes["iv_method"] == 0).sum())
    stats["brent"] = int((quotes["iv_method"] == 1).sum())
    return quotes, stats
