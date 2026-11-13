"""Delta-hedged straddle backtest: does selling (or buying) implied volatility pay after hedging and costs?

Strategy
--------
Every `tenor_days` calendar days, open one at-the-money straddle (call + put, strike = spot) on SPX,
priced with Black-Scholes at the VIX level, delta-hedge it at every daily close, hold it to expiry, settle
at intrinsic value, and immediately roll into the next one. Trades never overlap.

Why a delta-hedged straddle: hedging removes the bet on direction, leaving a bet on the *size* of moves.
For a hedged long option the daily P&L is approximately

    1/2 * Gamma * S^2 * (realised daily variance - implied daily variance) + Vega * d(IV)

so a seller wins when realised vol comes in below implied, and loses when it is above.

Realised volatility of a trade
------------------------------
Close-to-close log returns from entry to expiry, squared and summed (no mean subtracted, as for variance
swaps), annualised by 365 / calendar days held, square-rooted. Calendar days because the option's time to
expiry, and VIX, are measured in calendar time.

Execution
---------
`execution=None` is the frictionless ideal: fractional straddles, a hedge that matches delta exactly, zero
costs. Passing an `Execution` trades the way a desk would: a whole number of SPX option contracts
(x100 multiplier) sized to `notional`, hedged with a whole number of index futures contracts (E-mini S&P
by default), paying the option bid-ask spread and fees at entry and the futures spread and fees on every
hedge trade. Options are held to expiry and cash-settle, so there is no exit spread on them.

Units
-----
P&L is a fraction of the dollar notional at entry, so 36 years of trades are comparable even though SPX
went from ~350 to ~7,500. P&L is summed, not compounded.

Simplifications
---------------
* VIX stands in for the 30-day ATM implied vol. VIX is a variance-swap rate that includes the skew, so it
  usually sits *above* ATM IV; `iv_haircut` lowers it to test how much that matters.
* The same VIX level marks the option every day even as its remaining life shrinks (no term structure).
* Flat r and q; hedging only at closes; SPX itself stands in for the future (no basis); today's contract
  specs and spreads are applied to the whole history (E-mini futures only launched in 1997, and option
  spreads were wider in the 1990s).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .black_scholes import bs_greeks, bs_price

HISTORY_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{symbol}_History.csv"
_CP = np.array([True, False])  # call, put
SPX_MULTIPLIER = 100  # $ per index point per SPX option contract


@dataclass(frozen=True)
class Execution:
    """How the trade is actually executed. Defaults: E-mini S&P hedge, approximate 2026 fee schedules.

    notional                 $ of SPX notional per trade; sets the whole number of straddles traded
    option_half_spread_vol   half the straddle's bid-ask, in vol points (measure it: `measure_option_spread`)
    option_fee               $ per option contract per side (exchange + index licence + clearing + broker)
    fut_multiplier           $ per index point per futures contract: 50 = E-mini (ES), 5 = Micro (MES)
    fut_half_spread_pts      half the futures bid-ask in index points: ES trades one 0.25 tick wide
    fut_fee                  $ per futures contract per side (exchange + clearing + broker)
    """
    notional: float = 10_000_000
    option_half_spread_vol: float = 0.05
    option_fee: float = 1.50
    fut_multiplier: float = 50.0
    fut_half_spread_pts: float = 0.125
    fut_fee: float = 2.00


def load_history(cache_dir: str | Path = "data", refresh: bool = False, timeout: float = 30) -> pd.DataFrame:
    """Daily SPX and VIX closes from CBOE's public history files, cached in `cache_dir`.

    Returns a DataFrame indexed by date with columns `spot` (SPX close) and `iv` (VIX / 100).
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    series = {}
    for symbol, col in (("SPX", "SPX"), ("VIX", "CLOSE")):
        path = cache / f"{symbol}_History.csv"
        if refresh or not path.exists():
            resp = requests.get(HISTORY_URL.format(symbol=symbol), timeout=timeout)
            resp.raise_for_status()
            path.write_bytes(resp.content)
        df = pd.read_csv(path)
        series[symbol] = pd.Series(df[col].astype(float).to_numpy(),
                                   index=pd.to_datetime(df["DATE"], format="%m/%d/%Y"))
    out = pd.DataFrame({"spot": series["SPX"], "iv": series["VIX"] / 100}).dropna()
    out = out[(out["spot"] > 0) & (out["iv"] > 0)]
    return out[~out.index.duplicated()].sort_index()


def measure_option_spread(quotes: pd.DataFrame, target_days: float = 30, n: int = 4) -> dict:
    """Half bid-ask of near-ATM options at the expiry closest to `target_days`, from a cleaned chain
    (`prepare_quotes` output). In vol points, the straddle's half-spread equals each leg's."""
    T = quotes["T"].iloc[int((quotes["T"] * 365 - target_days).abs().argmin())]
    g = quotes[quotes["T"] == T]
    atm = g.iloc[np.argsort(g["k"].abs().to_numpy())[:n]]
    return {
        "days": float(T * 365),
        "atm_iv": float(atm["iv"].median()),
        "half_spread_vol": float(((atm["iv_ask"] - atm["iv_bid"]) / 2).median() * 100),
        "half_spread_pts": float(((atm["ask"] - atm["bid"]) / 2).median()),
        "strikes": atm["strike"].tolist(),
    }


def straddle(S: float, K: float, T: float, r: float, sigma: float, q: float) -> dict:
    """Value and Greeks of one call plus one put. At or after expiry (T <= 0) it is the payoff |S - K|."""
    if T <= 0:
        return {"value": abs(S - K), "delta": float(np.sign(S - K)), "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    K2 = np.array([K, K], dtype=float)
    g = bs_greeks(S, K2, T, r, sigma, q, _CP)
    value = bs_price(S, K2, T, r, sigma, q, _CP).sum()
    return {"value": float(value), **{k: float(g[k].sum()) for k in ("delta", "gamma", "vega", "theta")}}


@dataclass
class BacktestResult:
    daily: pd.DataFrame   # one row per date: pnl and its attribution, as fractions of notional
    trades: pd.DataFrame  # one row per straddle
    params: dict = field(default_factory=dict)


ATTRIBUTION = ["realised_vs_implied", "vega", "delta", "residual", "costs"]
COST_PARTS = ["option_spread", "option_fees", "futures_spread", "futures_fees"]


def run_backtest(
    market: pd.DataFrame,
    side: int = -1,
    tenor_days: int = 30,
    r: float = 0.03,
    q: float = 0.015,
    iv_haircut: float = 0.0,
    execution: Execution | None = None,
    hedge: bool = True,
) -> BacktestResult:
    """Run back-to-back delta-hedged ATM straddles over `market` (columns `spot`, `iv`).

    side        -1 = sell straddles (short vol), +1 = buy them (long vol)
    tenor_days  calendar days from entry to expiry
    iv_haircut  vol points subtracted from `iv` before pricing (VIX -> ATM IV adjustment)
    execution   None = frictionless; an `Execution` = whole contracts, real spreads and fees
    hedge       False = hold the straddle unhedged (for comparison)
    """
    if side not in (-1, 1):
        raise ValueError("side must be -1 (short) or +1 (long)")
    ex = execution
    dates = market.index
    S = market["spot"].to_numpy(float)
    IV = np.maximum(market["iv"].to_numpy(float) - iv_haircut / 100, 0.01)
    n = len(dates)
    rows, trades = [], []

    i = 0
    while True:
        expiry = dates[i] + pd.Timedelta(days=tenor_days)
        j = int(dates.searchsorted(expiry))  # first close on or after expiry: settlement day
        if j >= n:
            break
        tid, K = len(trades), S[i]
        # qty = index units of straddle held. Frictionless: $1 of notional. Real: whole SPX contracts.
        n_straddles = max(1, round(ex.notional / (SPX_MULTIPLIER * S[i]))) if ex else None
        qty = n_straddles * SPX_MULTIPLIER if ex else 1.0 / S[i]
        notional = qty * S[i]
        lot = ex.fut_multiplier / qty if ex else 0.0  # one futures contract, in hedge-ratio units

        def target(delta):
            if not hedge:
                return 0.0
            return round(delta / lot) * lot if ex else delta

        def hedge_trade(dh, spot):
            """Futures contracts traded and their cost (spread, fees) as fractions of notional."""
            if not ex:
                return 0, 0.0, 0.0
            contracts = round(abs(dh) / lot)
            return (contracts, contracts * ex.fut_half_spread_pts * ex.fut_multiplier / notional,
                    contracts * ex.fut_fee / notional)

        prev = straddle(S[i], K, tenor_days / 365, r, IV[i], q)
        premium = prev["value"] * qty / notional
        h = target(prev["delta"])
        parts = dict.fromkeys(COST_PARTS, 0.0)
        if ex:
            parts["option_spread"] = ex.option_half_spread_vol / 100 * prev["vega"] * qty / notional
            parts["option_fees"] = 2 * n_straddles * ex.option_fee / notional  # a call and a put, entry only
        contracts, fs, ff = hedge_trade(h, S[i])
        parts["futures_spread"] += fs
        parts["futures_fees"] += ff
        futures_traded = contracts
        entry_cost = sum(parts.values())
        rows.append({"date": dates[i], "pnl": -entry_cost, "costs": -entry_cost})
        trade_pnl, sum_r2 = -entry_cost, 0.0
        scale = qty / notional  # per-index-unit P&L -> fraction of notional

        for k in range(i, j):
            dt = (dates[k + 1] - dates[k]).days / 365
            T_left = (expiry - dates[k + 1]).days / 365
            nxt = straddle(S[k + 1], K, T_left, r, IV[k + 1], q)
            dS = S[k + 1] - S[k]
            sum_r2 += np.log(S[k + 1] / S[k]) ** 2

            # Book: `side` straddles and -side*h index units. Cash earns r on the premium and on the
            # short-sale proceeds, and the index position pays/receives the dividend yield q.
            option = nxt["value"] - prev["value"]
            hedge_pnl = -h * dS
            carry = (-r * prev["value"] + (r - q) * h * S[k]) * dt
            gross = side * (option + hedge_pnl + carry) * scale

            rvi = side * 0.5 * prev["gamma"] * (dS**2 - IV[k] ** 2 * S[k] ** 2 * dt) * scale
            vega = side * prev["vega"] * (IV[k + 1] - IV[k]) * scale
            delta = side * (prev["delta"] - h) * dS * scale  # unhedged, or hedge rounded to whole contracts

            new_h = 0.0 if k + 1 == j else target(nxt["delta"])  # unwind the hedge at expiry
            contracts, fs, ff = hedge_trade(new_h - h, S[k + 1])
            parts["futures_spread"] += fs
            parts["futures_fees"] += ff
            futures_traded += contracts
            rows.append({"date": dates[k + 1], "pnl": gross - fs - ff, "realised_vs_implied": rvi, "vega": vega,
                         "delta": delta, "residual": gross - rvi - vega - delta, "costs": -(fs + ff)})
            trade_pnl += gross - fs - ff
            prev, h = nxt, new_h

        cost = sum(parts.values())
        trades.append({
            "entry": dates[i], "expiry": dates[j], "strike": K, "spot_at_expiry": S[j],
            "entry_iv": IV[i], "realised_vol": np.sqrt(sum_r2 * 365 / (dates[j] - dates[i]).days),
            "premium": premium, "pnl": trade_pnl, "pnl_before_costs": trade_pnl + cost, "costs": -cost,
            "straddles": n_straddles, "notional_usd": notional if ex else np.nan,
            "futures_traded": futures_traded, **parts,
        })
        i = j

    if not trades:
        raise ValueError("history is shorter than one straddle tenor")
    daily = pd.DataFrame(rows).fillna(0.0).groupby("date")[["pnl", *ATTRIBUTION]].sum()
    params = dict(side=side, tenor_days=tenor_days, r=r, q=q, iv_haircut=iv_haircut, hedge=hedge,
                  execution=asdict(ex) if ex else None)
    return BacktestResult(daily, pd.DataFrame(trades), params)


def summarize(res: BacktestResult) -> dict:
    """Headline statistics. Returns are in % of notional per year (additive, not compounded)."""
    pnl, t = res.daily["pnl"], res.trades
    years = (pnl.index[-1] - pnl.index[0]).days / 365.25
    per_year = len(pnl) / years
    cum = pnl.cumsum()
    worst = t.loc[t["pnl"].idxmin()]
    return {
        "period": f"{pnl.index[0]:%Y-%m-%d} to {pnl.index[-1]:%Y-%m-%d}",
        "trades": len(t),
        "annual return %": 100 * pnl.sum() / years,
        "annual vol %": 100 * pnl.std() * np.sqrt(per_year),
        "sharpe": pnl.mean() / pnl.std() * np.sqrt(per_year) if pnl.std() > 0 else np.nan,
        "max drawdown %": 100 * (cum - cum.cummax()).min(),
        "win rate %": 100 * (t["pnl"] > 0).mean(),
        "avg premium %": 100 * t["premium"].mean(),
        "avg implied - realised (vol pts)": 100 * (t["entry_iv"] - t["realised_vol"]).mean(),
        "implied > realised (% of trades)": 100 * (t["entry_iv"] > t["realised_vol"]).mean(),
        "costs, annual %": 100 * t["costs"].sum() / years,
        "worst trade %": 100 * worst["pnl"],
        "worst trade entry": f"{worst['entry']:%Y-%m-%d}",
    }


def cost_breakdown(res: BacktestResult) -> pd.DataFrame:
    """Where the transaction costs go: per year, per trade, in $ (at the trade's notional) and vs gross P&L."""
    t = res.trades
    years = (res.daily.index[-1] - res.daily.index[0]).days / 365.25
    gross = t["pnl_before_costs"].sum()
    usd = t["notional_usd"] if t["notional_usd"].notna().all() else pd.Series(np.nan, index=t.index)
    rows = {}
    for c in COST_PARTS:
        rows[c.replace("_", " ")] = {
            "% of notional / yr": 100 * t[c].sum() / years,
            "$ per trade": (t[c] * usd).mean(),
            "$ per yr": (t[c] * usd).sum() / years,
            "% of gross P&L": 100 * t[c].sum() / gross if gross else np.nan,
        }
    out = pd.DataFrame(rows).T
    out.loc["TOTAL"] = out.sum()
    return out


def measure_live(rate: float = 0.04, target_days: float = 30) -> dict:
    """Today's ATM SPX option half-spread and VIX vs ATM IV, from CBOE's delayed quotes."""
    from .data import fetch_cboe_chain, prepare_quotes

    spot, asof, chain = fetch_cboe_chain("_SPX")
    quotes, _ = prepare_quotes(chain, spot, asof, rate=rate)
    out = measure_option_spread(quotes, target_days)
    vix = fetch_cboe_chain("_VIX")[0] / 100
    return {**out, "asof": asof, "spot": spot, "vix": vix, "vix_minus_atm": 100 * (vix - out["atm_iv"])}
