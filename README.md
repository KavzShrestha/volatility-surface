# Options Pricing & Volatility Surface Engine

Python · NumPy · SciPy · pandas · matplotlib · Streamlit · Plotly

```
python main.py                # live SPX chain from CBOE's public delayed-quotes REST API
python main.py --synthetic    # chain priced off a known SSVI surface: errors measured against the truth
python main.py --rate 0.043   # set the flat risk-free rate used for discounting
python -m pytest -q           # 22 unit tests
python backtest.py            # delta-hedged straddle backtest on SPX/VIX since 1990 (~1 min)
streamlit run app.py          # interactive dashboard (pip install streamlit plotly)
```
Outputs go to `output/` (or `--out DIR`): `surface_3d.png`, `smiles.png`, `term_structure.png`,
`greeks.png`, `surface_grid.csv`, `clean_quotes.csv`.

## Pipeline

| Step | File | What it does |
|---|---|---|
| Pricer | `engine/black_scholes.py` | Vectorised Black-Scholes-Merton prices plus delta, gamma, vega, theta, rho, vanna, volga |
| IV solver | `engine/implied_vol.py` | Vectorised Newton-Raphson; quotes it can't solve (tiny vega, step outside bounds) go to Brent's method. NaN for prices outside no-arbitrage bounds |
| Ingestion | `engine/data.py` | CBOE REST chain → parse OCC symbols → filter (zero bids, wide spreads, <5 days) → **forward implied from put-call parity** per expiry → keep OTM side only → IV of bid, mid and ask |
| Surface | `engine/surface.py` | Per expiry: cubic smoothing spline of IV in standardised log-moneyness k/√T, weighted by 1/spread², smoothest curve that fits within half the bid-ask. Across expiries: linear in total variance at fixed k. Calendar floor; butterfly/monotonicity diagnostics |
| Plots | `engine/plots.py` | 3D surface, smiles vs market, term structure, Greeks vs strike |

## How correctness is checked

1. **Textbook values**: Hull Example 15.6 (call 4.7594, put 0.8086).
2. **Put-call parity** on 1M random inputs: residual ~1e-14. *Caveat: BS satisfies parity by
   construction, so this only catches coding errors such as a sign or discounting slip. It is not model validation.*
3. **Greeks vs finite differences**: every analytic Greek is bumped and compared (rtol 1e-4).
4. **IV round trip**: price at a known vol → invert → recover it to 1e-6 over 20k random cases.
   This test caught a real bug: the Newton stopping rule tested price error rather than vol step.
5. **Brent path forced** (`max_iter=1`) to prove the fallback works.
6. **Synthetic truth test**: quotes generated from a known arbitrage-free SSVI surface, with noise
   and spreads. The engine recovers the dividend yield (1.2%) and the true vols to ~0.1 vol pt RMSE
   inside the quoted region, including at expiries that were never quoted.
7. **Independent benchmark**: our IVs vs CBOE's published IVs, median difference ~0.05 vol pts.
8. **Out-of-sample**: fit on 80% of live quotes and predict the other 20%: ~0.14 vol pt RMSE,
   ~92% of predictions fall inside the bid-ask (results vary with the day's data).
9. **No-arbitrage diagnostics**: call prices convex and decreasing in K; total variance non-decreasing in T.

## Does selling implied vol make money? A delta-hedged straddle backtest

`engine/backtest.py`, `backtest.py`. Every 30 days, sell an ATM SPX straddle priced at VIX, delta-hedge it
at each close, hold to expiry, roll. Hedging removes the bet on direction, so each trade's P&L is
≈ ½Γ S² (realised variance − implied variance) + vega·ΔIV. Sized at $1 notional per straddle; P&L in % of notional.
Daily SPX and VIX closes come from CBOE's public history files (cached in `data/`).

Execution is modelled the way a desk would trade it: a $10M book in whole SPX option contracts, hedged
with whole E-mini S&P futures. Costs are the option half-spread **measured from today's live chain**
(about ±0.05 vol pts for 30-day ATM), ~$1.50 per option contract, half a 0.25-point tick plus ~$2 per futures
contract on every hedge trade. 1990–2026:

| | |
|---|---|
| Implied above subsequently realised | 85% of months, by 4.2 vol pts on average |
| Annual return / vol / Sharpe | +10.8% / 4.8% / 2.3 (frictionless: +11.6% / 2.4); losing years: 1 of 37 (2008) |
| Trading costs | 0.78% of notional a year, ~7% of gross P&L; ~$78k/yr on $10M. Futures hedging is 75% of it (median 489 contracts per trade) |
| Max drawdown | −20%, about 4× the annual vol: returns are steady with fat left tails |
| Worst trades | Sep 2008 (−9.2%), Mar 2020 (−8.9%), Feb 2018 "Volmageddon" (−4.6%) |

**Read the headline as an upper bound: the VIX shortcut matters far more than trading costs.** VIX is a
variance-swap rate that includes the put skew, so it sits above the ATM implied vol a straddle actually trades at:
on 2026-11-13 the live chain showed VIX 20.3 vs 30-day ATM IV 15.2, a 5-point gap. Annual return and Sharpe by
haircut: VIX − 1: +8.0% / 1.7; VIX − 2: +5.3% / 1.1; VIX − 3: +2.5% / 0.5; VIX − 5: −3.1% / −0.6. The real answer
needs historical ATM option prices (OptionMetrics, CBOE DataShop). Other simplifications: one VIX level marks the
option for its whole life, flat r and q, hedging only at closes, today's spreads and contract specs applied to all
years (E-minis launched in 1997; 1990s option spreads were wider), no margin.

Tests (`tests/test_backtest.py`) run the backtest on simulated GBM paths with known answers: zero expected
P&L when realised = implied; short wins and long loses when implied > realised; average P&L matches
vega·(σ_r² − σ_i²)/2σ_i within 25%; a flat spot path loses exactly the premium; attribution and each cost component add up;
whole-contract hedging tracks the ideal hedge, more closely for bigger books.

## Known limitations (good interview material)

- Flat rate input, not an OIS/SOFR curve; flat-vol extrapolation in the wings (2–8 vol pt error vs truth there).
- A smoothing spline doesn't *guarantee* no butterfly arbitrage, it only checks for it. SVI/SSVI with
  constraints, or fitting directly in price space with convexity constraints, would.
- Sticky-strike Greeks: they ignore how the smile moves with spot (no skew/"shadow" delta).
- SPY/single-stock options are American; this engine is European. Use SPX, or a binomial/PDE model with discrete dividends.
- Delayed (~15 min) snapshot data; no handling of stale quotes or bid/ask timestamps.
