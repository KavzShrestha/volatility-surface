# Options Pricing & Volatility Surface Engine

Python · NumPy · SciPy · pandas · matplotlib · Streamlit · Plotly

```
python main.py                # live SPX chain from CBOE's public delayed-quotes REST API
python main.py --synthetic    # chain priced off a known SSVI surface: errors measured against the truth
python main.py --rate 0.043   # set the flat risk-free rate used for discounting
python -m pytest -q           # 14 unit tests
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

## Known limitations (good interview material)

- Flat rate input, not an OIS/SOFR curve; flat-vol extrapolation in the wings (2–8 vol pt error vs truth there).
- A smoothing spline doesn't *guarantee* no butterfly arbitrage, it only checks for it. SVI/SSVI with
  constraints, or fitting directly in price space with convexity constraints, would.
- Sticky-strike Greeks: they ignore how the smile moves with spot (no skew/"shadow" delta).
- SPY/single-stock options are American; this engine is European. Use SPX, or a binomial/PDE model with discrete dividends.
- Delayed (~15 min) snapshot data; no handling of stale quotes or bid/ask timestamps.
