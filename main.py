"""End-to-end run: validate pricer -> ingest chain -> implied vols -> surface -> checks -> plots.

    python main.py                 # live SPX chain from CBOE (falls back to synthetic if offline)
    python main.py --synthetic     # known SSVI surface, so errors can be measured against the truth
    python main.py --symbol SPY    # any CBOE symbol (SPY options are American: see README)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from engine import VolSurface, bs_greeks, bs_price, put_call_parity_residual
from engine.data import fetch_cboe_chain, prepare_quotes, synthetic_chain
from engine.plots import plot_greeks, plot_smiles, plot_surface_3d, plot_term_structure

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def self_checks():
    section("1. Pricer self-checks")
    # Textbook value: Hull, Options Futures & Other Derivatives, Example 15.6
    c = bs_price(42, 40, 0.5, 0.10, 0.20, 0, True)
    p = bs_price(42, 40, 0.5, 0.10, 0.20, 0, False)
    print(f"Hull example  S=42 K=40 T=0.5 r=10% vol=20%:  call {c:.4f} (book 4.76)  put {p:.4f} (book 0.81)")

    rng = np.random.default_rng(0)
    n = 1_000_000
    S, K = rng.uniform(50, 150, n), rng.uniform(50, 150, n)
    T, r, q, v = rng.uniform(0.01, 3, n), rng.uniform(-0.01, 0.08, n), rng.uniform(0, 0.05, n), rng.uniform(0.05, 1.0, n)
    t0 = time.perf_counter()
    bs_price(S, K, T, r, v, q, True)
    dt = time.perf_counter() - t0
    res = np.abs(put_call_parity_residual(S, K, T, r, v, q)).max()
    print(f"Priced {n:,} random calls in {dt * 1000:.0f} ms (vectorised); max |put-call parity residual| = {res:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="_SPX")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="output")
    ap.add_argument("--rate", type=float, default=0.04, help="flat risk-free rate (production: OIS/SOFR curve)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    self_checks()

    section("2. Ingest option chain")
    truth = None
    if args.synthetic:
        spot, asof, chain, truth = synthetic_chain()
        label = "SYNTHETIC (SSVI)"
    else:
        try:
            spot, asof, chain = fetch_cboe_chain(args.symbol)
            label = args.symbol.lstrip("_")
        except Exception as e:  # network down, API change, etc.
            print(f"Live fetch failed ({e!r}); using synthetic chain instead")
            spot, asof, chain, truth = synthetic_chain()
            label = "SYNTHETIC (SSVI)"
    print(f"Underlying {label}: spot {spot:,.2f} as of {asof}, {len(chain):,} listed options, "
          f"{chain['expiry'].nunique()} expiries")

    section("3. Clean quotes, imply forwards from put-call parity, invert implied vols")
    quotes, st = prepare_quotes(chain, spot, asof, rate=args.rate)
    print(f"{st['raw']:,} raw -> {st['clean']:,} clean OTM quotes across {st['expiries_used']} expiries")
    print(f"IV solver: {st['newton']:,} by Newton-Raphson, {st['brent']:,} by Brent fallback; "
          f"parity-forward fallbacks: {st['forward_fallbacks']}")
    fwd = quotes.groupby("T").agg(days=("T", lambda t: t.iloc[0] * 365), forward=("forward", "first"),
                                  r_input=("r", "first"), q_implied=("q", "first"), n=("iv", "size"))
    print("\nParity-implied forward and dividend yield per expiry (first 8):")
    print(fwd.head(8).round(4).to_string())
    if quotes["vendor_iv"].notna().any():
        d = (quotes["iv"] - quotes["vendor_iv"]).abs() * 100
        print(f"\nOur IV vs CBOE's published IV: median |diff| {d.median():.2f} vol pts, 90th pct {d.quantile(0.9):.2f}")

    section("4. Out-of-sample test: fit on 80% of quotes, predict the held-out 20%")
    rng = np.random.default_rng(1)
    test = rng.random(len(quotes)) < 0.2
    s_train = VolSurface(quotes[~test], spot)
    held = quotes[test]
    pred = s_train.iv(held["strike"].to_numpy(), held["T"].to_numpy())
    err = (pred - held["iv"].to_numpy()) * 100
    inside = (pred >= held["iv_bid"].fillna(0).to_numpy() - 1e-9) & (pred <= held["iv_ask"].to_numpy() + 1e-9)
    print(f"{test.sum()} held-out quotes: RMSE {np.sqrt(np.mean(err**2)):.2f} vol pts, "
          f"median |err| {np.median(np.abs(err)):.2f}; {inside.mean() * 100:.1f}% priced inside the bid-ask")

    section("5. Full surface + arbitrage diagnostics")
    surf = VolSurface(quotes, spot)
    fit = (surf.iv(quotes["strike"].to_numpy(), quotes["T"].to_numpy()) - quotes["iv"].to_numpy()) * 100
    print(f"In-sample fit: RMSE {np.sqrt(np.mean(fit**2)):.2f} vol pts")
    arb = surf.arbitrage_report()
    print(f"Butterfly violations (non-convex calls) {arb['butterfly_viol'].sum()} / "
          f"{len(arb) * 198} grid points; monotonicity violations {arb['monotonic_viol'].sum()}; "
          f"raw calendar violations {arb['calendar_viol_raw'].sum()} (removed by variance floor)")
    bad = arb[(arb[["butterfly_viol", "monotonic_viol", "calendar_viol_raw"]].sum(axis=1)) > 0]
    if len(bad):
        print("Expiries with any violation:\n" + bad.to_string(index=False))
    if truth is not None:
        m = np.linspace(0.85, 1.10, 26)
        Tg = np.linspace(surf.T[0], surf.T[-1], 40)
        KK, TT = np.meshgrid(m * spot, Tg)
        e = (surf.iv(KK, TT) - truth(KK, TT)) * 100
        # "Quoted region": within ~2 standard deviations of the forward, where options actually trade.
        inside = np.abs(np.log(KK / surf.forward(TT))) <= 2 * 0.2 * np.sqrt(TT)
        print(f"vs TRUE synthetic surface on a {e.size}-point off-grid mesh:")
        print(f"   inside quoted region: RMSE {np.sqrt(np.mean(e[inside]**2)):.2f} vol pts, "
              f"max |err| {np.abs(e[inside]).max():.2f}")
        print(f"   far wings (extrapolated flat): RMSE {np.sqrt(np.mean(e[~inside]**2)):.2f}, "
              f"max |err| {np.abs(e[~inside]).max():.2f}")

    section("6. Price off-market strikes / expiries from the surface")
    targets = [(1.025, 45), (0.93, 100), (1.00, 200), (0.85, 300)]
    rows = []
    for m, days in targets:
        K = round(spot * m / 5) * 5 + 2.5  # deliberately a strike that is not normally listed
        T = days / 365
        for cp in (True, False):
            g = surf.greeks(K, T, cp)
            row = {"type": "call" if cp else "put", "K": K, "days": days, "iv_%": surf.iv(K, T) * 100,
                   "price": surf.price(K, T, cp), "delta": g["delta"], "gamma": g["gamma"],
                   "vega/pt": g["vega"] / 100, "theta/day": g["theta"] / 365, "rho/1%": g["rho"] / 100,
                   "vanna/pt": g["vanna"] / 100, "volga/pt": g["volga"] / 1e4}
            rows.append({k: (v if isinstance(v, str) else float(v)) for k, v in row.items()})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    section("7. Outputs")
    surf.grid().to_csv(out / "surface_grid.csv")
    quotes.to_csv(out / "clean_quotes.csv", index=False)
    plot_surface_3d(surf, out / "surface_3d.png", f"{label} implied volatility surface ({asof:%Y-%m-%d})")
    plot_smiles(surf, quotes, out / "smiles.png", truth=truth)
    plot_term_structure(surf, out / "term_structure.png")
    plot_greeks(surf, 60 / 365, out / "greeks.png")
    print("IV grid (rows = years to expiry, columns = K/S), every 5th row:")
    print((surf.grid(np.array([0.85, 0.9, 0.95, 1.0, 1.05, 1.1])) * 100).iloc[::5].round(2).to_string())
    print(f"\nWrote {', '.join(sorted(p.name for p in out.iterdir()))} to {out.resolve()}")


if __name__ == "__main__":
    main()
