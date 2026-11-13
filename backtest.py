"""Delta-hedged straddle backtest on SPX, priced at VIX.

    python backtest.py                    # sell 30-day ATM straddles since 1990, hedge daily
    python backtest.py --side long        # buy them instead
    python backtest.py --haircut 1.5      # price at VIX - 1.5 vol pts (VIX sits above ATM IV)
    python backtest.py --no-hedge         # same straddles, never hedged
    python backtest.py --notional 1e6 --futures MES   # a $1M book hedged with Micro E-minis
    python backtest.py --opt-spread 0.25  # override the option spread measured from today's chain
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from engine.backtest import Execution, cost_breakdown, load_history, measure_live, run_backtest, summarize
from engine.plots import plot_attribution, plot_backtest_equity, plot_pnl_vs_vol, plot_vrp


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", choices=["short", "long"], default="short")
    ap.add_argument("--tenor", type=int, default=30, help="calendar days to expiry")
    ap.add_argument("--start", default="1990-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--rate", type=float, default=0.03, help="flat risk-free rate")
    ap.add_argument("--div", type=float, default=0.015, help="flat dividend yield")
    ap.add_argument("--haircut", type=float, default=0.0, help="vol pts subtracted from VIX to approximate ATM IV")
    ap.add_argument("--notional", type=float, default=10e6, help="$ notional per trade")
    ap.add_argument("--futures", choices=["ES", "MES"], default="ES", help="hedge with E-mini or Micro E-mini")
    ap.add_argument("--opt-spread", type=float, default=None,
                    help="option half bid-ask in vol pts (default: measured from today's SPX chain)")
    ap.add_argument("--opt-fee", type=float, default=1.50, help="$ per SPX option contract per side")
    ap.add_argument("--fut-fee", type=float, default=None, help="$ per futures contract per side (ES 2.00, MES 0.60)")
    ap.add_argument("--no-hedge", action="store_true")
    ap.add_argument("--out", default="output_backtest")
    ap.add_argument("--refresh", action="store_true", help="re-download SPX/VIX history")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    section("1. Data: SPX and VIX daily closes (CBOE public history files)")
    market = load_history(refresh=args.refresh).loc[args.start:args.end]
    print(f"{len(market):,} days, {market.index[0]:%Y-%m-%d} to {market.index[-1]:%Y-%m-%d}; "
          f"SPX {market['spot'].iloc[0]:,.0f} -> {market['spot'].iloc[-1]:,.0f}, "
          f"VIX mean {market['iv'].mean() * 100:.1f}, range {market['iv'].min() * 100:.1f}-{market['iv'].max() * 100:.1f}")

    section("2. Execution costs")
    measured = None
    if args.opt_spread is None:
        try:
            measured = measure_live(args.rate, args.tenor)
            print(f"Today's SPX chain ({measured['asof']:%Y-%m-%d %H:%M}): {measured['days']:.0f}-day ATM options "
                  f"(strikes {', '.join(f'{k:,.0f}' for k in measured['strikes'])}) quote "
                  f"+/-{measured['half_spread_pts']:.2f} pts around mid = +/-{measured['half_spread_vol']:.3f} vol pts")
            print(f"VIX now {measured['vix'] * 100:.2f} vs ATM IV {measured['atm_iv'] * 100:.2f}: "
                  f"VIX is {measured['vix_minus_atm']:+.2f} vol pts above ATM (try --haircut near this)")
        except Exception as e:
            print(f"Could not measure live spreads ({e!r}); using the default")
    opt_spread = args.opt_spread if args.opt_spread is not None else (
        measured["half_spread_vol"] if measured else Execution.option_half_spread_vol)
    mult, fee = (50.0, 2.00) if args.futures == "ES" else (5.0, 0.60)
    ex = Execution(notional=args.notional, option_half_spread_vol=opt_spread, option_fee=args.opt_fee,
                   fut_multiplier=mult, fut_half_spread_pts=0.125, fut_fee=args.fut_fee or fee)
    print(f"Book ${ex.notional:,.0f} per trade in whole SPX straddles (x{100}), hedged with whole {args.futures} "
          f"futures (${mult:.0f} x index, half-spread 0.125 pts, ${ex.fut_fee:.2f}/contract); option half-spread "
          f"{ex.option_half_spread_vol:.3f} vol pts + ${ex.option_fee:.2f}/contract")

    side = -1 if args.side == "short" else 1
    base = dict(side=side, tenor_days=args.tenor, r=args.rate, q=args.div, iv_haircut=args.haircut,
                hedge=not args.no_hedge)
    res = run_backtest(market, execution=ex, **base)
    gross = run_backtest(market, **base)

    section(f"3. {args.side.title()} {args.tenor}-day ATM straddles, "
            f"{'unhedged' if args.no_hedge else 'delta-hedged daily'}")
    s, g = summarize(res), summarize(gross)
    table = pd.DataFrame({"realistic execution": s, "frictionless": g})
    print(table.map(lambda v: f"{v:,.2f}" if isinstance(v, (float, np.floating)) else v).to_string())
    t = res.trades
    print(f"\nPer trade: {t['straddles'].median():.0f} straddles (median; {t['straddles'].iloc[0]:.0f} in "
          f"{t['entry'].iloc[0]:%Y}, {t['straddles'].iloc[-1]:.0f} in {t['entry'].iloc[-1]:%Y}), "
          f"{t['futures_traded'].median():.0f} futures contracts traded (median)")
    print("\nCost breakdown:")
    print(cost_breakdown(res).round(3).to_string())

    section("4. P&L attribution (total over the period, % of notional)")
    attr = res.daily[["realised_vs_implied", "vega", "delta", "residual", "costs", "pnl"]].sum() * 100
    print(attr.rename({"pnl": "TOTAL"}).round(2).to_string())

    section("5. Worst and best trades (after costs, % of notional)")
    cols = ["entry", "expiry", "entry_iv", "realised_vol", "premium", "pnl"]
    fmt = lambda df: df[cols].assign(entry_iv=df["entry_iv"] * 100, realised_vol=df["realised_vol"] * 100,
                                     premium=df["premium"] * 100, pnl=df["pnl"] * 100).round({c: 2 for c in cols[2:]}).to_string(index=False)
    print("Worst 5:\n" + fmt(res.trades.nsmallest(5, "pnl")))
    print("\nBest 5:\n" + fmt(res.trades.nlargest(5, "pnl")))

    section("6. By year (after costs, % of notional)")
    yearly = res.daily["pnl"].groupby(res.daily.index.year).sum() * 100
    print(" ".join(f"{y}:{v:+.1f}" for y, v in yearly.items()))
    print(f"Losing years: {(yearly < 0).sum()} of {len(yearly)}")

    section("7. Sensitivity: Sharpe (annual return %) by VIX haircut and option spread")
    grid = {}
    for hc in (0.0, 1.0, 2.0, 3.0, 5.0):
        row = {}
        for label, sp in ((f"spread {opt_spread:.2f}", opt_spread), ("spread 0.25", 0.25), ("spread 0.50", 0.50)):
            e = dataclasses.replace(ex, option_half_spread_vol=sp)
            r = summarize(run_backtest(market, execution=e, **{**base, "iv_haircut": hc}))
            row[label] = f"{r['sharpe']:.2f} ({r['annual return %']:+.2f}%)"
        grid[f"VIX - {hc:.0f} pts"] = row
    print(pd.DataFrame(grid).T.to_string())

    section("8. Outputs")
    res.trades.to_csv(out / "trades.csv", index=False)
    res.daily.to_csv(out / "daily_pnl.csv")
    plot_backtest_equity(res, out / "equity.png", gross,
                         title=f"{args.side.title()} {args.tenor}d ATM SPX straddle, "
                               f"{'unhedged' if args.no_hedge else 'delta-hedged'}")
    plot_vrp(res.trades, out / "implied_vs_realised.png")
    plot_pnl_vs_vol(gross.trades, out / "pnl_vs_vol_spread.png", side)
    plot_attribution(res, out / "attribution.png")
    print(f"Wrote {', '.join(sorted(p.name for p in out.iterdir()))} to {out.resolve()}")


if __name__ == "__main__":
    main()
