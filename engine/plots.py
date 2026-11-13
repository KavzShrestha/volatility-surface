"""Matplotlib visualisations: 3D surface, smile slices, ATM term structure, Greeks."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # write files; no window needed
import matplotlib.pyplot as plt
import numpy as np

from .surface import VolSurface


def plot_surface_3d(s: VolSurface, path, title="Implied volatility surface"):
    m = np.linspace(0.80, 1.15, 60)
    T = np.linspace(s.T[0], s.T[-1], 60)
    MM, TT = np.meshgrid(m, T)
    IV = np.where(s.in_quoted_range(MM * s.spot, TT), s.iv(MM * s.spot, TT), np.nan)  # no extrapolation
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(projection="3d")
    surf = ax.plot_surface(MM, TT * 365, IV * 100, cmap="viridis", linewidth=0, antialiased=True,
                           vmin=np.nanmin(IV) * 100, vmax=np.nanmax(IV) * 100)
    ax.set_xlabel("Strike / Spot")
    ax.set_ylabel("Days to expiry")
    ax.set_zlabel("Implied vol (%)")
    ax.set_title(title)
    ax.view_init(elev=25, azim=-130)
    fig.colorbar(surf, shrink=0.5, label="IV (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_smiles(s: VolSurface, quotes, path, n=6, truth=None):
    idx = np.unique(np.linspace(0, len(s.T) - 1, n).round().astype(int))
    fig, axes = plt.subplots(2, (len(idx) + 1) // 2, figsize=(14, 7), sharey=False)
    for ax, i in zip(axes.ravel(), idx):
        T = s.T[i]
        g = quotes[np.isclose(quotes["T"], T)]
        K = g["strike"].to_numpy()
        ax.errorbar(K / s.spot, g["iv"] * 100,
                    yerr=[(g["iv"] - g["iv_bid"]).clip(lower=0) * 100, (g["iv_ask"] - g["iv"]).clip(lower=0) * 100],
                    fmt=".", ms=3, color="tab:gray", alpha=0.7, label="market mid (bid/ask bars)")
        Kf = np.linspace(K.min(), K.max(), 300)
        ax.plot(Kf / s.spot, s.iv(Kf, T) * 100, color="tab:blue", lw=2, label="fitted surface")
        if truth is not None:
            ax.plot(Kf / s.spot, truth(Kf, T) * 100, "--", color="tab:red", lw=1.2, label="true (synthetic)")
        ax.set_title(f"{T * 365:.0f} days")
        ax.set_xlabel("K / S")
        ax.set_ylabel("IV (%)")
        ax.grid(alpha=0.3)
    axes.ravel()[0].legend(fontsize=8)
    for ax in axes.ravel()[len(idx):]:
        ax.axis("off")
    fig.suptitle("Volatility smile (skew) by expiry")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_term_structure(s: VolSurface, path):
    T = np.linspace(s.T[0], s.T[-1], 200)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m, style in ((0.90, ":"), (0.95, "--"), (1.00, "-"), (1.05, "-.")):
        ax.plot(T * 365, s.iv(m * s.forward(T), T) * 100, style, label=f"K/F = {m:.2f}")
    ax.scatter(s.T * 365, s.iv(s.forward(s.T), s.T) * 100, s=12, color="k", zorder=3, label="quoted expiries (ATM)")
    ax.set_xlabel("Days to expiry")
    ax.set_ylabel("Implied vol (%)")
    ax.set_title("Term structure of implied volatility")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_greeks(s: VolSurface, T, path):
    K = np.linspace(0.8, 1.2, 300) * s.spot
    gc = s.greeks(K, T, True)
    gp = s.greeks(K, T, False)
    names = ["delta", "gamma", "vega", "theta", "rho", "vanna", "volga"]
    scale = {"vega": 0.01, "vanna": 0.01, "volga": 0.0001, "theta": 1 / 365, "rho": 0.01}
    units = {"vega": "per vol pt", "vanna": "Δ per vol pt", "volga": "vega per vol pt",
             "theta": "per day", "rho": "per 1% rate"}
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, n in zip(axes.ravel(), names):
        sc = scale.get(n, 1.0)
        ax.plot(K / s.spot, gc[n] * sc, label="call")
        if n not in ("gamma", "vega", "vanna", "volga"):
            ax.plot(K / s.spot, gp[n] * sc, "--", label="put")
        ax.axvline(s.forward(T) / s.spot, color="gray", lw=0.6)
        ax.set_title(f"{n} ({units.get(n, '')})".replace(" ()", ""))
        ax.set_xlabel("K / S")
        ax.grid(alpha=0.3)
    axes.ravel()[0].legend()
    ax = axes.ravel()[-1]
    ax.plot(K / s.spot, s.iv(K, T) * 100, color="tab:green")
    ax.set_title("surface vol used (%)")
    ax.set_xlabel("K / S")
    ax.grid(alpha=0.3)
    fig.suptitle(f"Greeks vs strike at {T * 365:.0f} days (priced off the surface)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- delta-hedged backtest
def _shade_worst(ax, trades, n=3):
    for _, t in trades.nsmallest(n, "pnl").iterrows():
        ax.axvspan(t["entry"], t["expiry"], color="tab:red", alpha=0.15, lw=0)


def plot_backtest_equity(res, path, res_no_cost=None, title="Delta-hedged ATM straddle"):
    d = res.daily
    cum = d["pnl"].cumsum() * 100
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    if res_no_cost is not None:
        ax.plot(res_no_cost.daily["pnl"].cumsum() * 100, color="tab:gray", lw=1, label="before costs")
    ax.plot(cum, color="tab:blue", lw=1.5, label="after costs")
    _shade_worst(ax, res.trades)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("Cumulative P&L (% of notional)")
    ax.set_title(f"{title}: cumulative P&L (red = 3 worst trades)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax2.fill_between(cum.index, cum - cum.cummax(), 0, color="tab:red", alpha=0.5)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_vrp(trades, path):
    t = trades.set_index("entry")
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax.plot(t["entry_iv"] * 100, color="tab:blue", lw=1.2, label="implied at entry (VIX)")
    ax.plot(t["realised_vol"] * 100, color="tab:orange", lw=1.2, label="realised over the next 30 days")
    ax.set_ylabel("Volatility (%)")
    ax.set_title("Implied vs subsequently realised volatility, per straddle")
    ax.legend()
    ax.grid(alpha=0.3)
    spread = (t["entry_iv"] - t["realised_vol"]) * 100
    ax2.bar(spread.index, spread, width=25, color=np.where(spread >= 0, "tab:green", "tab:red"))
    ax2.axhline(spread.mean(), color="k", ls="--", lw=1, label=f"mean {spread.mean():.1f} vol pts")
    ax2.set_ylabel("Implied - realised")
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_pnl_vs_vol(trades, path, side=-1):
    x = (trades["entry_iv"] - trades["realised_vol"]) * 100
    y = trades["pnl_before_costs"] * 100
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, s=12, alpha=0.6, color="tab:blue")
    slope, icpt = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 2)
    ax.plot(xs, icpt + slope * xs, color="tab:orange", lw=2,
            label=f"fit: {slope:.2f}% of notional per vol pt, corr {np.corrcoef(x, y)[0, 1]:.2f}")
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("Implied at entry - realised over trade (vol pts)")
    ax.set_ylabel(f"Trade P&L before costs (% of notional), {'short' if side < 0 else 'long'} straddle")
    ax.set_title("Hedged straddle P&L is a bet on implied vs realised vol")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_attribution(res, path):
    from .backtest import ATTRIBUTION

    d = res.daily
    fig, ax = plt.subplots(figsize=(12, 5.5))
    labels = {"realised_vs_implied": "realised vs implied (gamma - theta)", "vega": "vega (IV changes)",
              "delta": "unhedged delta", "residual": "residual (vanna/volga cross-terms, discrete hedging)",
              "costs": "transaction costs"}
    for col in ATTRIBUTION:
        if d[col].abs().sum() > 0:
            ax.plot(d[col].cumsum() * 100, lw=1.3, label=labels[col])
    ax.plot(d["pnl"].cumsum() * 100, color="k", lw=2, label="total")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("Cumulative P&L (% of notional)")
    ax.set_title("Where the P&L comes from")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
