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
