"""Interactive dashboard for the options pricing & volatility surface engine.

    streamlit run app.py

Tabs:
  1. Black-Scholes calculator   price + Greeks for any inputs, and invert a market price to IV
  2. Market data                where the chain comes from, how it is cleaned, raw vs clean quotes
  3. Volatility surface         3D surface, smile per expiry vs market bid/ask, term structure
  4. IV explorer                pick any strike & expiry: surface IV, price, Greeks; or step through
                                the Newton solve for one listed quote
  5. Diagnostics                arbitrage checks, out-of-sample fit
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import VolSurface, bs_greeks, bs_price, implied_vol, put_call_parity_residual
from engine.data import CBOE_URL, YEAR_SECONDS, fetch_cboe_chain, prepare_quotes, synthetic_chain

st.set_page_config(page_title="Vol Surface Engine", layout="wide")

# Colour roles (validated categorical palette): market = blue, model = orange, truth = aqua.
MARKET, MODEL, TRUTH, BAND = "#2a78d6", "#eb6834", "#1baf7a", "rgba(42,120,214,0.25)"
GREEK_HELP = {
    "delta": "dV/dS", "gamma": "d²V/dS²", "vega/pt": "per 1 vol point", "theta/day": "per calendar day",
    "rho/1%": "per 1% rate", "vanna/pt": "d(delta)/d(vol), per vol pt", "volga/pt": "d(vega)/d(vol), per vol pt²",
}


def scaled_greeks(g: dict) -> dict:
    """Engine Greeks are per unit; show desk-style units."""
    return {"delta": g["delta"], "gamma": g["gamma"], "vega/pt": g["vega"] / 100, "theta/day": g["theta"] / 365,
            "rho/1%": g["rho"] / 100, "vanna/pt": g["vanna"] / 100, "volga/pt": g["volga"] / 1e4}


def fmt(x, nd=2):
    return f"{float(x):,.{nd}f}"


# ============================================================================ data loading (cached)
@st.cache_data(ttl=900, show_spinner="Downloading chain from CBOE…")
def load_live(symbol: str):
    spot, asof, chain = fetch_cboe_chain(symbol)
    return spot, asof, chain


@st.cache_data(show_spinner="Cleaning quotes and solving implied vols…")
def load_quotes(chain, spot, asof, rate, min_days, max_days, min_bid, max_rel_spread, max_abs_k):
    return prepare_quotes(chain, spot, asof, rate=rate, min_days=min_days, max_days=max_days,
                          min_bid=min_bid, max_rel_spread=max_rel_spread, max_abs_k=max_abs_k)


@st.cache_resource(show_spinner="Fitting surface…")
def build_surface(quotes, spot, smooth, enforce_calendar):
    return VolSurface(quotes, spot, smooth=smooth, enforce_calendar=enforce_calendar)


# ============================================================================ sidebar
with st.sidebar:
    st.header("Data source")
    source = st.radio("Option chain", ["Live CBOE (delayed ~15 min)", "Synthetic (known SSVI truth)"])
    symbol = st.text_input("CBOE symbol", "_SPX", help="Indices take a leading underscore: _SPX, _NDX, _RUT. "
                           "Stocks/ETFs like SPY are American-style; this engine is European.",
                           disabled=source.startswith("Synthetic"))
    if st.button("Refresh live data", disabled=source.startswith("Synthetic")):
        load_live.clear()

    st.header("Model inputs")
    rate = st.number_input("Risk-free rate (flat)", value=0.04, step=0.0025, format="%.4f")

    st.header("Cleaning filters")
    min_days, max_days = st.slider("Days to expiry", 0, 1000, (5, 800))
    min_bid = st.number_input("Min bid", value=0.05, step=0.05, format="%.2f")
    max_rel_spread = st.slider("Max relative bid-ask spread", 0.05, 1.0, 0.30, 0.05)
    max_abs_k = st.slider("Max |log-moneyness|", 0.1, 1.5, 0.6, 0.05)

    st.header("Surface fit")
    smooth = st.toggle("Smoothing spline (vs exact interpolation)", True)
    enforce_cal = st.toggle("Enforce calendar no-arbitrage", True)

truth = None
fetch_error = None
if source.startswith("Live"):
    try:
        spot, asof, chain = load_live(symbol)
        label, origin = symbol.lstrip("_"), CBOE_URL.format(symbol=symbol)
    except Exception as e:  # offline, bad symbol, API change
        fetch_error = repr(e)
if source.startswith("Synthetic") or fetch_error:
    spot, asof, chain, truth = synthetic_chain()
    label, origin = "SYNTHETIC", "engine.data.synthetic_chain(): SSVI surface + bid/ask noise, generated locally"
if fetch_error:
    st.warning(f"Live fetch failed ({fetch_error}). Showing the synthetic chain instead.")

try:
    quotes, stats = load_quotes(chain, spot, asof, rate, min_days, max_days, min_bid, max_rel_spread, max_abs_k)
    surf = build_surface(quotes, spot, smooth, enforce_cal)
except ValueError as e:
    st.error(f"No expiries survive the cleaning filters ({e}). Loosen the filters in the sidebar.")
    st.stop()

expiries = quotes.groupby("T").agg(expiry=("expiry", "first"), n=("iv", "size"),
                                   forward=("forward", "first"), r=("r", "first"), q=("q", "first")).reset_index()
expiries["days"] = expiries["T"] * 365
exp_labels = [f"{e:%Y-%m-%d}  ({d:.0f}d, {n} quotes)" for e, d, n in zip(expiries["expiry"], expiries["days"], expiries["n"])]

st.title(f"{label} volatility surface engine")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Spot", f"{spot:,.2f}")
c2.metric("As of", f"{asof:%Y-%m-%d %H:%M}")
c3.metric("Listed options", f"{stats['raw']:,}")
c4.metric("Clean OTM quotes", f"{stats['clean']:,}")
c5.metric("Expiries fitted", stats["expiries_used"])

tab_bs, tab_data, tab_surf, tab_iv, tab_diag = st.tabs(
    ["Black-Scholes calculator", "Market data", "Volatility surface", "IV explorer", "Diagnostics"])

# ============================================================================ 1. Black-Scholes
with tab_bs:
    st.subheader("Black-Scholes-Merton calculator")
    st.caption("European option. Pre-filled with the market spot; change anything.")
    a, b, c = st.columns(3)
    S = a.number_input("Spot S", value=float(round(spot, 2)), min_value=0.01, format="%.2f")
    K = a.number_input("Strike K", value=float(round(spot, 0)), min_value=0.01, format="%.2f")
    days = b.number_input("Days to expiry", value=30.0, min_value=0.01, step=1.0)
    sigma = b.number_input("Volatility σ (%)", value=20.0, min_value=0.01, step=0.5) / 100
    r_in = c.number_input("Rate r (%)", value=rate * 100, step=0.25) / 100
    q_in = c.number_input("Dividend yield q (%)", value=0.0, step=0.25) / 100
    T = days / 365
    is_call = st.radio("Type", ["Call", "Put"], horizontal=True) == "Call"

    px = float(bs_price(S, K, T, r_in, sigma, q_in, is_call))
    g = scaled_greeks(bs_greeks(S, K, T, r_in, sigma, q_in, is_call))
    cols = st.columns(8)
    cols[0].metric("Price", fmt(px))
    for col, (name, val) in zip(cols[1:], g.items()):
        col.metric(name, fmt(val), help=GREEK_HELP[name])
    other = float(bs_price(S, K, T, r_in, sigma, q_in, not is_call))
    parity = float(put_call_parity_residual(S, K, T, r_in, sigma, q_in))
    st.caption(f"{'Put' if is_call else 'Call'} at same inputs: {fmt(other)}. "
               f"Put-call parity residual C − P − (Se⁻ᵠᵀ − Ke⁻ʳᵀ) = {parity:.2f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Price / Greek vs spot**")
        what = st.selectbox("Plot", ["price", *g.keys()])
        Ss = np.linspace(0.6 * K, 1.4 * K, 300)
        ys = bs_price(Ss, K, T, r_in, sigma, q_in, is_call) if what == "price" \
            else scaled_greeks(bs_greeks(Ss, K, T, r_in, sigma, q_in, is_call))[what]
        fig = go.Figure(go.Scatter(x=Ss, y=ys, mode="lines", line=dict(color=MARKET, width=2), name=what,
                                   hovertemplate="S %{x:,.2f}<br>" + what + " %{y:,.2f}<extra></extra>"))
        if what == "price":
            intrinsic = np.maximum(Ss - K, 0) if is_call else np.maximum(K - Ss, 0)
            fig.add_scatter(x=Ss, y=intrinsic, mode="lines", name="intrinsic",
                            line=dict(color="gray", width=1, dash="dot"))
        fig.add_vline(x=S, line=dict(color="gray", width=1, dash="dash"))
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Spot",
                          yaxis_title=what, hovermode="x unified")
        st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown("**Implied vol from a market price**")
        st.caption("Inverts Black-Scholes with the engine's Newton-Raphson solver (Brent fallback).")
        mkt = st.number_input("Market option price", value=float(round(px, 2)), min_value=0.0, format="%.2f")
        iv, method = implied_vol(mkt, S, K, T, r_in, q_in, is_call, return_method=True)
        iv, method = float(iv), int(method)
        if np.isfinite(iv):
            st.metric("Implied volatility", f"{iv * 100:.2f}%",
                      help="Solved by " + {0: "Newton-Raphson", 1: "Brent's method (fallback)"}[method])
            st.caption(f"Solved by {'Newton-Raphson' if method == 0 else 'Brent fallback'}; "
                       f"repricing at this vol gives {fmt(bs_price(S, K, T, r_in, iv, q_in, is_call))}")
        else:
            dS, dK = S * np.exp(-q_in * T), K * np.exp(-r_in * T)
            lo = max(dS - dK, 0) if is_call else max(dK - dS, 0)
            hi = dS if is_call else dK
            st.error(f"No implied vol: price must lie strictly between the no-arbitrage bounds "
                     f"{fmt(lo)} and {fmt(hi)}.")
        vols = np.linspace(0.01, 1.5, 300)
        fig = go.Figure(go.Scatter(x=vols * 100, y=bs_price(S, K, T, r_in, vols, q_in, is_call), mode="lines",
                                   line=dict(color=MARKET, width=2), name="BS price(σ)",
                                   hovertemplate="σ %{x:.2f}%<br>price %{y:,.2f}<extra></extra>"))
        fig.add_hline(y=mkt, line=dict(color=MODEL, width=1, dash="dash"), annotation_text="market price")
        if np.isfinite(iv):
            fig.add_scatter(x=[iv * 100], y=[mkt], mode="markers", marker=dict(color=MODEL, size=10), name="IV")
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Volatility (%)",
                          yaxis_title="Option price", showlegend=False)
        st.plotly_chart(fig, width="stretch")

# ============================================================================ 2. Market data
with tab_data:
    st.subheader("Where the data comes from")
    if label == "SYNTHETIC":
        st.info(f"**Source:** {origin}. Spot 100, r = 4%, q = 1.2%, 10 expiries from 9 to 548 days. "
                "Because the true surface is known, the engine's error can be measured exactly.")
    else:
        st.info(f"**Source:** CBOE public delayed-quotes REST API  \n`GET {origin}`  \n"
                f"Snapshot timestamp {asof} (US/Eastern, ~15 min delayed). OCC option symbols are parsed "
                "into root / expiry / call-put / strike. For SPX only the SPXW (PM-settled) root is kept.")

    st.markdown("**Pipeline**")
    in_window = chain.assign(T=((chain["expiry"] + pd.Timedelta(hours=16)) - asof).dt.total_seconds() / YEAR_SECONDS)
    two_sided = (in_window["bid"] > 0) & (in_window["ask"] > in_window["bid"])
    window = two_sided & (in_window["T"] * 365 >= min_days) & (in_window["T"] * 365 <= max_days)
    funnel = pd.DataFrame({
        "step": ["Listed options", "Two-sided market (bid > 0, ask > bid)", f"Expiry {min_days}–{max_days} days",
                 f"OTM side only, bid ≥ {min_bid}, spread ≤ {max_rel_spread:.0%}, |k| ≤ {max_abs_k}, IV solvable"],
        "rows": [len(chain), int(two_sided.sum()), int(window.sum()), stats["clean"]],
    })
    fig = go.Figure(go.Bar(x=funnel["rows"], y=funnel["step"], orientation="h", marker_color=MARKET,
                           text=funnel["rows"].map("{:,}".format), textposition="outside",
                           hovertemplate="%{y}<br>%{x:,} rows<extra></extra>"))
    fig.update_layout(height=230, margin=dict(l=10, r=40, t=10, b=10), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Forwards come from put-call parity per expiry (C − P = D·(F − K), median over the 6 strikes "
               f"nearest spot), so the dividend yield is implied by the market rather than assumed. "
               f"Parity fallbacks: {stats['forward_fallbacks']}. IV solver: {stats['newton']:,} Newton, "
               f"{stats['brent']:,} Brent.")

    st.markdown("**Per-expiry forward and implied carry**")
    st.dataframe(expiries.assign(r=expiries["r"] * 100, q=expiries["q"] * 100)
                 .rename(columns={"r": "r %", "q": "implied q %", "n": "quotes"})
                 [["expiry", "days", "quotes", "forward", "r %", "implied q %"]],
                 hide_index=True, width="stretch",
                 column_config={"expiry": st.column_config.DateColumn(format="YYYY-MM-DD"),
                                "days": st.column_config.NumberColumn(format="%.2f"),
                                "forward": st.column_config.NumberColumn(format="%.2f"),
                                "r %": st.column_config.NumberColumn(format="%.2f"),
                                "implied q %": st.column_config.NumberColumn(format="%.2f")})

    view = st.radio("Table", ["Clean quotes (with IV)", "Raw chain"], horizontal=True)
    exp_filter = st.multiselect("Filter expiries", exp_labels)
    if view.startswith("Clean"):
        df = quotes.copy()
        if exp_filter:
            df = df[df["T"].isin(expiries["T"].iloc[[exp_labels.index(e) for e in exp_filter]])]
        df = df.assign(type=np.where(df["is_call"], "C", "P"), days=df["T"] * 365,
                       iv=df["iv"] * 100, iv_bid=df["iv_bid"] * 100, iv_ask=df["iv_ask"] * 100,
                       vendor_iv=df["vendor_iv"] * 100,
                       solver=df["iv_method"].map({0: "Newton", 1: "Brent"}))
        cols = ["expiry", "days", "type", "strike", "k", "bid", "ask", "mid", "iv_bid", "iv", "iv_ask",
                "vendor_iv", "solver", "volume", "open_interest"]
        st.dataframe(df[cols], hide_index=True, width="stretch", height=420,
                     column_config={"expiry": st.column_config.DateColumn(format="YYYY-MM-DD"),
                                    "days": st.column_config.NumberColumn(format="%.2f"),
                                    "k": st.column_config.NumberColumn("k = ln(K/F)", format="%.2f"),
                                    "iv": st.column_config.NumberColumn("IV mid %", format="%.2f"),
                                    "iv_bid": st.column_config.NumberColumn("IV bid %", format="%.2f"),
                                    "iv_ask": st.column_config.NumberColumn("IV ask %", format="%.2f"),
                                    "vendor_iv": st.column_config.NumberColumn("CBOE IV %", format="%.2f")})
    else:
        df = chain.assign(type=np.where(chain["is_call"], "C", "P"), vendor_iv=chain["vendor_iv"] * 100).round(2)
        if exp_filter:
            df = df[df["expiry"].isin(expiries["expiry"].iloc[[exp_labels.index(e) for e in exp_filter]])]
        st.dataframe(df[["expiry", "type", "strike", "bid", "ask", "volume", "open_interest", "vendor_iv"]],
                     hide_index=True, width="stretch", height=420,
                     column_config={"expiry": st.column_config.DateColumn(format="YYYY-MM-DD"),
                                    "vendor_iv": st.column_config.NumberColumn("CBOE IV %", format="%.2f")})
    st.download_button("Download clean quotes CSV", quotes.to_csv(index=False), "clean_quotes.csv", "text/csv")

# ============================================================================ 3. Surface
with tab_surf:
    st.subheader("Implied volatility surface")
    a, b, c = st.columns([2, 2, 1])
    m_lo, m_hi = a.slider("Strike range (K / spot)", 0.5, 1.5, (0.8, 1.15), 0.01)
    t_lo, t_hi = b.slider("Expiry range (days)", float(max(1, surf.T[0] * 365)), float(surf.T[-1] * 365),
                          (float(surf.T[0] * 365), float(surf.T[-1] * 365)))
    mask_extrap = c.toggle("Hide extrapolated", True, help="Blank out points outside the quoted moneyness range")
    show_pts = c.toggle("Show market IVs", True)

    m = np.linspace(m_lo, m_hi, 60)
    Td = np.linspace(t_lo, t_hi, 50)
    KK, TT = np.meshgrid(m * spot, Td / 365)
    Z = surf.iv(KK, TT) * 100
    if mask_extrap:
        Z = np.where(surf.in_quoted_range(KK, TT), Z, np.nan)
    fig = go.Figure(go.Surface(x=m, y=Td, z=Z, colorscale="Blues", reversescale=True, colorbar=dict(title="IV %"),
                               hovertemplate="K/S %{x:.2f}<br>%{y:.0f} days<br>IV %{z:.2f}%<extra>surface</extra>"))
    if show_pts:
        pts = quotes[(quotes["strike"] / spot).between(m_lo, m_hi) & (quotes["T"] * 365).between(t_lo, t_hi)]
        fig.add_scatter3d(x=pts["strike"] / spot, y=pts["T"] * 365, z=pts["iv"] * 100, mode="markers",
                          marker=dict(size=2.5, color=MODEL), name="market mid IV",
                          customdata=np.c_[pts["strike"], pts["mid"]],
                          hovertemplate="K %{customdata[0]:,.0f}<br>%{y:.0f} days<br>mid %{customdata[1]:,.2f}"
                                        "<br>IV %{z:.2f}%<extra>market</extra>")
    fig.update_layout(height=620, margin=dict(l=0, r=0, t=0, b=0),
                      scene=dict(xaxis_title="Strike / spot", yaxis_title="Days to expiry", zaxis_title="IV (%)",
                                 camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8))))
    st.plotly_chart(fig, width="stretch")
    st.caption("Drag to rotate, scroll to zoom, hover for values. Surface: smoothing spline per expiry in "
               "k/√T, linear in total variance across expiries.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Smile at one expiry: fit vs market bid/ask**")
        i = st.selectbox("Expiry", range(len(exp_labels)), format_func=lambda j: exp_labels[j], key="smile_exp")
        t = expiries["T"].iloc[i]
        g_q = quotes[quotes["T"] == t]
        x_axis = st.radio("x-axis", ["Strike", "log-moneyness k"], horizontal=True)
        Kgrid = np.linspace(g_q["strike"].min(), g_q["strike"].max(), 300)
        xg = Kgrid if x_axis == "Strike" else np.log(Kgrid / g_q["forward"].iloc[0])
        xq = g_q["strike"] if x_axis == "Strike" else g_q["k"]
        fig = go.Figure()
        fig.add_scatter(x=xq, y=g_q["iv"] * 100, mode="markers", name="market mid", marker=dict(color=MARKET, size=8),
                        error_y=dict(type="data", symmetric=False, color=BAND,
                                     array=(g_q["iv_ask"] - g_q["iv"]) * 100,
                                     arrayminus=(g_q["iv"] - g_q["iv_bid"]).fillna(0) * 100),
                        customdata=np.c_[g_q["strike"], g_q["bid"], g_q["ask"], g_q["iv_bid"] * 100, g_q["iv_ask"] * 100],
                        hovertemplate="K %{customdata[0]:,.2f}<br>bid/ask %{customdata[1]:.2f} / %{customdata[2]:.2f}"
                                      "<br>IV %{customdata[3]:.2f} – %{y:.2f} – %{customdata[4]:.2f}%<extra></extra>")
        fig.add_scatter(x=xg, y=surf.iv(Kgrid, np.full_like(Kgrid, t)) * 100, mode="lines", name="fitted surface",
                        line=dict(color=MODEL, width=2), hovertemplate="IV %{y:.2f}%<extra>fit</extra>")
        if truth is not None:
            fig.add_scatter(x=xg, y=truth(Kgrid, t) * 100, mode="lines", name="true SSVI",
                            line=dict(color=TRUTH, width=2, dash="dash"))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=x_axis,
                          yaxis_title="Implied vol (%)", legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown("**Term structure at fixed strike / spot**")
        picks = st.multiselect("K / spot", [0.85, 0.9, 0.95, 1.0, 1.05, 1.1], default=[0.9, 1.0, 1.05])
        Tg = np.linspace(surf.T[0], surf.T[-1], 200)
        fig = go.Figure()
        palette = [MARKET, MODEL, TRUTH, "#eda100", "#e87ba4", "#008300"]
        for col, mm in zip(palette, sorted(picks)):
            fig.add_scatter(x=Tg * 365, y=surf.iv(mm * spot, Tg) * 100, mode="lines", name=f"{mm:.2f}",
                            line=dict(color=col, width=2), hovertemplate="%{x:.0f}d  IV %{y:.2f}%<extra>" + f"{mm:.2f}</extra>")
        for d in expiries["days"]:
            fig.add_vline(x=d, line=dict(color="rgba(128,128,128,0.25)", width=1))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Days to expiry",
                          yaxis_title="Implied vol (%)", hovermode="x unified", legend=dict(title="K/S"))
        st.plotly_chart(fig, width="stretch")
        st.caption("Faint vertical lines mark quoted expiries; between them the curve is interpolated in total variance.")

# ============================================================================ 4. IV explorer
with tab_iv:
    st.subheader("Look up any strike and expiry")
    mode = st.radio("Pick", ["Any point on the surface (interpolated)", "A listed market quote"], horizontal=True)

    if mode.startswith("Any"):
        a, b = st.columns(2)
        K_pick = a.slider("Strike", float(round(0.6 * spot)), float(round(1.4 * spot)), float(round(spot)),
                          step=float(max(0.5, round(spot / 1000, 1))))
        d_pick = b.slider("Days to expiry", float(max(1, round(surf.T[0] * 365))), float(round(surf.T[-1] * 365 * 1.2)),
                          float(round(np.median(surf.T) * 365)), 1.0)
        cp_pick = st.radio("Option", ["Call", "Put"], horizontal=True, key="cp_pick") == "Call"
        T_pick = d_pick / 365
        iv_p = float(surf.iv(K_pick, T_pick))
        inside = bool(surf.in_quoted_range(K_pick, T_pick)) and surf.T[0] <= T_pick <= surf.T[-1]
        F_p = float(surf.forward(T_pick))
        cols = st.columns(5)
        cols[0].metric("Surface IV", f"{iv_p * 100:.2f}%")
        cols[1].metric("Model price", fmt(surf.price(K_pick, T_pick, cp_pick)))
        cols[2].metric("Forward", f"{F_p:,.2f}")
        cols[3].metric("k = ln(K/F)", f"{np.log(K_pick / F_p):.2f}")
        cols[4].metric("Region", "Interpolated" if inside else "Extrapolated")
        if not inside:
            st.warning("This point is outside the quoted strikes/expiries: vol is held flat, so treat it with caution.")
        g = scaled_greeks(surf.greeks(K_pick, T_pick, cp_pick))
        cols = st.columns(7)
        for col, (name, val) in zip(cols, g.items()):
            col.metric(name, fmt(val), help=GREEK_HELP[name])

        # Neighbouring market quotes: nearest two expiries, strikes around K
        j = np.searchsorted(surf.T, T_pick)
        near_T = surf.T[max(0, j - 1):j + 1]
        nb = quotes[quotes["T"].isin(near_T)].copy()
        nb = nb.iloc[np.argsort(np.abs(nb["strike"].to_numpy() - K_pick))].groupby("T").head(4).sort_values(["T", "strike"])
        st.markdown("**Nearest market quotes the surface interpolates between**")
        st.dataframe(nb.assign(days=nb["T"] * 365, type=np.where(nb["is_call"], "C", "P"), iv=nb["iv"] * 100,
                               fit=surf.iv(nb["strike"].to_numpy(), nb["T"].to_numpy()) * 100)
                     [["expiry", "days", "type", "strike", "bid", "ask", "mid", "iv", "fit"]],
                     hide_index=True, width="stretch",
                     column_config={"expiry": st.column_config.DateColumn(format="YYYY-MM-DD"),
                                    "days": st.column_config.NumberColumn(format="%.2f"),
                                    "iv": st.column_config.NumberColumn("market IV %", format="%.2f"),
                                    "fit": st.column_config.NumberColumn("surface IV %", format="%.2f")})
    else:
        a, b = st.columns(2)
        i = a.selectbox("Expiry", range(len(exp_labels)), format_func=lambda j: exp_labels[j], key="q_exp")
        g_q = quotes[quotes["T"] == expiries["T"].iloc[i]].reset_index(drop=True)
        atm = int(np.argmin(np.abs(g_q["strike"] - spot)))
        row_i = b.selectbox("Strike", range(len(g_q)), index=atm,
                            format_func=lambda j: f"{g_q['strike'][j]:,.2f} {'C' if g_q['is_call'][j] else 'P'}"
                                                  f"   bid {g_q['bid'][j]:.2f} / ask {g_q['ask'][j]:.2f}")
        row = g_q.iloc[row_i]
        S0, Kq, Tq, rq, qq, cq, mid = spot, row["strike"], row["T"], row["r"], row["q"], bool(row["is_call"]), row["mid"]

        st.markdown("**Inputs to the solver**")
        cols = st.columns(7)
        for col, (n, v) in zip(cols, [("Spot", f"{S0:,.2f}"), ("Strike", f"{Kq:,.2f}"), ("Type", "Call" if cq else "Put"),
                                      ("T (years)", f"{Tq:.2f}"), ("r", f"{rq * 100:.2f}%"),
                                      ("q (parity)", f"{qq * 100:.2f}%"), ("Mid price", f"{mid:.2f}")]):
            col.metric(n, v)

        # Re-run Newton step by step for display, same start as the engine (Manaster-Koehler).
        F = S0 * np.exp((rq - qq) * Tq)
        s = float(np.clip(np.sqrt(abs(2 * np.log(F / Kq)) / Tq) or 0.2, 0.05, 2.0))
        s = s if s > 1e-3 else 0.2
        steps = []
        for it in range(20):
            p = float(bs_price(S0, Kq, Tq, rq, s, qq, cq))
            v = float(bs_greeks(S0, Kq, Tq, rq, s, qq, cq)["vega"])
            steps.append({"iter": it, "σ %": s * 100, "BS price": p, "price error": p - mid, "vega": v,
                          "step σ %": -(p - mid) / v * 100 if v > 0 else np.nan})
            if v < 1e-8 or abs(p - mid) / v < 1e-10:
                break
            s -= (p - mid) / v
        st.markdown("**Newton-Raphson iterations**  σₙ₊₁ = σₙ − (BS(σₙ) − mid) / vega(σₙ)")
        st.dataframe(pd.DataFrame(steps).round(2), hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.2f") for c in
                                    ["σ %", "BS price", "price error", "vega", "step σ %"]})

        cols = st.columns(5)
        cols[0].metric("IV (mid)", f"{row['iv'] * 100:.2f}%", help="Solved by " + ("Newton" if row["iv_method"] == 0 else "Brent"))
        cols[1].metric("IV (bid)", f"{row['iv_bid'] * 100:.2f}%" if np.isfinite(row["iv_bid"]) else "n/a")
        cols[2].metric("IV (ask)", f"{row['iv_ask'] * 100:.2f}%")
        cols[3].metric("Surface fit", f"{float(surf.iv(Kq, Tq)) * 100:.2f}%",
                       delta=f"{(float(surf.iv(Kq, Tq)) - row['iv']) * 100:+.2f} pts", delta_color="off")
        cols[4].metric("CBOE published IV", f"{row['vendor_iv'] * 100:.2f}%" if np.isfinite(row["vendor_iv"]) else "n/a")
        if truth is not None:
            st.caption(f"True SSVI vol at this point: {float(truth(Kq, Tq)) * 100:.2f}%")

# ============================================================================ 5. Diagnostics
with tab_diag:
    st.subheader("Fit quality and no-arbitrage checks")
    fit = (surf.iv(quotes["strike"].to_numpy(), quotes["T"].to_numpy()) - quotes["iv"].to_numpy()) * 100
    inside = (fit / 100 + quotes["iv"] >= quotes["iv_bid"].fillna(0) - 1e-9) & (fit / 100 + quotes["iv"] <= quotes["iv_ask"] + 1e-9)
    cols = st.columns(4)
    cols[0].metric("In-sample RMSE", f"{np.sqrt(np.mean(fit ** 2)):.2f} vol pts")
    cols[1].metric("Fit inside bid-ask", f"{inside.mean() * 100:.2f}%")

    @st.cache_data(show_spinner="Out-of-sample test…")
    def oos(quotes, spot, smooth, enforce_cal):
        test = np.random.default_rng(1).random(len(quotes)) < 0.2
        s_tr = VolSurface(quotes[~test], spot, smooth=smooth, enforce_calendar=enforce_cal)
        h = quotes[test]
        e = (s_tr.iv(h["strike"].to_numpy(), h["T"].to_numpy()) - h["iv"].to_numpy()) * 100
        return float(np.sqrt(np.mean(e ** 2))), int(test.sum())

    rmse_oos, n_test = oos(quotes, spot, smooth, enforce_cal)
    cols[2].metric("Out-of-sample RMSE", f"{rmse_oos:.2f} vol pts", help=f"Fit on 80%, predict {n_test} held-out quotes")
    if quotes["vendor_iv"].notna().any():
        cols[3].metric("Median |ours − CBOE IV|", f"{((quotes['iv'] - quotes['vendor_iv']).abs() * 100).median():.2f} pts")

    fig = go.Figure(go.Scatter(x=quotes["k"], y=fit, mode="markers",
                               marker=dict(color=quotes["T"] * 365, colorscale="Blues", size=6,
                                           colorbar=dict(title="days")),
                               customdata=np.c_[quotes["strike"], quotes["T"] * 365],
                               hovertemplate="K %{customdata[0]:,.2f}, %{customdata[1]:.0f}d<br>"
                                             "fit − market %{y:.2f} pts<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="gray", width=1))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), title="Residuals: surface − market mid IV",
                      xaxis_title="log-moneyness k", yaxis_title="vol points")
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Arbitrage report per expiry** (raw fitted slices, before the calendar floor)")
    arb = surf.arbitrage_report()
    st.dataframe(arb.round(2), hide_index=True, width="stretch")
    st.caption("butterfly: call price not convex in strike (negative density). monotonic: call price rising with "
               "strike. calendar: total variance falling vs the previous expiry.")
