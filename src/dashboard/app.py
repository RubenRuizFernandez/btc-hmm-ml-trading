"""Streamlit dashboard — Functional BTC Regime Trading Strategy.

Run with:
    streamlit run src/dashboard/app.py
or:
    python scripts/run_dashboard.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import REGIME_LABELS, REGIME_MULTIPLIER, TREND_SCORE_THRESHOLDS, MIN_REGIME_BARS
from src.dashboard.plots import (
    REGIME_COLORS, REGIME_COLORS_RGBA, monthly_returns_heatmap,
    regime_distribution_chart, regime_transition_heatmap, regime_duration_violin,
)
from src.data.loader import load_raw
from src.regime.trend_regime import compute_trend_score, score_to_regime, smooth_regimes
from src.regime.regime_labels import compute_regime_stats, compute_transition_matrix
from src.backtest.functional import run_functional_strategy, DEFAULT_REGIME_UNITS
from src.backtest.metrics import compute_metrics, monthly_returns

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BTC Regime Trading",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetric"] { background: #161b22; border-radius: 8px; padding: 12px; }
</style>
""", unsafe_allow_html=True)


# ─── Cached data ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading OHLCV data...")
def get_raw_data():
    return load_raw()

@st.cache_data(show_spinner="Computing trend score...")
def get_trend_score(_df):
    return compute_trend_score(_df)

@st.cache_data(show_spinner="Computing regimes...")
def get_regimes(_score, min_bars):
    regime = score_to_regime(_score)
    return smooth_regimes(regime, min_bars)


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙ Strategy Controls")
    st.markdown("---")

    default_start = pd.Timestamp("2023-04-07").date()
    default_end = pd.Timestamp("2026-04-07").date()
    date_range = st.date_input("Date range", value=[default_start, default_end])

    st.markdown("---")
    st.markdown("**Account & Risk**")
    account_size = st.number_input("Account size ($)", value=100_000, step=10_000)
    margin_pct = st.slider("Margin / buying power (%)", 10, 80, 40, 5) / 100
    max_loss_pct = st.slider("Hard stop-loss (% of account)", 1.0, 10.0, 3.0, 0.5) / 100
    sl_cooldown = st.slider("SL cooldown (hours)", 0, 72, 24, 4)

    min_bars = st.slider("Min regime duration (hours)", 1, 48, MIN_REGIME_BARS, 1)

    st.markdown("---")
    n_units = 3
    unit_size = account_size * margin_pct / n_units
    max_loss_usd = account_size * max_loss_pct

    st.markdown("**Position Sizing**")
    st.markdown(f"Unit size: **${unit_size:,.0f}**")
    st.markdown(f"Hard SL: **${max_loss_usd:,.0f}**")
    for i, label in enumerate(REGIME_LABELS):
        units = DEFAULT_REGIME_UNITS[i]
        if units == 0:
            st.markdown(f'<span style="color:{REGIME_COLORS[i]}">■</span> {label} → **Flat**',
                        unsafe_allow_html=True)
        else:
            direction = "LONG" if units > 0 else "SHORT"
            notional = abs(units) * unit_size
            sl_move_pct = max_loss_usd / notional * 100
            st.markdown(
                f'<span style="color:{REGIME_COLORS[i]}">■</span> {label} → '
                f'**{direction} x{abs(units)}** (${notional:,.0f}) · SL {sl_move_pct:.1f}%',
                unsafe_allow_html=True,
            )


# ─── Load & compute ─────────────────────────────────────────────────────────

df = get_raw_data()
trend_score = get_trend_score(df)
regime_state = get_regimes(trend_score, min_bars)

# Filter by date
if len(date_range) == 2:
    start_ts = pd.Timestamp(date_range[0], tz="UTC")
    end_ts = pd.Timestamp(date_range[1], tz="UTC")
else:
    start_ts, end_ts = df.index[0], df.index[-1]

mask = (df.index >= start_ts) & (df.index <= end_ts)
common_idx = regime_state.dropna().index.intersection(df.index[mask])
df_view = df.loc[common_idx]
rs_view = regime_state.loc[common_idx]
score_view = trend_score.loc[common_idx]

if len(df_view) < 2:
    st.error("No data in selected date range.")
    st.stop()

# ─── Run functional strategy ────────────────────────────────────────────────

result = run_functional_strategy(
    df_view["close"], rs_view,
    account_size=account_size,
    margin_pct=margin_pct,
    max_loss_pct=max_loss_pct,
    sl_cooldown_bars=sl_cooldown,
)
m = result.metrics

# B&H: invest the same max notional ($40K) in BTC
bh_notional = account_size * margin_pct
bh_return = df_view["close"].iloc[-1] / df_view["close"].iloc[0] - 1
bh_pnl = bh_notional * bh_return
bh_equity = account_size + bh_notional * (df_view["close"] / df_view["close"].iloc[0] - 1)
bh_m = compute_metrics(bh_equity / bh_equity.iloc[0])
bh_m["net_pnl"] = float(bh_pnl)
bh_m["starting_capital"] = float(account_size)
bh_m["ending_capital"] = float(account_size + bh_pnl)

# ─── Header ─────────────────────────────────────────────────────────────────

st.title("📊 BTC/USD — Functional Regime Trading")
st.caption(
    f"Showing: {df_view.index[0].date()} → {df_view.index[-1].date()} · "
    f"{len(df_view):,} bars · Account ${account_size:,.0f} · "
    f"Margin {margin_pct:.0%} · SL {max_loss_pct:.0%}"
)

# ─── Top metrics ─────────────────────────────────────────────────────────────

st.markdown("### Strategy vs Buy & Hold")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Net P&L", f"${m['net_pnl']:+,.0f}",
          delta=f"${m['net_pnl'] - bh_m['net_pnl']:+,.0f} vs B&H", delta_color="normal")
c2.metric("Account Value", f"${result.equity_curve.iloc[-1]:,.0f}")
c3.metric("Total Return", f"{m['total_return_pct']:.1f}%",
          delta=f"{m['total_return_pct'] - bh_m['total_return_pct']:+.1f}%", delta_color="normal")
c4.metric("Sharpe", f"{m['sharpe']:.2f}",
          delta=f"{m['sharpe'] - bh_m['sharpe']:+.2f}", delta_color="normal")
c5.metric("Max Drawdown", f"{m['max_drawdown_pct']:.1f}%",
          delta=f"{m['max_drawdown_pct'] - bh_m['max_drawdown_pct']:+.1f}%", delta_color="inverse")
c6.metric("Trades", f"{m['n_trades']}")

# B&H reference
st.markdown(
    f"<div style='background:#1a1a2e; padding:8px 16px; border-radius:6px; margin-bottom:12px; font-size:0.85em;'>"
    f"<b>Buy & Hold ({margin_pct:.0%} capital):</b> &nbsp; "
    f"P&L: ${bh_m['net_pnl']:+,.0f} &nbsp;|&nbsp; "
    f"Return: {bh_m['total_return_pct']:.1f}% &nbsp;|&nbsp; "
    f"Sharpe: {bh_m['sharpe']:.2f} &nbsp;|&nbsp; "
    f"Max DD: {bh_m['max_drawdown_pct']:.1f}%"
    f"</div>",
    unsafe_allow_html=True,
)

# Open position banner
if result.open_position:
    op = result.open_position
    pnl_color = "#00c853" if op["unrealized_pnl"] >= 0 else "#ef5350"
    st.markdown(
        f"<div style='background:#1b2838; padding:10px 16px; border-radius:6px; "
        f"border-left:4px solid {pnl_color}; margin-bottom:12px;'>"
        f"<b>Open Position:</b> {op['direction']} x{op['units']} · "
        f"Entry: ${op['entry_price']:,.2f} · SL: ${op['sl_price']:,.2f} · "
        f"Current: ${op['current_price']:,.2f} · "
        f"P&L: <span style='color:{pnl_color}'>${op['unrealized_pnl']:+,.2f}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ─── Tabs ────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(
    ["🕯 Price & Regime", "📈 Equity & P&L", "📋 Trade Log", "🔍 Regime Analysis"]
)

# ── Tab 1: Price & Regime ────────────────────────────────────────────────────
with tab1:
    max_candles = 2000
    idx = df_view.index
    if len(idx) > max_candles:
        step = len(idx) // max_candles
        idx = idx[::step]

    df_plot = df_view.loc[idx]
    reg_plot = rs_view.reindex(idx, method="ffill")

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.04, row_heights=[0.75, 0.25],
        subplot_titles=["BTC/USD + Regime Bands", "Trend Score"],
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_plot.index, open=df_plot["open"], high=df_plot["high"],
        low=df_plot["low"], close=df_plot["close"],
        name="BTC/USD", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        showlegend=False,
    ), row=1, col=1)

    # Regime bands
    prev_state, seg_start = None, idx[0]
    for i, ts in enumerate(idx):
        state = reg_plot.iloc[i]
        if pd.isna(state):
            continue
        if state != prev_state or i == len(idx) - 1:
            if prev_state is not None and not pd.isna(prev_state):
                fig.add_vrect(x0=seg_start, x1=ts,
                              fillcolor=REGIME_COLORS_RGBA[int(prev_state)],
                              layer="below", line_width=0, row=1, col=1)
            seg_start = ts
            prev_state = state

    # Trade markers
    if len(result.trades) > 0:
        for _, t in result.trades.iterrows():
            color = "#00c853" if t["direction"] == "LONG" else "#c62828"
            # Entry
            fig.add_trace(go.Scatter(
                x=[t["entry_time"]], y=[t["entry_price"]],
                mode="markers", marker=dict(
                    symbol="triangle-up" if t["direction"] == "LONG" else "triangle-down",
                    size=10, color=color,
                ),
                name=f"Entry {t['direction']}", showlegend=False,
                hovertemplate=f"{t['direction']} x{t['units']}<br>Entry ${t['entry_price']:,.0f}<br>SL ${t['sl_price']:,.0f}",
            ), row=1, col=1)
            # Exit
            exit_color = "#ff7043" if t["exit_reason"] == "STOP LOSS" else "#9e9e9e"
            fig.add_trace(go.Scatter(
                x=[t["exit_time"]], y=[t["exit_price"]],
                mode="markers", marker=dict(symbol="x", size=8, color=exit_color),
                name=f"Exit ({t['exit_reason']})", showlegend=False,
                hovertemplate=f"Exit: {t['exit_reason']}<br>${t['exit_price']:,.0f}<br>P&L ${t['net_pnl']:+,.0f}",
            ), row=1, col=1)

    # Trend score
    score_plot = score_view.reindex(idx, method="ffill")
    fig.add_trace(go.Scatter(
        x=score_plot.index, y=score_plot.values,
        mode="lines", name="Trend Score",
        line=dict(color="#42a5f5", width=1.5),
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="solid", line_color="#555", row=2, col=1)
    for t in TREND_SCORE_THRESHOLDS:
        fig.add_hline(y=t, line_dash="dot", line_color="#333", row=2, col=1)

    fig.update_layout(
        height=700, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#fafafa"), xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=20, t=40, b=20),
    )
    fig.update_yaxes(gridcolor="#1e2128")
    fig.update_xaxes(gridcolor="#1e2128")
    st.plotly_chart(fig, use_container_width=True)

    st.plotly_chart(regime_distribution_chart(rs_view), use_container_width=True)


# ── Tab 2: Equity & P&L ─────────────────────────────────────────────────────
with tab2:
    fig_eq = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.05, row_heights=[0.7, 0.3],
        subplot_titles=["Account Value ($)", "Drawdown (%)"],
    )
    fig_eq.add_trace(go.Scatter(
        x=result.equity_curve.index, y=result.equity_curve.values,
        name="Strategy", line=dict(color="#42a5f5", width=2.5),
    ), row=1, col=1)
    fig_eq.add_trace(go.Scatter(
        x=bh_equity.index, y=bh_equity.values,
        name=f"B&H ({margin_pct:.0%} capital)", line=dict(color="#9e9e9e", width=2, dash="dot"),
    ), row=1, col=1)
    fig_eq.add_hline(y=account_size, line_dash="dash", line_color="#555",
                     annotation_text="Starting capital", row=1, col=1)

    # Drawdown
    rm = result.equity_curve.cummax()
    dd = (result.equity_curve - rm) / rm * 100
    fig_eq.add_trace(go.Scatter(
        x=dd.index, y=dd.values, name="Drawdown %", fill="tozeroy",
        line=dict(color="#ef5350", width=1), fillcolor="rgba(239,83,80,0.15)",
    ), row=2, col=1)

    fig_eq.update_layout(
        height=550, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#fafafa"), margin=dict(l=60, r=20, t=40, b=20),
    )
    fig_eq.update_yaxes(gridcolor="#1e2128", tickformat="$,.0f", row=1, col=1)
    fig_eq.update_yaxes(gridcolor="#1e2128", row=2, col=1)
    st.plotly_chart(fig_eq, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        monthly_tbl = monthly_returns(result.equity_curve / result.equity_curve.iloc[0])
        st.plotly_chart(monthly_returns_heatmap(monthly_tbl), use_container_width=True)
    with col_b:
        if len(result.trades) > 0:
            # Trade P&L bar chart
            fig_pnl = go.Figure(go.Bar(
                x=result.trades["trade_no"],
                y=result.trades["net_pnl"],
                marker_color=["#00c853" if p > 0 else "#ef5350" for p in result.trades["net_pnl"]],
                hovertemplate="Trade #%{x}<br>P&L: $%{y:+,.0f}<extra></extra>",
            ))
            fig_pnl.add_hline(y=0, line_dash="dash", line_color="#555")
            fig_pnl.update_layout(
                title="Trade P&L ($)", height=400,
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                font=dict(color="#fafafa"), xaxis_title="Trade #", yaxis_title="P&L ($)",
                yaxis_tickformat="$,.0f",
            )
            st.plotly_chart(fig_pnl, use_container_width=True)

    # Metrics comparison table
    st.subheader("Detailed Metrics")
    def _f(v, fmt=".1f"):
        return f"{v:{fmt}}" if isinstance(v, (int, float)) and not np.isnan(v) else "N/A"

    metrics_rows = {
        "Metric": ["Starting Capital", "Ending Capital", "Net P&L",
                    "Total Return (%)", "Annualized Return (%)",
                    "Sharpe", "Calmar", "Max Drawdown (%)",
                    "Win Rate", "Profit Factor", "# Trades"],
        "Strategy": [
            f"${m['starting_capital']:,.0f}", f"${m['ending_capital']:,.0f}", f"${m['net_pnl']:+,.0f}",
            _f(m["total_return_pct"]), _f(m["annualized_return_pct"]),
            _f(m["sharpe"], ".2f"), _f(m["calmar"], ".2f"), _f(m["max_drawdown_pct"]),
            _f(m["win_rate"], ".1%") if not np.isnan(m.get("win_rate", float("nan"))) else "N/A",
            _f(m["profit_factor"], ".2f") if not np.isnan(m.get("profit_factor", float("nan"))) else "N/A",
            str(m["n_trades"]),
        ],
        "Buy & Hold": [
            f"${bh_m['starting_capital']:,.0f}", f"${bh_m['ending_capital']:,.0f}", f"${bh_m['net_pnl']:+,.0f}",
            _f(bh_m["total_return_pct"]), _f(bh_m["annualized_return_pct"]),
            _f(bh_m["sharpe"], ".2f"), _f(bh_m["calmar"], ".2f"), _f(bh_m["max_drawdown_pct"]),
            "N/A", "N/A", "1",
        ],
    }
    st.dataframe(pd.DataFrame(metrics_rows).set_index("Metric"), use_container_width=True)


# ── Tab 3: Trade Log ─────────────────────────────────────────────────────────
with tab3:
    if len(result.trades) > 0:
        # Summary stats
        trades = result.trades
        sl_trades = trades[trades["exit_reason"] == "STOP LOSS"]
        regime_trades = trades[trades["exit_reason"] == "REGIME CHANGE"]
        winners = trades[trades["net_pnl"] > 0]
        losers = trades[trades["net_pnl"] <= 0]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Trades", len(trades))
        c2.metric("Winners", f"{len(winners)} ({len(winners)/len(trades)*100:.0f}%)")
        c3.metric("Stop Losses", f"{len(sl_trades)}")
        c4.metric("Avg Win", f"${winners['net_pnl'].mean():+,.0f}" if len(winners) else "N/A")
        c5.metric("Avg Loss", f"${losers['net_pnl'].mean():+,.0f}" if len(losers) else "N/A")

        # Full trade log
        st.subheader("Trade Log")
        display_cols = [
            "trade_no", "entry_time", "exit_time", "direction", "units",
            "entry_price", "exit_price", "sl_price", "notional",
            "net_pnl", "exit_reason", "account_after",
        ]
        st.dataframe(
            trades[display_cols].style.applymap(
                lambda v: "color: #00c853" if isinstance(v, (int, float)) and v > 0
                else "color: #ef5350" if isinstance(v, (int, float)) and v < 0
                else "",
                subset=["net_pnl"],
            ).format({
                "entry_price": "${:,.2f}",
                "exit_price": "${:,.2f}",
                "sl_price": "${:,.2f}",
                "notional": "${:,.0f}",
                "net_pnl": "${:+,.2f}",
                "account_after": "${:,.0f}",
            }),
            use_container_width=True,
            height=600,
        )
    else:
        st.info("No trades in selected period.")


# ── Tab 4: Regime Analysis ───────────────────────────────────────────────────
with tab4:
    col_a, col_b = st.columns(2)
    with col_a:
        trans_mat = compute_transition_matrix(rs_view)
        st.plotly_chart(regime_transition_heatmap(trans_mat), use_container_width=True)
    with col_b:
        feat_df_dur = pd.DataFrame({"regime_state": rs_view})
        st.plotly_chart(regime_duration_violin(feat_df_dur), use_container_width=True)

    st.subheader("Regime Statistics")
    feat_df = pd.DataFrame({
        "regime_state": rs_view,
        "regime_confidence": (score_view.abs() / 7.0).clip(0, 1),
        "forward_return": np.log(df_view["close"].shift(-8) / df_view["close"]),
    })
    stats = compute_regime_stats(feat_df)
    st.dataframe(
        stats[["regime_label", "count", "directional_accuracy",
               "mean_confidence", "mean_duration_bars"]].style.format({
            "directional_accuracy": "{:.1%}",
            "mean_confidence": "{:.1%}",
            "mean_duration_bars": "{:.1f}",
        }),
        use_container_width=True,
    )
