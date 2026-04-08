"""Plotly chart builders for the Streamlit dashboard."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.config import REGIME_LABELS

# 7-colour palette for regimes
REGIME_COLORS = [
    "#00c853",  # 0 Super Bull   — vivid green
    "#66bb6a",  # 1 Strong Bull  — medium green
    "#a5d6a7",  # 2 Bull         — light green
    "#bdbdbd",  # 3 Sideways     — grey
    "#ef9a9a",  # 4 Bear         — light red
    "#e57373",  # 5 Strong Bear  — medium red
    "#c62828",  # 6 Super Bear   — dark red
]
REGIME_COLORS_RGBA = [
    f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.18)"
    for c in REGIME_COLORS
]


def _sample_frame(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df

    step = max(len(df) // max_points, 1)
    sampled = df.iloc[::step]
    if sampled.index[-1] != df.index[-1]:
        sampled = pd.concat([sampled, df.iloc[[-1]]])
    return sampled


def btc_regime_price_chart(
    df: pd.DataFrame,
    regime_state: pd.Series,
    trades: pd.DataFrame | None = None,
    max_points: int = 5000,
) -> go.Figure:
    """Color-coded BTC close chart with visible trade markers."""
    plot_df = pd.DataFrame(
        {
            "close": df["close"].astype(float),
            "regime_state": regime_state.reindex(df.index).ffill().bfill().astype(int),
        },
        index=df.index,
    )
    plot_df = _sample_frame(plot_df, max_points)

    fig = go.Figure()
    seen_states: set[int] = set()
    run_start = 0

    for idx in range(1, len(plot_df) + 1):
        is_boundary = idx == len(plot_df) or (
            plot_df["regime_state"].iat[idx] != plot_df["regime_state"].iat[idx - 1]
        )
        if not is_boundary:
            continue

        segment = plot_df.iloc[run_start:idx]
        state = int(segment["regime_state"].iat[0])
        fig.add_vrect(
            x0=segment.index[0],
            x1=segment.index[-1],
            fillcolor=REGIME_COLORS_RGBA[state],
            layer="below",
            line_width=0,
        )
        fig.add_trace(
            go.Scatter(
                x=segment.index,
                y=segment["close"],
                mode="lines",
                name=REGIME_LABELS[state],
                legendgroup=REGIME_LABELS[state],
                showlegend=state not in seen_states,
                line=dict(color=REGIME_COLORS[state], width=3),
                hovertemplate=(
                    f"{REGIME_LABELS[state]}<br>"
                    "Time: %{x}<br>"
                    "BTC: $%{y:,.0f}<extra></extra>"
                ),
            )
        )
        seen_states.add(state)
        run_start = idx

    if trades is not None and len(trades) > 0:
        long_entries = trades[trades["direction"] == "LONG"]
        short_entries = trades[trades["direction"] == "SHORT"]
        stop_exits = trades[trades["exit_reason"] == "STOP LOSS"]
        regime_exits = trades[trades["exit_reason"] == "REGIME CHANGE"]

        if len(long_entries) > 0:
            fig.add_trace(
                go.Scatter(
                    x=long_entries["entry_time"],
                    y=long_entries["entry_price"],
                    mode="markers",
                    name="Long entry",
                    marker=dict(
                        symbol="triangle-up",
                        size=15,
                        color="#f8fafc",
                        line=dict(color="#00c853", width=2),
                    ),
                    hovertemplate=(
                        "Long entry<br>"
                        "Time: %{x}<br>"
                        "Entry: $%{y:,.2f}<extra></extra>"
                    ),
                )
            )

        if len(short_entries) > 0:
            fig.add_trace(
                go.Scatter(
                    x=short_entries["entry_time"],
                    y=short_entries["entry_price"],
                    mode="markers",
                    name="Short entry",
                    marker=dict(
                        symbol="triangle-down",
                        size=15,
                        color="#f8fafc",
                        line=dict(color="#c62828", width=2),
                    ),
                    hovertemplate=(
                        "Short entry<br>"
                        "Time: %{x}<br>"
                        "Entry: $%{y:,.2f}<extra></extra>"
                    ),
                )
            )

        if len(regime_exits) > 0:
            fig.add_trace(
                go.Scatter(
                    x=regime_exits["exit_time"],
                    y=regime_exits["exit_price"],
                    mode="markers",
                    name="Regime exit",
                    marker=dict(
                        symbol="circle",
                        size=11,
                        color="#f2b134",
                        line=dict(color="#111827", width=1),
                    ),
                    hovertemplate=(
                        "Regime exit<br>"
                        "Time: %{x}<br>"
                        "Exit: $%{y:,.2f}<extra></extra>"
                    ),
                )
            )

        if len(stop_exits) > 0:
            fig.add_trace(
                go.Scatter(
                    x=stop_exits["exit_time"],
                    y=stop_exits["exit_price"],
                    mode="markers",
                    name="Stop loss",
                    marker=dict(
                        symbol="x",
                        size=13,
                        color="#ff6b57",
                        line=dict(color="#ffe4dd", width=1),
                    ),
                    hovertemplate=(
                        "Stop loss<br>"
                        "Time: %{x}<br>"
                        "Exit: $%{y:,.2f}<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title="BTC Price Map by Regime",
        height=620,
        paper_bgcolor="#07111b",
        plot_bgcolor="#07111b",
        font=dict(color="#f5f7fa"),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(7,17,27,0.0)",
        ),
        margin=dict(l=40, r=20, t=70, b=20),
    )
    fig.update_yaxes(gridcolor="#203042", tickprefix="$", separatethousands=True)
    fig.update_xaxes(gridcolor="#203042")
    return fig


def trade_pnl_chart(trades: pd.DataFrame) -> go.Figure:
    """Trade-by-trade PnL with cumulative line."""
    cumulative = trades["net_pnl"].cumsum()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=trades["trade_no"],
            y=trades["net_pnl"],
            name="Trade P&L",
            marker_color=["#00c853" if pnl > 0 else "#ff6b57" for pnl in trades["net_pnl"]],
            hovertemplate="Trade #%{x}<br>P&L: $%{y:+,.0f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=trades["trade_no"],
            y=cumulative,
            name="Cumulative P&L",
            mode="lines+markers",
            line=dict(color="#8ab4ff", width=2.5),
            marker=dict(size=6),
            hovertemplate="After trade %{x}<br>Cumulative: $%{y:+,.0f}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#5c6b7a")
    fig.update_layout(
        title="Trade P&L Evolution",
        height=420,
        paper_bgcolor="#07111b",
        plot_bgcolor="#07111b",
        font=dict(color="#f5f7fa"),
        margin=dict(l=40, r=40, t=60, b=20),
    )
    fig.update_yaxes(gridcolor="#203042", tickprefix="$", separatethousands=True, secondary_y=False)
    fig.update_yaxes(gridcolor="#203042", tickprefix="$", separatethousands=True, secondary_y=True)
    fig.update_xaxes(gridcolor="#203042", title="Trade #")
    return fig


def exit_reason_chart(trades: pd.DataFrame) -> go.Figure:
    """Donut chart of trade exits."""
    counts = trades["exit_reason"].value_counts()
    color_map = {
        "REGIME CHANGE": "#f2b134",
        "STOP LOSS": "#ff6b57",
    }
    fig = go.Figure(
        go.Pie(
            labels=counts.index.tolist(),
            values=counts.values,
            hole=0.62,
            marker=dict(colors=[color_map.get(label, "#8ab4ff") for label in counts.index]),
            textinfo="label+percent",
        )
    )
    fig.update_layout(
        title="Exit Breakdown",
        height=320,
        paper_bgcolor="#07111b",
        plot_bgcolor="#07111b",
        font=dict(color="#f5f7fa"),
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False,
    )
    return fig


# ─── Tab 1: Price & Regime Overlay ────────────────────────────────────────────

def regime_price_chart(
    df: pd.DataFrame,
    regime_state: pd.Series,
    regime_confidence: pd.Series,
    posteriors: pd.DataFrame | None = None,
    signals: pd.Series | None = None,
    confidence_threshold: float = 0.75,
    max_candles: int = 2000,
) -> go.Figure:
    """
    Multi-panel chart:
    Row 1: Candlestick + regime color bands + signals
    Row 2: Regime confidence line
    Row 3: Stacked posterior area (if posteriors provided)
    """
    n_rows = 3 if posteriors is not None else 2
    row_heights = [0.6, 0.2, 0.2] if n_rows == 3 else [0.7, 0.3]

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=["BTC/USD + Regime", "Confidence", "Regime Posteriors"] if n_rows == 3
                       else ["BTC/USD + Regime", "Confidence"],
    )

    # Downsample for performance
    idx = df.index
    if len(idx) > max_candles:
        step = len(idx) // max_candles
        idx = idx[::step]

    df_plot = df.loc[idx]
    reg_plot = regime_state.reindex(idx, method="ffill")
    conf_plot = regime_confidence.reindex(idx, method="ffill")

    # ── Candlestick ────────────────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=df_plot.index,
            open=df_plot["open"],
            high=df_plot["high"],
            low=df_plot["low"],
            close=df_plot["close"],
            name="BTC/USD",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # ── Regime background bands ────────────────────────────────────────────────
    prev_state = None
    seg_start = idx[0]

    for i, ts in enumerate(idx):
        state = reg_plot.iloc[i]
        if pd.isna(state):
            continue
        if state != prev_state or i == len(idx) - 1:
            if prev_state is not None and not pd.isna(prev_state):
                fig.add_vrect(
                    x0=seg_start, x1=ts,
                    fillcolor=REGIME_COLORS_RGBA[int(prev_state)],
                    layer="below",
                    line_width=0,
                    row=1, col=1,
                )
            seg_start = ts
            prev_state = state

    # ── Buy/sell signals ───────────────────────────────────────────────────────
    if signals is not None:
        sig_plot = signals.reindex(idx, method="ffill").fillna(0)
        longs = sig_plot[sig_plot > 0]
        shorts = sig_plot[sig_plot < 0]
        if len(longs):
            fig.add_trace(
                go.Scatter(
                    x=longs.index,
                    y=df_plot["close"].reindex(longs.index),
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=8, color="#00c853"),
                    name="Long entry",
                ),
                row=1, col=1,
            )
        if len(shorts):
            fig.add_trace(
                go.Scatter(
                    x=shorts.index,
                    y=df_plot["close"].reindex(shorts.index),
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=8, color="#c62828"),
                    name="Short entry",
                ),
                row=1, col=1,
            )

    # ── Confidence line ────────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=conf_plot.index, y=conf_plot.values,
            mode="lines", name="Confidence",
            line=dict(color="#42a5f5", width=1.5),
        ),
        row=2, col=1,
    )
    fig.add_hline(
        y=confidence_threshold, line_dash="dash",
        line_color="#ff7043", row=2, col=1,
        annotation_text=f"Threshold {confidence_threshold:.0%}",
    )

    # ── Posterior stacked area ─────────────────────────────────────────────────
    if posteriors is not None and n_rows == 3:
        post_plot = posteriors.reindex(idx, method="ffill")
        for s in range(len(REGIME_LABELS)):
            col_name = s if s in post_plot.columns else str(s)
            if col_name not in post_plot.columns:
                continue
            fig.add_trace(
                go.Scatter(
                    x=post_plot.index,
                    y=post_plot[col_name].values,
                    stackgroup="one",
                    name=REGIME_LABELS[s],
                    line=dict(width=0),
                    fillcolor=REGIME_COLORS[s],
                    mode="lines",
                ),
                row=3, col=1,
            )

    fig.update_layout(
        height=750 if n_rows == 3 else 550,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=60, b=20),
    )
    fig.update_yaxes(gridcolor="#1e2128")
    fig.update_xaxes(gridcolor="#1e2128")
    return fig


# ─── Tab 2: Performance Comparison ──────────────────────────────────────────

def equity_comparison_chart(
    strat_equity: pd.Series,
    bh_equity: pd.Series,
) -> go.Figure:
    """Strategy vs Buy & Hold equity curves with drawdown."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=["Strategy vs Buy & Hold", "Drawdown (%)"],
    )
    fig.add_trace(
        go.Scatter(x=strat_equity.index, y=strat_equity.values,
                   name="Regime Strategy",
                   line=dict(color="#42a5f5", width=2.5)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=bh_equity.index, y=bh_equity.values,
                   name="Buy & Hold",
                   line=dict(color="#9e9e9e", width=2, dash="dot")),
        row=1, col=1,
    )

    # Strategy drawdown
    rm_strat = strat_equity.cummax()
    dd_strat = (strat_equity - rm_strat) / rm_strat * 100
    fig.add_trace(
        go.Scatter(x=dd_strat.index, y=dd_strat.values,
                   name="Strategy DD", fill="tozeroy",
                   line=dict(color="#42a5f5", width=1),
                   fillcolor="rgba(66,165,245,0.15)"),
        row=2, col=1,
    )
    # B&H drawdown
    rm_bh = bh_equity.cummax()
    dd_bh = (bh_equity - rm_bh) / rm_bh * 100
    fig.add_trace(
        go.Scatter(x=dd_bh.index, y=dd_bh.values,
                   name="B&H DD",
                   line=dict(color="#9e9e9e", width=1, dash="dot")),
        row=2, col=1,
    )

    fig.update_layout(
        height=550, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#fafafa"), margin=dict(l=40, r=20, t=60, b=20),
    )
    fig.update_yaxes(gridcolor="#1e2128")
    return fig


def equity_chart(
    equity_curve: pd.Series,
    buy_hold: pd.Series | None = None,
) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=["Equity Curve", "Drawdown (%)"],
    )
    fig.add_trace(
        go.Scatter(x=equity_curve.index, y=equity_curve.values,
                   name="Strategy", line=dict(color="#42a5f5", width=2)),
        row=1, col=1,
    )
    if buy_hold is not None:
        fig.add_trace(
            go.Scatter(x=buy_hold.index, y=buy_hold.values,
                       name="Buy & Hold", line=dict(color="#9e9e9e", width=1.5, dash="dot")),
            row=1, col=1,
        )
    # Drawdown
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max * 100
    fig.add_trace(
        go.Scatter(x=drawdown.index, y=drawdown.values,
                   name="Drawdown %", fill="tozeroy",
                   line=dict(color="#ef5350", width=1), fillcolor="rgba(239,83,80,0.2)"),
        row=2, col=1,
    )
    fig.update_layout(
        height=500, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#fafafa"), margin=dict(l=40, r=20, t=60, b=20),
    )
    fig.update_yaxes(gridcolor="#1e2128")
    return fig


def monthly_returns_heatmap(monthly_tbl: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=monthly_tbl.values,
            x=monthly_tbl.columns.tolist(),
            y=monthly_tbl.index.tolist(),
            colorscale="RdYlGn",
            zmid=0,
            text=np.round(monthly_tbl.values, 1),
            texttemplate="%{text:.1f}%",
            showscale=True,
        )
    )
    fig.update_layout(
        title="Monthly Returns (%)",
        height=400, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
    )
    return fig


def trade_histogram(trades: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Histogram(
            x=trades["pnl_pct"] * 100,
            nbinsx=50,
            marker_color="#42a5f5",
            name="Trade PnL %",
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#ff7043")
    fig.update_layout(
        title="Trade PnL Distribution (%)",
        height=300, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        xaxis_title="PnL (%)", yaxis_title="Count",
    )
    return fig


def regime_distribution_chart(regime_state: pd.Series) -> go.Figure:
    """Bar chart showing time spent in each regime."""
    counts = regime_state.value_counts().sort_index()
    total = counts.sum()
    labels = [REGIME_LABELS[int(i)] for i in counts.index]
    pcts = (counts / total * 100).values

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=pcts,
            marker_color=[REGIME_COLORS[int(i)] for i in counts.index],
            text=[f"{p:.1f}%" for p in pcts],
            textposition="auto",
        )
    )
    fig.update_layout(
        title="Time in Each Regime (%)",
        height=350, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        yaxis_title="% of Time",
        showlegend=False,
    )
    return fig


# ─── Walk-Forward Matrix ────────────────────────────────────────────────────

def wf_matrix_heatmap(summary: pd.DataFrame) -> go.Figure:
    metric_cols = [
        "sharpe", "calmar", "total_return_pct", "max_drawdown_pct",
        "win_rate", "profit_factor", "avg_trade_pct", "n_trades",
    ]
    metric_cols = [c for c in metric_cols if c in summary.columns]
    fold_labels = [f"F{i}" for i in summary["fold_idx"]]

    z = summary[metric_cols].T.values
    text = np.round(z, 2).astype(str)

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=fold_labels,
            y=metric_cols,
            colorscale="RdYlGn",
            zmid=0,
            text=text,
            texttemplate="%{text}",
            showscale=True,
        )
    )
    fig.update_layout(
        title="Walk-Forward Matrix (OOS metrics per fold)",
        height=450, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
    )
    return fig


def wf_sharpe_line(summary: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=summary["fold_idx"], y=summary["sharpe"],
            mode="lines+markers", name="OOS Sharpe",
            line=dict(color="#42a5f5", width=2),
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#9e9e9e")
    fig.add_hline(y=1, line_dash="dot", line_color="#66bb6a")
    fig.update_layout(
        title="OOS Sharpe per Fold",
        height=300, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        xaxis_title="Fold", yaxis_title="Sharpe",
    )
    return fig


def wf_return_bars(summary: pd.DataFrame) -> go.Figure:
    colors = [
        "#66bb6a" if v > 0 else "#ef5350"
        for v in summary["total_return_pct"]
    ]
    fig = go.Figure(
        go.Bar(
            x=summary["fold_idx"],
            y=summary["total_return_pct"],
            marker_color=colors,
            name="OOS Total Return %",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#9e9e9e")
    fig.update_layout(
        title="OOS Total Return (%) per Fold",
        height=300, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        xaxis_title="Fold", yaxis_title="Return (%)",
    )
    return fig


# ─── Model Insights ─────────────────────────────────────────────────────────

def shap_importance_chart(importance: pd.Series, top_n: int = 20) -> go.Figure:
    top = importance.head(top_n).sort_values()
    fig = go.Figure(
        go.Bar(
            x=top.values,
            y=top.index.tolist(),
            orientation="h",
            marker_color="#42a5f5",
        )
    )
    fig.update_layout(
        title=f"Feature Importance (top {top_n}, LightGBM gain)",
        height=450, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        xaxis_title="Importance (gain)",
    )
    return fig


def regime_transition_heatmap(trans_mat: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=trans_mat.values * 100,
            x=REGIME_LABELS,
            y=REGIME_LABELS,
            colorscale="Blues",
            text=np.round(trans_mat.values * 100, 1),
            texttemplate="%{text:.1f}%",
            showscale=True,
        )
    )
    fig.update_layout(
        title="Regime Transition Probability (%)",
        height=450, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        xaxis_title="To Regime", yaxis_title="From Regime",
    )
    return fig


def regime_duration_violin(df: pd.DataFrame, regime_col: str = "regime_state") -> go.Figure:
    """Violin plot of regime spell durations (in bars/hours)."""
    from itertools import groupby

    regime_arr = df[regime_col].values
    durations = []
    for state, group in groupby(regime_arr):
        length = sum(1 for _ in group)
        durations.append({"regime": REGIME_LABELS[int(state)], "duration": length})

    dur_df = pd.DataFrame(durations)
    fig = go.Figure()
    for s, label in enumerate(REGIME_LABELS):
        vals = dur_df[dur_df["regime"] == label]["duration"].values
        if len(vals) == 0:
            continue
        fig.add_trace(
            go.Violin(
                y=vals, name=label,
                box_visible=True, meanline_visible=True,
                fillcolor=REGIME_COLORS[s], opacity=0.7,
                line_color="white",
            )
        )
    fig.update_layout(
        title="Regime Spell Duration Distribution (hours)",
        height=400, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        yaxis_title="Duration (bars/hours)",
        showlegend=False,
    )
    return fig
