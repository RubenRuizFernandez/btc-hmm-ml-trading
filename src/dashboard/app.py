"""Streamlit dashboard for the functional BTC regime trading app."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.backtest.functional import (
    DEFAULT_REGIME_UNITS,
    FunctionalTradeConfig,
    run_functional_strategy,
)
from src.backtest.metrics import compute_metrics, monthly_returns
from src.config import (
    APP_ACCOUNT_SIZE,
    APP_MAX_BUYING_POWER_PCT,
    APP_RISK_PER_TRADE_PCT,
    MIN_REGIME_BARS,
    REGIME_LABELS,
)
from src.dashboard.plots import (
    REGIME_COLORS,
    btc_regime_price_chart,
    equity_comparison_chart,
    exit_reason_chart,
    monthly_returns_heatmap,
    regime_distribution_chart,
    regime_duration_violin,
    regime_transition_heatmap,
    trade_pnl_chart,
)
from src.data.loader import load_raw
from src.regime.regime_labels import compute_regime_stats, compute_transition_matrix
from src.regime.trend_regime import compute_trend_score, score_to_regime, smooth_regimes


SWEEP_MIN_BARS = 1
SWEEP_MAX_BARS = 48


st.set_page_config(
    page_title="BTC Regime Trading",
    page_icon="BTC",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp {
        background:
            radial-gradient(circle at 0% 0%, rgba(11, 140, 102, 0.22), transparent 26%),
            radial-gradient(circle at 100% 0%, rgba(242, 177, 52, 0.16), transparent 24%),
            linear-gradient(180deg, #07111b 0%, #091522 100%);
        color: #f5f7fa;
    }
    .block-container {
        max-width: 1500px;
        padding-top: 1.25rem;
        padding-bottom: 3rem;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(13, 22, 34, 0.95), rgba(10, 17, 27, 0.92));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 14px 16px;
        box-shadow: 0 18px 40px rgba(0, 0, 0, 0.18);
    }
    .hero-card {
        background: linear-gradient(135deg, rgba(13, 22, 34, 0.94), rgba(17, 31, 47, 0.88));
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 24px;
        padding: 24px 28px;
        margin-bottom: 1rem;
        box-shadow: 0 24px 60px rgba(0, 0, 0, 0.18);
    }
    .hero-kicker {
        color: #f2b134;
        font-size: 0.84rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.15;
        color: #f8fafc;
        margin-bottom: 0.55rem;
    }
    .hero-subtitle {
        color: #9db2c8;
        font-size: 1rem;
        line-height: 1.5;
    }
    .panel-card {
        background: linear-gradient(180deg, rgba(13, 22, 34, 0.95), rgba(11, 18, 28, 0.93));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 22px;
        padding: 20px 22px;
        box-shadow: 0 22px 48px rgba(0, 0, 0, 0.14);
    }
    .panel-title {
        color: #f8fafc;
        font-size: 1.02rem;
        font-weight: 700;
        margin-bottom: 0.9rem;
    }
    .snapshot-row {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        padding: 0.55rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.07);
    }
    .snapshot-row:last-child {
        border-bottom: none;
        padding-bottom: 0;
    }
    .snapshot-label {
        color: #9db2c8;
        font-size: 0.94rem;
    }
    .snapshot-value {
        color: #f8fafc;
        font-size: 0.96rem;
        font-weight: 700;
        text-align: right;
    }
    .regime-pill {
        display: inline-block;
        border-radius: 999px;
        padding: 0.36rem 0.8rem;
        font-size: 0.82rem;
        font-weight: 800;
        color: #081018;
        margin-bottom: 0.85rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading OHLCV data...")
def get_raw_data():
    return load_raw()


@st.cache_data(show_spinner="Computing trend score...")
def get_trend_score(df):
    return compute_trend_score(df)


@st.cache_data(show_spinner="Computing raw regimes...")
def get_raw_regime_state(score):
    return score_to_regime(score)


@st.cache_data(show_spinner="Smoothing regimes...")
def get_regimes(raw_regime_state, min_bars):
    return smooth_regimes(raw_regime_state, min_bars)


@st.cache_data(show_spinner="Running trading strategy...")
def get_strategy_result(market_data, regime_state):
    return run_functional_strategy(market_data, regime_state, config=FunctionalTradeConfig())


@st.cache_data(show_spinner="Sweeping regime duration...")
def get_duration_sweep(market_data, raw_regime_state, start_ts, end_ts):
    mask = (market_data.index >= start_ts) & (market_data.index <= end_ts)
    rows: list[dict] = []

    for min_bars in range(SWEEP_MIN_BARS, SWEEP_MAX_BARS + 1):
        smoothed = smooth_regimes(raw_regime_state, min_bars)
        common_idx = smoothed.dropna().index.intersection(market_data.index[mask])
        if len(common_idx) < 2:
            continue

        result = run_functional_strategy(
            market_data.loc[common_idx, ["open", "high", "low", "close"]],
            smoothed.loc[common_idx],
            config=FunctionalTradeConfig(),
        )
        metrics = result.metrics
        rows.append(
            {
                "min_bars": min_bars,
                "sharpe": metrics["sharpe"],
                "total_return_pct": metrics["total_return_pct"],
                "max_drawdown_pct": metrics["max_drawdown_pct"],
                "win_rate_pct": metrics["win_rate"] * 100 if not pd.isna(metrics["win_rate"]) else np.nan,
                "n_trades": metrics["n_trades"],
                "stagnation_days": metrics["stagnation_days"],
                "mean_open_hours": metrics["mean_open_hours"],
                "profit_factor": metrics["profit_factor"],
            }
        )

    return pd.DataFrame(rows)


def choose_best_duration(sweep_df: pd.DataFrame) -> pd.Series | None:
    if sweep_df.empty:
        return None

    valid = sweep_df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["sharpe", "total_return_pct", "max_drawdown_pct"]
    )
    if valid.empty:
        valid = sweep_df.copy()

    with_trades = valid[valid["n_trades"] > 0]
    if not with_trades.empty:
        valid = with_trades

    if valid.empty:
        return None

    ranked = valid.sort_values(
        by=["sharpe", "total_return_pct", "max_drawdown_pct", "stagnation_days", "n_trades"],
        ascending=[False, False, False, True, False],
    )
    return ranked.iloc[0]


def build_sweep_chart(sweep_df: pd.DataFrame, best_duration: int) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=sweep_df["min_bars"],
            y=sweep_df["sharpe"],
            name="Sharpe",
            mode="lines+markers",
            line=dict(color="#8ab4ff", width=2.5),
            marker=dict(size=6),
            hovertemplate="Duration %{x}h<br>Sharpe %{y:.2f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=sweep_df["min_bars"],
            y=sweep_df["total_return_pct"],
            name="Total return",
            mode="lines",
            line=dict(color="#f2b134", width=2, dash="dot"),
            hovertemplate="Duration %{x}h<br>Return %{y:.1f}%<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.add_vline(
        x=best_duration,
        line_dash="dash",
        line_color="#00c853",
        annotation_text=f"Best {best_duration}h",
        annotation_position="top",
    )
    fig.update_layout(
        title="Regime Duration Sweep",
        height=360,
        paper_bgcolor="#07111b",
        plot_bgcolor="#07111b",
        font=dict(color="#f5f7fa"),
        margin=dict(l=40, r=40, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(title="Min regime duration (hours)", gridcolor="#203042")
    fig.update_yaxes(title="Sharpe", gridcolor="#203042", secondary_y=False)
    fig.update_yaxes(title="Total return (%)", gridcolor="#203042", secondary_y=True)
    return fig


def fmt_currency(value: float | int | None, decimals: int = 0) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"${value:,.{decimals}f}"


def fmt_percent(value: float | int | None, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}%"


def fmt_ratio(value: float | int | None, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"


def fmt_hours(value: float | int | None, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f} h"


def fmt_days(value: float | int | None, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f} d"


def build_metrics_table(metrics: dict, buy_hold_metrics: dict) -> pd.DataFrame:
    rows = [
        ("Selected regime duration", f"{metrics.get('selected_min_bars', 'N/A')} h", "N/A"),
        ("Starting capital", fmt_currency(metrics["starting_capital"]), fmt_currency(buy_hold_metrics["starting_capital"])),
        ("Ending capital", fmt_currency(metrics["ending_capital"]), fmt_currency(buy_hold_metrics["ending_capital"])),
        ("Net P&L", fmt_currency(metrics["net_pnl"]), fmt_currency(buy_hold_metrics["net_pnl"])),
        ("Total return", fmt_percent(metrics["total_return_pct"]), fmt_percent(buy_hold_metrics["total_return_pct"])),
        ("Annualized return", fmt_percent(metrics["annualized_return_pct"]), fmt_percent(buy_hold_metrics["annualized_return_pct"])),
        ("Sharpe", fmt_ratio(metrics["sharpe"]), fmt_ratio(buy_hold_metrics["sharpe"])),
        ("Calmar", fmt_ratio(metrics["calmar"]), fmt_ratio(buy_hold_metrics["calmar"])),
        ("Max drawdown", fmt_percent(metrics["max_drawdown_pct"]), fmt_percent(buy_hold_metrics["max_drawdown_pct"])),
        ("Stagnation days", fmt_days(metrics["stagnation_days"]), fmt_days(buy_hold_metrics["stagnation_days"])),
        ("Win rate", fmt_percent(metrics["win_rate"] * 100 if not pd.isna(metrics["win_rate"]) else np.nan), "N/A"),
        ("Mean win", fmt_currency(metrics["avg_win_pnl"], 0), "N/A"),
        ("Mean loss", fmt_currency(metrics["avg_loss_pnl"], 0), "N/A"),
        ("Mean open time", fmt_hours(metrics["mean_open_hours"]), "N/A"),
        ("Profit factor", fmt_ratio(metrics["profit_factor"]), "N/A"),
        ("Average trade", fmt_percent(metrics["avg_trade_pct"]), "N/A"),
        ("Trades", str(metrics["n_trades"]), "1"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Strategy", "Buy & Hold"]).set_index("Metric")


def build_trade_log_table(trades: pd.DataFrame) -> pd.DataFrame:
    if len(trades) == 0:
        return pd.DataFrame()

    holding_hours = (
        (trades["exit_time"] - trades["entry_time"]).dt.total_seconds() / 3600.0
    ).round(1)

    display = pd.DataFrame(
        {
            "Trade #": trades["trade_no"].astype(int),
            "Entry": trades["entry_time"].dt.strftime("%Y-%m-%d %H:%M"),
            "Exit": trades["exit_time"].dt.strftime("%Y-%m-%d %H:%M"),
            "Side": trades["direction"],
            "Units": trades["units"].astype(int),
            "Entry px": trades["entry_price"].map(lambda x: fmt_currency(x, 2)),
            "Exit px": trades["exit_price"].map(lambda x: fmt_currency(x, 2)),
            "Stop px": trades["sl_price"].map(lambda x: fmt_currency(x, 2)),
            "Notional": trades["notional"].map(lambda x: fmt_currency(x, 0)),
            "Max loss": trades["max_loss"].map(lambda x: fmt_currency(x, 0)),
            "Net P&L": trades["net_pnl"].map(lambda x: fmt_currency(x, 2)),
            "R multiple": trades["risk_multiple"].map(lambda x: f"{x:+.2f}R"),
            "Hold (h)": holding_hours.map(lambda x: f"{x:.1f}"),
            "Exit reason": trades["exit_reason"],
            "Account after": trades["account_after"].map(lambda x: fmt_currency(x, 0)),
        }
    )
    return display


def stop_distance_pct(units: int, config: FunctionalTradeConfig) -> float:
    notional = abs(units) * config.unit_notional
    if notional == 0:
        return 0.0
    fee_buffer = 2 * notional * config.fee_rate
    return max(config.risk_budget - fee_buffer, 0.0) / notional * 100


def render_snapshot_card(
    current_regime_label: str,
    current_regime_color: str,
    trades: pd.DataFrame,
    metrics: dict,
    result,
) -> None:
    stop_count = int((trades["exit_reason"] == "STOP LOSS").sum()) if len(trades) else 0
    regime_exit_count = int((trades["exit_reason"] == "REGIME CHANGE").sum()) if len(trades) else 0
    best_trade = trades["net_pnl"].max() if len(trades) else np.nan
    worst_trade = trades["net_pnl"].min() if len(trades) else np.nan
    open_position_label = "Open" if result.open_position else "Flat"

    st.markdown(
        f"""
<div class="panel-card">
    <div class="regime-pill" style="background:{current_regime_color};">{current_regime_label}</div>
    <div class="panel-title">Strategy Snapshot</div>
    <div class="snapshot-row"><span class="snapshot-label">Portfolio state</span><span class="snapshot-value">{open_position_label}</span></div>
    <div class="snapshot-row"><span class="snapshot-label">Chosen duration</span><span class="snapshot-value">{metrics.get('selected_min_bars', 'N/A')} h</span></div>
    <div class="snapshot-row"><span class="snapshot-label">Completed trades</span><span class="snapshot-value">{metrics['n_trades']}</span></div>
    <div class="snapshot-row"><span class="snapshot-label">Stop exits</span><span class="snapshot-value">{stop_count}</span></div>
    <div class="snapshot-row"><span class="snapshot-label">Regime exits</span><span class="snapshot-value">{regime_exit_count}</span></div>
    <div class="snapshot-row"><span class="snapshot-label">Mean open time</span><span class="snapshot-value">{fmt_hours(metrics['mean_open_hours'])}</span></div>
    <div class="snapshot-row"><span class="snapshot-label">Stagnation</span><span class="snapshot-value">{fmt_days(metrics['stagnation_days'])}</span></div>
    <div class="snapshot-row"><span class="snapshot-label">Best trade</span><span class="snapshot-value">{fmt_currency(best_trade, 0)}</span></div>
    <div class="snapshot-row"><span class="snapshot-label">Worst trade</span><span class="snapshot-value">{fmt_currency(worst_trade, 0)}</span></div>
</div>
""",
        unsafe_allow_html=True,
    )


strategy_config = FunctionalTradeConfig()
account_size = strategy_config.account_size
max_notional = strategy_config.max_notional

df = get_raw_data()
data_start = df.index[0].date()
data_end = df.index[-1].date()
default_start_ts = max(df.index[0], df.index[-1] - pd.Timedelta(days=365 * 3))

with st.sidebar:
    st.title("Strategy Controls")
    st.markdown("---")
    date_range = st.date_input(
        "Date range",
        value=[default_start_ts.date(), data_end],
        min_value=data_start,
        max_value=data_end,
    )
    auto_duration = st.toggle("Use best duration from sweep", value=True)
    duration_status = st.empty()
    duration_control = st.empty()

    st.markdown("---")
    st.markdown("**Locked operating model**")
    st.markdown(f"Account: **{fmt_currency(APP_ACCOUNT_SIZE, 0)}**")
    st.markdown(
        f"Static buying power: **{APP_MAX_BUYING_POWER_PCT:.0%}** "
        f"(**{fmt_currency(max_notional, 0)}** max notional)"
    )
    st.markdown(
        f"Risk per trade: **{APP_RISK_PER_TRADE_PCT:.1%}** "
        f"(**{fmt_currency(strategy_config.risk_budget, 0)}** max loss)"
    )
    st.caption("Trades exit at the hard stop or on regime change, whichever comes first.")

    st.markdown("---")
    st.markdown("**Regime sizing**")
    st.markdown(f"Unit notional: **{fmt_currency(strategy_config.unit_notional, 0)}**")
    for idx, label in enumerate(REGIME_LABELS):
        units = DEFAULT_REGIME_UNITS[idx]
        if units == 0:
            st.markdown(
                f'<span style="color:{REGIME_COLORS[idx]}">&#9632;</span> {label}: **Flat**',
                unsafe_allow_html=True,
            )
            continue
        direction = "LONG" if units > 0 else "SHORT"
        notional = abs(units) * strategy_config.unit_notional
        st.markdown(
            f'<span style="color:{REGIME_COLORS[idx]}">&#9632;</span> {label}: '
            f'**{direction} x{abs(units)}** ({fmt_currency(notional, 0)}) | '
            f'SL distance {stop_distance_pct(units, strategy_config):.1f}%',
            unsafe_allow_html=True,
        )

trend_score = get_trend_score(df)
raw_regime_state = get_raw_regime_state(trend_score)

if len(date_range) == 2:
    start_ts = pd.Timestamp(date_range[0], tz="UTC")
    end_ts = pd.Timestamp(date_range[1], tz="UTC")
else:
    start_ts = pd.Timestamp(data_start, tz="UTC")
    end_ts = pd.Timestamp(data_end, tz="UTC")

sweep_df = get_duration_sweep(df[["open", "high", "low", "close"]], raw_regime_state, start_ts, end_ts)
best_duration_row = choose_best_duration(sweep_df)

if auto_duration and best_duration_row is not None:
    selected_min_bars = int(best_duration_row["min_bars"])
    duration_status.markdown(
        f"**Selected duration:** `{selected_min_bars}h` from Sharpe sweep"
    )
    with duration_control.container():
        st.caption("Manual duration appears only when auto sweep is disabled.")
else:
    with duration_control.container():
        manual_min_bars = st.slider(
            "Manual regime duration (hours)",
            SWEEP_MIN_BARS,
            SWEEP_MAX_BARS,
            MIN_REGIME_BARS,
            1,
            key="manual_min_bars",
        )
    selected_min_bars = manual_min_bars
    duration_status.markdown(
        f"**Selected duration:** `{selected_min_bars}h` manual"
    )

regime_state = get_regimes(raw_regime_state, selected_min_bars)

mask = (df.index >= start_ts) & (df.index <= end_ts)
common_idx = regime_state.dropna().index.intersection(df.index[mask])
df_view = df.loc[common_idx]
rs_view = regime_state.loc[common_idx]
score_view = trend_score.loc[common_idx]

if len(df_view) < 2:
    st.error("No data in selected date range.")
    st.stop()

result = get_strategy_result(df_view[["open", "high", "low", "close"]], rs_view)
metrics = result.metrics
metrics["selected_min_bars"] = selected_min_bars
trades = result.trades.copy()

bh_return = df_view["close"].iloc[-1] / df_view["close"].iloc[0] - 1
bh_pnl = max_notional * bh_return
bh_equity = account_size + max_notional * (df_view["close"] / df_view["close"].iloc[0] - 1)
bh_metrics = compute_metrics(bh_equity / account_size)
bh_metrics["net_pnl"] = float(bh_pnl)
bh_metrics["starting_capital"] = float(account_size)
bh_metrics["ending_capital"] = float(account_size + bh_pnl)

current_regime = int(rs_view.iloc[-1])
current_regime_label = REGIME_LABELS[current_regime]
current_regime_color = REGIME_COLORS[current_regime]
monthly_tbl = monthly_returns(result.equity_curve / account_size)
metrics_table = build_metrics_table(metrics, bh_metrics)
trade_log_table = build_trade_log_table(trades.iloc[::-1]) if len(trades) else pd.DataFrame()

st.markdown(
    f"""
<div class="hero-card">
    <div class="hero-kicker">Functional Trading App</div>
    <div class="hero-title">Sweep-driven regime selection, strategy diagnostics, and BTC trade context.</div>
    <div class="hero-subtitle">
        Showing {df_view.index[0].date()} to {df_view.index[-1].date()} |
        {len(df_view):,} hourly bars |
        {len(trades)} completed trades |
        Best regime duration {selected_min_bars}h |
        Static buying power {APP_MAX_BUYING_POWER_PCT:.0%} |
        Risk per trade {APP_RISK_PER_TRADE_PCT:.1%}
    </div>
</div>
""",
    unsafe_allow_html=True,
)

if result.open_position:
    op = result.open_position
    pnl_color = "#00c853" if op["unrealized_pnl"] >= 0 else "#ff6b57"
    st.markdown(
        f"""
<div class="panel-card" style="margin-bottom: 1rem; border-left: 4px solid {pnl_color};">
    <div class="panel-title">Open Position</div>
    <div class="snapshot-row"><span class="snapshot-label">Direction</span><span class="snapshot-value">{op['direction']} x{op['units']}</span></div>
    <div class="snapshot-row"><span class="snapshot-label">Entry</span><span class="snapshot-value">{fmt_currency(op['entry_price'], 2)}</span></div>
    <div class="snapshot-row"><span class="snapshot-label">Current</span><span class="snapshot-value">{fmt_currency(op['current_price'], 2)}</span></div>
    <div class="snapshot-row"><span class="snapshot-label">Stop</span><span class="snapshot-value">{fmt_currency(op['sl_price'], 2)}</span></div>
    <div class="snapshot-row"><span class="snapshot-label">Unrealized P&L</span><span class="snapshot-value" style="color:{pnl_color};">{fmt_currency(op['unrealized_pnl'], 2)}</span></div>
</div>
""",
        unsafe_allow_html=True,
    )

kpi_row_1 = st.columns(4)
kpi_row_2 = st.columns(4)
kpi_row_3 = st.columns(4)
kpi_row_1[0].metric("Account value", fmt_currency(result.equity_curve.iloc[-1], 0))
kpi_row_1[1].metric(
    "Net P&L",
    fmt_currency(metrics["net_pnl"], 0),
    delta=f"{fmt_currency(metrics['net_pnl'] - bh_metrics['net_pnl'], 0)} vs B&H",
    delta_color="normal",
)
kpi_row_1[2].metric(
    "Total return",
    fmt_percent(metrics["total_return_pct"]),
    delta=f"{metrics['total_return_pct'] - bh_metrics['total_return_pct']:+.1f}%",
    delta_color="normal",
)
kpi_row_1[3].metric("Chosen duration", f"{selected_min_bars} h")

kpi_row_2[0].metric("Sharpe", fmt_ratio(metrics["sharpe"]))
kpi_row_2[1].metric(
    "Win rate",
    fmt_percent(metrics["win_rate"] * 100 if not pd.isna(metrics["win_rate"]) else np.nan),
)
kpi_row_2[2].metric("Max drawdown", fmt_percent(metrics["max_drawdown_pct"]))
kpi_row_2[3].metric("Trades", str(metrics["n_trades"]))

kpi_row_3[0].metric("Stagnation days", fmt_days(metrics["stagnation_days"]))
kpi_row_3[1].metric("Mean win", fmt_currency(metrics["avg_win_pnl"], 0))
kpi_row_3[2].metric("Mean loss", fmt_currency(metrics["avg_loss_pnl"], 0))
kpi_row_3[3].metric("Mean open time", fmt_hours(metrics["mean_open_hours"]))

overview_left, overview_right = st.columns([1.85, 1], gap="large")
with overview_left:
    st.markdown("### Equity Evolution")
    st.plotly_chart(
        equity_comparison_chart(result.equity_curve, bh_equity),
        use_container_width=True,
    )

with overview_right:
    st.markdown("### Strategy Snapshot")
    render_snapshot_card(current_regime_label, current_regime_color, trades, metrics, result)
    if len(trades) > 0:
        st.plotly_chart(exit_reason_chart(trades), use_container_width=True)

if best_duration_row is not None and not sweep_df.empty:
    sweep_left, sweep_right = st.columns([1.55, 1], gap="large")
    with sweep_left:
        st.markdown("### Regime Duration Sweep")
        st.plotly_chart(
            build_sweep_chart(sweep_df, selected_min_bars),
            use_container_width=True,
        )
    with sweep_right:
        st.markdown("### Best Sweep Result")
        best_stats = pd.DataFrame(
            {
                "Metric": [
                    "Best duration",
                    "Sharpe",
                    "Total return",
                    "Max drawdown",
                    "Win rate",
                    "Trades",
                    "Stagnation",
                    "Mean open time",
                ],
                "Value": [
                    f"{int(best_duration_row['min_bars'])} h",
                    fmt_ratio(best_duration_row["sharpe"]),
                    fmt_percent(best_duration_row["total_return_pct"]),
                    fmt_percent(best_duration_row["max_drawdown_pct"]),
                    fmt_percent(best_duration_row["win_rate_pct"]),
                    str(int(best_duration_row["n_trades"])),
                    fmt_days(best_duration_row["stagnation_days"]),
                    fmt_hours(best_duration_row["mean_open_hours"]),
                ],
            }
        ).set_index("Metric")
        st.dataframe(best_stats, use_container_width=True)

        with st.expander("Sweep table"):
            sweep_table = sweep_df.copy().sort_values("sharpe", ascending=False)
            sweep_table["sharpe"] = sweep_table["sharpe"].map(lambda x: fmt_ratio(x))
            sweep_table["total_return_pct"] = sweep_table["total_return_pct"].map(lambda x: fmt_percent(x))
            sweep_table["max_drawdown_pct"] = sweep_table["max_drawdown_pct"].map(lambda x: fmt_percent(x))
            sweep_table["win_rate_pct"] = sweep_table["win_rate_pct"].map(lambda x: fmt_percent(x))
            sweep_table["stagnation_days"] = sweep_table["stagnation_days"].map(lambda x: fmt_days(x))
            sweep_table["mean_open_hours"] = sweep_table["mean_open_hours"].map(lambda x: fmt_hours(x))
            sweep_table = sweep_table.rename(
                columns={
                    "min_bars": "Duration (h)",
                    "sharpe": "Sharpe",
                    "total_return_pct": "Return",
                    "max_drawdown_pct": "Max DD",
                    "win_rate_pct": "Win rate",
                    "n_trades": "Trades",
                    "stagnation_days": "Stagnation",
                    "mean_open_hours": "Mean open time",
                    "profit_factor": "Profit factor",
                }
            )
            st.dataframe(sweep_table, use_container_width=True, height=360)

st.markdown("### BTC Price and Trades")
st.caption("BTC is colored by active regime. Entry and exit markers are rendered directly on the price path.")
st.plotly_chart(
    btc_regime_price_chart(df_view, rs_view, trades, max_points=5000),
    use_container_width=True,
)

detail_left, detail_right = st.columns([1.45, 1], gap="large")
with detail_left:
    st.markdown("### Trade P&L Evolution")
    if len(trades) > 0:
        st.plotly_chart(trade_pnl_chart(trades), use_container_width=True)
    else:
        st.info("No trades in the selected period.")

with detail_right:
    st.markdown("### Monthly Return Map")
    st.plotly_chart(monthly_returns_heatmap(monthly_tbl), use_container_width=True)

st.markdown("### Strategy Statistics")
st.dataframe(metrics_table, use_container_width=True)

tab_trades, tab_regimes = st.tabs(["Trade Log", "Regime Analysis"])

with tab_trades:
    if len(trades) > 0:
        summary_cols = st.columns(5)
        winners = trades[trades["net_pnl"] > 0]
        losers = trades[trades["net_pnl"] <= 0]
        summary_cols[0].metric("Winning trades", str(len(winners)))
        summary_cols[1].metric("Losing trades", str(len(losers)))
        summary_cols[2].metric("Stop losses", str((trades["exit_reason"] == "STOP LOSS").sum()))
        summary_cols[3].metric("Best trade", fmt_currency(trades["net_pnl"].max(), 0))
        summary_cols[4].metric("Worst trade", fmt_currency(trades["net_pnl"].min(), 0))

        st.markdown("#### Recent Trades")
        st.dataframe(trade_log_table.head(20), use_container_width=True, height=420)

        st.markdown("#### Full Trade Log")
        st.dataframe(trade_log_table, use_container_width=True, height=620)
    else:
        st.info("No trades in the selected period.")

with tab_regimes:
    regime_col_1, regime_col_2 = st.columns(2)
    with regime_col_1:
        st.plotly_chart(regime_distribution_chart(rs_view), use_container_width=True)
        trans_mat = compute_transition_matrix(rs_view)
        st.plotly_chart(regime_transition_heatmap(trans_mat), use_container_width=True)
    with regime_col_2:
        feat_df_dur = pd.DataFrame({"regime_state": rs_view})
        st.plotly_chart(regime_duration_violin(feat_df_dur), use_container_width=True)

        feat_df = pd.DataFrame(
            {
                "regime_state": rs_view,
                "regime_confidence": (score_view.abs() / 7.0).clip(0, 1),
                "forward_return": np.log(df_view["close"].shift(-8) / df_view["close"]),
            }
        )
        stats = compute_regime_stats(feat_df)
        regime_stats_table = stats[
            [
                "regime_label",
                "count",
                "directional_accuracy",
                "mean_confidence",
                "mean_duration_bars",
            ]
        ].copy()
        regime_stats_table["directional_accuracy"] = regime_stats_table["directional_accuracy"].map(
            lambda x: f"{x:.1%}"
        )
        regime_stats_table["mean_confidence"] = regime_stats_table["mean_confidence"].map(
            lambda x: f"{x:.1%}"
        )
        regime_stats_table["mean_duration_bars"] = regime_stats_table["mean_duration_bars"].map(
            lambda x: f"{x:.1f}"
        )
        st.markdown("#### Regime Statistics")
        st.dataframe(regime_stats_table, use_container_width=True, height=320)
