"""Streamlit dashboard entry point.

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

from src.config import (
    REGIME_CONFIDENCE_THRESHOLD, REGIME_LABELS,
    FEATURES_PATH, REGIMES_PATH, WF_RESULTS,
)
from src.dashboard.plots import REGIME_COLORS
from src.data.loader import load_raw
from src.data.features import build_hmm_features, build_ml_features, add_target, add_regime_features
from src.regime.hmm_model import BTCHMMModel, extract_regime_series
from src.regime.regime_labels import (
    build_state_map, apply_state_map, compute_regime_stats, compute_transition_matrix
)
from src.models.ensemble import EnsembleSignalModel
from src.backtest.engine import run_backtest
from src.backtest.metrics import monthly_returns, compute_metrics
from src.walkforward.wf_engine import load_wf_summary
from src.dashboard.plots import (
    regime_price_chart, equity_chart, monthly_returns_heatmap,
    trade_histogram, wf_matrix_heatmap, wf_sharpe_line, wf_return_bars,
    shap_importance_chart, regime_transition_heatmap, regime_duration_violin,
)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BTC Quant | HMM Regime + ML",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-box { border: 1px solid #333; border-radius:8px; padding:12px; margin:4px; }
</style>
""", unsafe_allow_html=True)


# ─── Data loading (cached) ────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading OHLCV data...")
def get_raw_data():
    return load_raw()


@st.cache_data(show_spinner="Computing features & regimes...")
def get_features_and_regimes(_df):
    hmm_feat = build_hmm_features(_df)
    ml_feat = build_ml_features(_df)
    common_idx = hmm_feat.index.intersection(ml_feat.index)
    hmm_feat = hmm_feat.loc[common_idx]
    ml_feat = ml_feat.loc[common_idx]
    return hmm_feat, ml_feat


@st.cache_resource(show_spinner="Fitting HMM regime model...")
def get_hmm_model(_hmm_feat):
    from src.config import HMM_FEATURES
    hmm = BTCHMMModel()
    hmm.fit(_hmm_feat[HMM_FEATURES].values)
    return hmm


@st.cache_data(show_spinner="Extracting regimes...")
def get_regimes(_hmm_model, _hmm_feat):
    from src.config import HMM_FEATURES
    posteriors = _hmm_model.predict_proba(_hmm_feat[HMM_FEATURES].values)
    state_map = build_state_map(_hmm_model)
    raw_states, confidence, entropy = extract_regime_series(posteriors)
    regime_state = apply_state_map(raw_states, state_map)
    idx = _hmm_feat.index
    return (
        pd.Series(regime_state, index=idx, name="regime_state"),
        pd.Series(confidence, index=idx, name="regime_confidence"),
        pd.Series(entropy, index=idx, name="regime_entropy"),
        pd.DataFrame(posteriors, index=idx, columns=list(range(7))),
    )


@st.cache_data(show_spinner="Training ensemble signal model...")
def get_ensemble_and_signals(_ml_feat, _regime_state, _conf, _ent, _df):
    ml = add_regime_features(_ml_feat, _regime_state, _conf, _ent)
    ml = add_target(ml, _df["close"])
    ml = ml.dropna()

    feature_cols = [c for c in ml.columns if c not in ("target", "forward_return")]
    y = (ml["target"] == 1).astype(int)

    ensemble = EnsembleSignalModel()
    ensemble.fit(
        ml[feature_cols], y,
        _regime_state.loc[ml.index],
        _conf.loc[ml.index],
    )
    signals = ensemble.generate_signals(
        ml[feature_cols],
        _regime_state.loc[ml.index],
        _conf.loc[ml.index],
    )
    importance = ensemble.lgbm_feature_importance
    return ensemble, signals, importance, feature_cols


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙ Controls")
    st.markdown("---")

    date_range = st.date_input(
        "Date range",
        value=[pd.Timestamp("2018-01-01").date(), pd.Timestamp("2026-04-01").date()],
    )
    conf_threshold = st.slider(
        "Regime confidence threshold",
        min_value=0.50, max_value=0.95, value=REGIME_CONFIDENCE_THRESHOLD, step=0.01,
        format="%.2f",
    )
    show_signals = st.checkbox("Show trade signals on chart", value=True)
    show_posteriors = st.checkbox("Show posterior area chart", value=True)

    st.markdown("---")
    st.markdown("**Regime Colours**")
    for i, label in enumerate(REGIME_LABELS):
        st.markdown(
            f'<span style="color:{REGIME_COLORS[i]}">■</span> {label}',
            unsafe_allow_html=True,
        )


# ─── Load data ────────────────────────────────────────────────────────────────

df = get_raw_data()
hmm_feat, ml_feat = get_features_and_regimes(df)
hmm_model = get_hmm_model(hmm_feat)
regime_state, regime_confidence, regime_entropy, posteriors = get_regimes(hmm_model, hmm_feat)
ensemble, signals, importance, feature_cols = get_ensemble_and_signals(
    ml_feat, regime_state, regime_confidence, regime_entropy, df
)

# ─── Filter by date ───────────────────────────────────────────────────────────
if len(date_range) == 2:
    start_ts = pd.Timestamp(date_range[0], tz="UTC")
    end_ts = pd.Timestamp(date_range[1], tz="UTC")
else:
    start_ts = df.index[0]
    end_ts = df.index[-1]

mask = (df.index >= start_ts) & (df.index <= end_ts)
df_view = df[mask]
rs_view = regime_state[mask]
rc_view = regime_confidence[mask]
re_view = regime_entropy[mask]
post_view = posteriors[mask]
sig_view = signals.reindex(df_view.index, fill_value=0) if signals is not None else None

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("📊 BTC/USD — HMM Regime + ML Trading System")
st.caption(f"Data: {df.index[0].date()} → {df.index[-1].date()} | "
           f"{len(df):,} hourly bars | Confidence threshold: {conf_threshold:.0%}")

# ─── Quick metrics strip ──────────────────────────────────────────────────────
close_view = df_view["close"]
bh_equity = close_view / close_view.iloc[0]
strat_signals = sig_view if sig_view is not None else pd.Series(0, index=df_view.index)
strat_result = run_backtest(close_view, strat_signals, rc_view, kelly_base=0.25)
m = strat_result.metrics

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Sharpe", f"{m['sharpe']:.2f}")
col2.metric("Calmar", f"{m['calmar']:.2f}")
col3.metric("Total Return", f"{m['total_return_pct']:.1f}%")
col4.metric("Max Drawdown", f"{m['max_drawdown_pct']:.1f}%")
col5.metric("# Trades", f"{m['n_trades']}")

st.markdown("---")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["🕯 Price & Regime", "📈 Performance", "🔀 Walk-Forward Matrix", "🔍 Model Insights"]
)

# ── Tab 1: Price & Regime ──────────────────────────────────────────────────────
with tab1:
    st.plotly_chart(
        regime_price_chart(
            df_view,
            rs_view,
            rc_view,
            posteriors=post_view if show_posteriors else None,
            signals=sig_view if show_signals else None,
            confidence_threshold=conf_threshold,
        ),
        use_container_width=True,
    )

    st.subheader("Regime Statistics (filtered period)")
    feat_df = hmm_feat[mask].join(
        pd.DataFrame({
            "regime_state": rs_view,
            "regime_confidence": rc_view,
            "forward_return": np.log(df_view["close"].shift(-8) / df_view["close"]),
        }),
        how="inner",
    )
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


# ── Tab 2: Performance ────────────────────────────────────────────────────────
with tab2:
    st.plotly_chart(
        equity_chart(strat_result.equity_curve, bh_equity),
        use_container_width=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        monthly_tbl = monthly_returns(strat_result.equity_curve)
        st.plotly_chart(monthly_returns_heatmap(monthly_tbl), use_container_width=True)
    with col_b:
        if len(strat_result.trades) > 0:
            st.plotly_chart(trade_histogram(strat_result.trades), use_container_width=True)
        else:
            st.info("No trades in selected period.")

    st.subheader("Performance Metrics")
    st.dataframe(
        pd.DataFrame([m]).T.rename(columns={0: "Value"}).style.format("{:.3f}"),
        use_container_width=True,
    )


# ── Tab 3: Walk-Forward Matrix ────────────────────────────────────────────────
with tab3:
    wf_path = WF_RESULTS / "wf_summary.parquet"
    if wf_path.exists():
        wf_summary = load_wf_summary()
        st.plotly_chart(wf_matrix_heatmap(wf_summary), use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(wf_sharpe_line(wf_summary), use_container_width=True)
        with col_b:
            st.plotly_chart(wf_return_bars(wf_summary), use_container_width=True)

        # Summary stats
        valid = wf_summary.dropna(subset=["sharpe"])
        if len(valid):
            st.subheader("Walk-Forward Summary")
            agg = valid[["sharpe", "calmar", "total_return_pct",
                          "max_drawdown_pct", "win_rate", "profit_factor"]].agg(
                ["mean", "std", "min", "max"]
            )
            st.dataframe(agg.style.format("{:.3f}"), use_container_width=True)
    else:
        st.info(
            "Walk-forward results not found. Run:\n"
            "```\npython scripts/run_pipeline.py --folds 5\n```\n"
            "to generate results."
        )


# ── Tab 4: Model Insights ─────────────────────────────────────────────────────
with tab4:
    col_a, col_b = st.columns(2)

    with col_a:
        if importance is not None:
            st.plotly_chart(shap_importance_chart(importance), use_container_width=True)

    with col_b:
        trans_mat = compute_transition_matrix(rs_view)
        st.plotly_chart(regime_transition_heatmap(trans_mat), use_container_width=True)

    feat_df_dur = pd.DataFrame({"regime_state": rs_view})
    st.plotly_chart(regime_duration_violin(feat_df_dur), use_container_width=True)

    # Regime confidence distribution
    st.subheader("Confidence Distribution by Regime")
    conf_df = pd.DataFrame({
        "Regime": rs_view.map(lambda x: REGIME_LABELS[x]),
        "Confidence": rc_view,
    })
    import plotly.express as px
    fig_box = px.box(
        conf_df, x="Regime", y="Confidence",
        color="Regime",
        color_discrete_sequence=["#00c853","#66bb6a","#a5d6a7","#bdbdbd",
                                  "#ef9a9a","#e57373","#c62828"],
        template="plotly_dark",
    )
    fig_box.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#fafafa"), showlegend=False, height=350,
    )
    st.plotly_chart(fig_box, use_container_width=True)
