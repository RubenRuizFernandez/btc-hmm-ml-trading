"""BTC/USD price coloured by HMM regime — thick line, single panel."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.graph_objects as go
from src.data.loader import load_raw
from src.config import REGIME_LABELS

COLORS = {
    0: "#00e676",   # Strong Bull  — vivid green
    1: "#69f0ae",   # Bull         — light green
    2: "#b9f6ca",   # Weak Bull    — very light green
    3: "#90a4ae",   # Sideways     — grey
    4: "#ffb300",   # Weak Bear    — amber
    5: "#ff5252",   # Bear         — red
    6: "#b71c1c",   # Strong Bear  — dark red
}

# ── Load ──────────────────────────────────────────────────────────────────────
df   = load_raw()
reg  = pd.read_parquet(Path(__file__).parent.parent / "data/processed/regimes.parquet")
common = df.index.intersection(reg.index)
df, reg = df.loc[common], reg.loc[common]

# Filter from 2018
start = pd.Timestamp("2018-01-01", tz="UTC")
df  = df[df.index >= start]
reg = reg[reg.index >= start]

# Daily OHLCV for display
daily = df.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
regime_daily = reg["regime_state"].resample("1D").last().reindex(daily.index, method="ffill")

# ── Build one trace per consecutive regime spell ───────────────────────────────
fig = go.Figure()
legend_done = set()

vals  = regime_daily.values
idx   = daily.index
close = daily["close"].values

i = 0
while i < len(vals):
    s = int(vals[i]) if pd.notna(vals[i]) else 3
    j = i + 1
    while j < len(vals) and (pd.isna(vals[j]) or int(vals[j]) == s):
        j += 1

    # Overlap by 1 point so segments connect without gaps
    x_seg = idx[i : j + 1]
    y_seg = close[i : j + 1]

    show = s not in legend_done
    legend_done.add(s)

    fig.add_trace(go.Scatter(
        x=x_seg, y=y_seg,
        mode="lines",
        line=dict(color=COLORS[s], width=3),
        name=REGIME_LABELS[s],
        legendgroup=str(s),
        showlegend=show,
        hovertemplate="%{x|%Y-%m-%d}  $%{y:,.0f}<extra>" + REGIME_LABELS[s] + "</extra>",
    ))
    i = j

# ── Layout ────────────────────────────────────────────────────────────────────
fig.update_layout(
    title=dict(text="BTC/USD — HMM Regime (7 states)", x=0.5,
               font=dict(size=20, color="#ffffff")),
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(color="#c9d1d9", family="Arial, sans-serif"),
    height=550,
    margin=dict(l=60, r=30, t=70, b=40),
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="center", x=0.5,
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
    ),
    xaxis=dict(
        gridcolor="#21262d",
        showgrid=True,
        zeroline=False,
        rangeslider=dict(visible=True, bgcolor="#161b22", thickness=0.04),
    ),
    yaxis=dict(
        type="log",
        title="Price USD (log scale)",
        gridcolor="#21262d",
        tickformat="$,.0f",
        showgrid=True,
        zeroline=False,
    ),
)

out = Path(__file__).parent.parent / "btc_regime_chart.html"
fig.write_html(str(out), include_plotlyjs="cdn")

import webbrowser
webbrowser.open(out.resolve().as_uri())
print("Saved and opened:", out)
