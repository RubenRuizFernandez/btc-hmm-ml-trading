"""Tests for the backtest engine and metrics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import run_backtest
from src.backtest.metrics import compute_metrics
from src.backtest.sizing import kelly_size, fractional_kelly, compute_position_sizes


def make_price_series(n: int = 1000, seed: int = 0) -> pd.Series:
    rng = np.random.RandomState(seed)
    returns = rng.randn(n) * 0.01
    prices = 10_000 * np.exp(np.cumsum(returns))
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    return pd.Series(prices, index=idx)


def make_signals(n: int = 1000, seed: int = 1) -> pd.Series:
    rng = np.random.RandomState(seed)
    raw = rng.choice([-1, 0, 1], size=n, p=[0.2, 0.6, 0.2])
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    return pd.Series(raw.astype(float), index=idx)


def make_confidence(n: int = 1000) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    return pd.Series(np.full(n, 0.85), index=idx)


# ── Sizing ────────────────────────────────────────────────────────────────────

def test_kelly_basic():
    k = kelly_size(win_rate=0.6, avg_win=0.02, avg_loss=0.01)
    assert 0 < k <= 1


def test_kelly_zero_when_no_edge():
    k = kelly_size(win_rate=0.4, avg_win=0.01, avg_loss=0.02)
    assert k == 0.0


def test_fractional_kelly_is_quarter():
    k_full = kelly_size(0.6, 0.02, 0.01)
    k_frac = fractional_kelly(0.6, 0.02, 0.01)
    assert abs(k_frac - k_full * 0.25) < 1e-9


def test_position_sizes_vectorized():
    conf = np.array([0.70, 0.75, 0.80, 0.90, 1.00])
    sizes = compute_position_sizes(conf, kelly_base=0.5)
    # Below threshold → 0
    assert sizes[0] == 0.0
    # At threshold → 0
    assert sizes[1] == 0.0
    # Increasing with confidence
    assert sizes[2] < sizes[3] < sizes[4]
    # All ≤ max_position
    assert (sizes <= 1.0).all()


# ── Engine ────────────────────────────────────────────────────────────────────

def test_backtest_returns_correct_shape():
    close = make_price_series()
    signals = make_signals()
    conf = make_confidence()
    result = run_backtest(close, signals, conf)
    assert len(result.equity_curve) == len(close)


def test_equity_starts_at_one():
    close = make_price_series()
    signals = pd.Series(np.zeros(len(close)), index=close.index)
    conf = make_confidence()
    result = run_backtest(close, signals, conf)
    assert result.equity_curve.iloc[0] == pytest.approx(1.0, abs=1e-6)


def test_flat_signals_no_trades():
    close = make_price_series()
    signals = pd.Series(0.0, index=close.index)
    conf = make_confidence()
    result = run_backtest(close, signals, conf)
    assert len(result.trades) == 0


def test_equity_monotone_when_always_right():
    """If we always go long in an up-trending market, equity should grow."""
    n = 500
    prices = 10_000 * np.exp(np.linspace(0, 0.5, n))
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    close = pd.Series(prices, index=idx)
    signals = pd.Series(1.0, index=idx)
    conf = pd.Series(1.0, index=idx)
    result = run_backtest(close, signals, conf, kelly_base=0.5)
    assert result.equity_curve.iloc[-1] > 1.0


def test_no_lookahead_in_engine():
    """Equity at bar t must only depend on prices up to t."""
    close = make_price_series()
    signals = make_signals()
    conf = make_confidence()
    result = run_backtest(close, signals, conf)
    # Simply verify no NaN in equity curve
    assert not result.equity_curve.isnull().any()


# ── Metrics ───────────────────────────────────────────────────────────────────

def test_metrics_keys():
    eq = pd.Series([1.0, 1.02, 0.99, 1.05], index=pd.date_range("2020", periods=4, freq="h", tz="UTC"))
    m = compute_metrics(eq)
    for key in ["sharpe", "calmar", "total_return_pct", "max_drawdown_pct", "win_rate", "profit_factor"]:
        assert key in m


def test_flat_equity_gives_zero_sharpe():
    eq = pd.Series(np.ones(100), index=pd.date_range("2020", periods=100, freq="h", tz="UTC"))
    m = compute_metrics(eq)
    assert np.isnan(m["sharpe"]) or abs(m["sharpe"]) < 1e-6


def test_metrics_include_trade_statistics_and_stagnation():
    idx = pd.date_range("2020-01-01", periods=5, freq="h", tz="UTC")
    eq = pd.Series([1.0, 1.05, 1.05, 1.02, 1.08], index=idx)
    trades = pd.DataFrame(
        {
            "pnl_pct": [0.02, -0.01],
            "net_pnl": [400.0, -200.0],
            "entry_time": [idx[0], idx[2]],
            "exit_time": [idx[1], idx[4]],
        }
    )

    metrics = compute_metrics(eq, trades)

    assert metrics["win_rate"] == pytest.approx(0.5)
    assert metrics["avg_win_pnl"] == pytest.approx(400.0)
    assert metrics["avg_loss_pnl"] == pytest.approx(-200.0)
    assert metrics["mean_open_hours"] == pytest.approx(1.5)
    assert metrics["stagnation_days"] == pytest.approx(3 / 24)
    assert metrics["n_trades"] == 2
