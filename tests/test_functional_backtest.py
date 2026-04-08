"""Tests for the functional trading app backtest."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest

from src.backtest.functional import FunctionalTradeConfig, run_functional_strategy


def make_market(prices, highs=None, lows=None):
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="h", tz="UTC")
    highs = highs or prices
    lows = lows or prices
    return pd.DataFrame(
        {
            "open": prices,
            "high": highs,
            "low": lows,
            "close": prices,
        },
        index=idx,
    )


def make_regimes(states, market):
    return pd.Series(states, index=market.index, dtype=int)


def test_functional_strategy_uses_static_buying_power_and_risk_budget():
    cfg = FunctionalTradeConfig()
    market = make_market([100.0, 100.0, 100.0])
    regimes = make_regimes([0, 3, 3], market)

    result = run_functional_strategy(market, regimes, config=cfg)

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["notional"] == pytest.approx(cfg.max_notional, abs=0.01)
    assert trade["max_loss"] == pytest.approx(cfg.risk_budget, abs=0.01)
    assert trade["sl_price"] == pytest.approx(95.96, abs=0.01)


def test_functional_strategy_exits_on_regime_change_and_reopens_new_direction():
    cfg = FunctionalTradeConfig()
    market = make_market([100.0, 101.0, 101.0])
    regimes = make_regimes([2, 4, 4], market)

    result = run_functional_strategy(market, regimes, config=cfg)

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["direction"] == "LONG"
    assert trade["exit_reason"] == "REGIME CHANGE"
    assert result.open_position is not None
    assert result.open_position["direction"] == "SHORT"
    assert result.open_position["units"] == 1
    assert result.open_position["entry_time"] == market.index[1]


def test_functional_strategy_stop_loss_uses_intrabar_extreme_and_caps_loss():
    cfg = FunctionalTradeConfig()
    market = make_market(
        prices=[100.0, 99.5],
        highs=[100.0, 100.0],
        lows=[100.0, 95.0],
    )
    regimes = make_regimes([0, 0], market)

    result = run_functional_strategy(market, regimes, config=cfg)

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "STOP LOSS"
    assert trade["exit_price"] == pytest.approx(trade["sl_price"], abs=0.01)
    assert trade["net_pnl"] == pytest.approx(-cfg.risk_budget, abs=0.01)
    assert result.equity_curve.iloc[-1] == pytest.approx(
        cfg.account_size - cfg.risk_budget,
        abs=0.01,
    )
