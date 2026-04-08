"""Functional trading backtest with static buying power and fixed trade risk."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import (
    APP_ACCOUNT_SIZE,
    APP_MAX_BUYING_POWER_PCT,
    APP_RISK_PER_TRADE_PCT,
    FEE_RATE,
    SLIPPAGE_BPS,
)

DEFAULT_REGIME_UNITS: dict[int, int] = {0: 3, 1: 2, 2: 1, 3: 0, 4: -1, 5: -2, 6: -3}


@dataclass(frozen=True)
class FunctionalTradeConfig:
    """Static operating model for the trading app."""

    account_size: float = APP_ACCOUNT_SIZE
    buying_power_pct: float = APP_MAX_BUYING_POWER_PCT
    risk_per_trade_pct: float = APP_RISK_PER_TRADE_PCT
    regime_units: dict[int, int] = field(default_factory=lambda: DEFAULT_REGIME_UNITS.copy())
    fee_rate: float = FEE_RATE
    slippage_bps: float = SLIPPAGE_BPS

    def __post_init__(self) -> None:
        if self.account_size <= 0:
            raise ValueError("account_size must be positive")
        if not 0 < self.buying_power_pct <= 1:
            raise ValueError("buying_power_pct must be in (0, 1]")
        if not 0 < self.risk_per_trade_pct <= 1:
            raise ValueError("risk_per_trade_pct must be in (0, 1]")
        if not self.regime_units:
            raise ValueError("regime_units cannot be empty")
        if self.max_units <= 0:
            raise ValueError("regime_units must contain at least one non-zero position")

    @property
    def max_units(self) -> int:
        return max(abs(units) for units in self.regime_units.values())

    @property
    def max_notional(self) -> float:
        return self.account_size * self.buying_power_pct

    @property
    def unit_notional(self) -> float:
        return self.max_notional / self.max_units

    @property
    def risk_budget(self) -> float:
        return self.account_size * self.risk_per_trade_pct


@dataclass
class FunctionalResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    open_position: dict | None
    config: FunctionalTradeConfig
    _metrics: dict | None = None

    @property
    def metrics(self) -> dict:
        if self._metrics is None:
            from src.backtest.metrics import compute_metrics

            normalized_equity = self.equity_curve / self.config.account_size
            self._metrics = compute_metrics(
                normalized_equity,
                self.trades if len(self.trades) > 0 else None,
            )
            self._metrics["starting_capital"] = float(self.config.account_size)
            self._metrics["ending_capital"] = float(self.equity_curve.iloc[-1])
            self._metrics["net_pnl"] = float(self.equity_curve.iloc[-1] - self.config.account_size)
        return self._metrics


def run_functional_strategy(
    market_data: pd.DataFrame | pd.Series,
    regime_state: pd.Series,
    config: FunctionalTradeConfig | None = None,
) -> FunctionalResult:
    """Run the functional strategy using static buying power and hard stop risk."""
    cfg = config or FunctionalTradeConfig()
    market = _coerce_market_data(market_data).reindex(regime_state.index)
    regimes = regime_state.reindex(market.index)

    valid_mask = market[["close", "high", "low"]].notna().all(axis=1) & regimes.notna()
    market = market.loc[valid_mask]
    regimes = regimes.loc[valid_mask].astype(int)

    if market.empty:
        raise ValueError("No valid market data is available for the selected range")

    close = market["close"].to_numpy(dtype=float)
    high = market["high"].to_numpy(dtype=float)
    low = market["low"].to_numpy(dtype=float)
    regime_arr = regimes.to_numpy(dtype=int)
    index = market.index

    n_bars = len(market)
    slippage = cfg.slippage_bps * 1e-4
    entry_fee_rate = cfg.fee_rate
    exit_fee_rate = cfg.fee_rate

    equity = np.full(n_bars, cfg.account_size, dtype=float)
    cash = float(cfg.account_size)

    pos_units = 0
    pos_notional = 0.0
    entry_price = 0.0
    entry_fee = 0.0
    stop_price = 0.0
    entry_bar = -1
    entry_account_value = 0.0

    trade_rows: list[dict] = []

    def close_position(bar_idx: int, exit_price: float, reason: str) -> None:
        nonlocal cash, pos_units, pos_notional, entry_price, entry_fee
        nonlocal stop_price, entry_bar, entry_account_value

        exit_fee = pos_notional * exit_fee_rate
        gross_pnl = pos_notional * (exit_price / entry_price - 1.0) * np.sign(pos_units)
        cash += gross_pnl - exit_fee

        net_pnl = cash - entry_account_value
        trade_rows.append(
            {
                "trade_no": len(trade_rows) + 1,
                "entry_time": index[entry_bar],
                "exit_time": index[bar_idx],
                "direction": "LONG" if pos_units > 0 else "SHORT",
                "units": abs(pos_units),
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "sl_price": round(stop_price, 2),
                "notional": round(pos_notional, 2),
                "max_loss": round(cfg.risk_budget, 2),
                "gross_pnl": round(gross_pnl, 2),
                "fees": round(entry_fee + exit_fee, 2),
                "net_pnl": round(net_pnl, 2),
                "pnl_pct": net_pnl / cfg.account_size,
                "risk_multiple": net_pnl / cfg.risk_budget,
                "exit_reason": reason,
                "account_after": round(cash, 2),
            }
        )

        pos_units = 0
        pos_notional = 0.0
        entry_price = 0.0
        entry_fee = 0.0
        stop_price = 0.0
        entry_bar = -1
        entry_account_value = 0.0

    def open_position(bar_idx: int, units: int) -> None:
        nonlocal cash, pos_units, pos_notional, entry_price, entry_fee
        nonlocal stop_price, entry_bar, entry_account_value

        entry_account_value = cash
        pos_notional = abs(units) * cfg.unit_notional
        entry_price = _apply_entry_slippage(close[bar_idx], units, slippage)
        entry_fee = pos_notional * entry_fee_rate
        cash -= entry_fee

        pos_units = units
        entry_bar = bar_idx
        stop_price = _compute_stop_price(
            entry_price=entry_price,
            units=units,
            notional=pos_notional,
            risk_budget=cfg.risk_budget,
            entry_fee=entry_fee,
            exit_fee_rate=exit_fee_rate,
        )

    for i in range(n_bars):
        desired_units = cfg.regime_units.get(regime_arr[i], 0)

        if pos_units != 0 and _stop_was_hit(pos_units, low[i], high[i], stop_price):
            close_position(i, stop_price, "STOP LOSS")
            equity[i] = cash
            continue

        if pos_units != 0 and desired_units != pos_units:
            exit_price = _apply_exit_slippage(close[i], pos_units, slippage)
            close_position(i, exit_price, "REGIME CHANGE")

        if pos_units == 0 and desired_units != 0:
            open_position(i, desired_units)

        equity[i] = cash if pos_units == 0 else cash + _mark_to_market_pnl(
            close_price=close[i],
            entry_price=entry_price,
            units=pos_units,
            notional=pos_notional,
        )

    open_position_snapshot = None
    if pos_units != 0:
        unrealized_pnl = cash + _mark_to_market_pnl(
            close_price=close[-1],
            entry_price=entry_price,
            units=pos_units,
            notional=pos_notional,
        ) - entry_account_value
        open_position_snapshot = {
            "direction": "LONG" if pos_units > 0 else "SHORT",
            "units": abs(pos_units),
            "entry_time": index[entry_bar],
            "entry_price": round(entry_price, 2),
            "current_price": round(close[-1], 2),
            "sl_price": round(stop_price, 2),
            "notional": round(pos_notional, 2),
            "risk_budget": round(cfg.risk_budget, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
        }

    trade_columns = [
        "trade_no",
        "entry_time",
        "exit_time",
        "direction",
        "units",
        "entry_price",
        "exit_price",
        "sl_price",
        "notional",
        "max_loss",
        "gross_pnl",
        "fees",
        "net_pnl",
        "pnl_pct",
        "risk_multiple",
        "exit_reason",
        "account_after",
    ]
    trades = (
        pd.DataFrame(trade_rows)
        if trade_rows
        else pd.DataFrame(columns=trade_columns)
    )

    return FunctionalResult(
        equity_curve=pd.Series(equity, index=index, name="equity"),
        trades=trades,
        open_position=open_position_snapshot,
        config=cfg,
    )


def _coerce_market_data(market_data: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(market_data, pd.Series):
        return pd.DataFrame(
            {
                "close": market_data.astype(float),
                "high": market_data.astype(float),
                "low": market_data.astype(float),
            }
        )

    if not isinstance(market_data, pd.DataFrame):
        raise TypeError("market_data must be a pandas Series or DataFrame")

    if "close" not in market_data.columns:
        raise ValueError("market_data must contain a 'close' column")

    close = market_data["close"].astype(float)
    high = market_data["high"].astype(float) if "high" in market_data else close
    low = market_data["low"].astype(float) if "low" in market_data else close
    return pd.DataFrame({"close": close, "high": high, "low": low}, index=market_data.index)


def _apply_entry_slippage(price: float, units: int, slippage: float) -> float:
    return price * (1.0 + np.sign(units) * slippage)


def _apply_exit_slippage(price: float, units: int, slippage: float) -> float:
    return price * (1.0 - np.sign(units) * slippage)


def _mark_to_market_pnl(
    close_price: float,
    entry_price: float,
    units: int,
    notional: float,
) -> float:
    return notional * (close_price / entry_price - 1.0) * np.sign(units)


def _compute_stop_price(
    entry_price: float,
    units: int,
    notional: float,
    risk_budget: float,
    entry_fee: float,
    exit_fee_rate: float,
) -> float:
    exit_fee = notional * exit_fee_rate
    available_price_risk = max(risk_budget - entry_fee - exit_fee, 0.0)
    stop_move = available_price_risk / notional if notional else 0.0
    if units > 0:
        return entry_price * (1.0 - stop_move)
    return entry_price * (1.0 + stop_move)


def _stop_was_hit(units: int, low_price: float, high_price: float, stop_price: float) -> bool:
    if units > 0:
        return low_price <= stop_price
    return high_price >= stop_price
