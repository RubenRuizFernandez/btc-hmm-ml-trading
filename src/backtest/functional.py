"""Functional trading strategy backtest with fixed sizing and hard stop loss.

Rules
-----
Account   : $100,000 (configurable)
Position  : 40% margin → $40,000 max notional, divided into 3 units
              Super Bull / Super Bear → 3 units ($40K)
              Strong Bull / Strong Bear → 2 units ($26.7K)
              Bull / Bear → 1 unit ($13.3K)
              Sideways → flat
Stop-loss : Hard SL at ±3% of account ($3,000).  Checked every bar.
Entry     : When regime transitions to a new state.
Exit      : Regime change OR hard SL — whichever comes first.
Fees      : 0.1% per side on notional + 5 bps slippage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

from src.config import FEE_RATE, SLIPPAGE_BPS

SLIPPAGE = SLIPPAGE_BPS * 1e-4

# Regime → signed units: +3 long … -3 short
DEFAULT_REGIME_UNITS = {0: 3, 1: 2, 2: 1, 3: 0, 4: -1, 5: -2, 6: -3}


@dataclass
class FunctionalResult:
    equity_curve: pd.Series       # mark-to-market account value (hourly)
    trades: pd.DataFrame          # completed trade log
    open_position: dict | None    # position still open at end (or None)

    _metrics: dict | None = None

    @property
    def metrics(self) -> dict:
        if self._metrics is None:
            from src.backtest.metrics import compute_metrics
            self._metrics = compute_metrics(
                self.equity_curve / self.equity_curve.iloc[0],
                self.trades if len(self.trades) > 0 else None,
            )
            # Add dollar-specific metrics
            self._metrics["starting_capital"] = float(self.equity_curve.iloc[0])
            self._metrics["ending_capital"] = float(self.equity_curve.iloc[-1])
            self._metrics["net_pnl"] = float(
                self.equity_curve.iloc[-1] - self.equity_curve.iloc[0]
            )
        return self._metrics


def run_functional_strategy(
    close: pd.Series,
    regime_state: pd.Series,
    account_size: float = 100_000,
    margin_pct: float = 0.40,
    max_loss_pct: float = 0.03,
    regime_units: dict = DEFAULT_REGIME_UNITS,
    fee_rate: float = FEE_RATE,
    slippage_bps: float = SLIPPAGE_BPS,
    sl_cooldown_bars: int = 24,
) -> FunctionalResult:
    """
    Run a functional dollar-denominated backtest.

    Parameters
    ----------
    close           : hourly BTC/USD closing prices
    regime_state    : integer regime per bar (0=Super Bull … 6=Super Bear)
    account_size    : starting account in USD
    margin_pct      : fraction of account used as max position (0.40 = 40%)
    max_loss_pct    : hard stop-loss as fraction of account (0.03 = 3%)
    regime_units    : mapping {regime_idx → signed units (-3 to +3)}
    fee_rate        : fee per side as fraction of notional
    slippage_bps    : adverse slippage in basis points
    sl_cooldown_bars: bars to wait before re-entering after a stop-loss

    Returns
    -------
    FunctionalResult with equity curve, trade log, and metrics.
    """
    slippage = slippage_bps * 1e-4
    n_units_max = max(abs(v) for v in regime_units.values())
    unit_size = account_size * margin_pct / n_units_max      # $ per unit
    max_loss = account_size * max_loss_pct                    # hard SL in $

    close = close.reindex(regime_state.index)
    n = len(close)
    px = close.values.astype(float)
    regimes = regime_state.values.astype(int)

    equity_arr = np.full(n, account_size)
    cash = float(account_size)      # realized cash balance
    pos_units = 0                   # signed units currently held
    pos_notional = 0.0              # absolute notional of open position
    entry_price = 0.0
    entry_bar = -1
    entry_time = None
    sl_cooldown = 0                 # bars remaining before re-entry allowed
    trade_rows: list[dict] = []

    def _close_position(bar_idx: int, reason: str):
        nonlocal cash, pos_units, pos_notional, entry_price, entry_bar
        fill = px[bar_idx] * (1.0 - np.sign(pos_units) * slippage)
        gross_pnl = pos_notional * (fill / entry_price - 1.0) * np.sign(pos_units)
        exit_fee = pos_notional * fee_rate
        net_pnl = gross_pnl - exit_fee
        cash += net_pnl

        sl_price = _compute_sl_price()
        trade_rows.append({
            "trade_no": len(trade_rows) + 1,
            "entry_time": close.index[entry_bar],
            "exit_time": close.index[bar_idx],
            "direction": "LONG" if pos_units > 0 else "SHORT",
            "units": abs(pos_units),
            "entry_price": round(entry_price, 2),
            "exit_price": round(fill, 2),
            "sl_price": round(sl_price, 2),
            "notional": round(pos_notional, 2),
            "gross_pnl": round(gross_pnl, 2),
            "fees": round(exit_fee + pos_notional * fee_rate, 2),
            "net_pnl": round(net_pnl, 2),
            "exit_reason": reason,
            "account_after": round(cash, 2),
        })
        pos_units = 0
        pos_notional = 0.0
        entry_price = 0.0
        entry_bar = -1

    def _open_position(bar_idx: int, units: int):
        nonlocal cash, pos_units, pos_notional, entry_price, entry_bar, entry_time
        pos_notional = abs(units) * unit_size
        entry_price = px[bar_idx] * (1.0 + np.sign(units) * slippage)
        entry_fee = pos_notional * fee_rate
        cash -= entry_fee
        pos_units = units
        entry_bar = bar_idx
        entry_time = close.index[bar_idx]

    def _compute_sl_price() -> float:
        """Hard-SL price for the current position."""
        if pos_units == 0 or entry_price == 0:
            return 0.0
        # max_loss = notional * |price_move / entry_price|
        # price_move = max_loss / notional * entry_price
        move = max_loss / pos_notional * entry_price
        if pos_units > 0:
            return entry_price - move   # long: SL below entry
        else:
            return entry_price + move   # short: SL above entry

    # ── Main loop ────────────────────────────────────────────────────────────
    for i in range(n):
        desired_units = regime_units.get(regimes[i], 0)

        # ── Check hard stop-loss ─────────────────────────────────────────────
        if pos_units != 0:
            unrealized = pos_notional * (px[i] / entry_price - 1.0) * np.sign(pos_units)
            if unrealized <= -max_loss:
                _close_position(i, "STOP LOSS")
                sl_cooldown = sl_cooldown_bars
                # Mark equity and skip to next bar
                equity_arr[i] = cash
                continue

        # ── Cooldown after stop-loss ─────────────────────────────────────────
        if sl_cooldown > 0:
            sl_cooldown -= 1
            if pos_units == 0:
                equity_arr[i] = cash
                continue

        # ── Regime changed → adjust position ─────────────────────────────────
        if desired_units != pos_units:
            # Close current
            if pos_units != 0:
                _close_position(i, "REGIME CHANGE")

            # Open new
            if desired_units != 0:
                _open_position(i, desired_units)

        # ── Mark-to-market equity ────────────────────────────────────────────
        if pos_units != 0 and entry_price > 0:
            unrealized = pos_notional * (px[i] / entry_price - 1.0) * np.sign(pos_units)
            equity_arr[i] = cash + unrealized
        else:
            equity_arr[i] = cash

    # ── Close open position at final bar ─────────────────────────────────────
    open_pos = None
    if pos_units != 0:
        sl_price = _compute_sl_price()
        unrealized = pos_notional * (px[-1] / entry_price - 1.0) * np.sign(pos_units)
        open_pos = {
            "direction": "LONG" if pos_units > 0 else "SHORT",
            "units": abs(pos_units),
            "entry_time": close.index[entry_bar],
            "entry_price": round(entry_price, 2),
            "current_price": round(px[-1], 2),
            "sl_price": round(sl_price, 2),
            "notional": round(pos_notional, 2),
            "unrealized_pnl": round(unrealized, 2),
        }

    eq_series = pd.Series(equity_arr, index=close.index, name="equity")
    trades_df = (
        pd.DataFrame(trade_rows) if trade_rows
        else pd.DataFrame(columns=[
            "trade_no", "entry_time", "exit_time", "direction", "units",
            "entry_price", "exit_price", "sl_price", "notional",
            "gross_pnl", "fees", "net_pnl", "exit_reason", "account_after",
        ])
    )
    return FunctionalResult(eq_series, trades_df, open_pos)
