"""Vectorized backtest engine with mark-to-market equity.

Simulates a bar-by-bar strategy with:
- Single-direction positions (long=+1, short=-1, flat=0)
- Fractional Kelly position sizing scaled by regime confidence
- 0.1% per-side fee + 5bps slippage on fill
- Mark-to-market equity curve (unrealized PnL tracked every bar)
"""
import numpy as np
import pandas as pd

from src.config import FEE_RATE, SLIPPAGE_BPS
from src.backtest.metrics import compute_metrics
from src.backtest.sizing import compute_position_sizes


SLIPPAGE = SLIPPAGE_BPS * 1e-4


class BacktestResult:
    def __init__(
        self,
        equity_curve: pd.Series,
        trades: pd.DataFrame,
        signals: pd.Series,
    ):
        self.equity_curve = equity_curve
        self.trades = trades
        self.signals = signals
        self._metrics = None

    @property
    def metrics(self) -> dict:
        if self._metrics is None:
            self._metrics = compute_metrics(self.equity_curve, self.trades)
        return self._metrics


def run_backtest(
    close: pd.Series,
    signals: pd.Series,
    regime_confidence: pd.Series,
    kelly_base: float = 0.25,
    initial_capital: float = 1.0,
) -> BacktestResult:
    """
    Run vectorized mark-to-market backtest.

    Parameters
    ----------
    close              : hourly closing prices
    signals            : +1 (long), -1 (short), 0 (flat)
    regime_confidence  : per-bar confidence in [0, 1]
    kelly_base         : base Kelly fraction (before confidence scaling)
    initial_capital    : starting portfolio value (default 1.0)

    Execution model
    ---------------
    - Signals are acted on at the same bar's close.
    - Entry: fill at close * (1 ± SLIPPAGE); pay FEE_RATE on notional.
    - Exit : fill at close * (1 ∓ SLIPPAGE); pay FEE_RATE on notional.
    - Position sizing: confidence-scaled Kelly at entry.
    - Equity updated EVERY bar via mark-to-market (unrealized PnL visible).
    - Trade PnL recorded at close (realized).
    """
    close = close.reindex(signals.index)
    conf = regime_confidence.reindex(signals.index).fillna(0)

    n = len(signals)
    sig = signals.values.astype(float)
    px = close.values.astype(float)
    conf_arr = conf.values.astype(float)

    # Confidence-scaled position sizes
    sizes = compute_position_sizes(conf_arr, kelly_base)

    equity_arr = np.full(n, initial_capital)
    cash = float(initial_capital)   # portion not in position
    position = 0.0                  # signed position fraction of portfolio
    entry_price = 0.0
    entry_bar = -1
    trade_rows = []

    for i in range(n):
        desired = sig[i] * sizes[i]

        # ── Step 1: close current position if signal changes direction ────────
        if position != 0.0 and not _same_direction(desired, position):
            fill_px = _exit_price(px[i], position)
            # Realized PnL on the position fraction
            gross_pnl = abs(position) * (fill_px / entry_price - 1.0) * np.sign(position)
            net_pnl = gross_pnl - abs(position) * FEE_RATE  # exit fee
            cash = cash * (1.0 + net_pnl)
            trade_rows.append({
                "entry_bar": entry_bar,
                "exit_bar": i,
                "direction": position,
                "entry_price": entry_price,
                "exit_price": fill_px,
                "pnl_pct": net_pnl,
                "pnl": cash * net_pnl,
            })
            position = 0.0
            entry_price = 0.0

        # ── Step 2: open new position if signal and currently flat ────────────
        if desired != 0.0 and position == 0.0:
            fill_px = _entry_price(px[i], desired)
            # Deduct entry fee immediately
            cash *= (1.0 - abs(desired) * FEE_RATE)
            entry_price = fill_px
            position = desired
            entry_bar = i

        # ── Step 3: mark-to-market equity ─────────────────────────────────────
        if position != 0.0 and entry_price > 0:
            mtm_pnl = abs(position) * (px[i] / entry_price - 1.0) * np.sign(position)
            equity_arr[i] = cash * (1.0 + mtm_pnl)
        else:
            equity_arr[i] = cash

    # ── Close any open position at the final bar ──────────────────────────────
    if position != 0.0:
        i = n - 1
        fill_px = _exit_price(px[i], position)
        gross_pnl = abs(position) * (fill_px / entry_price - 1.0) * np.sign(position)
        net_pnl = gross_pnl - abs(position) * FEE_RATE
        cash = cash * (1.0 + net_pnl)
        equity_arr[i] = cash
        trade_rows.append({
            "entry_bar": entry_bar,
            "exit_bar": i,
            "direction": position,
            "entry_price": entry_price,
            "exit_price": fill_px,
            "pnl_pct": net_pnl,
            "pnl": cash * net_pnl,
        })

    idx = signals.index
    eq_series = pd.Series(equity_arr, index=idx, name="equity")
    trades_df = (
        pd.DataFrame(trade_rows)
        if trade_rows
        else pd.DataFrame(
            columns=["entry_bar", "exit_bar", "direction",
                     "entry_price", "exit_price", "pnl_pct", "pnl"]
        )
    )
    return BacktestResult(eq_series, trades_df, signals)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _entry_price(px: float, desired: float) -> float:
    """Adverse slippage on entry: long buys higher, short sells lower."""
    return px * (1.0 + np.sign(desired) * SLIPPAGE)


def _exit_price(px: float, position: float) -> float:
    """Adverse slippage on exit: long sells lower, short buys higher."""
    return px * (1.0 - np.sign(position) * SLIPPAGE)


def _same_direction(desired: float, position: float) -> bool:
    """True if desired and current position are in the same direction."""
    return np.sign(desired) == np.sign(position) and desired != 0.0
