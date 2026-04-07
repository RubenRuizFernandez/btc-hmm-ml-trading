"""Performance metrics for backtests."""
import numpy as np
import pandas as pd
from src.config import RISK_FREE_RATE, HOURS_PER_YEAR


def compute_metrics(
    equity_curve: pd.Series,
    trades: pd.DataFrame | None = None,
    risk_free_rate: float = RISK_FREE_RATE,
) -> dict:
    """
    Compute a full set of performance metrics.

    Parameters
    ----------
    equity_curve : pd.Series  — portfolio value over time (starts at 1.0)
    trades       : pd.DataFrame with columns ['pnl', 'pnl_pct']  (optional)
    """
    if len(equity_curve) < 2:
        return _empty_metrics()

    returns = equity_curve.pct_change().dropna()
    rf_per_bar = risk_free_rate / HOURS_PER_YEAR

    # ── Sharpe ────────────────────────────────────────────────────────────────
    excess = returns - rf_per_bar
    ret_std = returns.std()
    if ret_std < 1e-10 or len(returns) < 30:
        sharpe = 0.0
    else:
        sharpe = excess.mean() / ret_std * np.sqrt(HOURS_PER_YEAR)

    # ── Drawdown ──────────────────────────────────────────────────────────────
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / (rolling_max + 1e-12)
    max_drawdown = float(drawdown.min())

    # ── Calmar ────────────────────────────────────────────────────────────────
    n_years = len(equity_curve) / HOURS_PER_YEAR
    total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)
    annualized_return = (1 + total_return) ** (1 / max(n_years, 1e-6)) - 1
    calmar = annualized_return / (abs(max_drawdown) + 1e-12)

    # ── Trade-level ───────────────────────────────────────────────────────────
    if trades is not None and len(trades) > 0:
        wins = trades["pnl_pct"] > 0
        losses = trades["pnl_pct"] <= 0
        win_rate = float(wins.mean())
        avg_win = float(trades.loc[wins, "pnl_pct"].mean()) if wins.any() else 0.0
        avg_loss = float(trades.loc[losses, "pnl_pct"].mean()) if losses.any() else 0.0
        gross_profit = trades.loc[wins, "pnl_pct"].sum()
        gross_loss = trades.loc[losses, "pnl_pct"].abs().sum()
        profit_factor = gross_profit / (gross_loss + 1e-12)
        avg_trade = float(trades["pnl_pct"].mean())
        n_trades = len(trades)
    else:
        win_rate = avg_win = avg_loss = profit_factor = avg_trade = np.nan
        n_trades = 0

    return {
        "sharpe": float(sharpe),
        "calmar": float(calmar),
        "total_return_pct": total_return * 100,
        "annualized_return_pct": annualized_return * 100,
        "max_drawdown_pct": max_drawdown * 100,
        "win_rate": win_rate,
        "avg_win_pct": avg_win * 100 if not np.isnan(avg_win) else np.nan,
        "avg_loss_pct": avg_loss * 100 if not np.isnan(avg_loss) else np.nan,
        "profit_factor": float(profit_factor) if not np.isnan(profit_factor) else np.nan,
        "avg_trade_pct": avg_trade * 100 if not np.isnan(avg_trade) else np.nan,
        "n_trades": n_trades,
    }


def monthly_returns(equity_curve: pd.Series) -> pd.DataFrame:
    """Pivot table of monthly returns: rows=year, cols=month."""
    ret = equity_curve.resample("ME").last().pct_change().dropna()
    ret.index = pd.DatetimeIndex(ret.index)
    tbl = ret.groupby([ret.index.year, ret.index.month]).first().unstack()
    tbl.index.name = "Year"
    tbl.columns = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ][: tbl.shape[1]]
    return tbl * 100


def _empty_metrics() -> dict:
    return {
        "sharpe": np.nan, "calmar": np.nan,
        "total_return_pct": np.nan, "annualized_return_pct": np.nan,
        "max_drawdown_pct": np.nan, "win_rate": np.nan,
        "avg_win_pct": np.nan, "avg_loss_pct": np.nan,
        "profit_factor": np.nan, "avg_trade_pct": np.nan, "n_trades": 0,
    }
