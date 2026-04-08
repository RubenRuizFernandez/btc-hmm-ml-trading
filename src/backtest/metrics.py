"""Performance metrics for backtests."""
import numpy as np
import pandas as pd

from src.config import HOURS_PER_YEAR, RISK_FREE_RATE


def compute_metrics(
    equity_curve: pd.Series,
    trades: pd.DataFrame | None = None,
    risk_free_rate: float = RISK_FREE_RATE,
) -> dict:
    """
    Compute a full set of performance metrics.

    Parameters
    ----------
    equity_curve : pd.Series
        Portfolio value over time, normalized or dollar-denominated.
    trades : pd.DataFrame | None
        Optional trade log. Expected columns include `pnl_pct`, `net_pnl`,
        `entry_time`, and `exit_time`.
    """
    if len(equity_curve) < 2:
        return _empty_metrics()

    returns = equity_curve.pct_change().dropna()
    rf_per_bar = risk_free_rate / HOURS_PER_YEAR

    excess = returns - rf_per_bar
    ret_std = returns.std()
    if ret_std < 1e-10 or len(returns) < 30:
        sharpe = 0.0
    else:
        sharpe = excess.mean() / ret_std * np.sqrt(HOURS_PER_YEAR)

    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / (rolling_max + 1e-12)
    max_drawdown = float(drawdown.min())

    n_years = len(equity_curve) / HOURS_PER_YEAR
    total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)
    if total_return > -1:
        annualized_return = (1 + total_return) ** (1 / max(n_years, 1e-6)) - 1
    else:
        annualized_return = -1.0
    calmar = float(np.real(annualized_return)) / (abs(max_drawdown) + 1e-12)
    stagnation_days = _max_stagnation_days(equity_curve)

    if trades is not None and len(trades) > 0:
        wins = trades["pnl_pct"] > 0
        losses = trades["pnl_pct"] <= 0
        pnl_col = "net_pnl" if "net_pnl" in trades.columns else "pnl" if "pnl" in trades.columns else None
        win_rate = float(wins.mean())
        avg_win = float(trades.loc[wins, "pnl_pct"].mean()) if wins.any() else 0.0
        avg_loss = float(trades.loc[losses, "pnl_pct"].mean()) if losses.any() else 0.0
        if pnl_col is not None:
            avg_win_pnl = float(trades.loc[wins, pnl_col].mean()) if wins.any() else 0.0
            avg_loss_pnl = float(trades.loc[losses, pnl_col].mean()) if losses.any() else 0.0
        else:
            avg_win_pnl = np.nan
            avg_loss_pnl = np.nan
        gross_profit = trades.loc[wins, "pnl_pct"].sum()
        gross_loss = trades.loc[losses, "pnl_pct"].abs().sum()
        profit_factor = gross_profit / (gross_loss + 1e-12)
        avg_trade = float(trades["pnl_pct"].mean())
        n_trades = len(trades)

        if {"entry_time", "exit_time"}.issubset(trades.columns):
            open_hours = (trades["exit_time"] - trades["entry_time"]).dt.total_seconds() / 3600.0
            mean_open_hours = float(open_hours.mean()) if len(open_hours) > 0 else np.nan
        elif {"entry_bar", "exit_bar"}.issubset(trades.columns) and len(equity_curve.index) > 1:
            inferred_hours = _infer_bar_hours(equity_curve.index)
            open_hours = (trades["exit_bar"] - trades["entry_bar"]) * inferred_hours
            mean_open_hours = float(open_hours.mean()) if len(open_hours) > 0 else np.nan
        else:
            mean_open_hours = np.nan
    else:
        win_rate = avg_win = avg_loss = profit_factor = avg_trade = np.nan
        avg_win_pnl = avg_loss_pnl = mean_open_hours = np.nan
        n_trades = 0

    return {
        "sharpe": float(sharpe),
        "calmar": float(calmar),
        "total_return_pct": total_return * 100,
        "annualized_return_pct": annualized_return * 100,
        "max_drawdown_pct": max_drawdown * 100,
        "stagnation_days": stagnation_days,
        "win_rate": win_rate,
        "avg_win_pct": avg_win * 100 if not np.isnan(avg_win) else np.nan,
        "avg_loss_pct": avg_loss * 100 if not np.isnan(avg_loss) else np.nan,
        "avg_win_pnl": avg_win_pnl,
        "avg_loss_pnl": avg_loss_pnl,
        "profit_factor": float(profit_factor) if not np.isnan(profit_factor) else np.nan,
        "avg_trade_pct": avg_trade * 100 if not np.isnan(avg_trade) else np.nan,
        "mean_open_hours": mean_open_hours,
        "mean_open_days": mean_open_hours / 24 if not np.isnan(mean_open_hours) else np.nan,
        "n_trades": n_trades,
    }


def monthly_returns(equity_curve: pd.Series) -> pd.DataFrame:
    """Pivot table of monthly returns: rows=year, cols=month."""
    ret = equity_curve.resample("ME").last().pct_change().dropna()
    ret.index = pd.DatetimeIndex(ret.index)
    tbl = ret.groupby([ret.index.year, ret.index.month]).first().unstack()
    tbl.index.name = "Year"
    tbl.columns = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ][: tbl.shape[1]]
    return tbl * 100


def _empty_metrics() -> dict:
    return {
        "sharpe": np.nan,
        "calmar": np.nan,
        "total_return_pct": np.nan,
        "annualized_return_pct": np.nan,
        "max_drawdown_pct": np.nan,
        "stagnation_days": np.nan,
        "win_rate": np.nan,
        "avg_win_pct": np.nan,
        "avg_loss_pct": np.nan,
        "avg_win_pnl": np.nan,
        "avg_loss_pnl": np.nan,
        "profit_factor": np.nan,
        "avg_trade_pct": np.nan,
        "mean_open_hours": np.nan,
        "mean_open_days": np.nan,
        "n_trades": 0,
    }


def _max_stagnation_days(equity_curve: pd.Series) -> float:
    """Longest elapsed time without printing a new equity high."""
    if len(equity_curve) < 2:
        return np.nan

    running_max = equity_curve.cummax()
    new_high = running_max.gt(running_max.shift(1).fillna(-np.inf))
    high_times = equity_curve.index[new_high]
    if len(high_times) == 0:
        return 0.0

    stagnation_periods = []
    previous_high = high_times[0]
    for ts in high_times[1:]:
        stagnation_periods.append((ts - previous_high).total_seconds() / 86_400.0)
        previous_high = ts

    stagnation_periods.append((equity_curve.index[-1] - previous_high).total_seconds() / 86_400.0)
    return float(max(stagnation_periods, default=0.0))


def _infer_bar_hours(index: pd.Index) -> float:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return np.nan

    deltas = index.to_series().diff().dropna().dt.total_seconds() / 3600.0
    if len(deltas) == 0:
        return np.nan
    return float(deltas.median())
