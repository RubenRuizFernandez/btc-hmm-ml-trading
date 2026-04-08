"""MT5 execution infrastructure for the BTC regime strategy."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.functional import FunctionalTradeConfig, run_functional_strategy
from src.config import DATA_RAW, REGIME_LABELS, ROOT
from src.mt5.data_sync import MT5DataSyncConfig, sync_csv_from_mt5
from src.regime.trend_regime import compute_trend_score, score_to_regime, smooth_regimes


DEFAULT_SNAPSHOT_PATH = ROOT / "data" / "live" / "mt5_signal_snapshot.json"
DEFAULT_STATE_PATH = ROOT / "data" / "live" / "mt5_controller_state.json"
DEFAULT_JOURNAL_PATH = ROOT / "data" / "live" / "mt5_execution_journal.jsonl"
SUCCESS_RETCODE_NAMES = ("TRADE_RETCODE_DONE", "TRADE_RETCODE_DONE_PARTIAL", "TRADE_RETCODE_PLACED")


@dataclass(frozen=True)
class MT5ConnectionConfig:
    """Connection settings for the local MT5 terminal."""

    path: str | None = None
    login: int | None = None
    password: str | None = None
    server: str | None = None
    timeout_ms: int = 30_000


@dataclass(frozen=True)
class MT5RuntimeConfig:
    """Runtime settings for live execution."""

    symbol: str = "BTCUSD"
    timeframe: str = "H1"
    history_bars: int = 24 * 450
    data_path: Path = DATA_RAW
    sync_data_to_csv: bool = True
    sync_full_history_on_start: bool = True
    history_start: str = "2010-01-01T00:00:00Z"
    sync_chunk_days: int = 365
    sync_overlap_bars: int = 24 * 14
    signal_min_bars: int = 1
    auto_optimize_duration: bool = False
    optimize_min_bars: int = 1
    optimize_max_bars: int = 48
    optimize_lookback_bars: int = 24 * 365
    magic_number: int = 26040801
    deviation_points: int = 200
    poll_seconds: int = 30
    dry_run: bool = True
    comment_prefix: str = "btc-rg"
    cooldown_after_sl_hours: int = 24
    max_tick_age_seconds: int = 180
    max_spread_bps: float = 15.0
    max_margin_pct: float | None = None
    snapshot_path: Path = DEFAULT_SNAPSHOT_PATH
    state_path: Path = DEFAULT_STATE_PATH
    journal_path: Path = DEFAULT_JOURNAL_PATH


@dataclass(frozen=True)
class ControllerState:
    """Persistent controller state for live trading safeguards."""

    cooldown_until: pd.Timestamp | None = None
    last_stop_loss_time: pd.Timestamp | None = None
    last_stop_loss_ticket: int | None = None
    blocked_reason: str | None = None
    updated_at: pd.Timestamp | None = None

    @property
    def cooldown_active(self) -> bool:
        if self.cooldown_until is None:
            return False
        return pd.Timestamp.now(tz="UTC") < self.cooldown_until


@dataclass(frozen=True)
class LiveSignal:
    """Latest signal computed from MT5 market data."""

    bar_time: pd.Timestamp
    close_price: float
    trend_score: float
    raw_regime_state: int
    regime_state: int
    regime_label: str
    desired_units: int
    selected_min_bars: int
    target_notional: float

    @property
    def side(self) -> str:
        if self.desired_units > 0:
            return "BUY"
        if self.desired_units < 0:
            return "SELL"
        return "FLAT"


@dataclass(frozen=True)
class ManagedPosition:
    """MT5 position that belongs to this strategy instance."""

    ticket: int
    side: str
    volume: float
    price_open: float
    sl: float
    comment: str
    symbol: str
    magic: int
    signed_units: int | None = None


@dataclass(frozen=True)
class ExecutionAction:
    """Single broker action produced by the reconciliation step."""

    kind: str
    reason: str
    ticket: int | None = None
    side: str | None = None
    volume: float | None = None
    price: float | None = None
    sl: float | None = None
    comment: str | None = None


@dataclass(frozen=True)
class ExecutionPlan:
    """Target execution state for the current signal."""

    signal: LiveSignal
    reference_price: float | None
    target_volume: float
    actual_notional: float
    max_notional_cap: float
    stop_loss: float | None
    actions: tuple[ExecutionAction, ...]
    sweep_summary: pd.DataFrame | None = None


@dataclass(frozen=True)
class ExecutionReport:
    """Output from a dry-run or live execution cycle."""

    signal: LiveSignal
    effective_signal: LiveSignal
    plan: ExecutionPlan
    controller_state: ControllerState
    current_positions: tuple[ManagedPosition, ...]
    dry_run: bool
    spread_bps: float | None = None
    tick_age_seconds: float | None = None
    max_margin_pct: float | None = None
    max_margin_allowed_usd: float | None = None
    margin_required: float | None = None
    margin_free_before: float | None = None
    margin_free_after: float | None = None
    broker_results: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        sweep_top = None
        if self.plan.sweep_summary is not None and not self.plan.sweep_summary.empty:
            sweep_top = self.plan.sweep_summary.head(5).to_dict(orient="records")

        return {
            "signal": {
                **_json_safe(asdict(self.signal)),
                "side": self.signal.side,
            },
            "effective_signal": {
                **_json_safe(asdict(self.effective_signal)),
                "side": self.effective_signal.side,
            },
            "plan": {
                "reference_price": self.plan.reference_price,
                "target_volume": self.plan.target_volume,
                "actual_notional": self.plan.actual_notional,
                "max_notional_cap": self.plan.max_notional_cap,
                "stop_loss": self.plan.stop_loss,
                "actions": [_json_safe(asdict(action)) for action in self.plan.actions],
                "sweep_top": _json_safe(sweep_top),
            },
            "controller_state": _json_safe(asdict(self.controller_state)),
            "current_positions": [_json_safe(asdict(position)) for position in self.current_positions],
            "dry_run": self.dry_run,
            "spread_bps": self.spread_bps,
            "tick_age_seconds": self.tick_age_seconds,
            "max_margin_pct": self.max_margin_pct,
            "max_margin_allowed_usd": self.max_margin_allowed_usd,
            "margin_required": self.margin_required,
            "margin_free_before": self.margin_free_before,
            "margin_free_after": self.margin_free_after,
            "broker_results": [_json_safe(result) for result in self.broker_results],
        }


def sweep_regime_durations(
    market_data: pd.DataFrame,
    raw_regime_state: pd.Series,
    strategy_config: FunctionalTradeConfig,
    min_bars: int,
    max_bars: int,
    lookback_bars: int | None = None,
) -> pd.DataFrame:
    """Evaluate regime smoothing duration over a trailing window."""
    if min_bars > max_bars:
        raise ValueError("min_bars must be <= max_bars")

    eval_index = market_data.index[-lookback_bars:] if lookback_bars else market_data.index
    rows: list[dict[str, Any]] = []
    for duration in range(min_bars, max_bars + 1):
        smoothed = smooth_regimes(raw_regime_state, duration)
        common_idx = smoothed.dropna().index.intersection(eval_index)
        if len(common_idx) < 2:
            continue

        result = run_functional_strategy(
            market_data.loc[common_idx, ["open", "high", "low", "close"]],
            smoothed.loc[common_idx],
            config=strategy_config,
        )
        metrics = result.metrics
        rows.append(
            {
                "min_bars": duration,
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
    """Apply the same ranking policy used in the dashboard sweep."""
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


def build_live_signal(
    market_data: pd.DataFrame,
    strategy_config: FunctionalTradeConfig,
    runtime_config: MT5RuntimeConfig,
) -> tuple[LiveSignal, pd.DataFrame | None]:
    """Compute the current live signal from MT5 OHLCV data."""
    trend_score = compute_trend_score(market_data)
    raw_regime_state = score_to_regime(trend_score)

    sweep_df = None
    selected_min_bars = runtime_config.signal_min_bars
    if runtime_config.auto_optimize_duration:
        sweep_df = sweep_regime_durations(
            market_data=market_data,
            raw_regime_state=raw_regime_state,
            strategy_config=strategy_config,
            min_bars=runtime_config.optimize_min_bars,
            max_bars=runtime_config.optimize_max_bars,
            lookback_bars=runtime_config.optimize_lookback_bars,
        )
        best = choose_best_duration(sweep_df)
        if best is not None:
            selected_min_bars = int(best["min_bars"])

    smoothed = smooth_regimes(raw_regime_state, selected_min_bars)
    regime_state = int(smoothed.iloc[-1])
    desired_units = int(strategy_config.regime_units.get(regime_state, 0))

    signal = LiveSignal(
        bar_time=smoothed.index[-1],
        close_price=float(market_data["close"].iloc[-1]),
        trend_score=float(trend_score.iloc[-1]),
        raw_regime_state=int(raw_regime_state.iloc[-1]),
        regime_state=regime_state,
        regime_label=REGIME_LABELS[regime_state],
        desired_units=desired_units,
        selected_min_bars=selected_min_bars,
        target_notional=abs(desired_units) * strategy_config.unit_notional,
    )
    return signal, sweep_df


def apply_controller_overrides(
    signal: LiveSignal,
    controller_state: ControllerState,
    now: pd.Timestamp | None = None,
) -> tuple[LiveSignal, str | None]:
    """Flatten new entries while a controller cooldown is active."""
    current_time = now or pd.Timestamp.now(tz="UTC")
    if controller_state.cooldown_until is None or current_time >= controller_state.cooldown_until:
        return signal, None

    blocked_reason = (
        "Cooldown active after stop-loss execution until "
        f"{controller_state.cooldown_until.isoformat()}"
    )
    effective_signal = replace(signal, desired_units=0, target_notional=0.0)
    return effective_signal, blocked_reason


def build_execution_plan(
    signal: LiveSignal,
    positions: list[ManagedPosition],
    symbol_info: Any,
    tick: Any,
    strategy_config: FunctionalTradeConfig,
    runtime_config: MT5RuntimeConfig,
    sweep_summary: pd.DataFrame | None = None,
) -> ExecutionPlan:
    """Turn the latest signal into broker actions."""
    step = _volume_step(symbol_info)
    actions: list[ExecutionAction] = []
    reference_price: float | None = None
    target_volume = 0.0
    actual_notional = 0.0
    stop_loss: float | None = None
    open_comment: str | None = None

    if signal.desired_units != 0:
        reference_price = _reference_price_from_tick(tick, signal.side)
        target_volume = compute_target_volume_lots(signal, reference_price, symbol_info, strategy_config)
        actual_notional = compute_actual_notional(reference_price, target_volume, symbol_info)
        stop_loss = compute_live_stop_loss(
            signal.side,
            reference_price,
            actual_notional,
            strategy_config,
            symbol_info,
        )
        open_comment = build_strategy_comment(runtime_config.comment_prefix, signal)

    if not positions:
        if signal.desired_units == 0:
            actions.append(ExecutionAction(kind="NOOP", reason="No managed positions and signal is flat"))
        else:
            actions.append(
                ExecutionAction(
                    kind="OPEN",
                    reason="No managed position and non-flat signal",
                    side=signal.side,
                    volume=target_volume,
                    price=reference_price,
                    sl=stop_loss,
                    comment=open_comment,
                )
            )
        return ExecutionPlan(
            signal,
            reference_price,
            target_volume,
            actual_notional,
            strategy_config.max_notional,
            stop_loss,
            tuple(actions),
            sweep_summary,
        )

    if signal.desired_units == 0:
        for position in positions:
            actions.append(
                ExecutionAction(
                    kind="CLOSE",
                    reason="Signal is flat",
                    ticket=position.ticket,
                    side=_close_side(position.side),
                    volume=position.volume,
                    comment=f"{runtime_config.comment_prefix}|flat",
                )
            )
        return ExecutionPlan(
            signal,
            reference_price,
            target_volume,
            actual_notional,
            strategy_config.max_notional,
            stop_loss,
            tuple(actions),
            sweep_summary,
        )

    if len(positions) != 1:
        for position in positions:
            actions.append(
                ExecutionAction(
                    kind="CLOSE",
                    reason="Multiple managed positions; flatten before syncing",
                    ticket=position.ticket,
                    side=_close_side(position.side),
                    volume=position.volume,
                    comment=f"{runtime_config.comment_prefix}|sync",
                )
            )
        actions.append(
            ExecutionAction(
                kind="OPEN",
                reason="Rebuild position to target state",
                side=signal.side,
                volume=target_volume,
                price=reference_price,
                sl=stop_loss,
                comment=open_comment,
            )
        )
        return ExecutionPlan(
            signal,
            reference_price,
            target_volume,
            actual_notional,
            strategy_config.max_notional,
            stop_loss,
            tuple(actions),
            sweep_summary,
        )

    position = positions[0]
    same_side = position.side == signal.side
    same_units = position.signed_units == signal.desired_units if position.signed_units is not None else False
    same_volume = math.isclose(position.volume, target_volume, abs_tol=max(step / 2, 1e-9))
    same_stop = _same_stop(position.sl, stop_loss, symbol_info)

    if same_side and (same_units or same_volume):
        if not same_stop:
            actions.append(
                ExecutionAction(
                    kind="UPDATE_SL",
                    reason="Signal unchanged but stop must be refreshed",
                    ticket=position.ticket,
                    sl=stop_loss,
                    comment=open_comment,
                )
            )
        else:
            actions.append(
                ExecutionAction(
                    kind="NOOP",
                    reason="Managed position already matches target signal",
                    ticket=position.ticket,
                    comment=position.comment,
                )
            )
    else:
        actions.append(
            ExecutionAction(
                kind="CLOSE",
                reason="Managed position does not match target signal",
                ticket=position.ticket,
                side=_close_side(position.side),
                volume=position.volume,
                comment=f"{runtime_config.comment_prefix}|flip",
            )
        )
        actions.append(
            ExecutionAction(
                kind="OPEN",
                reason="Open target signal after sync",
                side=signal.side,
                volume=target_volume,
                price=reference_price,
                sl=stop_loss,
                comment=open_comment,
            )
        )

    return ExecutionPlan(
        signal,
        reference_price,
        target_volume,
        actual_notional,
        strategy_config.max_notional,
        stop_loss,
        tuple(actions),
        sweep_summary,
    )


def build_strategy_comment(prefix: str, signal: LiveSignal) -> str:
    """Compact comment for MT5 positions."""
    return f"{prefix}|u={signal.desired_units}|r={signal.regime_state}|mb={signal.selected_min_bars}"


def parse_strategy_comment(comment: str | None, prefix: str) -> int | None:
    """Extract signed units from an MT5 comment."""
    if not comment or not comment.startswith(prefix):
        return None
    for part in comment.split("|")[1:]:
        if part.startswith("u="):
            try:
                return int(part.split("=", 1)[1])
            except ValueError:
                return None
    return None


def compute_target_volume_lots(
    signal: LiveSignal,
    reference_price: float,
    symbol_info: Any,
    strategy_config: FunctionalTradeConfig,
) -> float:
    """Convert strategy notional into broker lots."""
    if signal.desired_units == 0:
        return 0.0

    raw_volume = signal.target_notional / (reference_price * _contract_size(symbol_info))
    volume = _round_volume(raw_volume, symbol_info)
    if volume <= 0:
        raise ValueError("Computed MT5 volume rounded down to zero")

    volume_min = float(getattr(symbol_info, "volume_min", 0.0) or 0.0)
    if volume < volume_min:
        raise ValueError(
            f"Computed MT5 volume {volume} is below minimum {volume_min} for {getattr(symbol_info, 'name', 'symbol')}"
        )
    return volume


def compute_actual_notional(reference_price: float, volume: float, symbol_info: Any) -> float:
    """Dollar exposure from MT5 lots."""
    return reference_price * volume * _contract_size(symbol_info)


def compute_live_stop_loss(
    side: str,
    reference_price: float,
    notional: float,
    strategy_config: FunctionalTradeConfig,
    symbol_info: Any,
) -> float:
    """Stop price matching the project's fixed max-loss budget."""
    fee_buffer = 2 * notional * strategy_config.fee_rate
    move = max(strategy_config.risk_budget - fee_buffer, 0.0) / max(notional, 1e-12)
    stop_price = reference_price * (1.0 - move) if side == "BUY" else reference_price * (1.0 + move)
    return _round_price(stop_price, symbol_info)


def extract_managed_positions(
    raw_positions: list[Any] | tuple[Any, ...] | None,
    runtime_config: MT5RuntimeConfig,
) -> list[ManagedPosition]:
    """Keep only the positions that belong to this strategy instance."""
    managed: list[ManagedPosition] = []
    if not raw_positions:
        return managed

    for position in raw_positions:
        comment = str(getattr(position, "comment", "") or "")
        magic = int(getattr(position, "magic", 0) or 0)
        if magic != runtime_config.magic_number or not comment.startswith(runtime_config.comment_prefix):
            continue

        side = "BUY" if int(getattr(position, "type")) == 0 else "SELL"
        managed.append(
            ManagedPosition(
                ticket=int(getattr(position, "ticket")),
                side=side,
                volume=float(getattr(position, "volume")),
                price_open=float(getattr(position, "price_open", 0.0) or 0.0),
                sl=float(getattr(position, "sl", 0.0) or 0.0),
                comment=comment,
                symbol=str(getattr(position, "symbol", "")),
                magic=magic,
                signed_units=parse_strategy_comment(comment, runtime_config.comment_prefix),
            )
        )
    return managed


class MT5Executor:
    """Live MT5 execution loop for the BTC regime strategy."""

    def __init__(
        self,
        connection_config: MT5ConnectionConfig | None = None,
        runtime_config: MT5RuntimeConfig | None = None,
        strategy_config: FunctionalTradeConfig | None = None,
        mt5_module: Any | None = None,
    ) -> None:
        self.connection_config = connection_config or MT5ConnectionConfig()
        self.runtime_config = runtime_config or MT5RuntimeConfig()
        self.strategy_config = strategy_config or FunctionalTradeConfig()
        self._mt5 = mt5_module
        self._initialized = False
        self._did_initial_data_sync = False

    @property
    def mt5(self) -> Any:
        if self._mt5 is None:
            try:
                import MetaTrader5 as mt5  # type: ignore
            except ImportError as exc:
                raise RuntimeError("MetaTrader5 package is not installed") from exc
            self._mt5 = mt5
        return self._mt5

    def initialize(self) -> None:
        if self._initialized:
            return

        kwargs: dict[str, Any] = {"timeout": self.connection_config.timeout_ms}
        if self.connection_config.path:
            kwargs["path"] = self.connection_config.path
        if self.connection_config.login is not None:
            kwargs["login"] = self.connection_config.login
        if self.connection_config.password is not None:
            kwargs["password"] = self.connection_config.password
        if self.connection_config.server is not None:
            kwargs["server"] = self.connection_config.server

        if not self.mt5.initialize(**kwargs):
            error = self.mt5.last_error()
            message = (
                "MT5 initialize failed. Open the MT5 terminal, log into the target account, "
                "and pass --path to terminal64.exe if the default terminal is not the correct one. "
                f"last_error={error}"
            )
            raise RuntimeError(message)
        if not self.mt5.symbol_select(self.runtime_config.symbol, True):
            raise RuntimeError(f"MT5 symbol_select failed for {self.runtime_config.symbol}: {self.mt5.last_error()}")
        self._initialized = True

    def shutdown(self) -> None:
        if self._initialized:
            self.mt5.shutdown()
            self._initialized = False

    def fetch_market_data(self) -> pd.DataFrame:
        timeframe = _mt5_timeframe(self.mt5, self.runtime_config.timeframe)
        rates = self.mt5.copy_rates_from_pos(
            self.runtime_config.symbol,
            timeframe,
            0,
            self.runtime_config.history_bars,
        )
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"MT5 copy_rates_from_pos returned no data: {self.mt5.last_error()}")

        df = pd.DataFrame(rates)
        df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"tick_volume": "volume"}).set_index("datetime")
        df = df[["open", "high", "low", "close", "volume"]].astype(float).sort_index()
        df = _drop_open_bar(df, self.runtime_config.timeframe)
        if len(df) < 300:
            raise RuntimeError("MT5 returned too little history to compute the regime signal")
        return df

    def sync_market_data(self, force_full: bool | None = None) -> pd.DataFrame:
        """Persist MT5 history to the shared CSV and return the refreshed frame."""
        full_sync = (
            self.runtime_config.sync_full_history_on_start and not self._did_initial_data_sync
            if force_full is None
            else force_full
        )
        sync_config = MT5DataSyncConfig(
            symbol=self.runtime_config.symbol,
            timeframe=self.runtime_config.timeframe,
            csv_path=self.runtime_config.data_path,
            history_start=self.runtime_config.history_start,
            chunk_days=self.runtime_config.sync_chunk_days,
            overlap_bars=self.runtime_config.sync_overlap_bars,
        )
        market_data = sync_csv_from_mt5(self.mt5, config=sync_config, force_full=full_sync)
        self._did_initial_data_sync = True
        if len(market_data) < 300:
            raise RuntimeError("MT5 CSV sync returned too little history to compute the regime signal")
        return market_data

    def get_positions(self) -> list[ManagedPosition]:
        raw_positions = self.mt5.positions_get(symbol=self.runtime_config.symbol)
        return extract_managed_positions(raw_positions, self.runtime_config)

    def _load_controller_state(self) -> ControllerState:
        path = self.runtime_config.state_path
        if not path.exists():
            return ControllerState()

        payload = json.loads(path.read_text(encoding="utf-8"))
        return ControllerState(
            cooldown_until=_parse_timestamp(payload.get("cooldown_until")),
            last_stop_loss_time=_parse_timestamp(payload.get("last_stop_loss_time")),
            last_stop_loss_ticket=payload.get("last_stop_loss_ticket"),
            blocked_reason=payload.get("blocked_reason"),
            updated_at=_parse_timestamp(payload.get("updated_at")),
        )

    def _save_controller_state(self, state: ControllerState) -> None:
        path = self.runtime_config.state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_json_safe(asdict(state)), indent=2), encoding="utf-8")

    def _refresh_controller_state(self, now: pd.Timestamp) -> ControllerState:
        state = self._load_controller_state()
        lookback_hours = max(self.runtime_config.cooldown_after_sl_hours + 24, 72)
        start = (now - pd.Timedelta(hours=lookback_hours)).to_pydatetime()
        end = now.to_pydatetime()
        deals = self.mt5.history_deals_get(start, end, group=f"*{self.runtime_config.symbol}*")

        latest_ticket = state.last_stop_loss_ticket
        latest_time = state.last_stop_loss_time
        if deals is not None:
            latest_deal = _latest_managed_stop_loss_deal(
                deals=deals,
                runtime_config=self.runtime_config,
                mt5=self.mt5,
            )
            if latest_deal is not None:
                latest_ticket = int(getattr(latest_deal, "ticket"))
                latest_time = _deal_timestamp(latest_deal)

        cooldown_until = (
            latest_time + pd.Timedelta(hours=self.runtime_config.cooldown_after_sl_hours)
            if latest_time is not None
            else None
        )
        blocked_reason = None
        if cooldown_until is not None and now < cooldown_until:
            blocked_reason = f"Cooldown active after stop-loss until {cooldown_until.isoformat()}"

        refreshed = ControllerState(
            cooldown_until=cooldown_until,
            last_stop_loss_time=latest_time,
            last_stop_loss_ticket=latest_ticket,
            blocked_reason=blocked_reason,
            updated_at=now,
        )
        self._save_controller_state(refreshed)
        return refreshed

    def _validate_environment(
        self,
        terminal_info: Any,
        account_info: Any,
        market_data: pd.DataFrame,
        tick: Any,
    ) -> tuple[float, float]:
        if terminal_info is None:
            raise RuntimeError("MT5 terminal_info returned None")
        if account_info is None:
            raise RuntimeError("MT5 account_info returned None")
        if not bool(getattr(terminal_info, "connected", False)):
            raise RuntimeError("MT5 terminal is not connected")

        if not self.runtime_config.dry_run:
            if bool(getattr(terminal_info, "tradeapi_disabled", False)):
                raise RuntimeError("MT5 Python trading API is disabled in the terminal")
            if not bool(getattr(terminal_info, "trade_allowed", False)):
                raise RuntimeError("MT5 terminal algo trading is disabled")
            if not bool(getattr(account_info, "trade_allowed", False)):
                raise RuntimeError("MT5 account is not allowed to trade")
            if not bool(getattr(account_info, "trade_expert", False)):
                raise RuntimeError("MT5 account does not allow expert or API trading")

        last_bar_time = market_data.index[-1]
        max_bar_age = _timeframe_delta(self.runtime_config.timeframe) * 3
        now = pd.Timestamp.now(tz="UTC")
        if now - last_bar_time > max_bar_age:
            raise RuntimeError(
                f"MT5 market data is stale: last closed bar {last_bar_time.isoformat()}"
            )

        if tick is None:
            raise RuntimeError(f"MT5 symbol_info_tick returned None for {self.runtime_config.symbol}")

        bid = float(getattr(tick, "bid", 0.0) or 0.0)
        ask = float(getattr(tick, "ask", 0.0) or 0.0)
        if bid <= 0 or ask <= 0 or ask <= bid:
            raise RuntimeError(f"MT5 returned invalid bid/ask for {self.runtime_config.symbol}: bid={bid} ask={ask}")

        tick_age_seconds = _tick_age_seconds(tick, now)
        if tick_age_seconds > self.runtime_config.max_tick_age_seconds:
            raise RuntimeError(
                f"MT5 quote is stale for {self.runtime_config.symbol}: age={tick_age_seconds:.1f}s"
            )

        spread_bps = _spread_bps_from_tick(tick)
        if spread_bps > self.runtime_config.max_spread_bps:
            raise RuntimeError(
                f"MT5 spread too wide for {self.runtime_config.symbol}: spread={spread_bps:.2f}bps"
            )

        return spread_bps, tick_age_seconds

    def run_once(self) -> ExecutionReport:
        self.initialize()
        now = pd.Timestamp.now(tz="UTC")
        if self.runtime_config.sync_data_to_csv:
            market_data = self.sync_market_data()
        else:
            market_data = self.fetch_market_data()
        signal, sweep_summary = build_live_signal(market_data, self.strategy_config, self.runtime_config)
        symbol_info = self.mt5.symbol_info(self.runtime_config.symbol)
        tick = self.mt5.symbol_info_tick(self.runtime_config.symbol)
        terminal_info = self.mt5.terminal_info()
        account_info = self.mt5.account_info()
        if symbol_info is None:
            raise RuntimeError(f"MT5 symbol_info returned None for {self.runtime_config.symbol}")

        spread_bps, tick_age_seconds = self._validate_environment(
            terminal_info=terminal_info,
            account_info=account_info,
            market_data=market_data,
            tick=tick,
        )

        controller_state = self._refresh_controller_state(now)
        effective_signal, blocked_reason = apply_controller_overrides(signal, controller_state, now)
        if blocked_reason is not None:
            controller_state = ControllerState(
                cooldown_until=controller_state.cooldown_until,
                last_stop_loss_time=controller_state.last_stop_loss_time,
                last_stop_loss_ticket=controller_state.last_stop_loss_ticket,
                blocked_reason=blocked_reason,
                updated_at=now,
            )
            self._save_controller_state(controller_state)
        positions = self.get_positions()
        plan = build_execution_plan(
            signal=effective_signal,
            positions=positions,
            symbol_info=symbol_info,
            tick=tick,
            strategy_config=self.strategy_config,
            runtime_config=self.runtime_config,
            sweep_summary=sweep_summary,
        )

        margin_required = None
        margin_free_before = float(getattr(account_info, "margin_free", np.nan)) if account_info is not None else None
        margin_free_after = None
        equity = float(getattr(account_info, "equity", np.nan)) if account_info is not None else None
        max_margin_allowed_usd = _margin_cap_usd(equity, self.runtime_config.max_margin_pct)
        if plan.target_volume > 0 and plan.reference_price is not None:
            order_type = self.mt5.ORDER_TYPE_BUY if effective_signal.side == "BUY" else self.mt5.ORDER_TYPE_SELL
            margin_required = self.mt5.order_calc_margin(
                order_type,
                self.runtime_config.symbol,
                plan.target_volume,
                plan.reference_price,
            )
            if margin_required is not None and margin_free_before is not None and not np.isnan(margin_free_before):
                margin_free_after = float(margin_free_before - margin_required)
            _ensure_margin_within_cap(margin_required, max_margin_allowed_usd, self.runtime_config.max_margin_pct)

        broker_results: tuple[dict[str, Any], ...] = ()
        if not self.runtime_config.dry_run:
            if margin_required is None:
                raise RuntimeError("MT5 could not estimate margin for the target order")
            if margin_free_after is not None and margin_free_after < 0:
                raise RuntimeError(
                    f"Insufficient free margin for target order: required={margin_required}, free={margin_free_before}"
                )
            broker_results = tuple(self._execute_plan(plan, symbol_info, tick))

        report = ExecutionReport(
            signal=signal,
            effective_signal=effective_signal,
            plan=plan,
            controller_state=controller_state,
            current_positions=tuple(positions),
            dry_run=self.runtime_config.dry_run,
            spread_bps=spread_bps,
            tick_age_seconds=tick_age_seconds,
            max_margin_pct=self.runtime_config.max_margin_pct,
            max_margin_allowed_usd=max_margin_allowed_usd,
            margin_required=margin_required,
            margin_free_before=margin_free_before,
            margin_free_after=margin_free_after,
            broker_results=broker_results,
        )
        self._write_snapshot(report)
        self._append_journal(report)
        return report

    def run_forever(self) -> None:
        self.initialize()
        last_bar: pd.Timestamp | None = None
        while True:
            market_data = self.fetch_market_data()
            current_bar = market_data.index[-1]
            if last_bar is None or current_bar > last_bar:
                report = self.run_once()
                print(json.dumps(report.to_dict(), indent=2))
                last_bar = current_bar
            time.sleep(self.runtime_config.poll_seconds)

    def _execute_plan(self, plan: ExecutionPlan, symbol_info: Any, tick: Any) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for action in plan.actions:
            if action.kind == "NOOP":
                results.append({"kind": action.kind, "reason": action.reason, "status": "skipped"})
                continue

            request = self._build_order_request(action, symbol_info, tick)
            request = self._prepare_trade_request(request, action, symbol_info)
            response = self.mt5.order_send(request)
            if response is None:
                raise RuntimeError(f"MT5 order_send returned None for action {action.kind}: {self.mt5.last_error()}")

            response_dict = response._asdict() if hasattr(response, "_asdict") else {"response": str(response)}
            results.append({"kind": action.kind, "request": request, "response": response_dict})

            retcode = response_dict.get("retcode")
            if retcode not in _success_retcodes(self.mt5):
                raise RuntimeError(f"MT5 order_send failed for {action.kind}: {response_dict}")
        return results

    def _prepare_trade_request(self, request: dict[str, Any], action: ExecutionAction, symbol_info: Any) -> dict[str, Any]:
        if action.kind not in {"OPEN", "CLOSE"}:
            return request

        for filling_mode in _candidate_filling_modes(self.mt5, symbol_info):
            candidate = dict(request)
            candidate["type_filling"] = filling_mode
            check = self.mt5.order_check(candidate)
            if check is None:
                continue

            retcode = int(getattr(check, "retcode", -1) or -1)
            if retcode != 10030:
                return candidate

        return request

    def _build_order_request(self, action: ExecutionAction, symbol_info: Any, tick: Any) -> dict[str, Any]:
        if action.kind == "OPEN":
            order_type = self.mt5.ORDER_TYPE_BUY if action.side == "BUY" else self.mt5.ORDER_TYPE_SELL
            return {
                "action": self.mt5.TRADE_ACTION_DEAL,
                "symbol": self.runtime_config.symbol,
                "volume": action.volume,
                "type": order_type,
                "price": _reference_price_from_tick(tick, action.side or "BUY"),
                "sl": action.sl,
                "deviation": self.runtime_config.deviation_points,
                "magic": self.runtime_config.magic_number,
                "comment": action.comment or self.runtime_config.comment_prefix,
                "type_time": self.mt5.ORDER_TIME_GTC,
                "type_filling": _filling_mode(self.mt5, symbol_info),
            }

        if action.kind == "CLOSE":
            close_side = action.side or "SELL"
            order_type = self.mt5.ORDER_TYPE_BUY if close_side == "BUY" else self.mt5.ORDER_TYPE_SELL
            return {
                "action": self.mt5.TRADE_ACTION_DEAL,
                "symbol": self.runtime_config.symbol,
                "position": action.ticket,
                "volume": action.volume,
                "type": order_type,
                "price": _reference_price_from_tick(tick, close_side),
                "deviation": self.runtime_config.deviation_points,
                "magic": self.runtime_config.magic_number,
                "comment": action.comment or f"{self.runtime_config.comment_prefix}|close",
                "type_time": self.mt5.ORDER_TIME_GTC,
                "type_filling": _filling_mode(self.mt5, symbol_info),
            }

        if action.kind == "UPDATE_SL":
            return {
                "action": self.mt5.TRADE_ACTION_SLTP,
                "symbol": self.runtime_config.symbol,
                "position": action.ticket,
                "sl": action.sl,
                "magic": self.runtime_config.magic_number,
                "comment": action.comment or self.runtime_config.comment_prefix,
            }

        raise ValueError(f"Unsupported action kind: {action.kind}")

    def _write_snapshot(self, report: ExecutionReport) -> None:
        path = self.runtime_config.snapshot_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")

    def _append_journal(self, report: ExecutionReport) -> None:
        path = self.runtime_config.journal_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(report.to_dict(), default=str) + "\n")


def _contract_size(symbol_info: Any) -> float:
    contract_size = float(getattr(symbol_info, "trade_contract_size", 0.0) or 0.0)
    return contract_size if contract_size > 0 else 1.0


def _close_side(side: str) -> str:
    return "SELL" if side == "BUY" else "BUY"


def _drop_open_bar(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if df.empty:
        return df
    if df.index[-1] + _timeframe_delta(timeframe) > pd.Timestamp.now(tz="UTC"):
        return df.iloc[:-1]
    return df


def _filling_mode(mt5: Any, symbol_info: Any) -> int:
    candidates = _candidate_filling_modes(mt5, symbol_info)
    return candidates[0]


def _candidate_filling_modes(mt5: Any, symbol_info: Any) -> list[int]:
    preferred = [
        int(getattr(mt5, "ORDER_FILLING_IOC")),
        int(getattr(mt5, "ORDER_FILLING_FOK")),
        int(getattr(mt5, "ORDER_FILLING_RETURN")),
    ]
    raw_mode = getattr(symbol_info, "filling_mode", None)
    if raw_mode is not None:
        try:
            preferred.append(int(raw_mode))
        except (TypeError, ValueError):
            pass

    deduped: list[int] = []
    for mode in preferred:
        if mode not in deduped:
            deduped.append(mode)
    return deduped


def _latest_managed_stop_loss_deal(
    deals: list[Any] | tuple[Any, ...],
    runtime_config: MT5RuntimeConfig,
    mt5: Any,
) -> Any | None:
    latest_deal = None
    latest_time = None
    for deal in deals:
        magic = int(getattr(deal, "magic", 0) or 0)
        symbol = str(getattr(deal, "symbol", ""))
        reason = int(getattr(deal, "reason", -1) or -1)
        entry = int(getattr(deal, "entry", -1) or -1)
        if magic != runtime_config.magic_number or symbol != runtime_config.symbol:
            continue
        if reason != int(getattr(mt5, "DEAL_REASON_SL")):
            continue
        if entry not in {
            int(getattr(mt5, "DEAL_ENTRY_OUT")),
            int(getattr(mt5, "DEAL_ENTRY_INOUT")),
        }:
            continue

        deal_time = _deal_timestamp(deal)
        if latest_time is None or deal_time > latest_time:
            latest_time = deal_time
            latest_deal = deal
    return latest_deal


def _deal_timestamp(deal: Any) -> pd.Timestamp:
    time_msc = getattr(deal, "time_msc", None)
    if time_msc:
        return pd.to_datetime(int(time_msc), unit="ms", utc=True)
    return pd.to_datetime(int(getattr(deal, "time")), unit="s", utc=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _ensure_margin_within_cap(
    margin_required: float | None,
    max_margin_allowed_usd: float | None,
    max_margin_pct: float | None,
) -> None:
    if margin_required is None or max_margin_allowed_usd is None:
        return
    if margin_required > max_margin_allowed_usd:
        raise RuntimeError(
            "Estimated margin exceeds configured cap: "
            f"required={margin_required:.2f}, max_margin_pct={max_margin_pct:.4f}, "
            f"max_margin_allowed_usd={max_margin_allowed_usd:.2f}"
        )


def _parse_timestamp(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _margin_cap_usd(equity: float | None, max_margin_pct: float | None) -> float | None:
    if equity is None or max_margin_pct is None or np.isnan(equity):
        return None
    return float(equity * max_margin_pct)


def _mt5_timeframe(mt5: Any, timeframe: str) -> int:
    name = f"TIMEFRAME_{timeframe.upper()}"
    if not hasattr(mt5, name):
        raise ValueError(f"Unsupported MT5 timeframe: {timeframe}")
    return int(getattr(mt5, name))


def _price_digits(symbol_info: Any) -> int:
    return int(getattr(symbol_info, "digits", 2) or 2)


def _reference_price_from_tick(tick: Any, side: str) -> float:
    return float(getattr(tick, "ask")) if side == "BUY" else float(getattr(tick, "bid"))


def _spread_bps_from_tick(tick: Any) -> float:
    bid = float(getattr(tick, "bid"))
    ask = float(getattr(tick, "ask"))
    mid = (bid + ask) / 2.0
    return (ask - bid) / max(mid, 1e-12) * 10_000.0


def _tick_age_seconds(tick: Any, now: pd.Timestamp) -> float:
    time_msc = getattr(tick, "time_msc", None)
    if time_msc:
        tick_time = pd.to_datetime(int(time_msc), unit="ms", utc=True)
    else:
        tick_time = pd.to_datetime(int(getattr(tick, "time")), unit="s", utc=True)
    return float(max((now - tick_time).total_seconds(), 0.0))


def _round_price(price: float, symbol_info: Any) -> float:
    return round(price, _price_digits(symbol_info))


def _round_volume(volume: float, symbol_info: Any) -> float:
    step = _volume_step(symbol_info)
    maximum = float(getattr(symbol_info, "volume_max", volume) or volume)
    floored = math.floor((volume + 1e-12) / step) * step
    decimals = max(0, _step_decimals(step))
    return round(min(max(floored, 0.0), maximum), decimals)


def _same_stop(current_sl: float, target_sl: float | None, symbol_info: Any) -> bool:
    if target_sl is None:
        return current_sl in (0.0, None)
    tolerance = max(float(getattr(symbol_info, "point", 0.01) or 0.01), 1e-9) * 2
    return math.isclose(float(current_sl or 0.0), float(target_sl), abs_tol=tolerance)


def _step_decimals(step: float) -> int:
    text = f"{step:.10f}".rstrip("0")
    return len(text.split(".", 1)[1]) if "." in text else 0


def _success_retcodes(mt5: Any) -> set[int]:
    return {int(getattr(mt5, name)) for name in SUCCESS_RETCODE_NAMES if hasattr(mt5, name)}


def _timeframe_delta(timeframe: str) -> pd.Timedelta:
    mapping = {
        "M1": pd.Timedelta(minutes=1),
        "M5": pd.Timedelta(minutes=5),
        "M15": pd.Timedelta(minutes=15),
        "M30": pd.Timedelta(minutes=30),
        "H1": pd.Timedelta(hours=1),
        "H4": pd.Timedelta(hours=4),
        "D1": pd.Timedelta(days=1),
    }
    key = timeframe.upper()
    if key not in mapping:
        raise ValueError(f"Unsupported timeframe delta for {timeframe}")
    return mapping[key]


def _volume_step(symbol_info: Any) -> float:
    step = float(getattr(symbol_info, "volume_step", 0.0) or 0.0)
    return step if step > 0 else 0.01
