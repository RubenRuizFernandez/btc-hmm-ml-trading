"""Synchronize MT5 OHLCV history into the project's raw CSV."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DATA_RAW, ROOT


DEFAULT_HISTORY_START = "2010-01-01T00:00:00Z"


@dataclass(frozen=True)
class MT5DataSyncConfig:
    """Settings for persisting MT5 OHLCV data to CSV."""

    symbol: str = "BTCUSD"
    timeframe: str = "H1"
    csv_path: Path = DATA_RAW
    history_start: str = DEFAULT_HISTORY_START
    chunk_days: int = 365
    overlap_bars: int = 24 * 14


def sync_csv_from_mt5(
    mt5: Any,
    config: MT5DataSyncConfig,
    force_full: bool = False,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Bootstrap or incrementally refresh the local raw CSV from MT5."""
    csv_path = Path(config.csv_path)
    existing = read_ohlcv_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    sync_now = _sync_now(now)
    full_sync = force_full or existing.empty

    if full_sync:
        start = _parse_utc_timestamp(config.history_start)
    else:
        timeframe_delta = _timeframe_delta(config.timeframe)
        overlap = timeframe_delta * max(int(config.overlap_bars), 0)
        start = max(existing.index[-1] - overlap, _parse_utc_timestamp(config.history_start))

    incoming = fetch_historical_rates(
        mt5=mt5,
        symbol=config.symbol,
        timeframe=config.timeframe,
        start=start,
        end=sync_now,
        chunk_days=config.chunk_days,
    )

    if incoming.empty and existing.empty:
        raise RuntimeError(
            f"MT5 returned no history for {config.symbol} {config.timeframe} "
            f"between {start.isoformat()} and {sync_now.isoformat()}"
        )

    merged = merge_ohlcv(existing, incoming)
    merged = _drop_open_bar(merged, config.timeframe, now=sync_now)
    if merged.empty:
        raise RuntimeError("No closed MT5 bars are available after synchronization")

    write_ohlcv_csv(merged, csv_path)
    return merged


def sync_csv_from_env(
    csv_path: Path = DATA_RAW,
    force_full: bool = False,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Initialize MT5 using `.env` and refresh the local raw CSV."""
    from src.mt5.dotenv import load_dotenv_file

    env_path = ROOT / ".env"
    load_dotenv_file(env_path)
    resolved_csv_path = _configured_csv_path(csv_path)

    try:
        import MetaTrader5 as mt5  # type: ignore
    except ImportError as exc:
        raise RuntimeError("MetaTrader5 package is not installed") from exc

    kwargs: dict[str, Any] = {"timeout": int(os.getenv("MT5_TIMEOUT_MS", "30000"))}
    terminal_path = os.getenv("MT5_TERMINAL_PATH")
    if terminal_path:
        kwargs["path"] = terminal_path

    login = _env_int("MT5_LOGIN")
    if login is not None:
        kwargs["login"] = login

    password = os.getenv("MT5_PASSWORD")
    if password:
        kwargs["password"] = password

    server = os.getenv("MT5_SERVER")
    if server:
        kwargs["server"] = server

    if not mt5.initialize(**kwargs):
        raise RuntimeError(
            "MT5 initialize failed while bootstrapping raw data. "
            "Open the terminal configured in `.env` and log into the target account. "
            f"last_error={mt5.last_error()}"
        )

    symbol = os.getenv("MT5_SYMBOL", "BTCUSD")
    try:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"MT5 symbol_select failed for {symbol}: {mt5.last_error()}")

        config = MT5DataSyncConfig(
            symbol=symbol,
            timeframe=os.getenv("MT5_TIMEFRAME", "H1"),
            csv_path=resolved_csv_path,
            history_start=os.getenv("MT5_HISTORY_START", DEFAULT_HISTORY_START),
            chunk_days=int(os.getenv("MT5_SYNC_CHUNK_DAYS", "365")),
            overlap_bars=int(os.getenv("MT5_SYNC_OVERLAP_BARS", str(24 * 14))),
        )
        return sync_csv_from_mt5(mt5, config=config, force_full=force_full, now=now)
    finally:
        mt5.shutdown()


def fetch_historical_rates(
    mt5: Any,
    symbol: str,
    timeframe: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunk_days: int,
) -> pd.DataFrame:
    """Fetch the maximum historical range available from MT5 in chunks."""
    if start >= end:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    timeframe_value = _mt5_timeframe(mt5, timeframe)
    chunk_days = max(int(chunk_days), 1)
    timeframe_delta = _timeframe_delta(timeframe)
    frames: list[pd.DataFrame] = []

    cursor = start
    while cursor < end:
        chunk_end = min(cursor + pd.Timedelta(days=chunk_days), end)
        rates = mt5.copy_rates_range(symbol, timeframe_value, cursor.to_pydatetime(), chunk_end.to_pydatetime())
        if rates is not None and len(rates) > 0:
            frames.append(_rates_to_frame(rates))
        if chunk_end >= end:
            break
        cursor = chunk_end - timeframe_delta

    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    combined = pd.concat(frames).sort_index()
    return combined[~combined.index.duplicated(keep="last")]


def read_ohlcv_csv(path: Path) -> pd.DataFrame:
    """Read a local OHLCV CSV into the canonical DataFrame shape."""
    df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "datetime"
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def write_ohlcv_csv(df: pd.DataFrame, path: Path) -> None:
    """Persist canonical OHLCV data to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    to_save = df.copy()
    to_save.index = pd.to_datetime(to_save.index, utc=True)
    to_save.index.name = "datetime"
    to_save.insert(0, "timestamp", (to_save.index.view("int64") // 10**9).astype("int64"))
    to_save.to_csv(path)


def merge_ohlcv(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Merge OHLCV frames, preferring fresh MT5 bars for overlapping timestamps."""
    if existing.empty:
        return incoming.copy()
    if incoming.empty:
        return existing.copy()

    merged = pd.concat([existing, incoming]).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged


def _rates_to_frame(rates: Any) -> pd.DataFrame:
    df = pd.DataFrame(rates)
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"tick_volume": "volume"}).set_index("datetime")
    for column in ("real_volume", "spread", "time"):
        if column in df.columns:
            df = df.drop(columns=[column])
    df = df[["open", "high", "low", "close", "volume"]].astype(float).sort_index()
    return df


def _drop_open_bar(df: pd.DataFrame, timeframe: str, now: pd.Timestamp | None = None) -> pd.DataFrame:
    if df.empty:
        return df
    reference_now = _sync_now(now)
    if df.index[-1] + _timeframe_delta(timeframe) > reference_now:
        return df.iloc[:-1]
    return df


def _env_int(name: str) -> int | None:
    value = os.getenv(name)
    return int(value) if value else None


def _configured_csv_path(csv_path: Path) -> Path:
    requested = Path(csv_path)
    if requested != DATA_RAW:
        return requested

    override = os.getenv("MT5_DATA_PATH")
    if not override:
        return requested

    configured = Path(override)
    if not configured.is_absolute():
        configured = ROOT / configured
    return configured


def _mt5_timeframe(mt5: Any, timeframe: str) -> int:
    name = f"TIMEFRAME_{timeframe.upper()}"
    if not hasattr(mt5, name):
        raise ValueError(f"Unsupported MT5 timeframe: {timeframe}")
    return int(getattr(mt5, name))


def _parse_utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _sync_now(now: pd.Timestamp | None) -> pd.Timestamp:
    reference = now or pd.Timestamp.now(tz="UTC")
    if reference.tzinfo is None:
        return reference.tz_localize("UTC")
    return reference.tz_convert("UTC")


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
