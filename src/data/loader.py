"""Load and clean the raw BTC/USD CSV."""
import os
import pandas as pd
from pathlib import Path
from src.config import DATA_RAW, ROOT


def ensure_raw_data(path: Path = DATA_RAW, force_sync: bool = False) -> Path:
    """Bootstrap the raw CSV from MT5 when it is missing or explicitly forced."""
    csv_path = _configured_raw_path(path)
    if csv_path.exists() and not force_sync:
        return csv_path

    from src.mt5.data_sync import sync_csv_from_env

    try:
        sync_csv_from_env(csv_path=csv_path, force_full=True)
    except Exception as exc:  # pragma: no cover - exercised via integration path
        if csv_path.exists() and not force_sync:
            return csv_path
        raise FileNotFoundError(
            f"Raw data file is missing and MT5 bootstrap failed for {csv_path}: {exc}"
        ) from exc

    return csv_path


def _configured_raw_path(path: Path) -> Path:
    requested = Path(path)
    if requested != DATA_RAW:
        return requested

    from src.mt5.dotenv import load_dotenv_file

    load_dotenv_file(ROOT / ".env")
    override = os.getenv("MT5_DATA_PATH")
    if not override:
        return requested

    configured = Path(override)
    if not configured.is_absolute():
        configured = ROOT / configured
    return configured


def load_raw(path: Path = DATA_RAW) -> pd.DataFrame:
    """
    Return a cleaned OHLCV DataFrame indexed by UTC datetime.

    Guarantees:
    - Monotonically increasing index (UTC, timezone-aware)
    - No duplicate timestamps
    - Forward-fill of sparse zero-volume early bars (volume=0 → use previous close)
    - All OHLCV columns are float64
    """
    path = ensure_raw_data(path)
    df = pd.read_csv(
        path,
        parse_dates=["datetime"],
        index_col="datetime",
    )
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "datetime"

    # Drop the raw unix timestamp column if present
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])

    df = df[["open", "high", "low", "close", "volume"]].astype(float)

    # Remove exact duplicate timestamps (keep first)
    df = df[~df.index.duplicated(keep="first")]

    # Ensure monotonically increasing
    df = df.sort_index()

    # Early data quality: rows where volume == 0 likely have stale OHLC;
    # forward-fill close from previous bar so log-returns are 0 (not NaN).
    zero_vol = df["volume"] == 0
    df.loc[zero_vol, ["open", "high", "low", "close"]] = None
    df[["open", "high", "low", "close"]] = (
        df[["open", "high", "low", "close"]].ffill()
    )

    # Drop any remaining NaNs at the very start
    df = df.dropna()

    return df
