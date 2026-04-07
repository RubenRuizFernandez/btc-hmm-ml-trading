"""Load and clean the raw BTC/USD CSV."""
import pandas as pd
from pathlib import Path
from src.config import DATA_RAW


def load_raw(path: Path = DATA_RAW) -> pd.DataFrame:
    """
    Return a cleaned OHLCV DataFrame indexed by UTC datetime.

    Guarantees:
    - Monotonically increasing index (UTC, timezone-aware)
    - No duplicate timestamps
    - Forward-fill of sparse zero-volume early bars (volume=0 → use previous close)
    - All OHLCV columns are float64
    """
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
