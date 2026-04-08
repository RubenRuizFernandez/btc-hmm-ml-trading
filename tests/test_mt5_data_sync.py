"""Tests for MT5 data synchronization into the shared CSV."""
import pandas as pd

from src.mt5.data_sync import MT5DataSyncConfig, merge_ohlcv, read_ohlcv_csv, sync_csv_from_mt5


def _rates_frame(index, close_offset=0.0):
    rows = []
    for ts in index:
        rows.append(
            {
                "time": int(ts.timestamp()),
                "open": 100.0 + close_offset,
                "high": 101.0 + close_offset,
                "low": 99.0 + close_offset,
                "close": 100.5 + close_offset,
                "tick_volume": 10,
                "spread": 0,
                "real_volume": 0,
            }
        )
    return rows


class FakeMT5:
    TIMEFRAME_H1 = 1

    def __init__(self, responses):
        self._responses = list(responses)

    def copy_rates_range(self, symbol, timeframe, start, end):
        if self._responses:
            return self._responses.pop(0)
        return []


def test_merge_ohlcv_prefers_fresh_overlapping_bars():
    idx = pd.date_range("2026-04-01 00:00:00", periods=3, freq="h", tz="UTC")
    existing = pd.DataFrame(
        {"open": [1.0, 2.0, 3.0], "high": [1.0, 2.0, 3.0], "low": [1.0, 2.0, 3.0], "close": [1.0, 2.0, 3.0], "volume": [1.0, 1.0, 1.0]},
        index=idx,
    )
    incoming = pd.DataFrame(
        {"open": [20.0, 30.0], "high": [20.0, 30.0], "low": [20.0, 30.0], "close": [20.0, 30.0], "volume": [2.0, 2.0]},
        index=idx[1:],
    )

    merged = merge_ohlcv(existing, incoming)

    assert len(merged) == 3
    assert merged.loc[idx[1], "close"] == 20.0
    assert merged.loc[idx[2], "close"] == 30.0


def test_sync_csv_from_mt5_bootstraps_full_history(tmp_path):
    bootstrap = _rates_frame(pd.date_range("2026-04-01 00:00:00", periods=6, freq="h", tz="UTC"))
    mt5 = FakeMT5([bootstrap])
    csv_path = tmp_path / "BTCUSD_1h.csv"
    config = MT5DataSyncConfig(
        symbol="BTCUSD",
        timeframe="H1",
        csv_path=csv_path,
        history_start="2026-04-01T00:00:00Z",
        chunk_days=1,
        overlap_bars=2,
    )

    synced = sync_csv_from_mt5(
        mt5=mt5,
        config=config,
        force_full=True,
        now=pd.Timestamp("2026-04-01 05:30:00", tz="UTC"),
    )

    reloaded = read_ohlcv_csv(csv_path)

    assert csv_path.exists()
    assert len(synced) == 5
    assert synced.index[-1] == pd.Timestamp("2026-04-01 04:00:00", tz="UTC")
    assert synced.loc[pd.Timestamp("2026-04-01 02:00:00", tz="UTC"), "close"] == 100.5
    assert reloaded.equals(synced)


def test_sync_csv_from_mt5_incrementally_extends_existing_csv(tmp_path):
    csv_path = tmp_path / "BTCUSD_1h.csv"
    seed_index = pd.date_range("2026-04-01 00:00:00", periods=3, freq="h", tz="UTC")
    seed = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [10.0, 10.0, 10.0],
        },
        index=seed_index,
    )
    seed.index.name = "datetime"
    seed.to_csv(csv_path)

    overlap = _rates_frame(pd.date_range("2026-04-01 01:00:00", periods=4, freq="h", tz="UTC"), close_offset=2.0)
    mt5 = FakeMT5([overlap])
    config = MT5DataSyncConfig(
        symbol="BTCUSD",
        timeframe="H1",
        csv_path=csv_path,
        history_start="2026-04-01T00:00:00Z",
        chunk_days=1,
        overlap_bars=2,
    )

    synced = sync_csv_from_mt5(
        mt5=mt5,
        config=config,
        force_full=False,
        now=pd.Timestamp("2026-04-01 05:05:00", tz="UTC"),
    )

    assert len(synced) == 5
    assert synced.index[0] == pd.Timestamp("2026-04-01 00:00:00", tz="UTC")
    assert synced.index[-1] == pd.Timestamp("2026-04-01 04:00:00", tz="UTC")
    assert synced.loc[pd.Timestamp("2026-04-01 01:00:00", tz="UTC"), "close"] == 102.5
