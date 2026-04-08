"""Tests for raw data loading helpers."""
from src.config import DATA_RAW, ROOT
from src.data.loader import _configured_raw_path


def test_configured_raw_path_uses_env_override(monkeypatch):
    monkeypatch.setenv("MT5_DATA_PATH", "data/raw/custom.csv")

    resolved = _configured_raw_path(DATA_RAW)

    assert resolved == ROOT / "data" / "raw" / "custom.csv"
