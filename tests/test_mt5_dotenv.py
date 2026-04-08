"""Tests for .env loading in the MT5 runner."""
import os
from pathlib import Path

from src.mt5.dotenv import load_dotenv_file


def test_load_dotenv_file_sets_values(tmp_path, monkeypatch):
    monkeypatch.delenv("MT5_TERMINAL_PATH", raising=False)
    monkeypatch.delenv("MT5_RISK_PER_TRADE_PCT", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text(
        'MT5_TERMINAL_PATH="C:\\Program Files\\TTP Trading MT5 Terminal\\terminal64.exe"\n'
        "MT5_RISK_PER_TRADE_PCT=0.014\n",
        encoding="utf-8",
    )

    loaded = load_dotenv_file(env_path)

    assert loaded["MT5_TERMINAL_PATH"] == r"C:\Program Files\TTP Trading MT5 Terminal\terminal64.exe"
    assert loaded["MT5_RISK_PER_TRADE_PCT"] == "0.014"
    assert os.environ["MT5_TERMINAL_PATH"] == loaded["MT5_TERMINAL_PATH"]


def test_load_dotenv_file_does_not_override_existing_values(tmp_path, monkeypatch):
    monkeypatch.setenv("MT5_MAX_MARGIN_PCT", "0.10")
    env_path = tmp_path / ".env"
    env_path.write_text("MT5_MAX_MARGIN_PCT=0.175\n", encoding="utf-8")

    loaded = load_dotenv_file(env_path)

    assert loaded["MT5_MAX_MARGIN_PCT"] == "0.10"
    assert os.environ["MT5_MAX_MARGIN_PCT"] == "0.10"
