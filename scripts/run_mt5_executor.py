"""Run the BTC regime strategy on MT5 in dry-run or live mode."""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.functional import FunctionalTradeConfig
from src.config import DATA_RAW, LIVE_RISK_PER_TRADE_PCT
from src.mt5 import MT5ConnectionConfig, MT5Executor, MT5RuntimeConfig, load_dotenv_file


ROOT = Path(__file__).parent.parent
ENV_PATH = ROOT / ".env"


def parse_args() -> argparse.Namespace:
    load_dotenv_file(ENV_PATH)
    parser = argparse.ArgumentParser(description="MT5 live executor for the BTC regime strategy")
    parser.add_argument("--symbol", default=os.getenv("MT5_SYMBOL", "BTCUSD"), help="MT5 symbol to trade")
    parser.add_argument("--timeframe", default=os.getenv("MT5_TIMEFRAME", "H1"), help="MT5 timeframe, default H1")
    parser.add_argument("--history-bars", type=int, default=int(os.getenv("MT5_HISTORY_BARS", 24 * 450)))
    parser.add_argument(
        "--data-path",
        default=os.getenv("MT5_DATA_PATH", str(DATA_RAW)),
        help="Local OHLCV CSV updated from MT5 before each execution cycle",
    )
    parser.add_argument("--history-start", default=os.getenv("MT5_HISTORY_START", "2010-01-01T00:00:00Z"))
    parser.add_argument("--sync-chunk-days", type=int, default=int(os.getenv("MT5_SYNC_CHUNK_DAYS", 365)))
    parser.add_argument("--sync-overlap-bars", type=int, default=int(os.getenv("MT5_SYNC_OVERLAP_BARS", 24 * 14)))
    parser.add_argument("--min-regime-bars", type=int, default=int(os.getenv("MT5_MIN_REGIME_BARS", 1)))
    parser.add_argument("--auto-duration", action="store_true", help="Optimize regime duration before trading")
    parser.add_argument("--optimize-min-bars", type=int, default=1)
    parser.add_argument("--optimize-max-bars", type=int, default=48)
    parser.add_argument("--optimize-lookback-bars", type=int, default=24 * 365)
    parser.add_argument("--magic-number", type=int, default=int(os.getenv("MT5_MAGIC_NUMBER", 26040801)))
    parser.add_argument("--deviation-points", type=int, default=int(os.getenv("MT5_DEVIATION_POINTS", 200)))
    parser.add_argument("--poll-seconds", type=int, default=int(os.getenv("MT5_POLL_SECONDS", 30)))
    parser.add_argument("--cooldown-hours", type=int, default=int(os.getenv("MT5_COOLDOWN_HOURS", 24)))
    parser.add_argument("--max-tick-age-seconds", type=int, default=int(os.getenv("MT5_MAX_TICK_AGE_SECONDS", 180)))
    parser.add_argument("--max-spread-bps", type=float, default=float(os.getenv("MT5_MAX_SPREAD_BPS", 15.0)))
    parser.add_argument(
        "--max-margin-pct",
        type=float,
        default=_env_float("MT5_MAX_MARGIN_PCT"),
        help="Maximum broker margin as a fraction of current MT5 equity",
    )
    parser.add_argument("--path", default=os.getenv("MT5_TERMINAL_PATH"), help="Optional MT5 terminal path")
    parser.add_argument("--login", type=int, default=_env_int("MT5_LOGIN"), help="Optional MT5 login")
    parser.add_argument("--password", default=os.getenv("MT5_PASSWORD"), help="Optional MT5 password")
    parser.add_argument("--server", default=os.getenv("MT5_SERVER"), help="Optional MT5 server")
    parser.add_argument("--snapshot-path", default=os.getenv("MT5_SNAPSHOT_PATH"), help="Optional JSON snapshot path")
    parser.add_argument("--state-path", default=os.getenv("MT5_STATE_PATH"), help="Optional controller state JSON path")
    parser.add_argument("--journal-path", default=os.getenv("MT5_JOURNAL_PATH"), help="Optional JSONL execution journal path")
    parser.add_argument(
        "--risk-per-trade-pct",
        type=float,
        default=float(os.getenv("MT5_RISK_PER_TRADE_PCT", LIVE_RISK_PER_TRADE_PCT)),
        help="Live risk budget as a fraction of the static 200k account",
    )
    parser.add_argument("--sync-data", dest="sync_data", action="store_true", help="Refresh the shared CSV from MT5")
    parser.add_argument("--no-sync-data", dest="sync_data", action="store_false", help="Skip CSV synchronization")
    parser.set_defaults(sync_data=_env_bool("MT5_SYNC_DATA_TO_CSV", True))
    parser.add_argument(
        "--full-sync-on-start",
        dest="full_sync_on_start",
        action="store_true",
        help="Rebuild the local CSV from the full MT5 history at startup",
    )
    parser.add_argument(
        "--no-full-sync-on-start",
        dest="full_sync_on_start",
        action="store_false",
        help="Only do incremental CSV refreshes",
    )
    parser.set_defaults(full_sync_on_start=_env_bool("MT5_FULL_SYNC_ON_START", True))
    parser.add_argument("--live", action="store_true", help="Send orders to MT5. Default is dry-run.")
    parser.add_argument("--loop", action="store_true", help="Keep polling and act only on new closed bars")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    connection = MT5ConnectionConfig(
        path=args.path,
        login=args.login,
        password=args.password,
        server=args.server,
    )
    runtime = MT5RuntimeConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        history_bars=args.history_bars,
        data_path=_resolve_path(args.data_path),
        sync_data_to_csv=args.sync_data,
        sync_full_history_on_start=args.full_sync_on_start,
        history_start=args.history_start,
        sync_chunk_days=args.sync_chunk_days,
        sync_overlap_bars=args.sync_overlap_bars,
        signal_min_bars=args.min_regime_bars,
        auto_optimize_duration=args.auto_duration,
        optimize_min_bars=args.optimize_min_bars,
        optimize_max_bars=args.optimize_max_bars,
        optimize_lookback_bars=args.optimize_lookback_bars,
        magic_number=args.magic_number,
        deviation_points=args.deviation_points,
        poll_seconds=args.poll_seconds,
        dry_run=not args.live,
        cooldown_after_sl_hours=args.cooldown_hours,
        max_tick_age_seconds=args.max_tick_age_seconds,
        max_spread_bps=args.max_spread_bps,
        max_margin_pct=args.max_margin_pct,
        snapshot_path=_resolve_path(args.snapshot_path) if args.snapshot_path else MT5RuntimeConfig().snapshot_path,
        state_path=_resolve_path(args.state_path) if args.state_path else MT5RuntimeConfig().state_path,
        journal_path=_resolve_path(args.journal_path) if args.journal_path else MT5RuntimeConfig().journal_path,
    )
    executor = MT5Executor(
        connection_config=connection,
        runtime_config=runtime,
        strategy_config=FunctionalTradeConfig(risk_per_trade_pct=args.risk_per_trade_pct),
    )

    try:
        if args.loop:
            executor.run_forever()
        else:
            report = executor.run_once()
            print(json.dumps(report.to_dict(), indent=2))
    finally:
        executor.shutdown()


def _env_int(name: str) -> int | None:
    value = os.getenv(name)
    return int(value) if value else None


def _env_float(name: str) -> float | None:
    value = os.getenv(name)
    return float(value) if value else None


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


if __name__ == "__main__":
    main()
