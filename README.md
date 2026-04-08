# BTC Regime Trading App

This repo can run from a fresh clone after you install the dependencies and configure `.env`.

## Setup

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Update `.env` with your MT5 terminal path and risk limits.

## What The MT5 Runner Does

`scripts/run_mt5_executor.py` now treats MT5 as the source of truth for the raw H1 data:

- On startup it pulls the maximum BTC H1 history available from the configured MT5 account.
- It writes that history into the shared CSV defined by `MT5_DATA_PATH`.
- In loop mode it keeps polling and refreshes the CSV with new closed hourly bars before each execution cycle.
- The dashboard, pipeline, and tests continue using the same shared CSV path.

Recommended `MT5_DATA_PATH`: `data/raw/BTCUSD_1h_mt5.csv`

## Main Commands

Dry-run with CSV synchronization:

```powershell
python scripts\run_mt5_executor.py
```

Live supervised trading:

```powershell
python scripts\run_mt5_executor.py --live --loop
```

Launch the dashboard:

```powershell
python scripts\run_dashboard.py
```

Run the test suite:

```powershell
python -m pytest tests -q
```

## Required `.env` Keys

- `MT5_TERMINAL_PATH`
- `MT5_RISK_PER_TRADE_PCT`
- `MT5_MAX_MARGIN_PCT`

## Important Optional `.env` Keys

- `MT5_DATA_PATH`
- `MT5_SYNC_DATA_TO_CSV`
- `MT5_FULL_SYNC_ON_START`
- `MT5_HISTORY_START`
- `MT5_SYNC_CHUNK_DAYS`
- `MT5_SYNC_OVERLAP_BARS`
- `MT5_SYMBOL`
- `MT5_TIMEFRAME`
- `MT5_COOLDOWN_HOURS`
- `MT5_MAX_TICK_AGE_SECONDS`
- `MT5_MAX_SPREAD_BPS`
