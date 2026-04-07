"""Central configuration — all hyperparameters and paths."""
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw" / "BTCUSD_1h_bitstamp.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
WF_RESULTS = DATA_PROCESSED / "wf_results"
FEATURES_PATH = DATA_PROCESSED / "features.parquet"
REGIMES_PATH = DATA_PROCESSED / "regimes.parquet"

# ─── HMM ──────────────────────────────────────────────────────────────────────
HMM_N_STATES = 7
HMM_COVARIANCE_TYPE = "full"
HMM_N_ITER = 200
HMM_TOL = 1e-4
HMM_N_RESTARTS = 10
HMM_RANDOM_SEED = 42

HMM_FEATURES = [
    "log_return_1h",
    "log_return_4h",
    "log_return_24h",
    "realized_vol_24h",
    "realized_vol_168h",
    "vol_ratio",
    "price_range_norm",
    "volume_zscore",
]

# Semantic labels sorted Strong Bull → Strong Bear
REGIME_LABELS = [
    "Strong Bull",
    "Bull",
    "Weak Bull",
    "Sideways",
    "Weak Bear",
    "Bear",
    "Strong Bear",
]

# Directional expectation per regime: +1 = bullish, -1 = bearish, 0 = neutral
REGIME_DIRECTION = {
    0: 1,   # Strong Bull
    1: 1,   # Bull
    2: 1,   # Weak Bull
    3: 0,   # Sideways
    4: -1,  # Weak Bear
    5: -1,  # Bear
    6: -1,  # Strong Bear
}

# ─── Regime gating ────────────────────────────────────────────────────────────
REGIME_CONFIDENCE_THRESHOLD = 0.75
TRADEABLE_REGIMES = [0, 1, 2, 4, 5, 6]   # excludes Sideways (3)

# ─── ML models ────────────────────────────────────────────────────────────────
FORWARD_RETURN_HORIZON = 8          # bars (hours) ahead
FEE_THRESHOLD_BPS = 20              # min net return to be a positive label (bps)

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 500,
    "learning_rate": 0.02,
    "num_leaves": 63,
    "max_depth": 6,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "class_weight": "balanced",
    "verbosity": -1,
    "n_jobs": -1,
}
LGBM_EARLY_STOPPING_ROUNDS = 50
LGBM_VAL_FRACTION = 0.15            # fraction of IS used for early stopping

LSTM_SEQ_LEN = 48                   # bars (2 days)
LSTM_HIDDEN = [64, 32]
LSTM_DROPOUT = 0.3
LSTM_EPOCHS = 10                    # reduced from 30 — further reduce with --fast
LSTM_BATCH_SIZE = 512               # larger batch → faster iteration
LSTM_LR = 1e-3
LSTM_RANDOM_SEED = 42

ENSEMBLE_C = 0.1                    # LogisticRegression regularisation

SIGNAL_LONG_THRESHOLD = 0.60
SIGNAL_SHORT_THRESHOLD = 0.40

# ─── Walk-forward ─────────────────────────────────────────────────────────────
WF_IS_MIN_YEARS = 5                 # minimum IS years before first fold
WF_OOS_MONTHS = 12                  # OOS window per fold (months)
WF_STEP_MONTHS = 3                  # fold step (months)

# ─── Backtest ─────────────────────────────────────────────────────────────────
FEE_RATE = 0.001                    # 0.1% per side
SLIPPAGE_BPS = 5                    # 5 bps adverse move on fill
KELLY_FRACTION = 0.25               # quarter-Kelly
MAX_POSITION = 1.0                  # max fraction of capital per trade

# ─── Risk-free rate (annualized) ──────────────────────────────────────────────
RISK_FREE_RATE = 0.04
HOURS_PER_YEAR = 8_760
