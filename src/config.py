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
    "log_return_168h",
    "realized_vol_24h",
    "realized_vol_168h",
    "vol_ratio",
    "trend_sma_720",
]

# Semantic labels sorted Super Bull → Super Bear
REGIME_LABELS = [
    "Super Bull",
    "Strong Bull",
    "Bull",
    "Sideways",
    "Bear",
    "Strong Bear",
    "Super Bear",
]

# Directional expectation per regime: +1 = bullish, -1 = bearish, 0 = neutral
REGIME_DIRECTION = {
    0: 1,   # Super Bull
    1: 1,   # Strong Bull
    2: 1,   # Bull
    3: 0,   # Sideways
    4: -1,  # Bear
    5: -1,  # Strong Bear
    6: -1,  # Super Bear
}

# ─── Regime-based position sizing ─────────────────────────────────────────────
# Multiplier per regime: buy x1/x2/x3 for bull, sell x1/x2/x3 for bear
REGIME_MULTIPLIER = {
    0: 3,    # Super Bull  → buy x3
    1: 2,    # Strong Bull → buy x2
    2: 1,    # Bull        → buy x1
    3: 0,    # Sideways    → flat
    4: -1,   # Bear        → sell x1
    5: -2,   # Strong Bear → sell x2
    6: -3,   # Super Bear  → sell x3
}

# Base position unit (multiplier * BASE_UNIT = actual position fraction)
# 3 * 0.50 = 1.50 (150% long), 2 * 0.50 = 1.00 (100%), 1 * 0.50 = 0.50 (50%)
BASE_POSITION_UNIT = 0.50

# Minimum bars a regime must persist before switching (smoothing)
MIN_REGIME_BARS = 12

# ─── Trend regime detector (daily EMA-based) ─────────────────────────────────
# Thresholds for mapping continuous trend score [-7, +7] → 7 regimes.
# Tuned so only strongly bearish conditions produce bear regimes.
TREND_SCORE_THRESHOLDS = [3.5, 1.5, -0.5, -2.5, -4.5, -5.5]
#   score > 3.5  → 0 (Super Bull)
#   score > 1.5  → 1 (Strong Bull)
#   score > -0.5 → 2 (Bull)
#   score > -2.5 → 3 (Sideways)
#   score > -4.5 → 4 (Bear)
#   score > -5.5 → 5 (Strong Bear)
#   score ≤ -5.5 → 6 (Super Bear)

# ─── Regime gating (legacy, used by ML ensemble) ─────────────────────────────
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
MAX_POSITION = 1.5                  # max fraction of capital per trade (allows modest leverage)

# ─── Risk-free rate (annualized) ──────────────────────────────────────────────
RISK_FREE_RATE = 0.04
HOURS_PER_YEAR = 8_760
