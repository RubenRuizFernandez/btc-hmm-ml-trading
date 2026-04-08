"""Trend-following regime detector using daily EMA crossovers.

Computes a composite trend score from 7 EMA/momentum indicators and maps it
to 7 regimes (Super Bull → Super Bear).  Much more reliable for directional
classification than the HMM, which tends to cluster by volatility structure
rather than by trend direction.

Usage:
    regime_state, score = compute_trend_regime(df)
"""
import numpy as np
import pandas as pd

from src.config import (
    REGIME_LABELS,
    TREND_SCORE_THRESHOLDS,
    MIN_REGIME_BARS,
)


def compute_trend_score(df: pd.DataFrame) -> pd.Series:
    """
    Compute a continuous trend score in [-7, +7] from daily EMAs and momentum.

    Indicators (each contributes ~[-1, +1]):
      1. close vs EMA-20d
      2. close vs EMA-50d
      3. close vs EMA-200d
      4. EMA-20d vs EMA-50d   (golden/death cross short-term)
      5. EMA-50d vs EMA-200d  (golden/death cross long-term)
      6. 30-day log momentum
      7.  7-day log momentum

    Parameters
    ----------
    df : OHLCV DataFrame with DatetimeIndex (hourly).

    Returns
    -------
    pd.Series (hourly index) — continuous trend score.
    """
    close = df["close"]

    # Resample to daily for clean EMA computation
    daily_close = close.resample("1D").last().dropna()

    ema20 = daily_close.ewm(span=20, min_periods=10).mean()
    ema50 = daily_close.ewm(span=50, min_periods=25).mean()
    ema200 = daily_close.ewm(span=200, min_periods=100).mean()
    mom30 = np.log(daily_close / daily_close.shift(30))
    mom7 = np.log(daily_close / daily_close.shift(7))

    score = pd.Series(0.0, index=daily_close.index)
    score += np.clip((daily_close / ema20 - 1) * 20, -1, 1)
    score += np.clip((daily_close / ema50 - 1) * 10, -1, 1)
    score += np.clip((daily_close / ema200 - 1) * 5, -1, 1)
    score += np.clip((ema20 / ema50 - 1) * 20, -1, 1)
    score += np.clip((ema50 / ema200 - 1) * 10, -1, 1)
    score += np.clip(mom30.fillna(0) * 5, -1, 1)
    score += np.clip(mom7.fillna(0) * 10, -1, 1)

    # Forward-fill daily score to hourly
    return score.reindex(close.index, method="ffill")


def score_to_regime(
    score: pd.Series,
    thresholds: list | None = None,
) -> pd.Series:
    """
    Map continuous score → integer regime index (0=Super Bull … 6=Super Bear).

    Default thresholds are tuned so that only strongly bearish conditions
    produce bear regimes — reflecting BTC's structural upward drift.
    """
    if thresholds is None:
        thresholds = TREND_SCORE_THRESHOLDS

    # thresholds is a list of 6 values: [t0, t1, t2, t3, t4, t5]
    # score > t0 → 0 (Super Bull)
    # t0 >= score > t1 → 1 (Strong Bull)
    # ...
    # t4 >= score > t5 → 5 (Strong Bear)
    # score <= t5 → 6 (Super Bear)

    regime = pd.Series(len(thresholds), index=score.index, dtype=int)
    # Process from smallest threshold first so larger thresholds overwrite
    for i in range(len(thresholds) - 1, -1, -1):
        regime[score > thresholds[i]] = i

    return regime


def smooth_regimes(
    regime_series: pd.Series,
    min_duration: int = MIN_REGIME_BARS,
) -> pd.Series:
    """
    Causal smoothing: require a new regime to persist for min_duration bars.
    """
    vals = regime_series.values.copy()
    current = int(vals[0])
    candidate = current
    count = 0

    for i in range(1, len(vals)):
        raw = int(vals[i])
        if raw == candidate:
            count += 1
        else:
            candidate = raw
            count = 1
        if count >= min_duration and candidate != current:
            current = candidate
        vals[i] = current

    return pd.Series(vals, index=regime_series.index, name="regime_state")


def compute_trend_regime(
    df: pd.DataFrame,
    thresholds: list | None = None,
    min_regime_bars: int = MIN_REGIME_BARS,
) -> tuple[pd.Series, pd.Series]:
    """
    Full pipeline: compute trend score → classify regime → smooth.

    Returns
    -------
    (regime_state, trend_score) — both hourly pd.Series
    """
    score = compute_trend_score(df)
    regime = score_to_regime(score, thresholds)
    regime = smooth_regimes(regime, min_regime_bars)
    return regime, score
