"""Feature engineering pipeline.

Two feature sets are produced:
1. HMM features   — 8 normalized market-structure features fed to GaussianHMM
2. ML features    — ~50 predictive features fed to LightGBM / LSTM

All features are strictly backward-looking (no center=True, no future leakage).
"""
import numpy as np
import pandas as pd
import ta as _ta

from src.config import (
    FORWARD_RETURN_HORIZON,
    FEE_THRESHOLD_BPS,
    HMM_FEATURES,
)


# ─── HMM feature set ──────────────────────────────────────────────────────────

def build_hmm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the 8-dimensional feature matrix used by the HMM."""
    out = pd.DataFrame(index=df.index)

    log_ret = np.log(df["close"] / df["close"].shift(1))
    out["log_return_1h"] = log_ret
    out["log_return_4h"] = np.log(df["close"] / df["close"].shift(4))
    out["log_return_24h"] = np.log(df["close"] / df["close"].shift(24))
    out["log_return_168h"] = np.log(df["close"] / df["close"].shift(168))

    out["realized_vol_24h"] = log_ret.rolling(24, min_periods=24).std() * np.sqrt(24)
    out["realized_vol_168h"] = log_ret.rolling(168, min_periods=168).std() * np.sqrt(168)
    out["vol_ratio"] = out["realized_vol_24h"] / (out["realized_vol_168h"] + 1e-10)

    # Trend position: where price sits relative to 30-day SMA
    sma_720 = df["close"].rolling(720, min_periods=168).mean()
    out["trend_sma_720"] = (df["close"] - sma_720) / (sma_720 + 1e-10)

    return out.dropna()


# ─── ML feature set ───────────────────────────────────────────────────────────

def build_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the full ~50-feature predictive matrix."""
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    log_ret = np.log(close / close.shift(1))

    # ── Momentum ──────────────────────────────────────────────────────────────
    out["rsi_14"] = _ta.momentum.RSIIndicator(close, window=14).rsi()
    out["rsi_48"] = _ta.momentum.RSIIndicator(close, window=48).rsi()

    macd = _ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    out["macd_line"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_hist"] = macd.macd_diff()

    for h in [6, 24, 72, 168]:
        out[f"roc_{h}h"] = np.log(close / close.shift(h))

    adx = _ta.trend.ADXIndicator(high, low, close, window=14)
    out["adx_14"] = adx.adx()

    # ── Trend / Bollinger ─────────────────────────────────────────────────────
    for w in [20, 50, 200]:
        sma = close.rolling(w, min_periods=w).mean()
        out[f"close_vs_sma{w}"] = (close - sma) / (sma + 1e-10)

    bb = _ta.volatility.BollingerBands(close, window=20, window_dev=2)
    out["bb_pct_b"] = bb.bollinger_pband()
    out["bb_bandwidth"] = bb.bollinger_wband()

    # ── Volume ────────────────────────────────────────────────────────────────
    obv = _ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    obv_max = obv.rolling(720, min_periods=24).max()
    out["obv_norm"] = obv / (obv_max.abs() + 1e-10)

    for w in [24, 168]:
        vol_sma = volume.rolling(w, min_periods=w).mean()
        out[f"vol_vs_sma{w}"] = volume / (vol_sma + 1e-10)

    # VWAP deviation (rolling 24h approximation)
    typical = (high + low + close) / 3
    cumtp_vol = (typical * volume).rolling(24, min_periods=1).sum()
    cumvol = volume.rolling(24, min_periods=1).sum()
    vwap = cumtp_vol / (cumvol + 1e-10)
    out["vwap_dev"] = (close - vwap) / (vwap + 1e-10)

    # ── Volatility ────────────────────────────────────────────────────────────
    atr = _ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    out["atr_norm"] = atr / (close + 1e-10)

    # Parkinson volatility
    park = np.sqrt(1 / (4 * np.log(2)) * np.log(high / low) ** 2)
    out["parkinson_vol_24h"] = park.rolling(24, min_periods=24).mean()

    for w in [24, 72, 168]:
        out[f"realized_vol_{w}h"] = (
            log_ret.rolling(w, min_periods=w).std() * np.sqrt(w)
        )
    out["vol_ratio_24_168"] = out["realized_vol_24h"] / (out["realized_vol_168h"] + 1e-10)

    # ── Microstructure / calendar ─────────────────────────────────────────────
    hour = df.index.hour
    dow = df.index.dayofweek
    out["sin_hour"] = np.sin(2 * np.pi * hour / 24)
    out["cos_hour"] = np.cos(2 * np.pi * hour / 24)
    out["sin_dow"] = np.sin(2 * np.pi * dow / 7)
    out["cos_dow"] = np.cos(2 * np.pi * dow / 7)

    rolling_52w_high = close.rolling(8760, min_periods=168).max()
    rolling_52w_low = close.rolling(8760, min_periods=168).min()
    out["close_vs_52w_high"] = (close - rolling_52w_high) / (rolling_52w_high + 1e-10)
    out["close_vs_52w_low"] = (close - rolling_52w_low) / (rolling_52w_low + 1e-10)

    return out


def add_target(df_feat: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """
    Add the binary classification target.

    target = 1  if forward log-return over FORWARD_RETURN_HORIZON bars
                 exceeds fee_threshold (long profitable)
           = -1 if forward log-return is below -fee_threshold (short profitable)
           = 0  otherwise (neutral / within cost band)

    The target is computed here but only used at training time — never at inference.
    """
    fwd = np.log(close.shift(-FORWARD_RETURN_HORIZON) / close)
    threshold = FEE_THRESHOLD_BPS * 1e-4
    df_feat = df_feat.copy()
    df_feat["forward_return"] = fwd
    df_feat["target"] = 0
    df_feat.loc[fwd > threshold, "target"] = 1
    df_feat.loc[fwd < -threshold, "target"] = -1
    return df_feat


def add_regime_features(
    df_feat: pd.DataFrame,
    regime_state: pd.Series,
    regime_confidence: pd.Series,
    regime_entropy: pd.Series,
    n_states: int = 7,
) -> pd.DataFrame:
    """Append HMM regime outputs as ML features."""
    df_feat = df_feat.copy()
    df_feat["regime_state"] = regime_state
    df_feat["regime_confidence"] = regime_confidence
    df_feat["regime_entropy"] = regime_entropy
    for s in range(n_states):
        df_feat[f"regime_{s}"] = (regime_state == s).astype(float)
    return df_feat


def build_full_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience: run HMM + ML feature engineering and attach target.
    Returns combined DataFrame (both feature sets merged by index).
    """
    hmm_feat = build_hmm_features(df)
    ml_feat = build_ml_features(df)
    combined = hmm_feat.join(ml_feat, how="inner")
    combined = add_target(combined, df["close"])
    return combined.dropna(subset=HMM_FEATURES)
