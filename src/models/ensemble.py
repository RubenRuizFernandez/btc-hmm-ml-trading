"""Stacking ensemble: LogisticRegression on top of LightGBM + LSTM scores."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import (
    ENSEMBLE_C,
    SIGNAL_LONG_THRESHOLD,
    SIGNAL_SHORT_THRESHOLD,
    REGIME_CONFIDENCE_THRESHOLD,
    TRADEABLE_REGIMES,
    HMM_N_STATES,
)
from src.models.lgbm_model import LGBMSignalModel
from src.models.lstm_model import LSTMSignalModel


class EnsembleSignalModel:
    """
    Stacking ensemble combining LightGBM, LSTM, and regime features.

    Meta-features:
        [lgbm_score, lstm_score, regime_confidence, regime_{0..6} one-hot]
    """

    def __init__(self, ensemble_c: float = ENSEMBLE_C):
        self.lgbm = LGBMSignalModel()
        self.lstm = LSTMSignalModel()
        self.meta = LogisticRegression(C=ensemble_c, max_iter=500)
        self._fitted = False

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        regime_state: pd.Series,
        regime_confidence: pd.Series,
        use_lstm: bool = True,
    ) -> "EnsembleSignalModel":
        """
        Fit all three layers on IS data.

        y must be binary: 1 = long-profitable, 0 = otherwise.
        use_lstm=False skips LSTM for speed (~50x faster per fold).
        """
        self._use_lstm = use_lstm
        # Layer 1: base models
        self.lgbm.fit(X, y)
        if use_lstm:
            self.lstm.fit(X, y)

        # Layer 2: build meta-features on IS (in-sample predictions)
        meta_X = self._build_meta(X, regime_state, regime_confidence)
        self.meta.fit(meta_X, y)
        self._fitted = True
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(
        self,
        X: pd.DataFrame,
        regime_state: pd.Series,
        regime_confidence: pd.Series,
    ) -> np.ndarray:
        """Return ensemble score ∈ [0, 1] for each row."""
        self._check_fitted()
        meta_X = self._build_meta(X, regime_state, regime_confidence)
        return self.meta.predict_proba(meta_X)[:, 1]

    def generate_signals(
        self,
        X: pd.DataFrame,
        regime_state: pd.Series,
        regime_confidence: pd.Series,
        long_threshold: float = SIGNAL_LONG_THRESHOLD,
        short_threshold: float = SIGNAL_SHORT_THRESHOLD,
        confidence_threshold: float = REGIME_CONFIDENCE_THRESHOLD,
        tradeable_regimes: list = TRADEABLE_REGIMES,
        min_hold_bars: int = 8,
        signal_smooth_window: int = 4,
    ) -> pd.Series:
        """
        Return signal series: +1 (long), -1 (short), 0 (flat).

        All three conditions must hold:
        1. ensemble_score (smoothed) > long_threshold OR < short_threshold
        2. regime_confidence > confidence_threshold
        3. regime_state in tradeable_regimes

        min_hold_bars   : minimum bars to hold a position (reduces churn)
        signal_smooth_window : rolling mean of scores before thresholding
        """
        scores = self.predict_proba(X, regime_state, regime_confidence)
        # Smooth scores over a window to reduce bar-to-bar noise
        score_s = pd.Series(scores, index=X.index)
        if signal_smooth_window > 1:
            score_s = score_s.rolling(signal_smooth_window, min_periods=1).mean()

        regime_ok = regime_state.isin(tradeable_regimes)
        conf_ok = regime_confidence > confidence_threshold
        gate = regime_ok & conf_ok

        raw_sig = pd.Series(0, index=X.index, dtype=float)
        raw_sig[gate & (score_s > long_threshold)] = 1.0
        raw_sig[gate & (score_s < short_threshold)] = -1.0

        # Enforce minimum hold: once a position is taken, hold for min_hold_bars
        if min_hold_bars <= 1:
            return raw_sig

        sig = raw_sig.copy()
        current = 0.0
        hold_count = 0
        for i in range(len(sig)):
            if hold_count > 0:
                sig.iloc[i] = current
                hold_count -= 1
            elif raw_sig.iloc[i] != 0.0:
                current = raw_sig.iloc[i]
                sig.iloc[i] = current
                hold_count = min_hold_bars - 1
            else:
                current = 0.0
        return sig

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_meta(
        self,
        X: pd.DataFrame,
        regime_state: pd.Series,
        regime_confidence: pd.Series,
    ) -> np.ndarray:
        lgbm_scores = self.lgbm.predict_proba(X)
        use_lstm = getattr(self, "_use_lstm", True)
        lstm_scores = self.lstm.predict_proba(X) if use_lstm else np.full(len(X), 0.5)

        regime_onehot = np.zeros((len(X), HMM_N_STATES))
        for s in range(HMM_N_STATES):
            regime_onehot[:, s] = (regime_state.values == s).astype(float)

        return np.column_stack([
            lgbm_scores,
            lstm_scores,
            regime_confidence.values,
            regime_onehot,
        ])

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

    @property
    def lgbm_feature_importance(self) -> pd.Series:
        return self.lgbm.feature_importance

    def compute_shap(self, X: pd.DataFrame) -> np.ndarray:
        return self.lgbm.compute_shap(X)
