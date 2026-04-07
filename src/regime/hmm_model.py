"""Gaussian HMM regime detector.

Trains a 7-state Gaussian HMM on the 8-dimensional HMM feature set.
Uses multiple random restarts to avoid local optima.
"""
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

from src.config import (
    HMM_N_STATES,
    HMM_COVARIANCE_TYPE,
    HMM_N_ITER,
    HMM_TOL,
    HMM_N_RESTARTS,
    HMM_RANDOM_SEED,
    HMM_FEATURES,
)


class BTCHMMModel:
    """Wrapper around hmmlearn GaussianHMM with restart logic and scaler."""

    def __init__(
        self,
        n_states: int = HMM_N_STATES,
        n_restarts: int = HMM_N_RESTARTS,
        random_seed: int = HMM_RANDOM_SEED,
    ):
        self.n_states = n_states
        self.n_restarts = n_restarts
        self.random_seed = random_seed
        self.scaler = StandardScaler()
        self.model: GaussianHMM | None = None
        self._best_log_likelihood = -np.inf

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "BTCHMMModel":
        """
        Fit GaussianHMM with multiple random restarts.

        Parameters
        ----------
        X : array of shape (T, n_features)
            Raw (un-scaled) HMM feature matrix from in-sample data only.
        """
        X_scaled = self.scaler.fit_transform(X)

        best_model = None
        best_ll = -np.inf

        rng = np.random.RandomState(self.random_seed)

        for i in range(self.n_restarts):
            seed = rng.randint(0, 100_000)
            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type=HMM_COVARIANCE_TYPE,
                n_iter=HMM_N_ITER,
                tol=HMM_TOL,
                random_state=seed,
            )
            try:
                model.fit(X_scaled)
                ll = model.score(X_scaled)
                if ll > best_ll:
                    best_ll = ll
                    best_model = model
            except Exception:
                continue

        if best_model is None:
            raise RuntimeError("All HMM restarts failed.")

        self.model = best_model
        self._best_log_likelihood = best_ll
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return Viterbi state sequence (raw internal state indices)."""
        self._check_fitted()
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return posterior state probabilities, shape (T, n_states)."""
        self._check_fitted()
        return self.model.predict_proba(self.scaler.transform(X))

    def score(self, X: np.ndarray) -> float:
        """Log-likelihood per sample on X."""
        self._check_fitted()
        return self.model.score(self.scaler.transform(X))

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "BTCHMMModel":
        return joblib.load(path)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

    @property
    def means_(self) -> np.ndarray:
        """Shape (n_states, n_features) — mean of each state in scaled space."""
        self._check_fitted()
        return self.model.means_

    @property
    def transmat_(self) -> np.ndarray:
        """Transition probability matrix (n_states × n_states)."""
        self._check_fitted()
        return self.model.transmat_

    @property
    def log_likelihood(self) -> float:
        return self._best_log_likelihood


def extract_regime_series(posteriors: np.ndarray) -> tuple:
    """
    From posterior probability matrix, extract:
    - state  : integer state index (argmax)
    - confidence : max posterior per bar
    - entropy    : Shannon entropy of posterior vector per bar
    """
    state = posteriors.argmax(axis=1)
    confidence = posteriors.max(axis=1)
    entropy = -np.sum(posteriors * np.log(posteriors + 1e-9), axis=1)
    return state, confidence, entropy
