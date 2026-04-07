"""Tests for HMM regime model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.data.loader import load_raw
from src.data.features import build_hmm_features
from src.regime.hmm_model import BTCHMMModel, extract_regime_series
from src.regime.regime_labels import build_state_map, apply_state_map, compute_transition_matrix
from src.config import HMM_FEATURES, HMM_N_STATES, REGIME_LABELS


@pytest.fixture(scope="module")
def fitted_hmm():
    df = load_raw()
    feat = build_hmm_features(df)
    # Use first 20k bars for speed in tests
    X = feat[HMM_FEATURES].values[:20_000]
    hmm = BTCHMMModel(n_restarts=2)
    hmm.fit(X)
    return hmm, feat[:20_000]


def test_hmm_fits(fitted_hmm):
    hmm, _ = fitted_hmm
    assert hmm.model is not None
    assert hmm.log_likelihood < 0   # log-likelihood is negative


def test_hmm_state_count(fitted_hmm):
    hmm, feat = fitted_hmm
    X = feat[HMM_FEATURES].values
    posteriors = hmm.predict_proba(X)
    assert posteriors.shape == (len(X), HMM_N_STATES)
    # Each row sums to ~1
    assert np.allclose(posteriors.sum(axis=1), 1.0, atol=1e-4)


def test_hmm_all_states_populated(fitted_hmm):
    hmm, feat = fitted_hmm
    X = feat[HMM_FEATURES].values
    posteriors = hmm.predict_proba(X)
    raw_states, _, _ = extract_regime_series(posteriors)
    unique_states = np.unique(raw_states)
    # With 20k bars we expect all 7 states to appear
    assert len(unique_states) == HMM_N_STATES, \
        f"Only {len(unique_states)} states found: {unique_states}"


def test_state_map_coverage(fitted_hmm):
    hmm, _ = fitted_hmm
    state_map = build_state_map(hmm)
    assert set(state_map.keys()) == set(range(HMM_N_STATES))
    assert set(state_map.values()) == set(range(HMM_N_STATES))


def test_confidence_in_range(fitted_hmm):
    hmm, feat = fitted_hmm
    X = feat[HMM_FEATURES].values
    posteriors = hmm.predict_proba(X)
    _, confidence, entropy = extract_regime_series(posteriors)
    assert (confidence >= 0).all() and (confidence <= 1).all()
    assert (entropy >= 0).all()


def test_regime_duration(fitted_hmm):
    """Mean regime spell should be > 4 bars (not switching every bar)."""
    hmm, feat = fitted_hmm
    X = feat[HMM_FEATURES].values
    posteriors = hmm.predict_proba(X)
    raw_states, _, _ = extract_regime_series(posteriors)
    state_map = build_state_map(hmm)
    labeled = apply_state_map(raw_states, state_map)

    # Compute mean spell length
    spells = []
    count = 1
    for i in range(1, len(labeled)):
        if labeled[i] == labeled[i - 1]:
            count += 1
        else:
            spells.append(count)
            count = 1
    spells.append(count)
    mean_dur = np.mean(spells)
    assert mean_dur > 4, f"Mean regime duration too short: {mean_dur:.1f} bars"


def test_transition_matrix_shape(fitted_hmm):
    hmm, feat = fitted_hmm
    X = feat[HMM_FEATURES].values
    posteriors = hmm.predict_proba(X)
    raw_states, _, _ = extract_regime_series(posteriors)
    state_map = build_state_map(hmm)
    labeled = pd.Series(apply_state_map(raw_states, state_map))
    trans = compute_transition_matrix(labeled)
    assert trans.shape == (HMM_N_STATES, HMM_N_STATES)
    assert np.allclose(trans.values.sum(axis=1), 1.0, atol=1e-4)
