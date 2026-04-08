"""Semantic regime labeling and accuracy metrics.

After fitting the HMM, raw state indices (0-6) are sorted by a directional
score derived from their mean feature vectors and relabeled:
  0 = Strong Bull, 1 = Bull, 2 = Weak Bull, 3 = Sideways,
  4 = Weak Bear,   5 = Bear, 6 = Strong Bear
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import REGIME_LABELS, REGIME_DIRECTION, HMM_FEATURES, MIN_REGIME_BARS


def build_state_map(
    hmm_model,
    X_raw: np.ndarray = None,
    raw_states: np.ndarray = None,
    forward_returns: np.ndarray = None,
    feature_names: list = HMM_FEATURES,
) -> dict:
    """
    Return a mapping {raw_state_index → semantic_label_index}.

    Method 1 (WF folds — forward_returns provided):
      Sort by mean forward 24h log-return per state from IS data.

    Method 2 (full-period fit — X_raw + raw_states, no forward_returns):
      Sort by concurrent feature means (log_return_24h + log_return_168h).
      No look-ahead bias.

    Method 3 (fallback):
      Use scaled-space means from HMM.
    """
    n_states = hmm_model.n_states

    if raw_states is not None and forward_returns is not None:
        # Method 1: sort by actual mean forward return (valid for IS data)
        scores = np.array([
            forward_returns[raw_states == s].mean() if (raw_states == s).any() else 0.0
            for s in range(n_states)
        ])
    elif X_raw is not None and raw_states is not None:
        # Method 2: sort by concurrent observed returns (no look-ahead)
        idx_ret24 = feature_names.index("log_return_24h")
        idx_ret168 = feature_names.index("log_return_168h") if "log_return_168h" in feature_names else idx_ret24
        scores = np.array([
            0.5 * X_raw[raw_states == s, idx_ret24].mean()
            + 0.5 * X_raw[raw_states == s, idx_ret168].mean()
            if (raw_states == s).any() else 0.0
            for s in range(n_states)
        ])
    else:
        # Method 3: fallback using scaled means
        means = hmm_model.means_
        idx_ret24 = feature_names.index("log_return_24h")
        idx_ret4  = feature_names.index("log_return_4h")
        scores = 0.6 * means[:, idx_ret24] + 0.4 * means[:, idx_ret4]

    # sort descending: highest score → Super Bull (label 0)
    sorted_raw = np.argsort(scores)[::-1]
    state_map = {int(raw): int(label) for label, raw in enumerate(sorted_raw)}
    return state_map


def smooth_regimes(
    regime_series: pd.Series,
    min_duration: int = MIN_REGIME_BARS,
) -> pd.Series:
    """
    Causal regime smoothing — require a new regime to persist for
    min_duration bars before confirming the switch.

    Prevents whipsaw from noisy HMM transitions.
    """
    smoothed = regime_series.copy()
    vals = smoothed.values.copy()
    current_regime = int(vals[0])
    candidate = current_regime
    candidate_count = 0

    for i in range(1, len(vals)):
        raw = int(vals[i])
        if raw == candidate:
            candidate_count += 1
        else:
            candidate = raw
            candidate_count = 1

        if candidate_count >= min_duration and candidate != current_regime:
            current_regime = candidate

        vals[i] = current_regime

    return pd.Series(vals, index=regime_series.index, name=regime_series.name)


def apply_state_map(raw_states: np.ndarray, state_map: dict) -> np.ndarray:
    """Convert raw HMM state indices → semantic label indices."""
    return np.vectorize(state_map.get)(raw_states)


def regime_label_name(label_idx: int) -> str:
    return REGIME_LABELS[label_idx]


def compute_regime_stats(
    df: pd.DataFrame,
    regime_col: str = "regime_state",
    confidence_col: str = "regime_confidence",
    forward_return_col: str = "forward_return",
) -> pd.DataFrame:
    """
    Compute per-regime statistics:
    - mean_return, vol_return
    - directional_accuracy (fraction matching expected direction)
    - mean_duration (avg bars in continuous spell)
    - mean_confidence
    - count
    """
    rows = []
    for state in range(len(REGIME_LABELS)):
        mask = df[regime_col] == state
        sub = df[mask]
        if len(sub) == 0:
            continue

        direction = REGIME_DIRECTION[state]
        if direction == 1:
            dir_acc = (sub[forward_return_col] > 0).mean() if forward_return_col in sub else np.nan
        elif direction == -1:
            dir_acc = (sub[forward_return_col] < 0).mean() if forward_return_col in sub else np.nan
        else:
            dir_acc = np.nan

        # Average duration: group consecutive same-regime bars
        transitions = mask.astype(int).diff().fillna(0).abs()
        spell_id = transitions.cumsum()
        spell_lengths = (
            mask.astype(int)
            .groupby(spell_id)
            .sum()
        )
        spell_lengths = spell_lengths[spell_lengths > 0]
        mean_dur = spell_lengths.mean() if len(spell_lengths) > 0 else np.nan

        rows.append(
            {
                "regime_idx": state,
                "regime_label": REGIME_LABELS[state],
                "count": int(mask.sum()),
                "mean_return": sub[forward_return_col].mean() if forward_return_col in sub else np.nan,
                "vol_return": sub[forward_return_col].std() if forward_return_col in sub else np.nan,
                "directional_accuracy": dir_acc,
                "mean_confidence": sub[confidence_col].mean(),
                "mean_duration_bars": mean_dur,
            }
        )

    return pd.DataFrame(rows).set_index("regime_idx")


def compute_transition_matrix(
    regime_series: pd.Series,
    n_states: int = 7,
) -> pd.DataFrame:
    """Empirical transition probability matrix from the labeled regime series."""
    mat = np.zeros((n_states, n_states))
    vals = regime_series.values
    for i in range(len(vals) - 1):
        a, b = int(vals[i]), int(vals[i + 1])
        if 0 <= a < n_states and 0 <= b < n_states:
            mat[a, b] += 1

    row_sums = mat.sum(axis=1, keepdims=True)
    mat = np.where(row_sums > 0, mat / row_sums, 0)
    return pd.DataFrame(mat, index=REGIME_LABELS, columns=REGIME_LABELS)
