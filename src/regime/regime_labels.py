"""Semantic regime labeling and accuracy metrics.

After fitting the HMM, raw state indices (0-6) are sorted by a directional
score derived from their mean feature vectors and relabeled:
  0 = Strong Bull, 1 = Bull, 2 = Weak Bull, 3 = Sideways,
  4 = Weak Bear,   5 = Bear, 6 = Strong Bear
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import REGIME_LABELS, REGIME_DIRECTION, HMM_FEATURES


def build_state_map(
    hmm_model,
    X_raw: np.ndarray = None,
    raw_states: np.ndarray = None,
    forward_returns: np.ndarray = None,
    feature_names: list = HMM_FEATURES,
) -> dict:
    """
    Return a mapping {raw_state_index → semantic_label_index}.

    Preferred method (when X_raw + raw_states + forward_returns are provided):
      Sort by mean forward 24h log-return actually observed in each state.
      This is data-driven and avoids scaled-space ordering errors.

    Fallback (means_ in scaled space):
      score = 0.5 * mean(log_return_24h) + 0.3 * mean(log_return_4h)
              - 0.2 * mean(realized_vol_24h)
    """
    n_states = hmm_model.n_states

    if raw_states is not None and forward_returns is not None:
        # Data-driven: sort states by their actual mean forward return
        scores = np.array([
            forward_returns[raw_states == s].mean() if (raw_states == s).any() else 0.0
            for s in range(n_states)
        ])
    else:
        # Fallback: use scaled means
        means = hmm_model.means_
        idx_ret24 = feature_names.index("log_return_24h")
        idx_ret4  = feature_names.index("log_return_4h")
        idx_vol24 = feature_names.index("realized_vol_24h")
        scores = (
            0.5 * means[:, idx_ret24]
            + 0.3 * means[:, idx_ret4]
            - 0.2 * means[:, idx_vol24]
        )

    # sort descending: highest score → Strong Bull (label 0)
    sorted_raw = np.argsort(scores)[::-1]
    state_map = {int(raw): int(label) for label, raw in enumerate(sorted_raw)}
    return state_map


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
