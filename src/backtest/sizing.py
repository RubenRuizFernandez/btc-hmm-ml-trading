"""Position sizing: fractional Kelly scaled by regime confidence."""
import numpy as np
from src.config import KELLY_FRACTION, MAX_POSITION, REGIME_CONFIDENCE_THRESHOLD


def kelly_size(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Classic Kelly criterion.

    Parameters
    ----------
    win_rate : probability of a winning trade
    avg_win  : average return on winning trades (positive fraction)
    avg_loss : average return on losing trades (positive fraction, i.e. abs value)

    Returns
    -------
    Kelly fraction ∈ [0, 1]
    """
    if avg_loss <= 0 or win_rate <= 0:
        return 0.0
    b = avg_win / avg_loss
    p = win_rate
    q = 1.0 - p
    k = (b * p - q) / b
    return float(np.clip(k, 0.0, 1.0))


def fractional_kelly(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Quarter-Kelly size."""
    return kelly_size(win_rate, avg_win, avg_loss) * KELLY_FRACTION


def confidence_scaled_size(
    kelly_base: float,
    regime_confidence: float,
    threshold: float = REGIME_CONFIDENCE_THRESHOLD,
) -> float:
    """
    Scale position size linearly with regime confidence above the threshold.

    At threshold → factor = 0 → position = 0
    At 1.0       → factor = 1 → position = kelly_base
    """
    if regime_confidence < threshold:
        return 0.0
    factor = (regime_confidence - threshold) / (1.0 - threshold + 1e-10)
    size = kelly_base * float(np.clip(factor, 0.0, 1.0))
    return float(np.clip(size, 0.0, MAX_POSITION))


def compute_position_sizes(
    regime_confidence: np.ndarray,
    kelly_base: float,
    threshold: float = REGIME_CONFIDENCE_THRESHOLD,
) -> np.ndarray:
    """Vectorized confidence-scaled Kelly sizes for an array of confidence values."""
    factor = np.clip(
        (regime_confidence - threshold) / (1.0 - threshold + 1e-10), 0.0, 1.0
    )
    sizes = kelly_base * factor
    return np.clip(sizes, 0.0, MAX_POSITION)
