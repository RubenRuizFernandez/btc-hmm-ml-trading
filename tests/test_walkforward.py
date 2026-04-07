"""Tests for walk-forward engine."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.data.loader import load_raw
from src.walkforward.wf_engine import _fold_dates, _empty_fold
from src.config import WF_IS_MIN_YEARS, WF_OOS_MONTHS, WF_STEP_MONTHS


@pytest.fixture(scope="module")
def df():
    return load_raw()


def test_fold_dates_generated(df):
    hmm_feat_stub = df.copy()  # use df as proxy; _fold_dates only needs .index
    folds = _fold_dates(hmm_feat_stub)
    assert len(folds) > 10, f"Expected >10 folds, got {len(folds)}"


def test_fold_dates_no_overlap(df):
    """OOS start must equal IS end for each fold."""
    folds = _fold_dates(df)
    for is_start, oos_start, oos_end in folds:
        assert oos_start <= oos_end
        assert is_start < oos_start


def test_fold_dates_advancing(df):
    """Each fold's OOS should advance by WF_STEP_MONTHS."""
    folds = _fold_dates(df)
    from dateutil.relativedelta import relativedelta
    for i in range(1, len(folds)):
        prev_oos = folds[i - 1][1]
        curr_oos = folds[i][1]
        expected = prev_oos + relativedelta(months=WF_STEP_MONTHS)
        assert curr_oos == expected, f"Fold {i}: expected {expected}, got {curr_oos}"


def test_fold_dates_min_is_coverage(df):
    """First fold IS must cover at least WF_IS_MIN_YEARS."""
    folds = _fold_dates(df)
    is_start, oos_start, _ = folds[0]
    years_covered = (oos_start - is_start).days / 365.25
    assert years_covered >= WF_IS_MIN_YEARS, \
        f"IS too short: {years_covered:.1f} years (min={WF_IS_MIN_YEARS})"


def test_empty_fold_valid():
    from datetime import datetime
    result = _empty_fold(0, datetime(2017,1,1), datetime(2018,1,1), datetime(2019,1,1))
    assert result.fold_idx == 0
    assert np.isnan(result.metrics["sharpe"])


def test_smoke_wf_2_folds(df):
    """Run 2 folds end-to-end to verify no crash (slow, uses real models)."""
    from src.walkforward.wf_engine import run_walk_forward
    results = run_walk_forward(df, n_folds=2, save_results=False)
    assert len(results) == 2
    for r in results:
        assert hasattr(r, "metrics")
        assert "sharpe" in r.metrics
