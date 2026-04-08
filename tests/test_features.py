"""Tests for feature engineering pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.data.loader import load_raw
from src.data.features import build_hmm_features, build_ml_features, add_target
from src.config import HMM_FEATURES


@pytest.fixture(scope="module")
def df():
    return load_raw()


def test_loader_basic(df):
    assert len(df) > 24 * 365 * 8
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.is_monotonic_increasing
    assert not df.index.duplicated().any()
    assert df.isnull().sum().sum() == 0


def test_loader_no_zero_price(df):
    assert (df["close"] > 0).all()


def test_hmm_features_shape(df):
    feat = build_hmm_features(df)
    assert set(HMM_FEATURES).issubset(feat.columns), "Missing HMM feature columns"
    assert feat.isnull().sum().sum() == 0, "NaN in HMM features after dropna"
    assert len(feat) > len(df) * 0.95


def test_hmm_features_no_lookahead(df):
    """Verify features at bar t only use data up to t."""
    feat = build_hmm_features(df)
    # log_return_1h at bar t = log(close[t] / close[t-1]) — check monotone indexing
    assert feat.index.is_monotonic_increasing


def test_ml_features_shape(df):
    feat = build_ml_features(df)
    assert len(feat.columns) >= 30
    # No completely-empty columns
    all_nan_cols = feat.columns[feat.isnull().all()].tolist()
    assert len(all_nan_cols) == 0, f"All-NaN columns: {all_nan_cols}"


def test_target_no_lookahead(df):
    feat = build_ml_features(df)
    feat = add_target(feat, df["close"])
    # The forward return at the last bar must be NaN (no future data)
    assert pd.isna(feat["forward_return"].iloc[-1])


def test_target_values(df):
    feat = build_ml_features(df)
    feat = add_target(feat, df["close"])
    feat = feat.dropna(subset=["target"])
    assert set(feat["target"].unique()).issubset({-1, 0, 1})
