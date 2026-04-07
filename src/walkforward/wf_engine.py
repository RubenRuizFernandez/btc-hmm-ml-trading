"""Walk-forward validation engine.

Anchored expanding IS window + rolling 12-month OOS window, stepping 3 months.
Each fold retrains the full pipeline (scaler, HMM, LightGBM, LSTM, ensemble).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

import src.config as _cfg
from src.config import (
    WF_IS_MIN_YEARS,
    WF_OOS_MONTHS,
    WF_STEP_MONTHS,
    WF_RESULTS,
    HMM_FEATURES,
    KELLY_FRACTION,
)
from src.data.features import build_hmm_features, build_ml_features, add_target, add_regime_features
from src.regime.hmm_model import BTCHMMModel, extract_regime_series
from src.regime.regime_labels import build_state_map, apply_state_map
from src.models.ensemble import EnsembleSignalModel
from src.backtest.engine import run_backtest
from src.backtest.sizing import fractional_kelly
from src.backtest.metrics import compute_metrics


@dataclass
class FoldResult:
    fold_idx: int
    is_start: str
    is_end: str
    oos_start: str
    oos_end: str
    metrics: dict
    n_oos_bars: int
    hmm_log_likelihood: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(self.metrics)
        return d


def _fold_dates(df: pd.DataFrame) -> list[tuple]:
    """Generate (is_end, oos_start, oos_end) tuples."""
    data_start = df.index[0].to_pydatetime().replace(tzinfo=None)
    data_end = df.index[-1].to_pydatetime().replace(tzinfo=None)

    # First IS must cover at least WF_IS_MIN_YEARS of data
    min_is_end = data_start + relativedelta(years=WF_IS_MIN_YEARS)

    folds = []
    oos_start = min_is_end
    while True:
        oos_end = oos_start + relativedelta(months=WF_OOS_MONTHS)
        if oos_end > data_end:
            break
        folds.append((data_start, oos_start, oos_end))
        oos_start += relativedelta(months=WF_STEP_MONTHS)

    return folds


def run_walk_forward(
    df: pd.DataFrame,
    n_folds: int | None = None,
    save_results: bool = True,
) -> list[FoldResult]:
    """
    Run the full walk-forward matrix.

    Parameters
    ----------
    df        : cleaned OHLCV DataFrame from loader.load_raw()
    n_folds   : limit to first N folds (useful for smoke tests)
    save_results : persist per-fold parquet files to WF_RESULTS

    Returns
    -------
    List of FoldResult objects
    """
    if save_results:
        WF_RESULTS.mkdir(parents=True, exist_ok=True)

    # Precompute HMM and ML features once on the full series
    hmm_feat_full = build_hmm_features(df)
    ml_feat_full = build_ml_features(df)

    # Align on common index
    common_idx = hmm_feat_full.index.intersection(ml_feat_full.index)
    hmm_feat_full = hmm_feat_full.loc[common_idx]
    ml_feat_full = ml_feat_full.loc[common_idx]
    df_aligned = df.loc[common_idx]

    folds = _fold_dates(df_aligned)
    if n_folds is not None:
        folds = folds[:n_folds]

    results = []

    for fold_idx, (is_start, oos_start, oos_end) in enumerate(
        tqdm(folds, desc="Walk-forward folds")
    ):
        result = _run_single_fold(
            fold_idx=fold_idx,
            is_start=is_start,
            oos_start=oos_start,
            oos_end=oos_end,
            df=df_aligned,
            hmm_feat=hmm_feat_full,
            ml_feat=ml_feat_full,
            save_results=save_results,
        )
        results.append(result)

    if save_results:
        _save_summary(results)

    return results


def _run_single_fold(
    fold_idx: int,
    is_start: datetime,
    oos_start: datetime,
    oos_end: datetime,
    df: pd.DataFrame,
    hmm_feat: pd.DataFrame,
    ml_feat: pd.DataFrame,
    save_results: bool,
) -> FoldResult:
    # ── Slice IS / OOS ────────────────────────────────────────────────────────
    is_mask = (hmm_feat.index >= pd.Timestamp(is_start, tz="UTC")) & \
              (hmm_feat.index < pd.Timestamp(oos_start, tz="UTC"))
    oos_mask = (hmm_feat.index >= pd.Timestamp(oos_start, tz="UTC")) & \
               (hmm_feat.index < pd.Timestamp(oos_end, tz="UTC"))

    X_hmm_is = hmm_feat.loc[is_mask, HMM_FEATURES].values
    X_hmm_oos = hmm_feat.loc[oos_mask, HMM_FEATURES].values

    if len(X_hmm_is) < 500 or len(X_hmm_oos) < 24:
        return _empty_fold(fold_idx, is_start, oos_start, oos_end)

    # ── HMM: fit on IS, infer on IS+OOS ───────────────────────────────────────
    hmm = BTCHMMModel()
    hmm.fit(X_hmm_is)

    full_oos_hmm = hmm_feat.loc[oos_mask, HMM_FEATURES].values
    post_is = hmm.predict_proba(X_hmm_is)
    post_oos = hmm.predict_proba(full_oos_hmm)

    raw_is, conf_is, ent_is = extract_regime_series(post_is)
    raw_oos, conf_oos, ent_oos = extract_regime_series(post_oos)

    # Data-driven state map: sort by actual mean 24h forward return per IS state
    idx_is = hmm_feat.index[is_mask]
    fwd_is = np.log(df.loc[idx_is, "close"].shift(-24) / df.loc[idx_is, "close"]).values
    state_map = build_state_map(hmm, raw_states=raw_is, forward_returns=fwd_is)

    regime_is = apply_state_map(raw_is, state_map)
    regime_oos = apply_state_map(raw_oos, state_map)

    idx_is = hmm_feat.index[is_mask]
    idx_oos = hmm_feat.index[oos_mask]

    regime_is_s = pd.Series(regime_is, index=idx_is)
    conf_is_s = pd.Series(conf_is, index=idx_is)
    ent_is_s = pd.Series(ent_is, index=idx_is)

    regime_oos_s = pd.Series(regime_oos, index=idx_oos)
    conf_oos_s = pd.Series(conf_oos, index=idx_oos)
    ent_oos_s = pd.Series(ent_oos, index=idx_oos)

    # ── ML features: add regime + target ──────────────────────────────────────
    X_ml_is = ml_feat.loc[is_mask].copy()
    X_ml_is = add_regime_features(X_ml_is, regime_is_s, conf_is_s, ent_is_s)
    X_ml_is = add_target(X_ml_is, df.loc[is_mask, "close"])
    X_ml_is = X_ml_is.dropna()

    X_ml_oos = ml_feat.loc[oos_mask].copy()
    X_ml_oos = add_regime_features(X_ml_oos, regime_oos_s, conf_oos_s, ent_oos_s)
    X_ml_oos = X_ml_oos.dropna()

    # Binary target: 1 = long profitable
    feature_cols = [c for c in X_ml_is.columns if c not in
                    ("target", "forward_return")]
    y_is = (X_ml_is["target"] == 1).astype(int)

    if y_is.sum() < 50 or (1 - y_is).sum() < 50:
        return _empty_fold(fold_idx, is_start, oos_start, oos_end)

    # ── Ensemble: fit on IS, predict on OOS ───────────────────────────────────
    use_lstm = getattr(_cfg, "LSTM_EPOCHS", 10) > 0
    ensemble = EnsembleSignalModel()
    ensemble.fit(
        X_ml_is[feature_cols],
        y_is,
        regime_is_s.loc[X_ml_is.index],
        conf_is_s.loc[X_ml_is.index],
        use_lstm=use_lstm,
    )

    oos_feat_idx = X_ml_oos.index.intersection(idx_oos)
    X_ml_oos_clean = X_ml_oos.loc[oos_feat_idx, feature_cols]
    regime_oos_clean = regime_oos_s.loc[oos_feat_idx]
    conf_oos_clean = conf_oos_s.loc[oos_feat_idx]

    signals = ensemble.generate_signals(
        X_ml_oos_clean, regime_oos_clean, conf_oos_clean
    )

    # ── Kelly sizing from IS trade statistics ─────────────────────────────────
    # Use a quick IS backtest to calibrate Kelly params
    is_signals = ensemble.generate_signals(
        X_ml_is[feature_cols], regime_is_s.loc[X_ml_is.index],
        conf_is_s.loc[X_ml_is.index]
    )
    is_res = run_backtest(
        df.loc[X_ml_is.index, "close"],
        is_signals,
        conf_is_s.loc[X_ml_is.index],
        kelly_base=KELLY_FRACTION,
    )
    is_m = is_res.metrics
    kelly_base = fractional_kelly(
        win_rate=is_m.get("win_rate") or 0.5,
        avg_win=abs(is_m.get("avg_win_pct") or 1) / 100,
        avg_loss=abs(is_m.get("avg_loss_pct") or 1) / 100,
    )

    # ── OOS backtest ──────────────────────────────────────────────────────────
    oos_close = df.loc[oos_feat_idx, "close"]
    oos_result = run_backtest(oos_close, signals, conf_oos_clean, kelly_base=kelly_base)
    metrics = oos_result.metrics

    fold_result = FoldResult(
        fold_idx=fold_idx,
        is_start=str(is_start.date()),
        is_end=str(oos_start.date()),
        oos_start=str(oos_start.date()),
        oos_end=str(oos_end.date()),
        metrics=metrics,
        n_oos_bars=len(oos_feat_idx),
        hmm_log_likelihood=hmm.log_likelihood,
    )

    if save_results:
        eq = oos_result.equity_curve.reset_index()
        eq.columns = ["datetime", "equity"]
        eq["fold"] = fold_idx
        eq.to_parquet(WF_RESULTS / f"fold_{fold_idx:03d}_equity.parquet", index=False)

    return fold_result


def _empty_fold(fold_idx, is_start, oos_start, oos_end) -> FoldResult:
    from src.backtest.metrics import _empty_metrics
    return FoldResult(
        fold_idx=fold_idx,
        is_start=str(is_start.date()) if hasattr(is_start, "date") else str(is_start),
        is_end=str(oos_start.date()) if hasattr(oos_start, "date") else str(oos_start),
        oos_start=str(oos_start.date()) if hasattr(oos_start, "date") else str(oos_start),
        oos_end=str(oos_end.date()) if hasattr(oos_end, "date") else str(oos_end),
        metrics=_empty_metrics(),
        n_oos_bars=0,
        hmm_log_likelihood=np.nan,
    )


def _save_summary(results: list[FoldResult]) -> None:
    rows = [r.to_dict() for r in results]
    df = pd.DataFrame(rows)
    df.to_parquet(WF_RESULTS / "wf_summary.parquet", index=False)
    df.to_csv(WF_RESULTS / "wf_summary.csv", index=False)


def load_wf_summary(path: Path = None) -> pd.DataFrame:
    path = path or WF_RESULTS / "wf_summary.parquet"
    return pd.read_parquet(path)
