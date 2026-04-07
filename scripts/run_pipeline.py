"""End-to-end pipeline runner.

Usage:
    python scripts/run_pipeline.py                       # full run (~13h)
    python scripts/run_pipeline.py --fast                # fast run (~20min, no LSTM)
    python scripts/run_pipeline.py --folds 2             # smoke test
    python scripts/run_pipeline.py --folds 5 --no-save
    python scripts/run_pipeline.py --skip-wf             # HMM only
"""
import sys
import argparse
from pathlib import Path

# Allow importing src from any cwd
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from src.data.loader import load_raw
from src.data.features import build_hmm_features, build_ml_features, add_target, add_regime_features
from src.regime.hmm_model import BTCHMMModel, extract_regime_series
from src.regime.regime_labels import (
    build_state_map, apply_state_map, compute_regime_stats, compute_transition_matrix
)
from src.config import (
    HMM_FEATURES, FEATURES_PATH, REGIMES_PATH, DATA_PROCESSED, HMM_N_STATES
)
from src.walkforward.wf_engine import run_walk_forward, load_wf_summary


def parse_args():
    p = argparse.ArgumentParser(description="BTC HMM + ML Trading System Pipeline")
    p.add_argument("--folds", type=int, default=None, help="Limit WF folds (default: all)")
    p.add_argument("--no-save", action="store_true", help="Skip saving parquet files")
    p.add_argument("--skip-wf", action="store_true", help="Skip walk-forward, only run full-period HMM")
    p.add_argument("--fast", action="store_true",
                   help="Fast mode: skip LSTM, reduce HMM restarts (~30s/fold vs ~23min/fold)")
    return p.parse_args()


def main():
    args = parse_args()
    save = not args.no_save

    # Fast mode overrides
    if args.fast:
        import src.config as cfg
        cfg.LSTM_EPOCHS = 0            # signal to wf_engine to skip LSTM
        cfg.HMM_N_RESTARTS = 3

    print("=" * 60)
    print("BTC/USD HMM Regime + ML Trading System")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/4] Loading data...")
    df = load_raw()
    print(f"      {len(df):,} bars | {df.index[0].date()} to {df.index[-1].date()}")

    # ── 2. Build features ─────────────────────────────────────────────────────
    print("\n[2/4] Building features...")
    hmm_feat = build_hmm_features(df)
    ml_feat = build_ml_features(df)

    common_idx = hmm_feat.index.intersection(ml_feat.index)
    hmm_feat = hmm_feat.loc[common_idx]
    ml_feat = ml_feat.loc[common_idx]

    print(f"      HMM features : {hmm_feat.shape}  ML features: {ml_feat.shape}")

    # ── 3. Full-period HMM (for dashboard / regime overlay) ──────────────────
    print("\n[3/4] Fitting full-period HMM regime model...")
    hmm_full = BTCHMMModel()
    hmm_full.fit(hmm_feat[HMM_FEATURES].values)
    print(f"      Log-likelihood: {hmm_full.log_likelihood:.4f}")

    posteriors = hmm_full.predict_proba(hmm_feat[HMM_FEATURES].values)
    raw_states, confidence, entropy = extract_regime_series(posteriors)

    # Data-driven state map: sort by actual observed forward return per state
    fwd_ret = np.log(df["close"].shift(-24) / df["close"]).reindex(hmm_feat.index).values
    state_map = build_state_map(
        hmm_full,
        raw_states=raw_states,
        forward_returns=fwd_ret,
    )

    regime_state = apply_state_map(raw_states, state_map)

    regime_s = pd.Series(regime_state, index=hmm_feat.index, name="regime_state")
    conf_s = pd.Series(confidence, index=hmm_feat.index, name="regime_confidence")
    ent_s = pd.Series(entropy, index=hmm_feat.index, name="regime_entropy")

    # Add regime features to ML features + target
    ml_feat_full = add_regime_features(ml_feat, regime_s, conf_s, ent_s)
    ml_feat_full = add_target(ml_feat_full, df["close"])

    if save:
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        # Save combined feature matrix (drop HMM cols already in ml_feat_full)
        hmm_unique_cols = [c for c in hmm_feat.columns if c not in ml_feat_full.columns]
        combined = hmm_feat[hmm_unique_cols].join(ml_feat_full, how="inner")
        combined.to_parquet(FEATURES_PATH)

        # Save regime series
        regime_df = pd.DataFrame({
            "regime_state": regime_s,
            "regime_confidence": conf_s,
            "regime_entropy": ent_s,
        })
        regime_df.to_parquet(REGIMES_PATH)
        print(f"      Saved features → {FEATURES_PATH.name}")
        print(f"      Saved regimes  → {REGIMES_PATH.name}")

    # Regime statistics — drop duplicate columns before joining
    hmm_unique = hmm_feat[[c for c in hmm_feat.columns if c not in ml_feat_full.columns]]
    feature_df = hmm_unique.join(ml_feat_full, how="inner")
    feature_df["regime_state"] = regime_s
    feature_df["regime_confidence"] = conf_s
    stats = compute_regime_stats(feature_df)
    print("\n      Regime Statistics:")
    print(stats[["regime_label", "count", "directional_accuracy", "mean_confidence", "mean_duration_bars"]]
          .to_string())

    # ── 4. Walk-forward ───────────────────────────────────────────────────────
    if not args.skip_wf:
        print(f"\n[4/4] Running walk-forward validation (folds={args.folds or 'all'})...")
        results = run_walk_forward(df, n_folds=args.folds, save_results=save)

        print(f"\n      Completed {len(results)} folds")
        valid = [r for r in results if not np.isnan(r.metrics.get("sharpe", np.nan))]
        if valid:
            sharpes = [r.metrics["sharpe"] for r in valid]
            calmars = [r.metrics["calmar"] for r in valid]
            print(f"      OOS Sharpe : mean={np.mean(sharpes):.2f}  std={np.std(sharpes):.2f}"
                  f"  positive={sum(s > 0 for s in sharpes)}/{len(sharpes)}")
            print(f"      OOS Calmar : mean={np.nanmean(calmars):.2f}  "
                  f"  > 1.0 in {sum(c > 1.0 for c in calmars if not np.isnan(c))}/{len(valid)} folds")
    else:
        print("\n[4/4] Skipped walk-forward (--skip-wf).")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
