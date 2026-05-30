"""Phase 1b (Section 1.4): preprocessing and feature engineering.

Reads ``data/raw/<SYMBOL>_<INTERVAL>.parquet``, runs the full pipeline (dedup,
synchronise, indicators, winsorise, split, normalise, sequence), and saves the
sequence arrays + the fitted scaler to ``data/processed/`` and ``artifacts/``.

Usage
-----
    python -m scripts.phase1_preprocess
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data.binance_rest import INTERVAL_TO_SECONDS
from data.preprocess import preprocess
from utils.config import load_config, resolve_path
from utils.logging import get_logger
from utils.seeding import set_global_seed

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--interval", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.seed)
    symbol = args.symbol or cfg.data.symbol
    interval = args.interval or cfg.data.interval

    raw_path = resolve_path(cfg, cfg.paths.raw_data_dir) / f"{symbol}_{interval}.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(f"Run scripts.phase1_fetch_data first; missing {raw_path}")
    log.info("Loading %s", raw_path)
    df = pd.read_parquet(raw_path)
    df.index = pd.to_datetime(df.index, utc=True)

    processed = preprocess(
        df,
        feature_list=list(cfg.preprocessing.feature_list),
        target_kind=cfg.lstm.prediction_target,
        sequence_length=cfg.lstm.sequence_length,
        horizon=cfg.lstm.prediction_horizon,
        train_frac=cfg.lstm.split.train,
        val_frac=cfg.lstm.split.val,
        normalize=cfg.preprocessing.normalize,
        outlier_z=cfg.preprocessing.outlier_z_threshold,
        interval_seconds=INTERVAL_TO_SECONDS[interval],
        fill=cfg.preprocessing.fill_method,
    )

    processed_dir = resolve_path(cfg, cfg.paths.processed_data_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Save sequences. If dual target, also persist the direction labels alongside
    # the regression targets so phase2 can load both heads' supervision.
    save_kwargs = dict(
        X_train=processed.X_train, y_train=processed.y_train,
        X_val=processed.X_val, y_val=processed.y_val,
        X_test=processed.X_test, y_test=processed.y_test,
    )
    if processed.target_name == "dual":
        save_kwargs.update(
            y_train_dir=processed.y_train_dir,
            y_val_dir=processed.y_val_dir,
            y_test_dir=processed.y_test_dir,
        )
    np.savez_compressed(
        processed_dir / f"{symbol}_{interval}_sequences.npz",
        **save_kwargs,
    )
    processed.train_df.to_parquet(processed_dir / f"{symbol}_{interval}_train.parquet")
    processed.val_df.to_parquet(processed_dir / f"{symbol}_{interval}_val.parquet")
    processed.test_df.to_parquet(processed_dir / f"{symbol}_{interval}_test.parquet")
    processed.raw.to_parquet(processed_dir / f"{symbol}_{interval}_features.parquet")

    scaler_path = resolve_path(cfg, cfg.paths.lstm_scaler)
    processed.scaler.save(scaler_path)

    log.info("Sequences saved to %s", processed_dir)
    log.info("Scaler saved to %s", scaler_path)
    log.info("Train/Val/Test sequence shapes: %s / %s / %s",
             processed.X_train.shape, processed.X_val.shape, processed.X_test.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
