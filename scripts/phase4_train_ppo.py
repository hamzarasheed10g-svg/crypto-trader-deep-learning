"""Phase 4 + 5 (Sections 1.7, 1.8, 1.11): hybrid PPO training.

Loads the trained LSTM, precomputes its predictions over the train and
validation splits (this is the hybrid integration from §1.8), then trains a
PPO agent in the trading env with these signals injected into the state.

Usage
-----
    python -m scripts.phase4_train_ppo
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from agents.ppo import train_ppo
from data.preprocess import FeatureScaler
from env.trading_env import LSTMSignalProvider
from models.train_lstm import load_lstm_checkpoint, resolve_device
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

    processed_dir = resolve_path(cfg, cfg.paths.processed_data_dir)
    train_df = pd.read_parquet(processed_dir / f"{symbol}_{interval}_train.parquet")
    val_df = pd.read_parquet(processed_dir / f"{symbol}_{interval}_val.parquet")
    scaler = FeatureScaler.load(resolve_path(cfg, cfg.paths.lstm_scaler))

    log.info("Loading LSTM forecaster")
    device = resolve_device(cfg.device)
    lstm_model, lstm_cfg = load_lstm_checkpoint(resolve_path(cfg, cfg.paths.lstm_model), device=device)
    provider = LSTMSignalProvider(lstm_model, scaler, sequence_length=cfg.lstm.sequence_length, device=device)

    log.info("Precomputing LSTM signals over %d train / %d val bars", len(train_df), len(val_df))
    train_signals = provider.predict_all(train_df)
    val_signals = provider.predict_all(val_df)

    save_path = resolve_path(cfg, cfg.paths.ppo_model)
    vecnorm_path = resolve_path(cfg, cfg.paths.ppo_vecnorm)
    train_ppo(
        cfg=cfg,
        train_raw=train_df,
        val_raw=val_df,
        scaler=scaler,
        train_lstm_signals=train_signals,
        val_lstm_signals=val_signals,
        save_path=save_path,
        vecnorm_path=vecnorm_path,
    )
    log.info("PPO training complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
