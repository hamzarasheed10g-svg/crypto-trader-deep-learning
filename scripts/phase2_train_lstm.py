"""Phase 2 (Section 1.5): train the LSTM price/return forecaster.

Loads the sequence arrays produced by ``phase1_preprocess`` and trains either:

- A single-head ``LSTMForecaster`` (legacy ``log_return``/``close``/``direction``
  targets), or
- A dual-head ``DualHeadLSTMForecaster`` (``prediction_target: dual``) trained
  with a combined MSE + BCE loss. This is the recommended setting and exposes
  both a magnitude forecast and a calibrated directional probability at
  inference time.

Saves the best-validation checkpoint to ``artifacts/lstm_model.pt`` and the
full training history (per-epoch losses + metrics + final test metrics) to
``artifacts/lstm_history.json``.

Usage
-----
    python -m scripts.phase2_train_lstm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from models.train_lstm import (
    evaluate_lstm, evaluate_lstm_dual,
    train_lstm, train_lstm_dual,
)
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
    seq_path = processed_dir / f"{symbol}_{interval}_sequences.npz"
    if not seq_path.exists():
        raise FileNotFoundError(f"Run scripts.phase1_preprocess first; missing {seq_path}")

    log.info("Loading sequences from %s", seq_path)
    blob = np.load(seq_path)
    X_train, y_train = blob["X_train"], blob["y_train"]
    X_val, y_val = blob["X_val"], blob["y_val"]
    X_test, y_test = blob["X_test"], blob["y_test"]

    save_path = resolve_path(cfg, cfg.paths.lstm_model)
    artifacts_dir = resolve_path(cfg, cfg.paths.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    is_dual = cfg.lstm.prediction_target == "dual"

    if is_dual:
        if "y_train_dir" not in blob.files:
            raise RuntimeError(
                "Config sets prediction_target=dual but the sequence file "
                "does not contain direction targets. Re-run "
                "scripts.phase1_preprocess after updating the config."
            )
        y_train_dir = blob["y_train_dir"]
        y_val_dir = blob["y_val_dir"]
        y_test_dir = blob["y_test_dir"]

        model, history = train_lstm_dual(
            cfg, X_train, y_train, y_train_dir,
            X_val, y_val, y_val_dir,
            save_path=save_path,
        )
        log.info("Evaluating dual-head model on held-out test set")
        test_metrics = evaluate_lstm_dual(model, X_test, y_test, y_test_dir)
        loss_kind = "dual"
    else:
        model, history, loss_kind = train_lstm(
            cfg, X_train, y_train, X_val, y_val, save_path=save_path,
        )
        log.info("Evaluating single-head model on held-out test set")
        test_metrics = evaluate_lstm(model, X_test, y_test, loss_kind=loss_kind)

    log.info("Test metrics: %s", test_metrics)

    history_path = artifacts_dir / "lstm_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump({
            "train_loss": history.train_loss,
            "val_loss": history.val_loss,
            "val_metrics": history.val_metrics,
            "best_epoch": history.best_epoch,
            "best_val_loss": history.best_val_loss,
            "test_metrics": test_metrics,
            "loss_kind": loss_kind,
            "prediction_target": cfg.lstm.prediction_target,
        }, f, indent=2, default=float)
    log.info("Saved training history to %s", history_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
