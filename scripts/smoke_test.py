"""End-to-end smoke test on synthetic data — no network, no API keys.

Runs every phase of the methodology against generated OHLCV data with reduced
hyperparameters so the whole pipeline completes in a few minutes on CPU:

    Phase 1   -> generate synthetic OHLCV
    Phase 1b  -> preprocess + sequence
    Phase 2   -> train a tiny LSTM (few epochs)
    Phase 3   -> Gymnasium env check + random rollout
    Phase 4   -> precompute LSTM signals, train a tiny PPO
    Phase 6   -> backtest PPO + baselines
    Phase 7-stub -> instantiate InferenceService and run one synthetic inference call

If this script completes cleanly, the wiring is correct and the only remaining
variable is *training quality*, which depends on real data + longer training.

Usage
-----
    python -m scripts.smoke_test
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from utils.logging import get_logger
from utils.seeding import set_global_seed

log = get_logger(__name__)


def _make_tiny_config(tmpdir: Path) -> SimpleNamespace:
    """Build an in-memory cfg with shrunk hyperparameters for fast smoke runs."""
    from utils.config import load_config
    cfg = load_config()  # start from default

    # Smaller everything
    cfg.lstm.hidden_size = 32
    cfg.lstm.num_layers = 1
    cfg.lstm.sequence_length = 30
    cfg.env.window_size = 30
    cfg.lstm.training.epochs = 3
    cfg.lstm.training.batch_size = 32
    cfg.lstm.training.early_stopping_patience = 2

    cfg.ppo.total_timesteps = 4096
    cfg.ppo.n_steps = 512
    cfg.ppo.batch_size = 64
    cfg.ppo.eval_freq = 1024
    cfg.ppo.n_eval_episodes = 1

    # Point all artifact paths into the tmpdir
    cfg.paths.raw_data_dir = str(tmpdir / "raw")
    cfg.paths.processed_data_dir = str(tmpdir / "processed")
    cfg.paths.artifacts_dir = str(tmpdir / "artifacts")
    cfg.paths.lstm_model = str(tmpdir / "artifacts" / "lstm_model.pt")
    cfg.paths.lstm_scaler = str(tmpdir / "artifacts" / "lstm_scaler.pkl")
    cfg.paths.ppo_model = str(tmpdir / "artifacts" / "ppo_model.zip")
    cfg.paths.ppo_vecnorm = str(tmpdir / "artifacts" / "ppo_vecnorm.pkl")
    cfg.paths.backtest_report = str(tmpdir / "artifacts" / "backtest_report.html")
    cfg.paths.metrics_json = str(tmpdir / "artifacts" / "metrics.json")
    cfg._project_root = str(tmpdir)  # make resolve_path treat tmpdir as project root

    return cfg


def main() -> int:
    set_global_seed(42)
    tmpdir = Path(tempfile.mkdtemp(prefix="crypto_smoke_"))
    log.info("Smoke test working directory: %s", tmpdir)
    cfg = _make_tiny_config(tmpdir)

    # ---- Phase 1: synthetic data ----
    from data.synthetic import make_synthetic_ohlcv
    log.info("Phase 1: generating 1500 synthetic bars")
    df = make_synthetic_ohlcv(n_bars=1500, interval="1h", seed=42)
    raw_dir = Path(cfg.paths.raw_data_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{cfg.data.symbol}_{cfg.data.interval}.parquet"
    df.to_parquet(raw_path)

    # ---- Phase 1b: preprocess ----
    from data.preprocess import preprocess
    from data.binance_rest import INTERVAL_TO_SECONDS
    log.info("Phase 1b: preprocessing")
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
        interval_seconds=INTERVAL_TO_SECONDS[cfg.data.interval],
        fill=cfg.preprocessing.fill_method,
    )
    processed_dir = Path(cfg.paths.processed_data_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed.train_df.to_parquet(processed_dir / f"{cfg.data.symbol}_{cfg.data.interval}_train.parquet")
    processed.val_df.to_parquet(processed_dir / f"{cfg.data.symbol}_{cfg.data.interval}_val.parquet")
    processed.test_df.to_parquet(processed_dir / f"{cfg.data.symbol}_{cfg.data.interval}_test.parquet")
    Path(cfg.paths.lstm_scaler).parent.mkdir(parents=True, exist_ok=True)
    processed.scaler.save(cfg.paths.lstm_scaler)

    # ---- Phase 2: train LSTM ----
    from models.train_lstm import train_lstm, evaluate_lstm
    log.info("Phase 2: training tiny LSTM")
    model, history, loss_kind = train_lstm(
        cfg, processed.X_train, processed.y_train,
        processed.X_val, processed.y_val,
        save_path=cfg.paths.lstm_model,
    )
    test_metrics = evaluate_lstm(model, processed.X_test, processed.y_test, loss_kind=loss_kind)
    log.info("LSTM test metrics: %s", test_metrics)

    # ---- Phase 3: env check ----
    from env.trading_env import CryptoTradingEnv, LSTMSignalProvider
    from gymnasium.utils.env_checker import check_env
    import torch
    device = torch.device("cpu")
    log.info("Phase 3: gymnasium env_checker")
    env = CryptoTradingEnv(processed.train_df, processed.scaler, None, cfg, seed=42)
    check_env(env)

    # ---- Phase 4: PPO training ----
    from agents.ppo import train_ppo, load_ppo
    provider = LSTMSignalProvider(model, processed.scaler, cfg.lstm.sequence_length, device=device)
    train_sig = provider.predict_all(processed.train_df)
    val_sig = provider.predict_all(processed.val_df)
    log.info("Phase 4: training tiny PPO")
    train_ppo(
        cfg=cfg,
        train_raw=processed.train_df,
        val_raw=processed.val_df,
        scaler=processed.scaler,
        train_lstm_signals=train_sig,
        val_lstm_signals=val_sig,
        save_path=cfg.paths.ppo_model,
        vecnorm_path=cfg.paths.ppo_vecnorm,
    )

    # ---- Phase 6: backtest ----
    from backtest.engine import backtest_policy, backtest_strategy
    from backtest.baselines import build_baselines

    test_sig = provider.predict_all(processed.test_df)
    ppo_model, vec_env = load_ppo(cfg.paths.ppo_model, cfg.paths.ppo_vecnorm,
                                  processed.test_df, processed.scaler, test_sig, cfg)
    log.info("Phase 6: backtesting policy + baselines")

    def env_factory():
        return CryptoTradingEnv(processed.test_df, processed.scaler, test_sig, cfg, seed=42)

    results = [
        backtest_policy("lstm_ppo", env_factory, ppo_model, vec_env=vec_env,
                        bars_per_year=cfg.backtest.bars_per_year),
    ]
    for strat in build_baselines(cfg):
        actions = strat.actions(processed.test_df)
        results.append(backtest_strategy(
            strat.name,
            lambda a=actions: CryptoTradingEnv(processed.test_df, processed.scaler, None, cfg, seed=42),
            actions, bars_per_year=cfg.backtest.bars_per_year,
        ))

    log.info("Backtest results:")
    for r in results:
        m = r.metrics
        log.info("  %-15s cum=%+.2f%% sharpe=%.2f maxDD=%.2f%% trades=%d",
                 r.name, m.cumulative_return*100, m.sharpe, m.max_drawdown*100, m.num_trades)

    # ---- Phase 7 stub: inference service ----
    from backend.inference import InferenceService
    log.info("Phase 7 stub: running inference service on a synthetic window")
    svc = InferenceService(cfg)
    svc.load()
    out = svc.infer_window(processed.test_df.iloc[-(cfg.env.window_size + 100):])
    log.info("Inference: %s", out)
    assert out["action"] in ("HOLD", "BUY", "SELL")
    assert out["latency_ms"] >= 0

    log.info("SMOKE TEST PASSED")
    log.info("Artifacts retained at: %s", tmpdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
