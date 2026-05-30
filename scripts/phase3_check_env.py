"""Phase 3 (Section 1.6): sanity-check the DRL trading environment.

Builds a ``CryptoTradingEnv`` on the training split and runs Gymnasium's
``check_env`` to verify that the observation/action spaces, ``reset``, and
``step`` all conform to the Gymnasium API. Then runs a 100-step random rollout
and prints summary stats.

Usage
-----
    python -m scripts.phase3_check_env
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data.preprocess import FeatureScaler
from env.trading_env import CryptoTradingEnv
from utils.config import load_config, resolve_path
from utils.logging import get_logger
from utils.seeding import set_global_seed

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--interval", default=None)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.seed)
    symbol = args.symbol or cfg.data.symbol
    interval = args.interval or cfg.data.interval

    processed_dir = resolve_path(cfg, cfg.paths.processed_data_dir)
    train_df = pd.read_parquet(processed_dir / f"{symbol}_{interval}_train.parquet")
    scaler = FeatureScaler.load(resolve_path(cfg, cfg.paths.lstm_scaler))

    env = CryptoTradingEnv(raw_df=train_df, scaler=scaler, lstm_signals=None, cfg=cfg, seed=cfg.seed)

    from gymnasium.utils.env_checker import check_env
    log.info("Running Gymnasium env_checker...")
    check_env(env)
    log.info("env_checker OK")

    log.info("Running random rollout for %d steps", args.steps)
    obs, _ = env.reset()
    total_r = 0.0
    for i in range(args.steps):
        a = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(a)
        total_r += r
        if terminated or truncated:
            break
    log.info("Random rollout: %d steps, total reward=%.6f, final equity=%.2f",
             i + 1, total_r, info.get("equity", float("nan")))
    log.info("Observation shape: %s, action space: %s", obs.shape, env.action_space)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
