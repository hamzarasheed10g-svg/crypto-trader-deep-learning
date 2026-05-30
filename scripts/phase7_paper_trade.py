"""Phase 7 (Section 1.13): live paper trading on the Binance testnet.

Connects to Binance's WebSocket kline feed, runs the LSTM+PPO inference pipeline
on each new closed bar, and either simulates fills locally (paper mode) or sends
market orders to the Spot Testnet (live mode).

Usage
-----
    # Pure paper mode (no API keys needed, no network orders):
    python -m scripts.phase7_paper_trade --paper

    # Send real testnet orders (requires .env keys):
    python -m scripts.phase7_paper_trade

Stop with Ctrl-C; the trader will flush a final state and exit cleanly.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import signal
from pathlib import Path

from dotenv import load_dotenv

from agents.ppo import load_ppo
from data.preprocess import FeatureScaler
from deploy.live_trader import LiveTrader, state_to_dict
from models.train_lstm import load_lstm_checkpoint, resolve_device
from utils.config import load_config, resolve_path
from utils.logging import get_logger
from utils.seeding import set_global_seed

log = get_logger(__name__)


async def _main(cfg, args) -> int:
    load_dotenv()
    set_global_seed(cfg.seed)

    log.info("Loading models for live trading")
    device = resolve_device(cfg.device)
    lstm_model, _ = load_lstm_checkpoint(resolve_path(cfg, cfg.paths.lstm_model), device=device)
    scaler = FeatureScaler.load(resolve_path(cfg, cfg.paths.lstm_scaler))

    # We need a non-empty raw_df to wire VecNormalize back together; the live
    # trader uses its own observation construction, so this is just scaffolding.
    import pandas as pd
    processed_dir = resolve_path(cfg, cfg.paths.processed_data_dir)
    sample_parquet = processed_dir / f"{cfg.data.symbol}_{cfg.data.interval}_test.parquet"
    if not sample_parquet.exists():
        raise FileNotFoundError(f"Run earlier phases first; missing {sample_parquet}")
    dummy_df = pd.read_parquet(sample_parquet)
    ppo_model, vec_env = load_ppo(
        resolve_path(cfg, cfg.paths.ppo_model),
        resolve_path(cfg, cfg.paths.ppo_vecnorm),
        dummy_df, scaler, None, cfg,
    )

    executor = None
    if not args.paper:
        api_key = os.getenv("BINANCE_TESTNET_API_KEY")
        api_secret = os.getenv("BINANCE_TESTNET_API_SECRET")
        if not api_key or not api_secret:
            raise SystemExit("Missing BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET in .env "
                             "(or pass --paper to run without exchange orders)")
        from deploy.binance_executor import BinanceExecutor
        executor = BinanceExecutor(api_key=api_key, api_secret=api_secret,
                                   symbol=cfg.data.symbol, use_testnet=True)
        log.info("Connected to Binance Spot Testnet with API key %s...", api_key[:6])

    async def on_state(s) -> None:
        # In paper-trade script we just log; the FastAPI backend has a richer fan-out.
        if s.bar_count % cfg.deploy.log_every_n_bars == 0:
            log.info("STATE %s", state_to_dict(s))

    trader = LiveTrader(
        cfg=cfg,
        lstm_model=lstm_model,
        ppo_model=ppo_model,
        ppo_vecnorm=vec_env,
        scaler=scaler,
        executor=executor,
        on_state_change=on_state,
        use_testnet=True,
    )

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _on_signal() -> None:
        log.info("Stop requested")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _on_signal)
        except NotImplementedError:
            pass  # Windows

    run_task = asyncio.create_task(trader.run())
    stop_task = asyncio.create_task(stop_event.wait())
    done, pending = await asyncio.wait({run_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass

    log.info("Final state: %s", state_to_dict(trader.state))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--paper", action="store_true",
                        help="Pure paper mode — no API keys / no exchange orders")
    args = parser.parse_args()
    cfg = load_config(args.config)
    return asyncio.run(_main(cfg, args))


if __name__ == "__main__":
    raise SystemExit(main())
