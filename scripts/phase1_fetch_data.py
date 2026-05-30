"""Phase 1 (Section 1.3): fetch raw historical OHLCV from Binance.

Usage
-----
    python -m scripts.phase1_fetch_data
    python -m scripts.phase1_fetch_data --symbol BTCUSDT --interval 1h --years 3
    python -m scripts.phase1_fetch_data --synthetic           # offline mode

Output
------
    data/raw/<SYMBOL>_<INTERVAL>.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

from utils.config import load_config, resolve_path
from utils.logging import get_logger
from utils.seeding import set_global_seed

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to YAML config (defaults to configs/default.yaml)")
    parser.add_argument("--symbol", default=None, help="Override cfg.data.symbol")
    parser.add_argument("--interval", default=None, help="Override cfg.data.interval (e.g. 1h)")
    parser.add_argument("--years", type=float, default=None, help="Override cfg.data.years_of_history")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic data instead of fetching from Binance")
    parser.add_argument("--testnet", action="store_true", help="Hit Binance testnet REST endpoints instead of mainnet")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.seed)
    symbol = args.symbol or cfg.data.symbol
    interval = args.interval or cfg.data.interval
    years = args.years if args.years is not None else cfg.data.years_of_history

    raw_dir = resolve_path(cfg, cfg.paths.raw_data_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / f"{symbol}_{interval}.parquet"

    if args.synthetic:
        from data.synthetic import make_synthetic_ohlcv
        from data.binance_rest import INTERVAL_TO_SECONDS
        secs = INTERVAL_TO_SECONDS[interval]
        n_bars = max(int(years * 365 * 86400 / secs), 2000)
        log.info("Generating %d synthetic bars (%s %s)", n_bars, symbol, interval)
        df = make_synthetic_ohlcv(n_bars=n_bars, interval=interval, seed=cfg.seed)
    else:
        from data.binance_rest import fetch_history
        log.info("Fetching %.2f years of %s %s from Binance (%s)",
                 years, symbol, interval, "testnet" if args.testnet else "mainnet")
        df = fetch_history(symbol=symbol, interval=interval, years=years,
                           cache_path=out_path, use_testnet=args.testnet)

    if not args.synthetic and out_path.exists():
        log.info("Already cached at %s (%d rows)", out_path, len(df))
        return 0

    df.to_parquet(out_path)
    log.info("Wrote %d rows to %s", len(df), out_path)
    log.info("Range: %s -> %s", df.index[0], df.index[-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
