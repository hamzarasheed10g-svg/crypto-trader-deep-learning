"""Binance REST kline fetcher with pagination, retry, and CSV caching.

Implements Section 1.3 of the methodology. Binance limits kline responses to
1000 candles per request, so we paginate by ``startTime`` to assemble multi-year
histories.

The fetcher is intentionally framework-light: it depends only on ``httpx``
(synchronous client) so it can run inside scripts without an event loop. The
live WebSocket pipeline lives in ``data/stream.py``.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import pandas as pd

from utils.logging import get_logger

log = get_logger(__name__)

BINANCE_SPOT_REST = "https://api.binance.com"
BINANCE_TESTNET_REST = "https://testnet.binance.vision"

# Map of interval -> seconds (Binance kline intervals)
INTERVAL_TO_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600, "8h": 28800, "12h": 43200,
    "1d": 86400, "3d": 259200, "1w": 604800,
}

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "n_trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


def _http_get_with_retry(client, url: str, params: dict, max_retries: int = 5) -> list:
    """GET with exponential backoff for transient errors (429, 5xx, network)."""
    delay = 1.0
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            r = client.get(url, params=params, timeout=15.0)
            if r.status_code == 429:
                wait_s = float(r.headers.get("Retry-After", delay))
                log.warning("Rate limited; sleeping %.1fs", wait_s)
                time.sleep(wait_s)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            log.warning("Attempt %d failed: %s. Retrying in %.1fs", attempt + 1, exc, delay)
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    raise RuntimeError(f"Failed to fetch klines after {max_retries} attempts: {last_exc}")


def fetch_klines(
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime | None = None,
    use_testnet: bool = False,
) -> pd.DataFrame:
    """Fetch OHLCV klines from Binance between ``start`` and ``end`` (UTC).

    Returns a DataFrame indexed by UTC timestamp with float columns
    open/high/low/close/volume.
    """
    import httpx  # local import keeps the module import-light

    if interval not in INTERVAL_TO_SECONDS:
        raise ValueError(f"Unknown interval {interval!r}; choose from {list(INTERVAL_TO_SECONDS)}")
    if end is None:
        end = datetime.now(timezone.utc)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    base = BINANCE_TESTNET_REST if use_testnet else BINANCE_SPOT_REST
    url = f"{base}/api/v3/klines"

    all_rows: List[list] = []
    cursor_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    step_ms = INTERVAL_TO_SECONDS[interval] * 1000

    log.info("Fetching %s %s klines from %s to %s", symbol, interval, start, end)
    with httpx.Client(headers={"User-Agent": "crypto-trader/1.0"}) as client:
        while cursor_ms < end_ms:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor_ms,
                "limit": 1000,
            }
            batch = _http_get_with_retry(client, url, params)
            if not batch:
                break
            all_rows.extend(batch)
            last_open = batch[-1][0]
            next_cursor = last_open + step_ms
            if next_cursor <= cursor_ms:  # safety against infinite loops
                break
            cursor_ms = next_cursor
            if len(batch) < 1000:
                # Reached the live edge; nothing more to fetch.
                break

    if not all_rows:
        raise RuntimeError(f"No klines returned for {symbol} {interval}")

    df = pd.DataFrame(all_rows, columns=KLINE_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    keep = ["open", "high", "low", "close", "volume"]
    df = df[keep]
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df[df.index <= end]
    log.info("Fetched %d rows (%s to %s)", len(df), df.index[0], df.index[-1])
    return df


def fetch_history(
    symbol: str,
    interval: str,
    years: float,
    cache_path: Path | None = None,
    use_testnet: bool = False,
) -> pd.DataFrame:
    """High-level wrapper: fetch the last ``years`` of history with optional caching."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(years * 365))

    if cache_path is not None and cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
            cached.index = pd.to_datetime(cached.index, utc=True)
            if not cached.empty and cached.index[-1] >= end - timedelta(days=1):
                log.info("Using cached data at %s (%d rows)", cache_path, len(cached))
                return cached
        except Exception as exc:  # noqa: BLE001
            log.warning("Cache read failed (%s); refetching", exc)

    df = fetch_klines(symbol, interval, start, end, use_testnet=use_testnet)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
        log.info("Cached %d rows to %s", len(df), cache_path)
    return df
