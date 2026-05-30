"""Binance WebSocket kline stream (Section 1.3 — real-time data ingestion).

Provides ``BinanceKlineStream``: an async iterator that yields completed klines
(``kline['x'] == True``) from Binance's combined stream endpoint.

The stream is robust to disconnects: it reconnects with exponential backoff,
and drops duplicate klines based on ``open_time``.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import AsyncIterator

from utils.logging import get_logger

log = get_logger(__name__)

BINANCE_WS_BASE = "wss://stream.binance.com:9443/ws"
BINANCE_TESTNET_WS_BASE = "wss://stream.testnet.binance.vision/ws"


@dataclass
class Kline:
    """Closed-kline record yielded by ``BinanceKlineStream``."""

    symbol: str
    interval: str
    open_time_ms: int
    close_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    @classmethod
    def from_payload(cls, payload: dict) -> "Kline":
        k = payload["k"]
        return cls(
            symbol=k["s"],
            interval=k["i"],
            open_time_ms=int(k["t"]),
            close_time_ms=int(k["T"]),
            open=float(k["o"]),
            high=float(k["h"]),
            low=float(k["l"]),
            close=float(k["c"]),
            volume=float(k["v"]),
        )


class BinanceKlineStream:
    """Subscribe to ``<symbol>@kline_<interval>`` and yield only *closed* klines."""

    def __init__(self, symbol: str, interval: str, use_testnet: bool = False):
        self.symbol = symbol.lower()
        self.interval = interval
        base = BINANCE_TESTNET_WS_BASE if use_testnet else BINANCE_WS_BASE
        self.url = f"{base}/{self.symbol}@kline_{self.interval}"
        self._seen_open_times: set[int] = set()

    async def __aiter__(self) -> AsyncIterator[Kline]:
        import websockets  # local import: ws-only dependency

        backoff = 1.0
        while True:
            try:
                async with websockets.connect(self.url, ping_interval=30, ping_timeout=20) as ws:
                    log.info("WebSocket connected: %s", self.url)
                    backoff = 1.0
                    async for raw in ws:
                        try:
                            payload = json.loads(raw)
                        except json.JSONDecodeError:
                            log.warning("Could not parse WS message: %r", raw[:200])
                            continue
                        if payload.get("e") != "kline":
                            continue
                        k = payload["k"]
                        if not k.get("x", False):
                            continue  # not closed yet
                        if k["t"] in self._seen_open_times:
                            continue
                        self._seen_open_times.add(int(k["t"]))
                        yield Kline.from_payload(payload)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("WebSocket error (%s); reconnecting in %.1fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)


async def stream_klines(symbol: str, interval: str, use_testnet: bool = False) -> AsyncIterator[Kline]:
    """Convenience wrapper to use ``async for`` directly on a function call."""
    stream = BinanceKlineStream(symbol, interval, use_testnet=use_testnet)
    async for k in stream:
        yield k


def kline_to_record(k: Kline) -> dict:
    return {
        "timestamp": k.open_time_ms / 1000.0,
        "open": k.open,
        "high": k.high,
        "low": k.low,
        "close": k.close,
        "volume": k.volume,
        "received_at": time.time(),
    }
