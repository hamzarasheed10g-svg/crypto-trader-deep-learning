"""Binance Spot order executor with strict testnet default (Section 1.9 / 1.13).

We thinly wrap ``python-binance``'s ``Client`` so that:

- Default endpoints are the public **Spot Testnet** (https://testnet.binance.vision).
  Switching to mainnet requires ``allow_mainnet=True`` AND ``BINANCE_USE_MAINNET=true``
  in the environment. Both gates must be open simultaneously — refusing to do
  anything irreversible without a deliberate, redundant opt-in is the whole point.
- All orders are MARKET orders sized in *quote* (USDT) terms for BUYs and *base*
  (BTC) quantity for SELLs, matching Binance's spot REST contract.
- Failures are logged and surfaced; we never silently swallow exchange errors.

This is a paper-trading layer: nothing in this codebase encourages live mainnet
trading, and the methodology project explicitly targets testnet validation.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional

from utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class OrderResult:
    success: bool
    order_id: Optional[int]
    side: str
    qty: float
    price: float
    fills: list
    raw: dict
    error: Optional[str] = None


class BinanceExecutor:
    """Async-friendly order executor for the Binance Spot Testnet."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbol: str,
        use_testnet: bool = True,
        allow_mainnet: bool = False,
    ):
        if not use_testnet:
            if not allow_mainnet:
                raise RuntimeError(
                    "Refusing to instantiate a mainnet client without allow_mainnet=True. "
                    "This project is configured for paper trading."
                )
            if os.getenv("BINANCE_USE_MAINNET", "false").lower() != "true":
                raise RuntimeError(
                    "Set BINANCE_USE_MAINNET=true to enable mainnet trading. "
                    "Default behaviour is testnet only."
                )
            log.warning("MAINNET TRADING ENABLED — every order touches real funds.")

        from binance.client import Client  # python-binance
        self._client = Client(api_key=api_key, api_secret=api_secret, testnet=use_testnet)
        self.symbol = symbol.upper()
        self.use_testnet = use_testnet
        self._symbol_info_cache: Optional[dict] = None

    # ----------------------------------------------------------------- helpers

    def get_symbol_info(self) -> dict:
        if self._symbol_info_cache is None:
            info = self._client.get_symbol_info(self.symbol)
            if info is None:
                raise RuntimeError(f"Symbol {self.symbol} not found on Binance")
            self._symbol_info_cache = info
        return self._symbol_info_cache

    def _step_size(self) -> float:
        for f in self.get_symbol_info().get("filters", []):
            if f.get("filterType") == "LOT_SIZE":
                return float(f["stepSize"])
        return 1e-8

    def _tick_size(self) -> float:
        for f in self.get_symbol_info().get("filters", []):
            if f.get("filterType") == "PRICE_FILTER":
                return float(f["tickSize"])
        return 0.01

    def _round_qty(self, qty: float) -> float:
        step = self._step_size()
        if step <= 0:
            return qty
        # Round DOWN to the nearest step (Binance rejects anything finer)
        return math.floor(qty / step) * step

    # ----------------------------------------------------------------- account

    def get_balances(self) -> dict[str, float]:
        acc = self._client.get_account()
        out = {}
        for b in acc.get("balances", []):
            free = float(b["free"])
            if free > 0:
                out[b["asset"]] = free
        return out

    def get_last_price(self) -> float:
        ticker = self._client.get_symbol_ticker(symbol=self.symbol)
        return float(ticker["price"])

    # ----------------------------------------------------------------- orders

    def market_buy_quote(self, quote_qty_usdt: float) -> OrderResult:
        """Buy ``quote_qty_usdt`` USDT worth of the base asset (Binance ``quoteOrderQty``)."""
        try:
            resp = self._client.order_market_buy(
                symbol=self.symbol,
                quoteOrderQty=round(quote_qty_usdt, 2),
            )
            return self._parse(resp, side="BUY")
        except Exception as exc:  # noqa: BLE001
            log.error("market_buy_quote failed: %s", exc)
            return OrderResult(success=False, order_id=None, side="BUY", qty=0.0, price=0.0, fills=[], raw={}, error=str(exc))

    def market_sell_qty(self, base_qty: float) -> OrderResult:
        """Sell ``base_qty`` units of the base asset."""
        qty = self._round_qty(base_qty)
        if qty <= 0:
            return OrderResult(success=False, order_id=None, side="SELL", qty=0.0, price=0.0, fills=[], raw={}, error="qty rounded to zero")
        try:
            resp = self._client.order_market_sell(symbol=self.symbol, quantity=qty)
            return self._parse(resp, side="SELL")
        except Exception as exc:  # noqa: BLE001
            log.error("market_sell_qty failed: %s", exc)
            return OrderResult(success=False, order_id=None, side="SELL", qty=0.0, price=0.0, fills=[], raw={}, error=str(exc))

    @staticmethod
    def _parse(resp: dict, side: str) -> OrderResult:
        fills = resp.get("fills", [])
        if fills:
            total_qty = sum(float(f["qty"]) for f in fills)
            total_quote = sum(float(f["qty"]) * float(f["price"]) for f in fills)
            avg_price = total_quote / total_qty if total_qty > 0 else 0.0
        else:
            total_qty = float(resp.get("executedQty", 0.0))
            cumm_quote = float(resp.get("cummulativeQuoteQty", 0.0))
            avg_price = cumm_quote / total_qty if total_qty > 0 else 0.0
        return OrderResult(
            success=True,
            order_id=int(resp.get("orderId", -1)),
            side=side,
            qty=total_qty,
            price=avg_price,
            fills=fills,
            raw=resp,
        )
