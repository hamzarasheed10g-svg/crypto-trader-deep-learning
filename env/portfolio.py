"""Portfolio state container shared between the trading env and the live executor."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Trade:
    timestamp: int          # bar index or unix-ms in live trading
    side: str               # "BUY" or "SELL"
    price: float
    qty: float
    fee: float
    notional: float
    reason: str = ""        # e.g. "agent", "stop_loss", "take_profit"


@dataclass
class Portfolio:
    """Simple long-only cash + position tracker.

    The PPO env keeps the agent long-only (Buy / Sell / Hold) — this matches the
    methodology's discrete action space. Shorting is not supported in this lab
    project but would slot in cleanly here.
    """

    cash: float
    position_qty: float = 0.0
    avg_entry_price: float = 0.0
    realized_pnl: float = 0.0
    trade_log: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    peak_equity: float = 0.0

    @property
    def is_long(self) -> bool:
        return self.position_qty > 1e-12

    @property
    def is_flat(self) -> bool:
        return not self.is_long

    def equity(self, price: float) -> float:
        return self.cash + self.position_qty * price

    def drawdown(self, price: float) -> float:
        """Current peak-to-current drawdown (positive number, e.g. 0.07 = 7%)."""
        eq = self.equity(price)
        if self.peak_equity <= 0:
            self.peak_equity = eq
        if eq > self.peak_equity:
            self.peak_equity = eq
            return 0.0
        return (self.peak_equity - eq) / self.peak_equity if self.peak_equity > 0 else 0.0

    def record(self, price: float) -> float:
        eq = self.equity(price)
        self.equity_curve.append(eq)
        if eq > self.peak_equity:
            self.peak_equity = eq
        return eq

    # ---------- order helpers ----------

    def market_buy(self, price: float, notional: float, fee_rate: float, ts: int = 0, reason: str = "") -> Trade | None:
        """Buy ``notional`` USDT worth (gross). Fee deducted from cash separately."""
        if notional <= 0 or self.cash <= 0:
            return None
        spend = min(notional, self.cash)
        fee = spend * fee_rate
        if fee >= spend:
            return None
        invest = spend - fee
        qty = invest / price if price > 0 else 0.0
        if qty <= 0:
            return None
        # weighted-average entry
        new_qty = self.position_qty + qty
        if new_qty > 0:
            self.avg_entry_price = (self.position_qty * self.avg_entry_price + qty * price) / new_qty
        self.position_qty = new_qty
        self.cash -= spend
        trade = Trade(timestamp=ts, side="BUY", price=price, qty=qty, fee=fee, notional=invest, reason=reason)
        self.trade_log.append(trade)
        return trade

    def market_sell(self, price: float, fraction: float, fee_rate: float, ts: int = 0, reason: str = "") -> Trade | None:
        """Sell ``fraction`` of the current position. ``fraction`` in (0, 1]."""
        if self.position_qty <= 1e-12 or fraction <= 0:
            return None
        fraction = min(fraction, 1.0)
        qty = self.position_qty * fraction
        proceeds = qty * price
        fee = proceeds * fee_rate
        net = proceeds - fee
        pnl = (price - self.avg_entry_price) * qty - fee
        self.realized_pnl += pnl
        self.position_qty -= qty
        self.cash += net
        if self.position_qty < 1e-12:
            self.position_qty = 0.0
            self.avg_entry_price = 0.0
        trade = Trade(timestamp=ts, side="SELL", price=price, qty=qty, fee=fee, notional=net, reason=reason)
        self.trade_log.append(trade)
        return trade
