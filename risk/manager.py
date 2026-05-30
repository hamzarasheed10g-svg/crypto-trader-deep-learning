"""Standalone risk manager (Section 1.9 of the methodology).

The trading env applies its own risk constraints during PPO training, but in
live paper trading we need a separate component that:

1. Validates each prospective order against portfolio safety limits
2. Watches existing positions and emits forced-close events on stop-loss /
   take-profit triggers
3. Halts trading if portfolio drawdown exceeds the maximum allowed

This module is intentionally framework-free (no torch / no gymnasium) so it can
be imported anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RiskAction(str, Enum):
    ALLOW = "allow"
    REJECT = "reject"
    FORCE_CLOSE = "force_close"
    HALT = "halt"


@dataclass
class RiskDecision:
    action: RiskAction
    reason: str = ""
    suggested_qty: Optional[float] = None


@dataclass
class RiskLimits:
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.05
    max_drawdown_pct: float = 0.15
    max_position_fraction: float = 1.0
    min_trade_notional: float = 10.0      # USDT
    cooldown_bars: int = 0


class RiskManager:
    """Enforces portfolio-level safety constraints around every order."""

    def __init__(self, limits: RiskLimits):
        self.limits = limits
        self._halted = False
        self._cooldown_remaining = 0
        self.peak_equity = 0.0

    @property
    def is_halted(self) -> bool:
        return self._halted

    def reset(self) -> None:
        self._halted = False
        self._cooldown_remaining = 0
        self.peak_equity = 0.0

    # ----------------------------------------------------------------- updates

    def update_equity(self, equity: float) -> RiskDecision:
        """Call once per bar with the current portfolio equity.

        Returns ``HALT`` if drawdown limit has been breached.
        """
        if equity > self.peak_equity:
            self.peak_equity = equity
        dd = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        if dd >= self.limits.max_drawdown_pct:
            self._halted = True
            return RiskDecision(RiskAction.HALT, reason=f"max_drawdown ({dd:.2%} >= {self.limits.max_drawdown_pct:.2%})")
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
        return RiskDecision(RiskAction.ALLOW)

    # ----------------------------------------------------------------- watches

    def check_open_position(self, avg_entry: float, current_price: float) -> RiskDecision:
        """Should we force-close the current long position?"""
        if avg_entry <= 0:
            return RiskDecision(RiskAction.ALLOW)
        move = (current_price - avg_entry) / avg_entry
        if move <= -self.limits.stop_loss_pct:
            self._cooldown_remaining = self.limits.cooldown_bars
            return RiskDecision(RiskAction.FORCE_CLOSE, reason=f"stop_loss ({move:.2%})")
        if move >= self.limits.take_profit_pct:
            self._cooldown_remaining = self.limits.cooldown_bars
            return RiskDecision(RiskAction.FORCE_CLOSE, reason=f"take_profit ({move:.2%})")
        return RiskDecision(RiskAction.ALLOW)

    # ----------------------------------------------------------------- gating

    def validate_buy(
        self,
        cash_available: float,
        equity: float,
        current_position_value: float,
        intended_notional: float,
    ) -> RiskDecision:
        """Check whether a proposed Buy order is allowed under the policy."""
        if self._halted:
            return RiskDecision(RiskAction.REJECT, reason="risk_halted")
        if self._cooldown_remaining > 0:
            return RiskDecision(RiskAction.REJECT, reason="cooldown")
        if intended_notional < self.limits.min_trade_notional:
            return RiskDecision(RiskAction.REJECT, reason=f"below_min_notional ({intended_notional:.2f})")
        if intended_notional > cash_available:
            intended_notional = cash_available
        # Check post-trade position fraction
        new_position_value = current_position_value + intended_notional
        if equity > 0 and new_position_value / equity > self.limits.max_position_fraction + 1e-9:
            allowable = max(0.0, self.limits.max_position_fraction * equity - current_position_value)
            if allowable < self.limits.min_trade_notional:
                return RiskDecision(RiskAction.REJECT, reason="max_position_fraction")
            return RiskDecision(RiskAction.ALLOW, reason="trimmed_to_limit", suggested_qty=allowable)
        return RiskDecision(RiskAction.ALLOW, suggested_qty=intended_notional)

    def validate_sell(self, position_qty: float) -> RiskDecision:
        if self._halted:
            # Permit closing positions even when halted (we want to exit).
            return RiskDecision(RiskAction.ALLOW, reason="halt_close")
        if position_qty <= 0:
            return RiskDecision(RiskAction.REJECT, reason="no_position")
        return RiskDecision(RiskAction.ALLOW)
