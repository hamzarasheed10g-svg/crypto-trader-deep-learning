"""Baseline trading strategies for comparison (Section 1.12).

All baselines emit a series of discrete actions (0=Hold, 1=Buy, 2=Sell) that
plug into the same backtest engine as the PPO policy. This guarantees an
apples-to-apples comparison: identical fees, slippage, risk constraints, and
equity-curve sampling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


class Strategy(Protocol):
    name: str
    def actions(self, df: pd.DataFrame) -> np.ndarray: ...


@dataclass
class BuyAndHold:
    name: str = "buy_and_hold"

    def actions(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        a = np.zeros(n, dtype=np.int64)
        a[0] = 1            # buy on bar 0
        # rest is hold; the backtester naturally sells at the end
        return a


@dataclass
class SMACrossover:
    """Classic moving-average crossover.

    Buy when fast SMA crosses above slow SMA, sell when it crosses back below.
    """
    fast: int = 20
    slow: int = 50
    name: str = "sma_crossover"

    def actions(self, df: pd.DataFrame) -> np.ndarray:
        close = df["close"].astype(float)
        fast = close.rolling(self.fast, min_periods=self.fast).mean()
        slow = close.rolling(self.slow, min_periods=self.slow).mean()
        prev_fast = fast.shift(1)
        prev_slow = slow.shift(1)
        cross_up = (prev_fast <= prev_slow) & (fast > slow)
        cross_down = (prev_fast >= prev_slow) & (fast < slow)
        n = len(df)
        a = np.zeros(n, dtype=np.int64)
        a[cross_up.fillna(False).values] = 1
        a[cross_down.fillna(False).values] = 2
        return a


@dataclass
class Momentum:
    """Long when k-bar momentum is positive, flat when negative."""
    lookback: int = 14
    name: str = "momentum"

    def actions(self, df: pd.DataFrame) -> np.ndarray:
        close = df["close"].astype(float)
        mom = close.pct_change(self.lookback)
        prev = mom.shift(1)
        long_signal = (mom > 0) & (prev <= 0)
        short_signal = (mom < 0) & (prev >= 0)
        n = len(df)
        a = np.zeros(n, dtype=np.int64)
        a[long_signal.fillna(False).values] = 1
        a[short_signal.fillna(False).values] = 2
        return a


def build_baselines(cfg) -> list[Strategy]:
    names = list(cfg.backtest.baselines)
    out: list[Strategy] = []
    for n in names:
        if n == "buy_and_hold":
            out.append(BuyAndHold())
        elif n == "sma_crossover":
            out.append(SMACrossover(fast=cfg.backtest.sma_fast, slow=cfg.backtest.sma_long))
        elif n == "momentum":
            out.append(Momentum(lookback=cfg.backtest.momentum_lookback))
        else:
            raise ValueError(f"Unknown baseline {n!r}")
    return out
