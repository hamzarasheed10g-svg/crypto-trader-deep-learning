"""Tests for ``backtest.baselines``."""
import numpy as np
import pandas as pd

from backtest.baselines import BuyAndHold, Momentum, SMACrossover


def test_buy_and_hold_buys_on_first_bar():
    df = pd.DataFrame({"close": np.arange(50, dtype=float)})
    a = BuyAndHold().actions(df)
    assert a[0] == 1
    assert (a[1:] == 0).all()


def test_sma_crossover_emits_signals():
    # The crossover only fires when fast SMA *transitions* across slow SMA, so
    # we need data with a reversal — e.g. down then up.
    down = np.linspace(200, 100, 60)
    up = np.linspace(100, 250, 60)
    df = pd.DataFrame({"close": np.concatenate([down, up])})
    actions = SMACrossover(fast=5, slow=20).actions(df)
    assert (actions == 1).any(), "expected at least one buy after the up-reversal"
    # Reverse: up then down
    df2 = pd.DataFrame({"close": np.concatenate([up, down])})
    a2 = SMACrossover(fast=5, slow=20).actions(df2)
    assert (a2 == 2).any(), "expected at least one sell after the down-reversal"


def test_momentum_emits_signals():
    df = pd.DataFrame({"close": np.r_[np.linspace(100, 80, 30), np.linspace(80, 130, 30)]})
    a = Momentum(lookback=5).actions(df)
    # Should buy somewhere in the second half (uptrend) and sell in the first (downtrend)
    assert (a == 1).any() or (a == 2).any()
