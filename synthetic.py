"""Synthetic OHLCV generator for offline development and tests.

Generates a geometric Brownian motion with stochastic volatility plus realistic
intra-bar high/low spreads. Output schema matches ``binance_rest.fetch_history``.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from data.binance_rest import INTERVAL_TO_SECONDS


def make_synthetic_ohlcv(
    n_bars: int = 2000,
    interval: str = "1h",
    start_price: float = 30000.0,
    annual_vol: float = 0.7,
    annual_drift: float = 0.05,
    seed: Optional[int] = 42,
    start_time: Optional[datetime] = None,
) -> pd.DataFrame:
    """Build a synthetic but realistic OHLCV time series."""
    rng = np.random.default_rng(seed)
    sec = INTERVAL_TO_SECONDS[interval]
    bars_per_year = 365.0 * 86400.0 / sec
    dt = 1.0 / bars_per_year
    mu = annual_drift
    sigma = annual_vol

    # Stochastic-vol multiplier: volatility-of-volatility ≈ 0.6, mean reverting to 1
    vol_mult = np.zeros(n_bars)
    vol_mult[0] = 1.0
    kappa = 0.05
    eta = 0.6
    for t in range(1, n_bars):
        vol_mult[t] = max(0.1, vol_mult[t - 1] + kappa * (1.0 - vol_mult[t - 1]) + eta * np.sqrt(dt) * rng.normal())

    shocks = rng.normal(size=n_bars)
    log_rets = (mu - 0.5 * (sigma * vol_mult) ** 2) * dt + sigma * vol_mult * np.sqrt(dt) * shocks
    close = start_price * np.exp(np.cumsum(log_rets))

    # Open = previous close (true for continuous-trading markets like crypto)
    open_ = np.concatenate([[start_price], close[:-1]])

    # Intra-bar high/low: scale by sqrt(2*dt)*sigma_t, expand symmetrically
    intrabar_range = np.abs(rng.normal(size=n_bars)) * sigma * vol_mult * np.sqrt(dt) * close
    high = np.maximum(open_, close) + intrabar_range * 0.5
    low = np.minimum(open_, close) - intrabar_range * 0.5
    low = np.clip(low, 1e-6, None)

    base_volume = 100.0
    volume = base_volume * (1 + 0.5 * np.abs(rng.normal(size=n_bars))) * vol_mult

    if start_time is None:
        start_time = datetime.now(timezone.utc) - timedelta(seconds=sec * n_bars)
    idx = pd.date_range(start=start_time, periods=n_bars, freq=pd.Timedelta(seconds=sec))

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    ).rename_axis("timestamp")
