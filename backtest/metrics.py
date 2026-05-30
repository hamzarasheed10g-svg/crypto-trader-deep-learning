"""Trading-evaluation metrics (Section 1.12 of the methodology).

All functions accept an ``equity_curve`` (1-d numpy array of portfolio equity
sampled at the bar frequency). The Sharpe / Sortino annualisation factor is
``sqrt(bars_per_year)``.

References
----------
- Sharpe (1994), Sortino & van der Meer (1991) for the ratios
- Magdon-Ismail & Atiya (2004) for max-drawdown definition
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def equity_to_returns(equity: np.ndarray) -> np.ndarray:
    """Per-bar simple returns from an equity curve. Length n-1."""
    eq = np.asarray(equity, dtype=np.float64)
    if len(eq) < 2:
        return np.array([], dtype=np.float64)
    return eq[1:] / eq[:-1] - 1.0


def log_equity_returns(equity: np.ndarray) -> np.ndarray:
    eq = np.asarray(equity, dtype=np.float64)
    if len(eq) < 2:
        return np.array([], dtype=np.float64)
    return np.log(eq[1:] / eq[:-1])


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def cumulative_return(equity: np.ndarray) -> float:
    eq = np.asarray(equity, dtype=np.float64)
    if len(eq) < 2 or eq[0] <= 0:
        return 0.0
    return float(eq[-1] / eq[0] - 1.0)


def annualised_return(equity: np.ndarray, bars_per_year: float) -> float:
    eq = np.asarray(equity, dtype=np.float64)
    if len(eq) < 2 or eq[0] <= 0:
        return 0.0
    n_bars = len(eq) - 1
    if n_bars <= 0:
        return 0.0
    total_return = eq[-1] / eq[0]
    years = n_bars / bars_per_year
    if years <= 0 or total_return <= 0:
        return 0.0
    return float(total_return ** (1.0 / years) - 1.0)


def annualised_volatility(equity: np.ndarray, bars_per_year: float) -> float:
    rets = equity_to_returns(equity)
    if len(rets) < 2:
        return 0.0
    return float(np.std(rets, ddof=1) * math.sqrt(bars_per_year))


def sharpe_ratio(equity: np.ndarray, bars_per_year: float, risk_free: float = 0.0) -> float:
    """Annualised Sharpe ratio (Sharpe, 1994)."""
    rets = equity_to_returns(equity)
    if len(rets) < 2:
        return 0.0
    rf_per_bar = risk_free / bars_per_year
    excess = rets - rf_per_bar
    std = np.std(excess, ddof=1)
    if std <= 1e-12:
        return 0.0
    return float(np.mean(excess) / std * math.sqrt(bars_per_year))


def sortino_ratio(equity: np.ndarray, bars_per_year: float, risk_free: float = 0.0) -> float:
    """Annualised Sortino ratio (Sortino & van der Meer, 1991)."""
    rets = equity_to_returns(equity)
    if len(rets) < 2:
        return 0.0
    rf_per_bar = risk_free / bars_per_year
    excess = rets - rf_per_bar
    downside = excess[excess < 0]
    if len(downside) < 1:
        return float("inf") if np.mean(excess) > 0 else 0.0
    downside_dev = math.sqrt(np.mean(downside ** 2))
    if downside_dev <= 1e-12:
        return 0.0
    return float(np.mean(excess) / downside_dev * math.sqrt(bars_per_year))


def max_drawdown(equity: np.ndarray) -> float:
    eq = np.asarray(equity, dtype=np.float64)
    if len(eq) == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.where(peak > 0, peak, 1.0)
    return float(np.max(dd))


def calmar_ratio(equity: np.ndarray, bars_per_year: float) -> float:
    mdd = max_drawdown(equity)
    if mdd <= 1e-12:
        return 0.0
    return float(annualised_return(equity, bars_per_year) / mdd)


def win_rate_and_profit_factor(trade_pnls: Sequence[float]) -> tuple[float, float]:
    """Trade-level win rate and profit factor.

    profit_factor = sum(positive PnL) / |sum(negative PnL)|
    """
    if len(trade_pnls) == 0:
        return 0.0, 0.0
    arr = np.asarray(trade_pnls, dtype=np.float64)
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    win_rate = float(len(wins) / len(arr))
    gross_profit = float(np.sum(wins)) if len(wins) else 0.0
    gross_loss = float(-np.sum(losses)) if len(losses) else 0.0
    if gross_loss <= 1e-12:
        profit_factor = float("inf") if gross_profit > 0 else 0.0
    else:
        profit_factor = gross_profit / gross_loss
    return win_rate, profit_factor


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

@dataclass
class TradingMetrics:
    cumulative_return: float
    annualised_return: float
    annualised_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float
    profit_factor: float
    num_trades: int

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Replace inf with a string for JSON safety
        for k, v in list(d.items()):
            if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
                d[k] = ("inf" if math.isinf(v) else "nan")
        return d


def compute_metrics(
    equity: np.ndarray,
    trade_pnls: Iterable[float],
    bars_per_year: float,
    risk_free: float = 0.0,
) -> TradingMetrics:
    trade_pnls_list = list(trade_pnls)
    wr, pf = win_rate_and_profit_factor(trade_pnls_list)
    return TradingMetrics(
        cumulative_return=cumulative_return(equity),
        annualised_return=annualised_return(equity, bars_per_year),
        annualised_volatility=annualised_volatility(equity, bars_per_year),
        sharpe=sharpe_ratio(equity, bars_per_year, risk_free),
        sortino=sortino_ratio(equity, bars_per_year, risk_free),
        max_drawdown=max_drawdown(equity),
        calmar=calmar_ratio(equity, bars_per_year),
        win_rate=wr,
        profit_factor=pf,
        num_trades=len(trade_pnls_list),
    )
