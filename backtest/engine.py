"""Backtest engine (Section 1.12).

Runs an action-emitting strategy (or a trained PPO policy) through the standard
``CryptoTradingEnv`` and returns the resulting equity curve, trade log, and
metric bundle.

This guarantees apples-to-apples comparison: every strategy pays the same fees,
suffers the same slippage, and is bound by the same risk constraints.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from backtest.metrics import TradingMetrics, compute_metrics
from utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class BacktestResult:
    name: str
    equity_curve: np.ndarray
    trade_pnls: List[float]
    trades: List[dict]
    metrics: TradingMetrics
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "metrics": self.metrics.to_dict(),
            "n_bars": len(self.equity_curve),
            "n_trades": len(self.trades),
            **self.extras,
        }


def _trade_pnls_from_log(env_trade_log: List[Any]) -> List[float]:
    """FIFO matching of BUY/SELL pairs to per-trade PnL.

    The portfolio object stores fills in chronological order; we walk them and
    realise PnL whenever a SELL closes some (or all) of the open quantity.
    """
    pnls: List[float] = []
    open_lots: List[tuple[float, float]] = []  # (qty, price_after_fee_attribution)
    for t in env_trade_log:
        if t.side == "BUY":
            # Effective cost-basis per unit = (notional + fee) / qty; we want fees-in price.
            unit_cost = (t.notional + t.fee) / t.qty if t.qty > 0 else 0.0
            open_lots.append((t.qty, unit_cost))
        elif t.side == "SELL":
            remaining = t.qty
            proceeds_per_unit = (t.notional) / t.qty if t.qty > 0 else 0.0
            # The notional in our portfolio is already net of fee; that's what we receive per share.
            while remaining > 1e-12 and open_lots:
                lot_qty, lot_cost = open_lots[0]
                take = min(lot_qty, remaining)
                pnls.append((proceeds_per_unit - lot_cost) * take)
                remaining -= take
                lot_qty -= take
                if lot_qty <= 1e-12:
                    open_lots.pop(0)
                else:
                    open_lots[0] = (lot_qty, lot_cost)
    return pnls


# ---------------------------------------------------------------------------
# Strategy runner: applies a precomputed action vector to the env
# ---------------------------------------------------------------------------

def run_actions(env, actions: np.ndarray) -> tuple[np.ndarray, List[Any]]:
    """Drive ``env`` step-by-step using the given action sequence."""
    obs, _ = env.reset()
    done = False
    truncated = False
    step_idx = env.current_step
    while not (done or truncated):
        # ``actions`` is aligned with the raw_df index, but env step advances
        # to current_step+1 each call; we read the action at the *current* index.
        idx = env.current_step
        a = int(actions[idx]) if idx < len(actions) else 0
        obs, _r, done, truncated, _info = env.step(a)
    return env.equity_curve, env.portfolio.trade_log


# ---------------------------------------------------------------------------
# Policy runner: applies an SB3 policy
# ---------------------------------------------------------------------------

def run_policy(env, model, deterministic: bool = True, vec_env=None) -> tuple[np.ndarray, List[Any]]:
    """Drive ``env`` using an SB3 policy. ``vec_env`` is the VecNormalize wrapper if any."""
    obs, _ = env.reset()
    done = False
    truncated = False
    while not (done or truncated):
        if vec_env is not None:
            # The policy expects normalised observations
            obs_norm = vec_env.normalize_obs(np.expand_dims(obs, 0))
            action, _ = model.predict(obs_norm, deterministic=deterministic)
            a = int(np.asarray(action).reshape(-1)[0])
        else:
            action, _ = model.predict(obs, deterministic=deterministic)
            a = int(np.asarray(action).reshape(-1)[0])
        obs, _r, done, truncated, _info = env.step(a)
    return env.equity_curve, env.portfolio.trade_log


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def backtest_strategy(
    name: str,
    env_factory: Callable[[], Any],
    actions: np.ndarray,
    bars_per_year: float,
    risk_free: float = 0.0,
) -> BacktestResult:
    env = env_factory()
    equity, trade_log = run_actions(env, actions)
    pnls = _trade_pnls_from_log(trade_log)
    metrics = compute_metrics(equity, pnls, bars_per_year, risk_free)
    trades = [dict(timestamp=t.timestamp, side=t.side, price=t.price, qty=t.qty, fee=t.fee, notional=t.notional, reason=t.reason)
              for t in trade_log]
    return BacktestResult(name=name, equity_curve=equity, trade_pnls=pnls, trades=trades, metrics=metrics)


def backtest_policy(
    name: str,
    env_factory: Callable[[], Any],
    model,
    vec_env=None,
    bars_per_year: float = 8760,
    risk_free: float = 0.0,
    deterministic: bool = True,
) -> BacktestResult:
    env = env_factory()
    equity, trade_log = run_policy(env, model, deterministic=deterministic, vec_env=vec_env)
    pnls = _trade_pnls_from_log(trade_log)
    metrics = compute_metrics(equity, pnls, bars_per_year, risk_free)
    trades = [dict(timestamp=t.timestamp, side=t.side, price=t.price, qty=t.qty, fee=t.fee, notional=t.notional, reason=t.reason)
              for t in trade_log]
    return BacktestResult(name=name, equity_curve=equity, trade_pnls=pnls, trades=trades, metrics=metrics)
