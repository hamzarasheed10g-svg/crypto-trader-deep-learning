"""Gymnasium trading environment (Sections 1.6, 1.8, 1.9 of the methodology).

State (per Section 1.6.1)
------------------------
- Window of scaled features over the past ``window_size`` bars
- LSTM-generated prediction for the next bar (injected into state space per §1.8)
- Portfolio info: cash fraction, position fraction, unrealised PnL, drawdown

Actions
-------
Discrete: 0 = Hold, 1 = Buy (open or add to long), 2 = Sell (close long)

Reward
------
Configurable: differential Sharpe ratio (Moody & Saffell, 2001), raw PnL, or
log-return of equity. Drawdown and per-step holding penalties are added on top.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces

from data.preprocess import FeatureScaler
from env.portfolio import Portfolio
from utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Optional LSTM signal generator
# ---------------------------------------------------------------------------

class LSTMSignalProvider:
    """Wraps an ``LSTMForecaster`` for in-env inference.

    During PPO training we precompute the predictions for the whole dataset once
    to keep the env's ``step`` deterministic and fast. ``predict_all`` returns an
    array of shape ``(n_bars,)`` with the LSTM's forecast for each bar's "next
    return / next close"; positions where the window is incomplete are filled
    with 0 (predict no movement).
    """

    def __init__(self, model, scaler: FeatureScaler, sequence_length: int, device: torch.device | None = None):
        self.model = model
        self.scaler = scaler
        self.sequence_length = sequence_length
        self.device = device or (next(model.parameters()).device)

    def predict_all(self, raw_df: pd.DataFrame, batch_size: int = 256) -> np.ndarray:
        feats = self.scaler.transform(raw_df)         # (n, n_features)
        n = len(feats)
        seq_len = self.sequence_length
        preds = np.zeros(n, dtype=np.float32)
        if n <= seq_len:
            return preds
        windows = np.stack([feats[i - seq_len + 1 : i + 1] for i in range(seq_len - 1, n)], axis=0)
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(windows), batch_size):
                batch = torch.from_numpy(windows[start : start + batch_size]).float().to(self.device)
                out = self.model(batch).cpu().numpy().reshape(-1)
                preds[seq_len - 1 + start : seq_len - 1 + start + len(out)] = out
        return preds


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

@dataclass
class DifferentialSharpe:
    """Moody & Saffell (2001) differential Sharpe ratio for online RL.

    Maintains EWMA of returns (A_t) and squared returns (B_t) with decay ``eta``,
    and returns the instantaneous change in the Sharpe ratio.
    """
    eta: float = 0.01
    A: float = 0.0
    B: float = 0.0

    def update(self, r: float) -> float:
        delta_A = r - self.A
        delta_B = r * r - self.B
        denom = (self.B - self.A * self.A) ** 1.5 + 1e-9
        ds = (self.B * delta_A - 0.5 * self.A * delta_B) / denom
        self.A += self.eta * delta_A
        self.B += self.eta * delta_B
        return float(ds)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class CryptoTradingEnv(gym.Env):
    """Long-only discrete trading environment with risk constraints.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Full feature DataFrame (post-indicator). Must contain ``close`` and the
        features in ``scaler.feature_names``.
    scaler : FeatureScaler
        Feature scaler fit on the training split. Used to normalise the window.
    lstm_signals : np.ndarray | None
        Optional 1-d array of length ``len(raw_df)`` containing the LSTM's
        forecast for each bar. If None, the signal channel is set to 0.
    cfg : SimpleNamespace
        Loaded config (we use ``cfg.env`` and ``cfg.risk``).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        raw_df: pd.DataFrame,
        scaler: FeatureScaler,
        lstm_signals: Optional[np.ndarray],
        cfg,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if "close" not in raw_df.columns:
            raise ValueError("raw_df must contain a 'close' column")
        self.raw_df = raw_df.reset_index(drop=True)
        self.scaler = scaler
        self.cfg = cfg
        self.window_size = cfg.env.window_size
        self.fee_rate = cfg.env.fee_rate
        self.slippage = cfg.env.slippage_bps / 10000.0
        self.initial_balance = float(cfg.env.initial_balance)
        self.holding_penalty = cfg.env.reward.holding_penalty
        self.dd_penalty = cfg.env.reward.drawdown_penalty
        self.reward_kind = cfg.env.reward.type
        self.sharpe_eta = cfg.env.reward.sharpe_eta

        # Risk caps
        self.stop_loss_pct = cfg.risk.stop_loss_pct
        self.take_profit_pct = cfg.risk.take_profit_pct
        self.max_drawdown_pct = cfg.risk.max_drawdown_pct

        # Precompute feature matrix
        self._features = scaler.transform(raw_df)          # (n, n_features)
        self._closes = raw_df["close"].to_numpy(dtype=np.float64)
        if lstm_signals is None:
            self._lstm = np.zeros(len(raw_df), dtype=np.float32)
        else:
            if len(lstm_signals) != len(raw_df):
                raise ValueError("lstm_signals length must match raw_df length")
            self._lstm = lstm_signals.astype(np.float32)

        self.n_features = self._features.shape[1]

        # Observation = flattened window + 5 portfolio scalars + 1 LSTM signal
        obs_dim = self.window_size * self.n_features + 6
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)

        # Action: discrete 3 options
        self.action_space = spaces.Discrete(3)

        # Internal state initialised in reset()
        self.portfolio: Portfolio | None = None
        self.current_step: int = 0
        self._diff_sharpe: DifferentialSharpe | None = None
        self._prev_equity: float = 0.0
        self._np_random, _ = gym.utils.seeding.np_random(seed)
        self._terminated_by_drawdown = False

    # ------------------------------------------------------------------ utils

    def _current_price(self) -> float:
        return float(self._closes[self.current_step])

    def _build_observation(self) -> np.ndarray:
        end = self.current_step + 1
        start = end - self.window_size
        window = self._features[start:end]                 # (window, n_features)
        flat = window.reshape(-1).astype(np.float32)

        price = self._current_price()
        equity = self.portfolio.equity(price)
        position_value = self.portfolio.position_qty * price
        position_frac = position_value / equity if equity > 0 else 0.0
        cash_frac = self.portfolio.cash / equity if equity > 0 else 0.0
        unrealised = 0.0
        if self.portfolio.is_long and self.portfolio.avg_entry_price > 0:
            unrealised = (price - self.portfolio.avg_entry_price) / self.portfolio.avg_entry_price
        drawdown = self.portfolio.drawdown(price)
        lstm_sig = float(self._lstm[self.current_step])

        portfolio_state = np.array(
            [position_frac, cash_frac, unrealised, drawdown, np.tanh(lstm_sig * 10.0), 1.0 if self.portfolio.is_long else 0.0],
            dtype=np.float32,
        )
        return np.concatenate([flat, portfolio_state]).astype(np.float32)

    # ------------------------------------------------------------------ gym api

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.portfolio = Portfolio(cash=self.initial_balance)
        self.current_step = self.window_size - 1
        self._diff_sharpe = DifferentialSharpe(eta=self.sharpe_eta)
        self._prev_equity = self.initial_balance
        self.portfolio.peak_equity = self.initial_balance
        self.portfolio.record(self._current_price())
        self._terminated_by_drawdown = False
        return self._build_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        assert self.portfolio is not None, "Call reset() before step()"
        price = self._current_price()
        ts = self.current_step

        # ---------- Risk-managed pre-trade checks (Section 1.9) ----------
        forced_close = False
        if self.portfolio.is_long and self.portfolio.avg_entry_price > 0:
            move = (price - self.portfolio.avg_entry_price) / self.portfolio.avg_entry_price
            if move <= -self.stop_loss_pct:
                self.portfolio.market_sell(price * (1 - self.slippage), 1.0, self.fee_rate, ts=ts, reason="stop_loss")
                forced_close = True
            elif move >= self.take_profit_pct:
                self.portfolio.market_sell(price * (1 - self.slippage), 1.0, self.fee_rate, ts=ts, reason="take_profit")
                forced_close = True

        # ---------- Apply agent action (unless we just forced a close) ----------
        if not forced_close:
            if action == 1 and self.portfolio.cash > self.cfg.risk.min_trade_notional:
                notional = self.portfolio.cash * self.cfg.env.max_position_fraction
                self.portfolio.market_buy(price * (1 + self.slippage), notional, self.fee_rate, ts=ts, reason="agent")
            elif action == 2 and self.portfolio.is_long:
                self.portfolio.market_sell(price * (1 - self.slippage), 1.0, self.fee_rate, ts=ts, reason="agent")
            # action == 0 (Hold): no-op

        # ---------- Advance time ----------
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= len(self._closes) - 1
        new_price = self._current_price() if not truncated else price
        equity = self.portfolio.record(new_price)

        # ---------- Reward ----------
        if self.reward_kind == "log_return":
            raw_r = math.log(equity / self._prev_equity) if self._prev_equity > 0 else 0.0
        elif self.reward_kind == "pnl":
            raw_r = (equity - self._prev_equity) / self.initial_balance
        elif self.reward_kind == "differential_sharpe":
            step_ret = (equity - self._prev_equity) / self._prev_equity if self._prev_equity > 0 else 0.0
            raw_r = self._diff_sharpe.update(step_ret)
        else:
            raise ValueError(f"Unknown reward type {self.reward_kind!r}")

        dd = self.portfolio.drawdown(new_price)
        reward = raw_r - self.dd_penalty * dd - self.holding_penalty
        self._prev_equity = equity

        # ---------- Drawdown circuit breaker ----------
        if dd >= self.max_drawdown_pct:
            terminated = True
            self._terminated_by_drawdown = True
            # Force-flatten so the recorded final equity reflects the exit.
            if self.portfolio.is_long:
                self.portfolio.market_sell(new_price * (1 - self.slippage), 1.0, self.fee_rate, ts=self.current_step, reason="max_drawdown")
            reward -= 1.0  # explicit penalty

        info = {
            "step": self.current_step,
            "price": new_price,
            "equity": equity,
            "cash": self.portfolio.cash,
            "position_qty": self.portfolio.position_qty,
            "drawdown": dd,
            "forced_close": forced_close,
            "terminated_by_drawdown": self._terminated_by_drawdown,
        }
        if truncated and self.portfolio.is_long:
            self.portfolio.market_sell(new_price * (1 - self.slippage), 1.0, self.fee_rate, ts=self.current_step, reason="episode_end")
        return self._build_observation(), float(reward), terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        if self.portfolio is None:
            return
        price = self._current_price()
        print(
            f"step={self.current_step} price={price:.2f} cash={self.portfolio.cash:.2f} "
            f"qty={self.portfolio.position_qty:.6f} eq={self.portfolio.equity(price):.2f} "
            f"dd={self.portfolio.drawdown(price):.3f}"
        )

    # Convenience
    @property
    def equity_curve(self) -> np.ndarray:
        return np.asarray(self.portfolio.equity_curve if self.portfolio else [], dtype=np.float64)
