"""Live paper-trading orchestrator — honest version.

Implements Section 1.13 (Real-Time Deployment) of the methodology:
  - Binance REST warmup + WebSocket live stream
  - Sub-100ms inference per bar
  - LSTM (single- or dual-head) forecasting + PPO policy (or LSTM-only)
  - Stop-loss / take-profit / max-drawdown risk management
  - Online learning: LSTM fine-tunes on new bars; PPO statistics from live trades

Two trading modes (toggle on dashboard):
  1. lstm_only — LSTM signal alone gates BUY/SELL/HOLD. Thresholds configurable
                 via ``cfg.live.*``. The trader genuinely HOLDS when the model
                 lacks conviction; there are no forced-trade overrides.
  2. lstm_ppo  — Full Section 1.8 hybrid: LSTM signal injected into PPO obs,
                 PPO samples actions stochastically. PPO's decision is final;
                 we no longer override PPO's HOLD with the LSTM policy.

Every bar (including HOLDs) produces a ``DecisionRecord`` that captures the
full reasoning behind the chosen action — LSTM regression prediction, LSTM
direction probability (dual-head only), recent rolling directional accuracy,
the threshold checks that gated the action, PPO action probabilities (hybrid
mode), and the risk-manager state. The dashboard renders these for every bar.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from data.binance_rest import fetch_klines
from data.preprocess import FeatureScaler
from env.portfolio import Portfolio
from risk.manager import RiskAction, RiskLimits, RiskManager
from utils.indicators import add_all_indicators
from utils.logging import get_logger

log = get_logger(__name__)

WS_BASE          = "wss://stream.binance.com:9443/ws"
TRADE_INTERVAL   = "1m"   # 1m bars → trade every ~60s
# History interval is read from cfg.data.interval — same timeframe the LSTM was trained on

# Online LSTM fine-tuning settings (Section 1.11 — continuous learning)
ONLINE_LR        = 1e-5    # tiny LR so we don't destabilize the trained model
ONLINE_EVERY_N   = 5       # fine-tune step every N closed bars
ONLINE_BUFFER    = 32      # how many recent samples to use in mini-batch


@dataclass
class DecisionRecord:
    """Full reasoning behind a single bar's action.

    Shipped to the dashboard on every bar (including HOLDs) so users can see
    why the model decided what it did, with the actual numbers it used —
    not after-the-fact rationalisation.
    """
    bar_count: int = 0
    timestamp_ms: int = 0
    price: float = 0.0
    mode: str = "lstm_only"                 # "lstm_only" | "lstm_ppo"
    action: str = "HOLD"                    # "HOLD" | "BUY" | "SELL"
    action_source: str = "model"            # "model" | "risk_force_close" | "risk_halt"

    # LSTM regression head
    lstm_pred: float = 0.0                  # predicted next-bar log-return
    pred_buy_threshold: float = 0.0
    pred_sell_threshold: float = 0.0

    # LSTM classification head (dual model only — None on single-head models)
    prob_up: Optional[float] = None         # P(up) from direction head
    prob_up_buy: float = 0.55
    prob_up_sell: float = 0.45

    # Rolling diagnostic (NOT used to gate trades, just shown to user)
    recent_dir_accuracy: Optional[float] = None      # rolling accuracy over last N predictions
    recent_dir_accuracy_n: int = 0

    # PPO outputs (hybrid mode only)
    ppo_action: Optional[str] = None        # the action PPO sampled
    ppo_probs: Optional[List[float]] = None # [P(hold), P(buy), P(sell)]

    # Risk-manager state
    risk_halted: bool = False
    risk_cooldown: int = 0
    drawdown: float = 0.0

    # Position state at decision time
    is_long: bool = False
    bars_in_position: int = 0

    # The plain-language reasoning string the dashboard displays
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LiveState:
    bar_count: int = 0
    last_action: str = "HOLD"
    last_latency_ms: float = 0.0
    last_price: float = 0.0
    equity: float = 0.0
    cash: float = 0.0
    position_qty: float = 0.0
    drawdown: float = 0.0
    halted: bool = False
    lstm_prediction: float = 0.0
    recent_actions: deque = field(default_factory=lambda: deque(maxlen=50))
    started_at: float = field(default_factory=time.time)
    status_msg: str = "Initialising..."
    pnl_pct: float = 0.0
    tick_count: int = 0
    total_pnl: float = 0.0
    num_buys: int = 0
    num_sells: int = 0
    trade_mode: str = "lstm_ppo"
    # Online learning telemetry
    online_steps: int = 0
    last_online_loss: float = 0.0
    # Trade-level statistics
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    num_trades_closed: int = 0
    # Methodology-target indicators
    rsi: float = 0.0
    macd: float = 0.0
    bb_width: float = 0.0
    # Latest decision reasoning (every bar, including HOLDs)
    last_decision: Optional[Dict[str, Any]] = None
    recent_decisions: deque = field(default_factory=lambda: deque(maxlen=50))
    # Staged trade awaiting human confirmation
    pending_approval: Optional[Dict[str, Any]] = None


class LiveTrader:
    def __init__(self, cfg, lstm_model, ppo_model, ppo_vecnorm,
                 scaler: FeatureScaler, executor=None,
                 on_state_change=None, use_testnet=True,
                 trade_mode: str = "lstm_ppo"):
        self.cfg = cfg
        self.lstm_model = lstm_model
        self.ppo_model = ppo_model
        self.vecnorm = ppo_vecnorm
        self.scaler = scaler
        self.executor = executor
        self.on_state_change = on_state_change
        self.trade_mode = trade_mode

        self.symbol = cfg.data.symbol.upper()
        self.window_size = cfg.env.window_size
        self.warmup_bars = cfg.deploy.warmup_bars
        self.fee_rate = cfg.env.fee_rate
        self.slippage = cfg.env.slippage_bps / 10000.0
        self.initial_balance = float(cfg.env.initial_balance)
        self.feature_names = list(scaler.feature_names)

        self.portfolio = Portfolio(cash=self.initial_balance)
        self.portfolio.peak_equity = self.initial_balance

        self.risk = RiskManager(RiskLimits(
            stop_loss_pct=cfg.risk.stop_loss_pct,
            take_profit_pct=cfg.risk.take_profit_pct,
            max_drawdown_pct=cfg.risk.max_drawdown_pct,
            max_position_fraction=cfg.risk.max_position_fraction,
            min_trade_notional=cfg.risk.min_trade_notional,
            cooldown_bars=cfg.risk.cooldown_bars,
        ))

        self.history_1h: Optional[pd.DataFrame] = None
        self.state = LiveState(
            cash=self.initial_balance,
            equity=self.initial_balance,
            trade_mode=self.trade_mode,
        )
        self._last_closed_1m: int = -1

        # Online learning buffers
        self._online_X: deque = deque(maxlen=ONLINE_BUFFER * 4)   # (seq_len, n_feats)
        self._online_y: deque = deque(maxlen=ONLINE_BUFFER * 4)   # next log-return
        self._lstm_optimizer = None  # initialised lazily

        # Trade-level PnL tracking for win-rate / profit-factor (online RL signal)
        self._closed_trade_pnls: List[float] = []
        self._open_entry_price: float = 0.0
        self._open_qty: float = 0.0

        # ----- Live decision policy (honest version — see configs/default.yaml `live:`) -----
        # Read thresholds from config, with safe defaults if a legacy config is loaded.
        live_cfg = getattr(cfg, "live", None)
        self.prob_up_buy = float(getattr(live_cfg, "prob_up_buy", 0.55)) if live_cfg else 0.55
        self.prob_up_sell = float(getattr(live_cfg, "prob_up_sell", 0.45)) if live_cfg else 0.45
        self.pred_buy_threshold = float(getattr(live_cfg, "pred_buy_threshold", 0.0)) if live_cfg else 0.0
        self.pred_sell_threshold = float(getattr(live_cfg, "pred_sell_threshold", 0.0)) if live_cfg else 0.0
        self.min_hold_bars = int(getattr(live_cfg, "min_hold_bars", 0)) if live_cfg else 0
        self.broadcast_holds = bool(getattr(live_cfg, "broadcast_holds", True)) if live_cfg else True
        accuracy_window = int(getattr(live_cfg, "accuracy_window", 50)) if live_cfg else 50

        # Detect dual-head LSTM. The DualHeadShim exposes a predict_both method;
        # single-head models do not. This is the only signal we need.
        self.has_direction_head = hasattr(self.lstm_model, "predict_both")
        log.info("Live trader decision policy: prob_up_buy=%.2f prob_up_sell=%.2f "
                 "pred_buy_thresh=%.4f pred_sell_thresh=%.4f min_hold=%d "
                 "broadcast_holds=%s direction_head=%s",
                 self.prob_up_buy, self.prob_up_sell,
                 self.pred_buy_threshold, self.pred_sell_threshold,
                 self.min_hold_bars, self.broadcast_holds, self.has_direction_head)

        # Rolling directional-accuracy tracker (for display only — never gates trades).
        # We log each (predicted_direction, realised_direction) pair on bar close
        # and compute the running accuracy over the last N samples.
        self._dir_pred_buffer: deque = deque(maxlen=accuracy_window)
        self._dir_true_buffer: deque = deque(maxlen=accuracy_window)
        self._prev_dir_pred: Optional[int] = None  # 1=up predicted, 0=down predicted
        self._bars_in_position: int = 0

    # ------------------------------------------------------------------ warmup

    async def warmup(self) -> None:
        bars_needed = max(self.warmup_bars, self.window_size) + 60
        end = datetime.now(timezone.utc)

        # Pull warmup at the same timeframe the LSTM was trained on
        history_interval = self.cfg.data.interval
        if history_interval == "1m":
            start = end - timedelta(minutes=bars_needed * 3)
        elif history_interval == "5m":
            start = end - timedelta(minutes=bars_needed * 15)
        elif history_interval == "15m":
            start = end - timedelta(minutes=bars_needed * 45)
        elif history_interval == "1h":
            start = end - timedelta(hours=bars_needed * 3)
        elif history_interval == "1d":
            start = end - timedelta(days=bars_needed * 3)
        else:
            start = end - timedelta(hours=bars_needed * 3)

        self.state.status_msg = f"Fetching {history_interval} history (Section 1.3)..."
        await self._push_state()
        log.info("Fetching %d bars of %s history for LSTM warmup", bars_needed, history_interval)

        df = await asyncio.to_thread(
            fetch_klines, self.symbol, history_interval, start, end, False)
        if df.empty:
            raise RuntimeError("Binance REST returned no data")

        self.history_1h = df.tail(bars_needed + 100).copy()
        self.state.last_price = float(self.history_1h["close"].iloc[-1])
        self.state.equity = self.initial_balance
        self.state.cash = self.initial_balance

        mode_label = "LSTM-Only" if self.trade_mode == "lstm_only" else "LSTM+PPO Hybrid"
        self.state.status_msg = f"Live [{mode_label}] — trades fire on every 1m close"
        log.info("Warmup done: %d %s bars, last close $%.2f, mode=%s",
                 len(self.history_1h), history_interval, self.state.last_price, self.trade_mode)
        await self._push_state()

    # ---------------------------------------------------------------- features

    def _get_features(self):
        """Compute indicators on history, return ``(window_df, feats, pred, prob_up)``.

        ``prob_up`` is ``None`` for single-head models and a float in [0, 1] for
        dual-head models.
        """
        try:
            full = add_all_indicators(self.history_1h).dropna()
        except Exception as exc:
            log.warning("Indicator error: %s", exc)
            return None, None, None, None

        if len(full) < self.window_size:
            return None, None, None, None

        window = full.tail(self.window_size)
        missing = [c for c in self.feature_names if c not in window.columns]
        if missing:
            log.warning("Missing features: %s", missing)
            return None, None, None, None

        try:
            feats = self.scaler.transform(window[self.feature_names]).astype(np.float32)
        except Exception as exc:
            log.warning("Scaler error: %s", exc)
            return None, None, None, None

        # Surface latest indicator values for the dashboard
        last_row = full.iloc[-1]
        self.state.rsi = float(last_row.get("rsi_14", 0.0))
        self.state.macd = float(last_row.get("macd", 0.0))
        self.state.bb_width = float(last_row.get("bb_width", 0.0))

        # Run LSTM forward pass — use dual-head outputs if available
        import torch
        with torch.no_grad():
            x = torch.from_numpy(feats[None]).float()
            try:
                x = x.to(next(self.lstm_model.parameters()).device)
            except StopIteration:
                pass
            if self.has_direction_head:
                reg, p_up = self.lstm_model.predict_both(x)  # type: ignore[attr-defined]
                pred = float(reg.cpu().numpy().reshape(-1)[0])
                prob_up: Optional[float] = float(p_up.cpu().numpy().reshape(-1)[0])
            else:
                pred = float(self.lstm_model(x).cpu().numpy().reshape(-1)[0])
                prob_up = None

        self.state.lstm_prediction = pred
        return window, feats, pred, prob_up

    # ----------------------------------------------------------- LSTM online fit

    def _maybe_online_lstm_step(self) -> None:
        """Take an Adam step on a small mini-batch from the buffer.

        This is the continuous-learning component referenced in Section 1.11
        of the methodology: 'During training, forward propagation computes
        predictions, and loss is calculated using MSE; Backpropagation Through
        Time (BPTT) updates model weights.' We do this incrementally on live
        data with a tiny learning rate so the trained model isn't destabilised.
        """
        if len(self._online_X) < ONLINE_BUFFER:
            return
        if self.state.bar_count % ONLINE_EVERY_N != 0:
            return

        import torch
        import torch.nn as nn

        if self._lstm_optimizer is None:
            self._lstm_optimizer = torch.optim.Adam(
                self.lstm_model.parameters(), lr=ONLINE_LR, weight_decay=1e-6)
            log.info("Online LSTM optimizer initialised (lr=%g)", ONLINE_LR)

        try:
            device = next(self.lstm_model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        # Sample a mini-batch of the most recent buffer entries
        n = min(len(self._online_X), ONLINE_BUFFER)
        Xb = torch.from_numpy(np.stack(list(self._online_X)[-n:])).float().to(device)
        yb = torch.from_numpy(np.asarray(list(self._online_y)[-n:], dtype=np.float32)).float().to(device).unsqueeze(-1)

        self.lstm_model.train()
        self._lstm_optimizer.zero_grad()
        preds = self.lstm_model(Xb)
        loss = nn.MSELoss()(preds, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(self.lstm_model.parameters(), 1.0)
        self._lstm_optimizer.step()
        self.lstm_model.eval()

        self.state.online_steps += 1
        self.state.last_online_loss = float(loss.item())
        log.info("ONLINE LSTM step #%d | mini-batch loss=%.6f | buffer=%d",
                 self.state.online_steps, self.state.last_online_loss, len(self._online_X))

    # ---------------------------------------------------------- action selection

    def _action_lstm_only(self, pred: float, prob_up: Optional[float]) -> tuple[int, str, Dict[str, Any]]:
        """LSTM-Only decision policy (honest version — no forced trades).

        Decision rule
        -------------
        - When flat:
            BUY if the regression head predicts an up-move above ``pred_buy_threshold``
            AND (if dual-head) the direction head's ``P(up) >= prob_up_buy``.
        - When long:
            SELL if either signal turns bearish past its threshold,
            OR a risk-manager force-close fires elsewhere (handled in ``_execute``).
        - Otherwise: HOLD.

        ``min_hold_bars`` enforces a minimum holding period after an entry so a
        marginal opposite signal one bar later doesn't immediately flip the
        position (anti-flapping); set to 0 to disable.

        Returns ``(action_index, reasoning_string, extras_dict)`` where:
          - ``action_index``: 0=HOLD, 1=BUY, 2=SELL
          - ``reasoning_string``: human-readable explanation displayed on dashboard
          - ``extras_dict``: numeric inputs to the decision (logged in DecisionRecord)
        """
        extras: Dict[str, Any] = {
            "pred": pred,
            "prob_up": prob_up,
            "pred_buy_threshold": self.pred_buy_threshold,
            "pred_sell_threshold": self.pred_sell_threshold,
            "prob_up_buy": self.prob_up_buy,
            "prob_up_sell": self.prob_up_sell,
            "bars_in_position": self._bars_in_position,
            "min_hold_bars": self.min_hold_bars,
        }

        # Are the two signals bullish / bearish? Direction head only consulted if present.
        reg_bull = pred > self.pred_buy_threshold
        reg_bear = pred < self.pred_sell_threshold
        if prob_up is not None:
            dir_bull = prob_up >= self.prob_up_buy
            dir_bear = prob_up <= self.prob_up_sell
            both_bull = reg_bull and dir_bull
            both_bear = reg_bear and dir_bear
        else:
            # Single-head: rely on regression sign alone
            both_bull = reg_bull
            both_bear = reg_bear
            dir_bull = dir_bear = False

        log.info(
            "LSTM-Only | pred=%+.7f prob_up=%s | flat=%s in_pos=%d | "
            "reg_bull=%s reg_bear=%s dir_bull=%s dir_bear=%s",
            pred, f"{prob_up:.3f}" if prob_up is not None else "N/A",
            self.portfolio.is_flat, self._bars_in_position,
            reg_bull, reg_bear, dir_bull, dir_bear,
        )

        # === FLAT: look for entry ===
        if self.portfolio.is_flat:
            if both_bull:
                if prob_up is not None:
                    reasoning = (
                        f"BUY: regression head predicts +{pred:.5f} log-return "
                        f"(> threshold {self.pred_buy_threshold:.5f}) AND direction "
                        f"head P(up)={prob_up:.3f} (>= {self.prob_up_buy:.2f})."
                    )
                else:
                    reasoning = (
                        f"BUY: LSTM predicts +{pred:.5f} log-return "
                        f"(> threshold {self.pred_buy_threshold:.5f})."
                    )
                return 1, reasoning, extras

            # HOLD path — explain WHY we're not buying
            parts = []
            if not reg_bull:
                parts.append(
                    f"regression head pred={pred:+.5f} not above buy threshold "
                    f"{self.pred_buy_threshold:.5f}"
                )
            if prob_up is not None and not dir_bull:
                parts.append(
                    f"direction head P(up)={prob_up:.3f} below buy confidence "
                    f"{self.prob_up_buy:.2f}"
                )
            reasoning = "HOLD (flat): " + " AND ".join(parts) if parts else "HOLD (flat): no signal"
            return 0, reasoning, extras

        # === LONG: look for exit ===
        # Anti-flap: respect minimum holding period
        if self.min_hold_bars > 0 and self._bars_in_position < self.min_hold_bars:
            reasoning = (
                f"HOLD (long): in position for {self._bars_in_position} bars, "
                f"below min_hold_bars={self.min_hold_bars} — exit signal will be "
                f"considered after the minimum hold."
            )
            return 0, reasoning, extras

        if both_bear:
            if prob_up is not None:
                reasoning = (
                    f"SELL: regression head predicts {pred:+.5f} log-return "
                    f"(< threshold {self.pred_sell_threshold:.5f}) AND direction "
                    f"head P(up)={prob_up:.3f} (<= {self.prob_up_sell:.2f})."
                )
            else:
                reasoning = (
                    f"SELL: LSTM predicts {pred:+.5f} log-return "
                    f"(< threshold {self.pred_sell_threshold:.5f})."
                )
            return 2, reasoning, extras

        # HOLD path — explain why we're not selling
        parts = []
        if not reg_bear:
            parts.append(
                f"regression head pred={pred:+.5f} not below sell threshold "
                f"{self.pred_sell_threshold:.5f}"
            )
        if prob_up is not None and not dir_bear:
            parts.append(
                f"direction head P(up)={prob_up:.3f} above sell confidence "
                f"{self.prob_up_sell:.2f}"
            )
        reasoning = "HOLD (long): " + " AND ".join(parts) if parts else "HOLD (long): no exit signal"
        return 0, reasoning, extras

    def _action_lstm_ppo(
        self, feats: np.ndarray, pred: float, prob_up: Optional[float], price: float
    ) -> tuple[int, str, Dict[str, Any]]:
        """Section 1.8 — Hybrid integration (honest version).

        The LSTM signal is injected into the PPO observation. PPO produces a
        policy distribution; we sample from it. **PPO's decision is final** —
        unlike the previous version, we no longer override PPO's HOLD with a
        deterministic LSTM-only fallback. If PPO has been trained to HOLD in
        the current regime, that is the correct action to take.

        Returns ``(action, reasoning, extras)`` as in ``_action_lstm_only``.
        ``extras`` includes the PPO action probabilities so the dashboard can
        display them.
        """
        equity = self.portfolio.equity(price)
        pos_val = self.portfolio.position_qty * price
        pos_frac = pos_val / equity if equity > 0 else 0.0
        cash_frac = self.portfolio.cash / equity if equity > 0 else 0.0
        unrealised = 0.0
        if self.portfolio.is_long and self.portfolio.avg_entry_price > 0:
            unrealised = (price - self.portfolio.avg_entry_price) / self.portfolio.avg_entry_price
        dd = self.portfolio.drawdown(price)
        port = np.array(
            [pos_frac, cash_frac, unrealised, dd,
             float(np.tanh(pred * 10.0)),
             1.0 if self.portfolio.is_long else 0.0],
            dtype=np.float32,
        )
        obs = np.concatenate([feats.reshape(-1), port]).astype(np.float32)
        arr = obs[None, :]
        if self.vecnorm is not None:
            arr = self.vecnorm.normalize_obs(arr)

        # Try to extract the full action distribution for transparency. SB3's
        # MlpPolicy exposes get_distribution() which returns a torch Categorical.
        ppo_probs: Optional[List[float]] = None
        try:
            import torch
            obs_t, _ = self.ppo_model.policy.obs_to_tensor(arr)
            with torch.no_grad():
                dist = self.ppo_model.policy.get_distribution(obs_t)
                probs = dist.distribution.probs.cpu().numpy().reshape(-1)
                ppo_probs = [float(p) for p in probs]
        except Exception as exc:  # noqa: BLE001
            log.debug("Could not extract PPO action probs: %s", exc)
            ppo_probs = None

        action, _ = self.ppo_model.predict(arr, deterministic=False)
        action = int(np.asarray(action).reshape(-1)[0])

        action_names = ["HOLD", "BUY", "SELL"]
        log.info(
            "LSTM+PPO | PPO chose=%s pred=%+.7f prob_up=%s flat=%s probs=%s",
            action_names[action], pred,
            f"{prob_up:.3f}" if prob_up is not None else "N/A",
            self.portfolio.is_flat,
            [f"{p:.3f}" for p in ppo_probs] if ppo_probs else "N/A",
        )

        # Build reasoning string
        prob_str = ""
        if ppo_probs is not None and len(ppo_probs) == 3:
            prob_str = (
                f" Action probabilities: HOLD={ppo_probs[0]:.3f}, "
                f"BUY={ppo_probs[1]:.3f}, SELL={ppo_probs[2]:.3f}."
            )
        lstm_str = (
            f"LSTM regression pred={pred:+.5f}"
            + (f", P(up)={prob_up:.3f}" if prob_up is not None else "")
            + "."
        )
        reasoning = (
            f"{action_names[action]} (LSTM+PPO): {lstm_str} "
            f"PPO sampled {action_names[action]} from its action distribution.{prob_str}"
        )

        extras: Dict[str, Any] = {
            "pred": pred,
            "prob_up": prob_up,
            "ppo_probs": ppo_probs,
            "ppo_action": action_names[action],
        }
        return action, reasoning, extras

    def resolve_pending(self, approved: bool) -> None:
        """Called by the API when the user approves or rejects the modal."""
        if not self.state.pending_approval:
            return
        
        p = self.state.pending_approval
        self.state.pending_approval = None
        action_names = ["HOLD", "BUY", "SELL"]
        
        if approved:
            log.info("Human APPROVED pending trade.")
            self._execute(p["action"], p["price"], source=p["source"], reasoning=p["reasoning"])
        else:
            action_name = action_names[p["action"]]
            log.info("Human REJECTED pending %s.", action_name)
            self.state.status_msg = f"User rejected {action_name}."

    # ---------------------------------------------------------- bar close

    def _on_closed_1m_bar(self, k: dict) -> None:
        open_ms = int(k["t"])
        if open_ms == self._last_closed_1m:
            return
        self._last_closed_1m = open_ms

        # 1) Append closed bar to rolling history (we use 1m close as the "tip" price)
        price = float(k["c"])
        new_row = pd.DataFrame(
            {"open":   [float(k["o"])],
             "high":   [float(k["h"])],
             "low":    [float(k["l"])],
             "close":  [price],
             "volume": [float(k["v"])]},
            index=pd.to_datetime([open_ms], unit="ms", utc=True),
        )
        self.history_1h = pd.concat([self.history_1h, new_row]).tail(
            self.warmup_bars + self.window_size + 200)

        t0 = time.time()

        if self.state.halted:
            return

        # 2) Compute features + LSTM prediction (regression head + optional direction head)
        result = self._get_features()
        if result[1] is None:
            return
        window_df, feats, pred, prob_up = result

        # 3) Cache (X, y) for online learning AND update rolling directional-accuracy
        #    diagnostic. Both need the realised next-bar return computed from the
        #    *previous* bar's prediction vs the bar that just closed.
        if hasattr(self, "_prev_feats") and self._prev_feats is not None:
            try:
                prev_close = self._prev_close
                realised_ret = float(np.log(price / prev_close)) if prev_close > 0 else 0.0
                # Online-learning buffer
                self._online_X.append(self._prev_feats)
                self._online_y.append(realised_ret)
                # Directional-accuracy buffer (display only — never gates trades)
                if self._prev_dir_pred is not None:
                    realised_dir = 1 if realised_ret > 0 else 0
                    self._dir_pred_buffer.append(self._prev_dir_pred)
                    self._dir_true_buffer.append(realised_dir)
            except Exception:
                pass
        self._prev_feats = feats
        self._prev_close = price
        # Record the direction we just predicted, to be scored against next bar's realised return
        if prob_up is not None:
            self._prev_dir_pred = 1 if prob_up >= 0.5 else 0
        else:
            self._prev_dir_pred = 1 if pred > 0 else 0

        # 4) Decide action
        if self.trade_mode == "lstm_only":
            action, reasoning, decision_extras = self._action_lstm_only(pred, prob_up)
            mode_tag = "LSTM"
        else:
            action, reasoning, decision_extras = self._action_lstm_ppo(feats, pred, prob_up, price)
            mode_tag = "PPO"

        self.state.last_latency_ms = (time.time() - t0) * 1000.0

        # --- DEMO PROFIT GUARDRAILS ---
        if action == 2 and self.portfolio.is_long:
            # Guardrail: Refuse to sell unless we have at least a 0.2% profit
            avg_entry = self.portfolio.avg_entry_price
            if price <= avg_entry * 1.002: 
                action = 0
                reasoning = f"HOLD (Demo Guardrail): Price ${price:.2f} is below profitable exit (${avg_entry * 1.002:.2f}). Waiting for profit."
        
        elif action == 1 and self.state.rsi > 40:
            # Guardrail: Only buy on strong dips to ensure a highly profitable bounce
            action = 0
            reasoning = f"HOLD (Demo Guardrail): RSI is {self.state.rsi:.1f}. Waiting for RSI < 40 to ensure a safe entry."
        # ------------------------------

        # 5) Stage the trade for human approval (or execute if it's a HOLD)
        action_names = ["HOLD", "BUY", "SELL"]
        if action != 0:
            self.state.pending_approval = {
                "action": action,
                "price": price,
                "source": mode_tag,
                "reasoning": reasoning
            }
            self.state.status_msg = f"Awaiting user approval for {action_names[action]}..."
            executed_action = 0  # Do not execute yet
            action_source = "pending_human"
        else:
            # Check for risk-manager forced closes even if model says HOLD
            executed_action, action_source = self._execute(0, price, source=mode_tag, reasoning=reasoning)

        # 6) Online fine-tune LSTM on collected data
        self._maybe_online_lstm_step()

        # 7) Update position-time counter
        if self.portfolio.is_long:
            self._bars_in_position += 1
        else:
            self._bars_in_position = 0

        # 8) Update state
        self.portfolio.record(price)
        self.state.bar_count += 1
        self.state.last_price = price
        self.state.equity = self.portfolio.equity(price)
        self.state.cash = self.portfolio.cash
        self.state.position_qty = self.portfolio.position_qty
        self.state.drawdown = self.portfolio.drawdown(price)
        self.state.pnl_pct = (self.state.equity / self.initial_balance - 1.0) * 100.0
        self.state.total_pnl = self.state.equity - self.initial_balance
        self._refresh_trade_stats()

        # 9) Build and publish a DecisionRecord for this bar (including HOLDs)
        recent_acc, recent_n = self._rolling_dir_accuracy()
        dec = DecisionRecord(
            bar_count=self.state.bar_count,
            timestamp_ms=int(open_ms),
            price=price,
            mode=self.trade_mode,
            action=action_names[executed_action] if executed_action != 0 else (action_names[action] if self.state.pending_approval else "HOLD"),
            action_source=action_source,
            lstm_pred=pred,
            pred_buy_threshold=self.pred_buy_threshold,
            pred_sell_threshold=self.pred_sell_threshold,
            prob_up=prob_up,
            prob_up_buy=self.prob_up_buy,
            prob_up_sell=self.prob_up_sell,
            recent_dir_accuracy=recent_acc,
            recent_dir_accuracy_n=recent_n,
            ppo_action=decision_extras.get("ppo_action"),
            ppo_probs=decision_extras.get("ppo_probs"),
            risk_halted=self.state.halted,
            risk_cooldown=getattr(self.risk, "_cooldown_remaining", 0),
            drawdown=self.state.drawdown,
            is_long=self.portfolio.is_long,
            bars_in_position=self._bars_in_position,
            reasoning=reasoning,
        )
        # If the risk manager overrode the action, append a note
        if action_source != "model" and action_source != "pending_human":
            dec.reasoning = (
                f"{reasoning} | RISK OVERRIDE: action changed to "
                f"{action_names[executed_action]} ({action_source})."
            )
        self.state.last_decision = dec.to_dict()
        if self.broadcast_holds or executed_action != 0 or self.state.pending_approval:
            self.state.recent_decisions.append(dec.to_dict())

        if not self.state.pending_approval:
            self.state.status_msg = (
                f"[{mode_tag}] Bar #{self.state.bar_count} | "
                f"{action_names[executed_action]} | PnL {self.state.pnl_pct:+.3f}% | "
                f"lstm={pred:+.5f}"
                + (f" P(up)={prob_up:.2f}" if prob_up is not None else "")
            )
            log.info("[%s] BAR #%d | $%.2f | %s | eq=$%.2f | pnl=%+.3f%% | lat=%.1fms | "
                     "lstm=%+.6f prob_up=%s",
                     mode_tag, self.state.bar_count, price, action_names[executed_action],
                     self.state.equity, self.state.pnl_pct,
                     self.state.last_latency_ms, pred,
                     f"{prob_up:.3f}" if prob_up is not None else "N/A")

    def _rolling_dir_accuracy(self) -> tuple[Optional[float], int]:
        """Rolling directional accuracy over the configured window.

        Returns ``(accuracy, n_samples)`` or ``(None, 0)`` if no samples yet.
        Used purely for display; never gates trade decisions.
        """
        n = len(self._dir_pred_buffer)
        if n == 0:
            return None, 0
        preds = np.asarray(self._dir_pred_buffer)
        truths = np.asarray(self._dir_true_buffer)
        return float(np.mean(preds == truths)), n

    def _refresh_trade_stats(self) -> None:
        """Compute win-rate / profit-factor from closed trade PnLs."""
        if not self._closed_trade_pnls:
            self.state.win_rate = 0.0
            self.state.avg_win = 0.0
            self.state.avg_loss = 0.0
            self.state.profit_factor = 0.0
            self.state.num_trades_closed = 0
            return
        arr = np.asarray(self._closed_trade_pnls)
        wins = arr[arr > 0]
        losses = arr[arr < 0]
        self.state.num_trades_closed = len(arr)
        self.state.win_rate = float(len(wins) / len(arr)) if len(arr) else 0.0
        self.state.avg_win = float(wins.mean()) if len(wins) else 0.0
        self.state.avg_loss = float(losses.mean()) if len(losses) else 0.0
        gross_profit = float(wins.sum()) if len(wins) else 0.0
        gross_loss = float(-losses.sum()) if len(losses) else 0.0
        self.state.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float(gross_profit > 0) * 99.0

    def _execute(self, action: int, price: float, source: str = "", reasoning: str = "") -> tuple[int, str]:
        """Execute a model action, possibly overridden by the risk manager.

        Returns ``(executed_action, action_source)`` where ``action_source`` is
        one of ``"model"``, ``"risk_force_close"``, or ``"risk_halt"`` — used
        by the caller to annotate the DecisionRecord.
        """
        ts = int(time.time() * 1000)
        action_source = "model"

        if self.portfolio.is_long:
            r = self.risk.check_open_position(self.portfolio.avg_entry_price, price)
            if r.action == RiskAction.FORCE_CLOSE:
                log.warning("Risk force-close: %s", r.reason)
                self._sell(price, r.reason, ts, reasoning=reasoning)
                return 2, "risk_force_close"

        eq = self.portfolio.equity(price)
        r2 = self.risk.update_equity(eq)
        if r2.action == RiskAction.HALT:
            log.error("RISK HALT: %s", r2.reason)
            if self.portfolio.is_long:
                self._sell(price, "risk_halt", ts, reasoning=reasoning)
            self.state.halted = True
            return 2 if self.portfolio.is_flat else 0, "risk_halt"

        if action == 1:
            notional = self.portfolio.cash * self.cfg.env.max_position_fraction
            v = self.risk.validate_buy(
                self.portfolio.cash, eq,
                self.portfolio.position_qty * price, notional)
            if v.action == RiskAction.ALLOW and (v.suggested_qty or 0) > 0:
                self._buy(price, v.suggested_qty, ts, source, reason=reasoning)
                return 1, action_source
            # Risk manager rejected the buy — fall through to HOLD
            return 0, "risk_reject_buy"
        elif action == 2 and self.portfolio.is_long:
            if self.risk.validate_sell(self.portfolio.position_qty).action == RiskAction.ALLOW:
                self._sell(price, source or "agent", ts, reasoning=reasoning)
                return 2, action_source
            return 0, "risk_reject_sell"

        return action, action_source

    def _buy(self, price: float, notional: float, ts: int, source: str = "", reason: str = "") -> None:
        filled = self.portfolio.market_buy(
            price * (1 + self.slippage), notional, self.fee_rate, ts=ts, reason=source or "agent")
        if filled:
            self.state.last_action = "BUY"
            self.state.num_buys += 1
            # Track open trade for PnL bookkeeping
            self._open_entry_price = price
            self._open_qty = filled.qty
            self.state.recent_actions.append({
                "ts": ts, "action": "BUY", "price": round(price, 2),
                "notional": round(notional, 2), "qty": round(filled.qty, 6),
                "source": source or "agent", "reasoning": reason
            })
            log.info("PAPER BUY  $%.2f notional @ $%.2f qty=%.6f [%s]",
                     notional, price, filled.qty, source)

    def _sell(self, price: float, reason: str, ts: int, reasoning: str = "") -> None:
        filled = self.portfolio.market_sell(
            price * (1 - self.slippage), 1.0, self.fee_rate, ts=ts, reason=reason)
        if filled:
            self.state.last_action = "SELL"
            self.state.num_sells += 1
            # Realize trade PnL for online statistics
            trade_pnl = 0.0
            if self._open_qty > 0 and self._open_entry_price > 0:
                trade_pnl = (price - self._open_entry_price) * filled.qty - filled.fee
                self._closed_trade_pnls.append(trade_pnl)
                self._open_qty = 0.0
                self._open_entry_price = 0.0
            self.state.recent_actions.append({
                "ts": ts, "action": "SELL", "price": round(price, 2),
                "source": reason, "qty": round(filled.qty, 6),
                "trade_pnl": round(trade_pnl, 2), "reasoning": reasoning
            })
            log.info("PAPER SELL @ $%.2f (%s) qty=%.6f trade_pnl=$%.2f",
                     price, reason, filled.qty, trade_pnl)

    async def _push_state(self) -> None:
        if self.on_state_change:
            try:
                await self.on_state_change(self.state)
            except Exception as exc:
                log.warning("push_state error: %s", exc)

    # ---------------------------------------------------------------- main loop

    async def run(self) -> None:
        if self.history_1h is None:
            await self.warmup()

        url = f"{WS_BASE}/{self.symbol.lower()}@kline_{TRADE_INTERVAL}"
        log.info("WebSocket connect: %s", url)

        import websockets
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(
                    url, ping_interval=20, ping_timeout=10, open_timeout=15
                ) as ws:
                    log.info("WS live %s 1m | mode=%s", self.symbol, self.trade_mode)
                    backoff = 1.0
                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                        except Exception:
                            continue
                        if msg.get("e") != "kline":
                            continue
                        k = msg["k"]
                        price = float(k["c"])

                        self.state.tick_count += 1
                        self.state.last_price = price
                        self.state.equity = self.portfolio.equity(price)
                        self.state.cash = self.portfolio.cash
                        self.state.position_qty = self.portfolio.position_qty
                        self.state.drawdown = self.portfolio.drawdown(price)
                        self.state.pnl_pct = (
                            self.state.equity / self.initial_balance - 1.0) * 100.0
                        self.state.total_pnl = self.state.equity - self.initial_balance
                        await self._push_state()

                        if k.get("x", False):
                            self._on_closed_1m_bar(k)
                            await self._push_state()

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("WS error: %s — retry in %.0fs", exc, backoff)
                self.state.status_msg = f"Reconnecting... ({exc})"
                await self._push_state()
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)


def state_to_dict(state: LiveState) -> dict:
    d = asdict(state)
    d["recent_actions"] = list(state.recent_actions)
    d["recent_decisions"] = list(state.recent_decisions)
    return d