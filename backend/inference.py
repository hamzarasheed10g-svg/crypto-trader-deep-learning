"""Stateless inference service used by the REST and WebSocket endpoints.

Keeps the LSTM + PPO in memory as singletons and exposes a single
``infer_window`` method that runs the full hybrid pipeline:

    OHLCV window -> indicators -> normalise -> LSTM -> build observation -> PPO -> action

This is the realtime equivalent of one ``LiveTrader`` bar.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from agents.ppo import load_ppo
from data.preprocess import FeatureScaler
from models.train_lstm import load_lstm_checkpoint
from utils.indicators import add_all_indicators
from utils.config import resolve_path
from utils.logging import get_logger

log = get_logger(__name__)


class InferenceService:
    def __init__(self, cfg):
        self.cfg = cfg
        self.symbol = cfg.data.symbol
        self.interval = cfg.data.interval
        self.window_size = cfg.env.window_size

        # Loaded lazily on first use
        self._lstm_model = None
        self._lstm_cfg = None
        self._ppo_model = None
        self._vecnorm = None
        self._scaler: Optional[FeatureScaler] = None
        self.loaded_at: Optional[float] = None

    @property
    def loaded(self) -> bool:
        return self._lstm_model is not None and self._ppo_model is not None

    def load(self) -> None:
        if self.loaded:
            return
        import torch  # noqa: F401  (verify torch available before doing anything)

        lstm_path = resolve_path(self.cfg, self.cfg.paths.lstm_model)
        scaler_path = resolve_path(self.cfg, self.cfg.paths.lstm_scaler)
        ppo_path = resolve_path(self.cfg, self.cfg.paths.ppo_model)
        vecnorm_path = resolve_path(self.cfg, self.cfg.paths.ppo_vecnorm)

        for p in (lstm_path, scaler_path, ppo_path, vecnorm_path):
            if not Path(p).exists():
                raise FileNotFoundError(f"Required artifact missing: {p}")

        log.info("Loading LSTM checkpoint from %s", lstm_path)
        self._lstm_model, self._lstm_cfg = load_lstm_checkpoint(lstm_path)
        self._scaler = FeatureScaler.load(scaler_path)

        # We need a dummy raw_df to satisfy load_ppo's env_factory. It is only
        # used to construct the observation/action spaces, not for inference.
        dummy = self._make_dummy_df()
        log.info("Loading PPO model from %s", ppo_path)
        self._ppo_model, vec_env = load_ppo(
            ppo_path, vecnorm_path, dummy, self._scaler, None, self.cfg,
        )
        self._vecnorm = vec_env
        self.loaded_at = time.time()
        log.info("Inference service ready")

    def _make_dummy_df(self) -> pd.DataFrame:
        from utils.config import resolve_path as rp
        processed_dir = rp(self.cfg, self.cfg.paths.processed_data_dir)
        for split in ("test", "train", "val"):
            p = Path(processed_dir) / f"{self.cfg.data.symbol}_{self.cfg.data.interval}_{split}.parquet"
            if p.exists():
                df = pd.read_parquet(p)
                df.index = pd.to_datetime(df.index, utc=True)
                return df
        raise FileNotFoundError(
            f"No processed parquet found in {processed_dir}. "
            "Run scripts.phase1_preprocess first."
    )

    def infer_window(self, ohlcv: pd.DataFrame, portfolio_state: Optional[dict] = None) -> dict:
        """Run inference on the *latest* window of an OHLCV DataFrame.

        ``portfolio_state`` should be a dict with keys:
            position_frac, cash_frac, unrealised, drawdown, is_long
        If omitted, defaults assume flat cash position.

        Returns a dict with keys:
            lstm_prediction  — regression head output (next-bar log-return)
            prob_up          — direction head P(up) if dual-head, else None
            action_index     — 0=HOLD, 1=BUY, 2=SELL
            action           — string label
            latency_ms       — end-to-end inference time
            reasoning        — human-readable explanation of the action chosen
        """
        if not self.loaded:
            self.load()
        assert self._lstm_model is not None and self._ppo_model is not None and self._scaler is not None

        start = time.time()
        full = add_all_indicators(ohlcv).dropna()
        if len(full) < self.window_size:
            raise ValueError(f"Need at least {self.window_size} indicator-ready bars, got {len(full)}")
        window = full.tail(self.window_size)
        feats = self._scaler.transform(window).astype(np.float32)

        import torch
        has_dual = hasattr(self._lstm_model, "predict_both")
        with torch.no_grad():
            x = torch.from_numpy(feats[None, :, :]).float()
            try:
                x = x.to(next(self._lstm_model.parameters()).device)
            except StopIteration:
                pass
            if has_dual:
                reg, p_up = self._lstm_model.predict_both(x)  # type: ignore[attr-defined]
                pred = float(reg.cpu().numpy().reshape(-1)[0])
                prob_up: Optional[float] = float(p_up.cpu().numpy().reshape(-1)[0])
            else:
                pred = float(self._lstm_model(x).cpu().numpy().reshape(-1)[0])
                prob_up = None

        ps = portfolio_state or {}
        pos_frac = float(ps.get("position_frac", 0.0))
        cash_frac = float(ps.get("cash_frac", 1.0))
        unrealised = float(ps.get("unrealised", 0.0))
        drawdown = float(ps.get("drawdown", 0.0))
        is_long = bool(ps.get("is_long", False))
        portfolio_state_arr = np.array(
            [pos_frac, cash_frac, unrealised, drawdown, np.tanh(pred * 10.0), 1.0 if is_long else 0.0],
            dtype=np.float32,
        )
        obs = np.concatenate([feats.reshape(-1), portfolio_state_arr]).astype(np.float32)
        obs_arr = obs[None, :]
        if self._vecnorm is not None:
            obs_arr = self._vecnorm.normalize_obs(obs_arr)
        action, _ = self._ppo_model.predict(obs_arr, deterministic=True)
        action_idx = int(np.asarray(action).reshape(-1)[0])
        action_name = ["HOLD", "BUY", "SELL"][action_idx]

        # Build a concise reasoning string
        if prob_up is not None:
            lstm_str = (f"LSTM regression pred={pred:+.5f}, "
                        f"direction head P(up)={prob_up:.3f}.")
        else:
            lstm_str = f"LSTM pred={pred:+.5f}."
        reasoning = f"{action_name}: {lstm_str} PPO policy (deterministic) chose {action_name}."

        latency_ms = (time.time() - start) * 1000.0
        return {
            "lstm_prediction": pred,
            "prob_up": prob_up,
            "action_index": action_idx,
            "action": action_name,
            "latency_ms": latency_ms,
            "reasoning": reasoning,
        }
