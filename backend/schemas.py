"""Pydantic schemas exposed by the FastAPI backend."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    symbol: Optional[str] = None
    interval: Optional[str] = None
    uptime_seconds: float


class InferenceRequest(BaseModel):
    open: List[float] = Field(..., min_length=1)
    high: List[float] = Field(..., min_length=1)
    low: List[float] = Field(..., min_length=1)
    close: List[float] = Field(..., min_length=1)
    volume: List[float] = Field(..., min_length=1)


class InferenceResponse(BaseModel):
    lstm_prediction: float
    prob_up: Optional[float] = None
    action: str
    action_index: int
    latency_ms: float
    reasoning: Optional[str] = None


class DecisionRecordSchema(BaseModel):
    bar_count: int = 0
    timestamp_ms: int = 0
    price: float = 0.0
    mode: str = "lstm_only"
    action: str = "HOLD"
    action_source: str = "model"
    lstm_pred: float = 0.0
    pred_buy_threshold: float = 0.0
    pred_sell_threshold: float = 0.0
    prob_up: Optional[float] = None
    prob_up_buy: float = 0.55
    prob_up_sell: float = 0.45
    recent_dir_accuracy: Optional[float] = None
    recent_dir_accuracy_n: int = 0
    ppo_action: Optional[str] = None
    ppo_probs: Optional[List[float]] = None
    risk_halted: bool = False
    risk_cooldown: int = 0
    drawdown: float = 0.0
    is_long: bool = False
    bars_in_position: int = 0
    reasoning: str = ""


class StateSnapshot(BaseModel):
    bar_count: int
    last_action: str
    last_latency_ms: float
    last_price: float
    equity: float
    cash: float
    position_qty: float
    drawdown: float
    halted: bool
    lstm_prediction: float
    started_at: float
    recent_actions: List[dict]
    last_decision: Optional[Dict[str, Any]] = None
    recent_decisions: List[dict] = []


class StartTradingRequest(BaseModel):
    use_testnet: bool = True
    paper_only: bool = True
    trade_mode: str = "lstm_ppo"
