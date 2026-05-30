"""FastAPI application (Section 1.10 of the methodology).

Exposes:
- ``GET  /health``        — liveness + model-loaded check
- ``GET  /docs``          — automatic OpenAPI documentation
- ``POST /infer``         — one-shot inference on a submitted OHLCV window
- ``POST /trade/start``   — kick off the live paper-trader in a background task
- ``POST /trade/stop``    — cancel the running trader
- ``GET  /trade/state``   — current LiveState snapshot
- ``POST /trade/resolve`` — resolve a staged trade via user approval
- ``WS   /ws/state``      — broadcasts LiveState updates as they happen

Run with: ``uvicorn backend.app:app --host 0.0.0.0 --port 8000``
"""
from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.inference import InferenceService
from backend.schemas import (
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    StartTradingRequest,
    StateSnapshot,
)
from utils.config import load_config
from utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

class AppState:
    def __init__(self):
        self.cfg = load_config(os.getenv("CRYPTO_TRADER_CONFIG"))
        self.started_at = time.time()
        self.inference: InferenceService = InferenceService(self.cfg)
        self.trader = None              # set when /trade/start is called
        self.trader_task: Optional[asyncio.Task] = None
        self.connected_websockets: set[WebSocket] = set()
        self.last_state: Optional[dict] = None

    async def broadcast(self, payload: dict) -> None:
        """Push a state snapshot to every connected websocket. Skip broken ones."""
        self.last_state = payload
        dead: list[WebSocket] = []
        for ws in list(self.connected_websockets):
            try:
                await ws.send_json(payload)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            self.connected_websockets.discard(ws)


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Backend starting up (symbol=%s interval=%s)", state.cfg.data.symbol, state.cfg.data.interval)
    # Try to load models eagerly so /health reports correctly. Failure is non-fatal:
    # the user might still be in the training phase.
    try:
        state.inference.load()
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not preload models (%s). /infer will retry on first call.", exc)
    yield
    # Shutdown — cancel trader if running
    if state.trader_task is not None and not state.trader_task.done():
        state.trader_task.cancel()
        try:
            await state.trader_task
        except (asyncio.CancelledError, Exception):
            pass
    log.info("Backend stopped")


app = FastAPI(
    title="Hybrid LSTM-PPO Crypto Trading Backend",
    version="1.0.0",
    description="Real-time inference and paper-trading endpoints for the hybrid LSTM-PPO framework.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(state.cfg.backend.cors_allow_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=state.inference.loaded,
        symbol=state.cfg.data.symbol,
        interval=state.cfg.data.interval,
        uptime_seconds=time.time() - state.started_at,
    )


@app.post("/infer", response_model=InferenceResponse)
async def infer(req: InferenceRequest) -> InferenceResponse:
    lengths = {len(req.open), len(req.high), len(req.low), len(req.close), len(req.volume)}
    if len(lengths) != 1:
        raise HTTPException(status_code=400, detail="All OHLCV arrays must have the same length")

    df = pd.DataFrame({
        "open": req.open, "high": req.high, "low": req.low,
        "close": req.close, "volume": req.volume,
    })
    df.index = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(df), freq="h", tz="UTC")

    try:
        if not state.inference.loaded:
            state.inference.load()
        result = state.inference.infer_window(df)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Models not yet trained: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("Inference failure")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return InferenceResponse(**result)


@app.get("/trade/state", response_model=Optional[StateSnapshot])
async def trade_state():
    return state.last_state


@app.post("/trade/start")
async def trade_start(req: StartTradingRequest):
    if state.trader_task is not None and not state.trader_task.done():
        raise HTTPException(status_code=409, detail="Trader already running")
    try:
        if not state.inference.loaded:
            state.inference.load()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Models not trained: {exc}") from exc

    from deploy.live_trader import LiveTrader, state_to_dict

    executor = None
    if not req.paper_only:
        api_key = os.getenv("BINANCE_TESTNET_API_KEY", "")
        api_secret = os.getenv("BINANCE_TESTNET_API_SECRET", "")
        if not api_key or not api_secret:
            raise HTTPException(status_code=400, detail="Missing BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET")
        from deploy.binance_executor import BinanceExecutor
        executor = BinanceExecutor(
            api_key=api_key, api_secret=api_secret,
            symbol=state.cfg.data.symbol,
            use_testnet=req.use_testnet,
        )

    async def on_state(live_state) -> None:
        await state.broadcast(state_to_dict(live_state))

    state.trader = LiveTrader(
        cfg=state.cfg,
        lstm_model=state.inference._lstm_model,
        ppo_model=state.inference._ppo_model,
        ppo_vecnorm=state.inference._vecnorm,
        scaler=state.inference._scaler,
        executor=executor,
        on_state_change=on_state,
        use_testnet=req.use_testnet,
        trade_mode=req.trade_mode,
    )

    async def _run():
        try:
            await state.trader.run()
        except asyncio.CancelledError:
            log.info("Trader cancelled")
            raise
        except Exception:  # noqa: BLE001
            log.exception("Trader crashed")

    state.trader_task = asyncio.create_task(_run())
    return {"status": "started", "paper_only": req.paper_only, "testnet": req.use_testnet}


@app.post("/trade/stop")
async def trade_stop():
    if state.trader_task is None or state.trader_task.done():
        return {"status": "not_running"}
    state.trader_task.cancel()
    try:
        await state.trader_task
    except (asyncio.CancelledError, Exception):
        pass
    state.trader_task = None
    state.trader = None
    return {"status": "stopped"}


class ApprovalRequest(BaseModel):
    approved: bool

@app.post("/trade/resolve")
async def trade_resolve(req: ApprovalRequest):
    if state.trader is None:
        raise HTTPException(status_code=400, detail="Trader not running")
    
    state.trader.resolve_pending(req.approved)
    
    from deploy.live_trader import state_to_dict
    await state.broadcast(state_to_dict(state.trader.state))
    return {"status": "resolved"}


# ---------------------------------------------------------------------------
# WebSocket endpoint for live state fan-out
# ---------------------------------------------------------------------------

@app.websocket("/ws/state")
async def ws_state(ws: WebSocket):
    await ws.accept()
    state.connected_websockets.add(ws)
    log.info("WS connected (%d total)", len(state.connected_websockets))
    try:
        # Send the most recent snapshot immediately so the client has something
        if state.last_state is not None:
            await ws.send_json(state.last_state)
        while True:
            # Keep the connection alive; we don't expect inbound messages but
            # consume them if the client sends pings or commands.
            msg = await ws.receive_text()
            if msg == "ping":
                await ws.send_json({"type": "pong", "ts": time.time()})
    except WebSocketDisconnect:
        pass
    finally:
        state.connected_websockets.discard(ws)
        log.info("WS disconnected (%d remaining)", len(state.connected_websockets))


# ---------------------------------------------------------------------------
# Frontend static files
# ---------------------------------------------------------------------------
import pathlib as _pathlib
_FRONTEND = _pathlib.Path(__file__).resolve().parent.parent / "frontend"
if _FRONTEND.exists():
    app.mount("/ui", StaticFiles(directory=str(_FRONTEND)), name="frontend")

@app.get("/", include_in_schema=False)
async def root():
    dashboard = _FRONTEND / "dashboard.html"
    if dashboard.exists():
        return FileResponse(str(dashboard))
    return {"message": "LSTM-PPO Trading API — visit /docs"}