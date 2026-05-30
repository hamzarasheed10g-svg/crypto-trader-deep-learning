# Hybrid LSTM–PPO Cryptocurrency Trading Framework

End-to-end implementation of the methodology in `Deep_Learning_Lab_Methodology.docx`:
LSTM for short-term price forecasting + PPO Deep Reinforcement Learning agent for
autonomous trading, served through a FastAPI + WebSocket distributed backend, with
risk-managed paper trading on Binance Testnet.

**Authors of methodology:** Ali Hamza (01-134232-035), Mehar Ali Musa (01-134232-097)
**Reference symbol/timeframe:** BTC/USDT, 1h

---

## 1. Project Layout

```
crypto_trader/
├── configs/             YAML config files (single source of truth)
├── data/                Data acquisition (REST + WebSocket) and feature engineering
├── env/                 Gymnasium-compatible trading environment (MDP)
├── models/              LSTM forecasting network + training loop
├── agents/              PPO agent (stable-baselines3) + callbacks
├── risk/                Risk manager: stop-loss, position sizing, drawdown limits
├── backtest/            Vectorised backtester + baseline strategies + metrics
├── backend/             FastAPI app, WebSocket streamer, inference service
├── deploy/              Live paper-trading orchestrator (Binance Testnet)
├── utils/               Logging, seeding, indicators, helpers
├── scripts/             CLI entrypoints (one per pipeline phase)
├── tests/               Unit + integration tests (synthetic data, no network)
├── artifacts/           Saved models, scalers, training curves (git-ignored)
└── notebooks/           Optional exploration notebooks
```

## 2. Setup

```bash
# Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                # then fill in BINANCE_TESTNET_API_KEY / SECRET
```

Binance Spot Testnet keys are free at: https://testnet.binance.vision/
**Never use mainnet keys with this code.** The codebase defaults to testnet endpoints.

## 3. Pipeline (7 phases from the methodology)

Each phase is a script in `scripts/` and can be run independently. They share state
via the `artifacts/` directory.

```bash
# Phase 1: Data Collection and Preprocessing
python -m scripts.phase1_fetch_data --symbol BTCUSDT --interval 1h --years 3
python -m scripts.phase1_preprocess

# Phase 2: LSTM-Based Predictive Model Development
python -m scripts.phase2_train_lstm

# Phase 3: Deep Reinforcement Learning Environment Construction
python -m scripts.phase3_check_env          # sanity-checks the Gymnasium env

# Phase 4 + 5: Hybrid LSTM-DRL Integration & PPO Training
python -m scripts.phase4_train_ppo

# Phase 6: Backtesting and Performance Evaluation
python -m scripts.phase6_backtest

# Phase 7: Real-Time Deployment and Validation
uvicorn backend.app:app --host 0.0.0.0 --port 8000     # in one shell
python -m scripts.phase7_paper_trade                    # in another shell
```

Sensible defaults are in `configs/default.yaml`; override with `--config path/to/file.yaml`.

## 4. End-to-end smoke test (no network needed)

Runs the full pipeline on synthetic OHLCV so you can verify everything wires up
before you point it at Binance:

```bash
python -m scripts.smoke_test
```

This trains a tiny LSTM (few epochs), a tiny PPO (few thousand steps), runs a
backtest, and exercises the FastAPI inference endpoints — all in < 5 minutes on CPU.

## 5. Running tests

```bash
pytest -q
```

## 6. Key design decisions

- **Pure PyTorch** for the LSTM (Paszke et al., 2019). No TensorFlow dependency —
  the methodology mentions both, but a single framework keeps the surface area small.
- **stable-baselines3** for PPO (Schulman et al., 2017). Battle-tested, GPU-aware,
  matches the clipped-surrogate objective in Eq. 1.7 of the methodology.
- **Gymnasium** for the MDP wrapper (Sutton & Barto, 2018). State includes the
  LSTM's prediction for the next bar, as specified in §1.8 of the methodology.
- **FastAPI + WebSockets** for the distributed backend (Ramírez 2018; RFC 6455).
  Async endpoints, JSON payloads, automatic OpenAPI docs at `/docs`.
- **Binance Testnet only by default.** Live mainnet trading requires explicitly
  setting `BINANCE_USE_MAINNET=true` in `.env` and is gated by a confirmation prompt.

## 7. Disclaimer

This is a research/lab project. Cryptocurrency trading involves substantial risk.
The PPO agent's policy is non-deterministic and trained on historical data — past
performance does not predict future returns. Use the testnet. Do not deploy this
against real funds without independent review.
