# LSTM-PPO Crypto Trading Framework — Setup Guide

This is a hybrid LSTM + PPO reinforcement learning system that trades BTC/USDT
using live data from Binance. It includes a real-time dashboard at localhost:8000.

## Requirements

- **Windows 10/11** (tested) or macOS/Linux
- **Python 3.10, 3.11, or 3.12** (NOT 3.13 — some ML libraries don't support it yet)
- **Internet connection** (for Binance data)
- **~5 GB disk space** (for dependencies + trained models + data cache)
- **~8 GB RAM minimum**

Check your Python version:
```powershell
python --version
```

If you don't have Python or have 3.13, install Python 3.12 from:
https://www.python.org/downloads/release/python-3128/

During install, **check the box "Add python.exe to PATH"**.

---

## Step 1 — Extract the project

Extract the zip file to a folder. Recommended location: `C:\crypto_trader` or `E:\crypto_trader`.

Avoid paths with spaces (like `Desktop` or `My Documents`) — they can break some Python imports.

---

## Step 2 — Open the project in VS Code

1. Install VS Code from https://code.visualstudio.com/ if you don't have it
2. Open VS Code → File → Open Folder → select the `crypto_trader` folder
3. Open the integrated terminal: View → Terminal (or ``Ctrl+` ``)

The terminal should open with the project folder as the current directory:
```
PS E:\crypto_trader>
```

---

## Step 3 — Create a virtual environment

In the VS Code terminal, run:

```powershell
python -m venv .venv
```

This takes ~30 seconds and creates a `.venv/` folder.

---

## Step 4 — Activate the virtual environment

```powershell
.venv\Scripts\Activate.ps1
```

You should see `(.venv)` appear at the start of the terminal prompt.

**If PowerShell blocks the script with an "execution policy" error**, run this once and try again:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Step 5 — Install dependencies

```powershell
pip install -r requirements.txt
pip install pyarrow
```

This downloads ~2 GB of packages and takes 5-15 minutes depending on internet speed.

**On AMD GPU / no NVIDIA**: torch will install in CPU mode automatically. That's fine — it just trains slower.

**On NVIDIA GPU**: if you want GPU acceleration, uninstall torch then reinstall the CUDA build:
```powershell
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Step 6 — Verify the trained models are present

```powershell
dir artifacts
```

You should see:
- `lstm_model.pt` (~5-15 MB)
- `lstm_scaler.pkl`
- `ppo_model.zip`
- `ppo_vecnorm.pkl`

**If any are missing**, the models weren't included in the zip. You'll need to train them yourself — see "Training from scratch" at the bottom.

---

## Step 7 — Start the backend

The models were trained with the **fast 1m config**, so set that as the active config before starting:

```powershell
$env:CRYPTO_TRADER_CONFIG = "configs/fast_1m.yaml"
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Wait until you see:
```
INFO:     Loading LSTM checkpoint from ...\artifacts\lstm_model.pt
INFO:     Loading PPO model from ...\artifacts\ppo_model.zip
INFO:     Inference service ready
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Keep this terminal open and running.** Don't close it.

> Note: every new terminal session forgets the env variable. If you close VS Code and reopen,
> you'll need to set `$env:CRYPTO_TRADER_CONFIG` again before starting uvicorn.

---

## Step 8 — Open the dashboard

In your web browser, go to:

```
http://localhost:8000
```

You should see the LSTM-PPO dashboard with:
- Live BTC price streaming in the topbar
- A green candlestick chart of BTC/USDT (streaming from Binance)
- API Health panel showing "loaded" and "ok"

If the page is blank or says "Backend offline", refresh with `Ctrl+Shift+R`.

---

## Step 9 — Start trading

On the dashboard:

1. Choose a trading mode (left sidebar):
   - **LSTM ONLY** — LSTM forecaster drives trades directly
   - **LSTM + PPO** — hybrid agent (full methodology)

2. Click **▶ START PAPER TRADING**

3. Wait 1-2 minutes for the first 1-minute candle to close. The first trade will fire and you'll see:
   - A popup notification flashing on screen
   - A row added to the Trade Log table
   - A green BUY arrow on the candlestick chart

4. Trades continue firing every 1-5 minutes as the LSTM forecast oscillates.

---

## Step 10 — Stop trading

Click **■ STOP TRADING** on the dashboard, or press `Ctrl+C` in the terminal running uvicorn.

---

## Troubleshooting

### "Backend offline" toast keeps appearing
The uvicorn server stopped. Re-run Step 7.

### "Port 8000 is already in use"
Another Python process is hogging the port. Kill it:
```powershell
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
```
Then re-run Step 7.

### "Model not loaded" warning banner
You forgot to set `$env:CRYPTO_TRADER_CONFIG = "configs/fast_1m.yaml"` before starting uvicorn,
OR the model files in `artifacts/` are missing/corrupted. Check `dir artifacts` and re-run Step 7.

### Browser says "Cannot reach 0.0.0.0:8000"
Use `localhost:8000` instead, not `0.0.0.0`.

### No trades fire after 5+ minutes
Stop trading, switch the mode (LSTM ONLY <-> LSTM + PPO), and start again. If still nothing, the LSTM prediction may be stuck — restart uvicorn and try again.

### Browser tab shows old version after edits
Hard refresh: `Ctrl+Shift+R`

---

## Training from scratch (only if model files are missing)

If you need to retrain the models, you have three options:

### Option 1 — Fast 1m retrain (~2-3 hours) [RECOMMENDED]
Best balance of training time and trading quality on 1m bars:
```powershell
.\retrain_fast.ps1
```

When it finishes, models are written to `artifacts/`. Then start the backend:
```powershell
$env:CRYPTO_TRADER_CONFIG = "configs/fast_1m.yaml"
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

### Option 2 — Heavy 1m retrain (~6-12 hours)
Larger model, 180 days of history. Marginal quality gain over fast on noisy 1m data:
```powershell
.\retrain_heavy.ps1
```
After:
```powershell
$env:CRYPTO_TRADER_CONFIG = "configs/heavy_1m.yaml"
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

### Option 3 — Default 1h retrain (~30-60 minutes)
Original methodology config, trains on 1h bars over 3 years:
```powershell
python -m scripts.phase1_fetch_data
python -m scripts.phase1_preprocess
python -m scripts.phase2_train_lstm
python -m scripts.phase3_check_env
python -m scripts.phase4_train_ppo
python -m scripts.phase6_backtest
```
After (no env variable needed — this uses the default config):
```powershell
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

---

## What's running

- **Live BTC price**: streamed from Binance public WebSocket (real production data)
- **Trade execution**: paper trading — simulated locally, no real money or exchange account needed
- **LSTM model**: PyTorch, trained on real historical BTC data
- **PPO model**: stable-baselines3, trained on real historical BTC data
- **Backend**: FastAPI + uvicorn
- **Dashboard**: HTML + Chart.js + TradingView Lightweight Charts

Compare the BTC price on the dashboard to https://www.binance.com/en/trade/BTC_USDT — they update at the same time because they're literally the same data.

---

## Quick reference — minimal commands

After everything is set up, you only need 3 commands to use the system:

```powershell
.venv\Scripts\Activate.ps1
$env:CRYPTO_TRADER_CONFIG = "configs/fast_1m.yaml"
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000` in your browser.
