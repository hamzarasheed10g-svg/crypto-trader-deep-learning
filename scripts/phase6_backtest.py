"""Phase 6 (Section 1.12): backtest the trained PPO + baselines on the held-out test split.

Runs the trained policy and every baseline through identical ``CryptoTradingEnv``
instances, computes the full metric bundle, and produces:

- ``artifacts/metrics.json``      — machine-readable metrics
- ``artifacts/backtest_report.html`` — human-readable comparison report with charts

Usage
-----
    python -m scripts.phase6_backtest
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from agents.ppo import load_ppo
from backtest.baselines import build_baselines
from backtest.engine import BacktestResult, backtest_policy, backtest_strategy
from data.preprocess import FeatureScaler
from env.trading_env import CryptoTradingEnv, LSTMSignalProvider
from models.train_lstm import load_lstm_checkpoint, resolve_device
from utils.config import load_config, resolve_path
from utils.logging import get_logger
from utils.seeding import set_global_seed

log = get_logger(__name__)


def _equity_plot_png(results: List[BacktestResult]) -> str:
    """Render an equity-curve comparison plot as a base64-encoded PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for r in results:
        ax.plot(r.equity_curve, label=r.name)
    ax.set_title("Equity Curve Comparison (test split)")
    ax.set_xlabel("Bar index")
    ax.set_ylabel("Portfolio equity (USDT)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _drawdown_plot_png(results: List[BacktestResult]) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    for r in results:
        eq = np.asarray(r.equity_curve, dtype=float)
        peak = np.maximum.accumulate(eq) if len(eq) else eq
        dd = (peak - eq) / np.where(peak > 0, peak, 1.0) if len(eq) else eq
        ax.plot(dd, label=r.name)
    ax.set_title("Drawdown")
    ax.set_xlabel("Bar index")
    ax.set_ylabel("Drawdown")
    ax.legend()
    ax.grid(True, alpha=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _render_html_report(results: List[BacktestResult], cfg, out_path: Path) -> None:
    eq_png = _equity_plot_png(results)
    dd_png = _drawdown_plot_png(results)

    rows_html = ""
    headers = [
        "Strategy", "Cum. return", "Ann. return", "Ann. vol.",
        "Sharpe", "Sortino", "Max DD", "Calmar", "Win rate", "Profit factor", "# Trades",
    ]
    rows_html += "<tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"
    for r in results:
        m = r.metrics
        def _fmt(v, pct=False):
            if isinstance(v, float):
                if pct:
                    return f"{v*100:.2f}%"
                return f"{v:.4f}"
            return str(v)
        rows_html += "<tr>"
        rows_html += f"<td><b>{r.name}</b></td>"
        rows_html += f"<td>{_fmt(m.cumulative_return, pct=True)}</td>"
        rows_html += f"<td>{_fmt(m.annualised_return, pct=True)}</td>"
        rows_html += f"<td>{_fmt(m.annualised_volatility, pct=True)}</td>"
        rows_html += f"<td>{_fmt(m.sharpe)}</td>"
        rows_html += f"<td>{_fmt(m.sortino)}</td>"
        rows_html += f"<td>{_fmt(m.max_drawdown, pct=True)}</td>"
        rows_html += f"<td>{_fmt(m.calmar)}</td>"
        rows_html += f"<td>{_fmt(m.win_rate, pct=True)}</td>"
        rows_html += f"<td>{_fmt(m.profit_factor)}</td>"
        rows_html += f"<td>{m.num_trades}</td>"
        rows_html += "</tr>"

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Backtest Report — {cfg.data.symbol} {cfg.data.interval}</title>
<style>
body {{ font-family: -apple-system, Segoe UI, sans-serif; max-width: 1100px; margin: 24px auto; color: #222; }}
h1, h2 {{ font-weight: 600; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 14px; }}
th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: right; }}
th {{ background: #f4f4f4; }}
td:first-child {{ text-align: left; }}
img {{ width: 100%; max-width: 1100px; }}
.small {{ color: #777; font-size: 12px; }}
</style></head><body>
<h1>Backtest Report</h1>
<div class="small">Symbol: {cfg.data.symbol} &nbsp; Interval: {cfg.data.interval} &nbsp; Test bars: {len(results[0].equity_curve)}</div>
<h2>Performance metrics</h2>
<table>{rows_html}</table>
<h2>Equity curves</h2>
<img alt="equity" src="data:image/png;base64,{eq_png}"/>
<h2>Drawdown</h2>
<img alt="drawdown" src="data:image/png;base64,{dd_png}"/>
</body></html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--interval", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.seed)
    symbol = args.symbol or cfg.data.symbol
    interval = args.interval or cfg.data.interval

    processed_dir = resolve_path(cfg, cfg.paths.processed_data_dir)
    test_df = pd.read_parquet(processed_dir / f"{symbol}_{interval}_test.parquet")
    scaler = FeatureScaler.load(resolve_path(cfg, cfg.paths.lstm_scaler))

    log.info("Loading models for backtest")
    device = resolve_device(cfg.device)
    lstm_model, _ = load_lstm_checkpoint(resolve_path(cfg, cfg.paths.lstm_model), device=device)
    provider = LSTMSignalProvider(lstm_model, scaler, sequence_length=cfg.lstm.sequence_length, device=device)
    test_signals = provider.predict_all(test_df)

    ppo_model, vec_env = load_ppo(
        resolve_path(cfg, cfg.paths.ppo_model),
        resolve_path(cfg, cfg.paths.ppo_vecnorm),
        test_df, scaler, test_signals, cfg,
    )

    def env_factory():
        return CryptoTradingEnv(raw_df=test_df, scaler=scaler, lstm_signals=test_signals, cfg=cfg, seed=cfg.seed)

    results: List[BacktestResult] = []

    log.info("Backtesting LSTM-PPO policy")
    ppo_result = backtest_policy(
        name="lstm_ppo",
        env_factory=env_factory,
        model=ppo_model,
        vec_env=vec_env,
        bars_per_year=cfg.backtest.bars_per_year,
        risk_free=cfg.backtest.risk_free_rate,
    )
    results.append(ppo_result)

    for strat in build_baselines(cfg):
        log.info("Backtesting baseline: %s", strat.name)
        actions = strat.actions(test_df)
        # Baselines run without LSTM signals (signals=None means env uses zeros).
        def factory(strat_actions=actions):
            return CryptoTradingEnv(raw_df=test_df, scaler=scaler, lstm_signals=None, cfg=cfg, seed=cfg.seed)
        res = backtest_strategy(
            name=strat.name,
            env_factory=factory,
            actions=actions,
            bars_per_year=cfg.backtest.bars_per_year,
            risk_free=cfg.backtest.risk_free_rate,
        )
        results.append(res)

    metrics_path = resolve_path(cfg, cfg.paths.metrics_json)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({r.name: r.to_dict() for r in results}, f, indent=2)
    log.info("Wrote metrics to %s", metrics_path)

    report_path = resolve_path(cfg, cfg.paths.backtest_report)
    _render_html_report(results, cfg, report_path)
    log.info("Wrote HTML report to %s", report_path)

    log.info("Summary:")
    for r in results:
        m = r.metrics
        log.info("  %-15s | cum_ret=%6.2f%% sharpe=%5.2f maxDD=%5.2f%% trades=%d",
                 r.name, m.cumulative_return * 100, m.sharpe, m.max_drawdown * 100, m.num_trades)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
