"""Tests for ``backtest.metrics``."""
import math

import numpy as np
import pytest

from backtest.metrics import (
    annualised_return, annualised_volatility, calmar_ratio, compute_metrics,
    cumulative_return, equity_to_returns, max_drawdown, sharpe_ratio,
    sortino_ratio, win_rate_and_profit_factor,
)


def test_equity_to_returns():
    eq = np.array([100, 110, 105])
    r = equity_to_returns(eq)
    np.testing.assert_allclose(r, [0.1, 105/110 - 1])


def test_cumulative_return():
    assert cumulative_return(np.array([100.0, 110.0])) == pytest.approx(0.1)
    assert cumulative_return(np.array([])) == 0.0
    assert cumulative_return(np.array([100.0])) == 0.0


def test_max_drawdown_known_curve():
    # Peak 200, trough 150 -> DD = 0.25
    eq = np.array([100, 150, 200, 180, 150, 170])
    assert max_drawdown(eq) == pytest.approx(0.25)


def test_sharpe_and_sortino_sign():
    # Steady upward drift
    eq = np.linspace(100, 200, 100)
    s = sharpe_ratio(eq, bars_per_year=252)
    so = sortino_ratio(eq, bars_per_year=252)
    assert s > 0
    assert so > 0 or math.isinf(so)


def test_sharpe_zero_volatility():
    # Identical equity values -> zero excess returns -> zero Sharpe
    eq = np.full(50, 100.0)
    assert sharpe_ratio(eq, bars_per_year=252) == 0.0


def test_win_rate_profit_factor():
    wr, pf = win_rate_and_profit_factor([100, -50, 200, -100, 50])
    assert wr == 0.6
    # gross profit 350 / gross loss 150 = 2.333
    assert pf == pytest.approx(350 / 150)


def test_win_rate_only_winners_infinite_profit_factor():
    wr, pf = win_rate_and_profit_factor([10, 20, 30])
    assert wr == 1.0
    assert math.isinf(pf)


def test_compute_metrics_serialises_inf():
    eq = np.array([100, 110, 120, 130])
    m = compute_metrics(eq, [10, 20], bars_per_year=8760)
    d = m.to_dict()
    # If profit_factor is infinite (no losers), it should become the string "inf"
    if m.profit_factor == math.inf:
        assert d["profit_factor"] == "inf"


def test_calmar_zero_drawdown_returns_zero():
    eq = np.full(10, 100.0)
    assert calmar_ratio(eq, bars_per_year=252) == 0.0


def test_annualised_return_and_vol_smoke():
    rng = np.random.default_rng(0)
    eq = 1000.0 * np.cumprod(1 + rng.normal(0.0005, 0.01, 1000))
    ar = annualised_return(eq, bars_per_year=252)
    vol = annualised_volatility(eq, bars_per_year=252)
    assert isinstance(ar, float)
    assert vol > 0
