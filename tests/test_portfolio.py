"""Tests for ``env.portfolio``."""
import pytest

from env.portfolio import Portfolio


def test_buy_consumes_cash_and_grows_position():
    p = Portfolio(cash=1000.0)
    trade = p.market_buy(price=100.0, notional=500.0, fee_rate=0.001)
    assert trade is not None
    assert trade.side == "BUY"
    # Fee = 500 * 0.001 = 0.50; invest = 499.50; qty = 4.995
    assert p.position_qty == pytest.approx(4.995, abs=1e-6)
    assert p.cash == pytest.approx(500.0)
    assert p.avg_entry_price == pytest.approx(100.0)


def test_sell_partial_then_full():
    p = Portfolio(cash=0.0)
    p.position_qty = 1.0
    p.avg_entry_price = 100.0
    p.cash = 0.0
    trade = p.market_sell(price=110.0, fraction=0.5, fee_rate=0.001)
    assert trade is not None
    assert p.position_qty == pytest.approx(0.5)
    # Profit per share = 10; realised on half share = 5 - fees
    assert p.realized_pnl > 0
    # Now sell the rest
    trade2 = p.market_sell(price=120.0, fraction=1.0, fee_rate=0.001)
    assert trade2 is not None
    assert p.position_qty == 0.0
    assert p.avg_entry_price == 0.0


def test_equity_curve_and_peak():
    p = Portfolio(cash=1000.0)
    p.market_buy(price=100.0, notional=500.0, fee_rate=0.0)
    # No fees: equity = cash + qty*price = 500 + 5*100 = 1000
    assert p.equity(100.0) == pytest.approx(1000.0)
    p.record(100.0)
    p.record(110.0)  # equity = 500 + 5*110 = 1050
    assert p.equity_curve[-1] == pytest.approx(1050.0)
    assert p.peak_equity >= 1050.0
    # Drawdown after a dip
    dd = p.drawdown(99.0)
    assert dd > 0


def test_no_buy_when_cash_is_zero():
    p = Portfolio(cash=0.0)
    res = p.market_buy(price=100.0, notional=100.0, fee_rate=0.001)
    assert res is None
    assert p.position_qty == 0.0


def test_no_buy_when_fee_eats_entire_notional():
    p = Portfolio(cash=10.0)
    # Fee rate of 1.0 means the whole notional gets eaten by fees
    res = p.market_buy(price=100.0, notional=5.0, fee_rate=1.0)
    assert res is None
    assert p.position_qty == 0.0


def test_weighted_average_entry_price():
    p = Portfolio(cash=10_000.0)
    p.market_buy(price=100.0, notional=1000.0, fee_rate=0.0)
    p.market_buy(price=200.0, notional=2000.0, fee_rate=0.0)
    # First fill: 10 units at 100 ; second fill: 10 units at 200 ; avg = 150
    assert p.avg_entry_price == pytest.approx(150.0)
    assert p.position_qty == pytest.approx(20.0)
