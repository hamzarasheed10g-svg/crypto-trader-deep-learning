"""Tests for ``risk.manager``."""
import pytest

from risk.manager import RiskAction, RiskLimits, RiskManager


@pytest.fixture
def rm() -> RiskManager:
    return RiskManager(RiskLimits(
        stop_loss_pct=0.02, take_profit_pct=0.05, max_drawdown_pct=0.10,
        max_position_fraction=1.0, min_trade_notional=10.0, cooldown_bars=0,
    ))


def test_buy_allowed_with_sufficient_cash(rm):
    d = rm.validate_buy(cash_available=1000, equity=1000, current_position_value=0, intended_notional=500)
    assert d.action == RiskAction.ALLOW
    assert d.suggested_qty == 500


def test_buy_rejected_below_min_notional(rm):
    d = rm.validate_buy(cash_available=1000, equity=1000, current_position_value=0, intended_notional=5)
    assert d.action == RiskAction.REJECT
    assert "below_min_notional" in d.reason


def test_buy_trimmed_to_position_limit(rm):
    # Position fraction limit is 100%; already have 800 of equity 1000 in BTC,
    # so we can only add 200 more.
    d = rm.validate_buy(cash_available=500, equity=1000, current_position_value=800, intended_notional=400)
    assert d.action == RiskAction.ALLOW
    assert d.reason == "trimmed_to_limit"
    assert d.suggested_qty == pytest.approx(200)


def test_stop_loss_triggers_force_close(rm):
    d = rm.check_open_position(avg_entry=100, current_price=97)  # -3% (> stop_loss 2%)
    assert d.action == RiskAction.FORCE_CLOSE
    assert "stop_loss" in d.reason


def test_take_profit_triggers_force_close(rm):
    d = rm.check_open_position(avg_entry=100, current_price=106)  # +6% (> take_profit 5%)
    assert d.action == RiskAction.FORCE_CLOSE
    assert "take_profit" in d.reason


def test_open_position_within_band(rm):
    d = rm.check_open_position(avg_entry=100, current_price=101)  # +1%
    assert d.action == RiskAction.ALLOW


def test_drawdown_halts(rm):
    rm.update_equity(1000.0)
    d = rm.update_equity(890.0)  # 11% drawdown — over 10% limit
    assert d.action == RiskAction.HALT
    assert rm.is_halted


def test_buy_rejected_when_halted(rm):
    rm.update_equity(1000.0); rm.update_equity(800.0)
    d = rm.validate_buy(cash_available=1000, equity=800, current_position_value=0, intended_notional=200)
    assert d.action == RiskAction.REJECT
    assert d.reason == "risk_halted"


def test_sell_allowed_when_halted(rm):
    """We always permit closing a position even after a halt, to exit cleanly."""
    rm.update_equity(1000.0); rm.update_equity(800.0)
    d = rm.validate_sell(position_qty=0.1)
    assert d.action == RiskAction.ALLOW
    assert d.reason == "halt_close"


def test_sell_rejected_with_no_position(rm):
    d = rm.validate_sell(position_qty=0)
    assert d.action == RiskAction.REJECT


def test_reset(rm):
    rm.update_equity(1000.0); rm.update_equity(800.0)
    assert rm.is_halted
    rm.reset()
    assert not rm.is_halted
    assert rm.peak_equity == 0.0


def test_cooldown_after_force_close():
    rm = RiskManager(RiskLimits(stop_loss_pct=0.02, take_profit_pct=0.05,
                                max_drawdown_pct=0.10, cooldown_bars=2))
    # Trigger a stop-loss
    rm.check_open_position(avg_entry=100, current_price=97)
    # Next buy attempt should be rejected for cooldown
    d = rm.validate_buy(1000, 1000, 0, 500)
    assert d.action == RiskAction.REJECT and d.reason == "cooldown"
    # Tick equity twice (each ticks down cooldown)
    rm.update_equity(1000.0); rm.update_equity(1000.0)
    d = rm.validate_buy(1000, 1000, 0, 500)
    assert d.action == RiskAction.ALLOW
