"""Tests for ``utils.indicators``."""
import numpy as np
import pandas as pd
import pytest

from utils.indicators import (
    add_all_indicators, atr, bollinger_bands, ema, log_returns, macd, rsi,
    rolling_volatility, vwap,
)


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    open_ = close + rng.normal(0, 0.3, n)
    vol = rng.uniform(10, 100, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol})


def test_ema_first_period_warmup(sample_ohlcv):
    e = ema(sample_ohlcv["close"], 14)
    assert e.iloc[:13].isna().all()
    assert not e.iloc[13:].isna().any()


def test_rsi_bounds(sample_ohlcv):
    r = rsi(sample_ohlcv["close"], 14).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_rsi_monotone_uptrend():
    # Strictly monotone increasing close -> RSI should saturate near 100
    close = pd.Series(np.arange(1, 100, dtype=float))
    r = rsi(close, 14).dropna()
    assert r.iloc[-1] > 95


def test_macd_components(sample_ohlcv):
    out = macd(sample_ohlcv["close"])
    assert {"macd", "macd_signal", "macd_hist"}.issubset(out.columns)
    # hist = macd - signal exactly (where not NaN)
    valid = out.dropna()
    np.testing.assert_allclose(valid["macd_hist"], valid["macd"] - valid["macd_signal"], atol=1e-10)


def test_bollinger_upper_above_lower(sample_ohlcv):
    bb = bollinger_bands(sample_ohlcv["close"]).dropna()
    assert (bb["bb_upper"] > bb["bb_lower"]).all()


def test_atr_positive(sample_ohlcv):
    a = atr(sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"]).dropna()
    assert (a > 0).all()


def test_vwap_between_low_and_high(sample_ohlcv):
    v = vwap(sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"], sample_ohlcv["volume"]).dropna()
    # VWAP need not be bounded by every single bar's low/high, but a cumulative
    # version should fall between the global min and max of typical prices.
    typ = (sample_ohlcv["high"] + sample_ohlcv["low"] + sample_ohlcv["close"]) / 3.0
    assert v.min() >= typ.min() - 1e-9
    assert v.max() <= typ.max() + 1e-9


def test_log_returns_basic():
    close = pd.Series([100.0, 110.0, 99.0])
    lr = log_returns(close)
    assert np.isnan(lr.iloc[0])
    np.testing.assert_allclose(lr.iloc[1], np.log(110 / 100))
    np.testing.assert_allclose(lr.iloc[2], np.log(99 / 110))


def test_rolling_volatility(sample_ohlcv):
    rv = rolling_volatility(sample_ohlcv["close"], 24).dropna()
    assert (rv >= 0).all()


def test_add_all_indicators_attaches_all_columns(sample_ohlcv):
    out = add_all_indicators(sample_ohlcv)
    for col in ["rsi_14", "macd", "macd_signal", "macd_hist", "ema_12", "ema_26",
                "bb_upper", "bb_lower", "bb_width", "atr_14", "vwap",
                "log_return", "rolling_vol_24"]:
        assert col in out.columns


def test_add_all_indicators_requires_ohlcv():
    with pytest.raises(ValueError, match="Missing required OHLCV columns"):
        add_all_indicators(pd.DataFrame({"close": [1.0, 2.0]}))
