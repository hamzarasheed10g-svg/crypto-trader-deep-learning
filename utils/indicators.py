"""Technical-indicator implementations referenced by Section 1.4 of the methodology.

All functions operate on ``pandas.Series`` or ``pandas.DataFrame`` inputs and
return outputs of matching length (NaN-padded at the start where the indicator
needs a warm-up period).

References
----------
- Wilder (1978) for RSI / ATR
- Appel (2005) for MACD
- Bollinger (2002) for Bollinger Bands
- Berkowitz et al. (1988) for VWAP intuition
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "rsi",
    "macd",
    "ema",
    "bollinger_bands",
    "atr",
    "vwap",
    "log_returns",
    "rolling_volatility",
    "lagged_returns",
    "volatility_ratio",
    "distance_from_ema",
    "hour_of_day_features",
    "add_all_indicators",
]


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average (Appel, 2005)."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder, 1978).

    Uses Wilder's smoothing (equivalent to an EMA with alpha = 1/period).
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder smoothing
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi_val = 100.0 - (100.0 / (1.0 + rs))
    # When avg_loss is zero the asset only went up -> RSI = 100
    rsi_val = rsi_val.where(avg_loss != 0.0, 100.0)
    return rsi_val


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD line, signal line, and histogram (Appel, 2005)."""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_hist": hist,
    })


def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Bollinger Bands (Bollinger, 2002). Returns upper/lower/width.

    Width = (upper - lower) / middle, a unitless volatility proxy.
    Implements Eq. 1.2 from the methodology.
    """
    ma = close.rolling(window=period, min_periods=period).mean()
    sd = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper = ma + num_std * sd
    lower = ma - num_std * sd
    width = (upper - lower) / ma.replace(0.0, np.nan)
    return pd.DataFrame({
        "bb_upper": upper,
        "bb_lower": lower,
        "bb_width": width,
    })


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (Wilder, 1978)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Cumulative Volume Weighted Average Price (Berkowitz et al., 1988).

    Note: classic VWAP resets daily. We provide a *rolling* cumulative VWAP
    here because the dataset is bar-indexed without session boundaries; the
    backtester treats it as a slowly-moving anchor price.
    """
    typical = (high + low + close) / 3.0
    pv = typical * volume
    cum_pv = pv.cumsum()
    cum_vol = volume.cumsum().replace(0.0, np.nan)
    return cum_pv / cum_vol


def log_returns(close: pd.Series) -> pd.Series:
    """log(p_t / p_{t-1}). NaN at index 0."""
    return np.log(close / close.shift(1))


def rolling_volatility(close: pd.Series, period: int = 24) -> pd.Series:
    """Standard deviation of log returns over the past ``period`` bars."""
    return log_returns(close).rolling(window=period, min_periods=period).std(ddof=0)


# ---------------------------------------------------------------------------
# Additional features for directional forecasting
# ---------------------------------------------------------------------------

def lagged_returns(close: pd.Series, lags: tuple[int, ...] = (1, 5, 24)) -> pd.DataFrame:
    """Past log-returns at multiple horizons.

    Recent multi-horizon returns are among the strongest *directional* features
    for short-horizon forecasting on crypto bars: the 1-bar return captures
    momentum continuation, the 5-bar return captures short-term trend, and the
    24-bar return captures session-level regime. The LSTM cannot easily derive
    these from raw prices alone because they require differencing and scaling.
    """
    out = pd.DataFrame(index=close.index)
    lr = log_returns(close)
    for k in lags:
        if k == 1:
            out[f"return_{k}"] = lr
        else:
            # Sum of k consecutive log returns = log(p_t / p_{t-k})
            out[f"return_{k}"] = lr.rolling(window=k, min_periods=k).sum()
    return out


def volatility_ratio(close: pd.Series, short: int = 6, long: int = 48) -> pd.Series:
    """Ratio of short-window to long-window realised volatility.

    A volatility regime indicator: values > 1 mean the market is currently
    more volatile than its recent baseline (often near reversals or breakouts);
    values < 1 mean a quiet regime. Standard in quantitative finance for
    detecting regime shifts (Andersen & Bollerslev, 1998).
    """
    lr = log_returns(close)
    vol_s = lr.rolling(window=short, min_periods=short).std(ddof=0)
    vol_l = lr.rolling(window=long, min_periods=long).std(ddof=0)
    return vol_s / vol_l.replace(0.0, np.nan)


def distance_from_ema(close: pd.Series, period: int = 50) -> pd.Series:
    """Percent distance of price from its longer-window EMA.

    Captures mean-reversion / overextension signals: large positive values
    mean price is far above its trend (potential pullback); large negative
    values mean price is far below (potential bounce). Normalised by the EMA
    so it is comparable across price levels.
    """
    e = ema(close, period)
    return (close - e) / e.replace(0.0, np.nan)


def hour_of_day_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Sin/cos encoded hour-of-day.

    Crypto markets have well-documented intraday patterns driven by
    overlapping trading sessions across regions. Sin/cos encoding gives the
    model a continuous representation of the hour so that 23:00 and 00:00
    are close in feature space (which integer encoding would not preserve).
    Two columns rather than one because a single periodic feature cannot
    be linearly separated.
    """
    hours = index.hour.values.astype(np.float32)
    angle = 2.0 * np.pi * hours / 24.0
    return pd.DataFrame(
        {
            "hour_sin": np.sin(angle),
            "hour_cos": np.cos(angle),
        },
        index=index,
    )


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Append every indicator the methodology references to a copy of the input.

    Input must contain columns ``open, high, low, close, volume``.
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {sorted(missing)}")

    out = df.copy()
    out["rsi_14"] = rsi(out["close"], period=14)
    out = out.join(macd(out["close"]))
    out["ema_12"] = ema(out["close"], 12)
    out["ema_26"] = ema(out["close"], 26)
    out = out.join(bollinger_bands(out["close"]))
    out["atr_14"] = atr(out["high"], out["low"], out["close"], 14)
    out["vwap"] = vwap(out["high"], out["low"], out["close"], out["volume"])
    out["log_return"] = log_returns(out["close"])
    out["rolling_vol_24"] = rolling_volatility(out["close"], 24)
    # New directional / regime features
    out = out.join(lagged_returns(out["close"], lags=(1, 5, 24)))
    out["vol_ratio_6_48"] = volatility_ratio(out["close"], short=6, long=48)
    out["dist_ema_50"] = distance_from_ema(out["close"], period=50)
    # Time-of-day features only if the index is datetime-like
    if isinstance(out.index, pd.DatetimeIndex):
        out = out.join(hour_of_day_features(out.index))
    return out
