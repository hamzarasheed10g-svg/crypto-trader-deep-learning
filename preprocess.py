"""Data preprocessing pipeline (Section 1.4 of the methodology).

Stages
------
1. Deduplicate and sort on timestamp
2. Synchronise timestamps to a regular grid; fill or drop gaps
3. Winsorise per-feature outliers (|z| > threshold)
4. Compute technical indicators (handled by ``utils.indicators.add_all_indicators``)
5. Drop warm-up NaNs
6. Min-max (or z-score) normalise — fit on train split only to avoid leakage
7. Slice into rolling windows of length ``sequence_length`` for LSTM training

Min-max normalisation matches Eq. 1.1 in the methodology:
    x' = (x - x_min) / (x_max - x_min)
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils.indicators import add_all_indicators
from utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def deduplicate_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df.index.duplicated(keep="first")].sort_index()


def synchronise_grid(df: pd.DataFrame, interval_seconds: int, fill: str = "linear") -> pd.DataFrame:
    """Reindex to a regular timestamp grid and fill missing bars.

    ``fill`` ∈ {"linear", "ffill", "drop"}.
    """
    if df.empty:
        return df
    freq = pd.Timedelta(seconds=interval_seconds)
    grid = pd.date_range(df.index[0], df.index[-1], freq=freq, tz=df.index.tz)
    out = df.reindex(grid)
    if fill == "drop":
        out = out.dropna()
    elif fill == "ffill":
        out = out.ffill()
    elif fill == "linear":
        out = out.interpolate(method="linear", limit_direction="forward")
        out = out.ffill().bfill()
    else:
        raise ValueError(f"Unknown fill method {fill!r}")
    out.index.name = df.index.name
    return out


def winsorise_outliers(df: pd.DataFrame, z_threshold: float, cols: Iterable[str]) -> pd.DataFrame:
    """Cap per-column outliers at ±z_threshold standard deviations (no row drops).

    Cryptocurrency data has heavy tails: dropping outliers throws away legitimate
    moves. Winsorising keeps the row but prevents wildly skewed normalisation.
    """
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        col = out[c]
        mu = col.mean()
        sd = col.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            continue
        upper = mu + z_threshold * sd
        lower = mu - z_threshold * sd
        out[c] = col.clip(lower=lower, upper=upper)
    return out


# ---------------------------------------------------------------------------
# Splitting and normalisation
# ---------------------------------------------------------------------------

def time_series_split(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split — no shuffling (data is a time series)."""
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train : n_train + n_val].copy()
    test = df.iloc[n_train + n_val :].copy()
    return train, val, test


@dataclass
class FeatureScaler:
    """Wraps a sklearn scaler + the feature order used to fit it.

    We persist the feature order so that inference uses identical columns.
    """

    method: str
    feature_names: List[str]
    _scaler: object = field(default=None, repr=False)

    def fit(self, df: pd.DataFrame) -> "FeatureScaler":
        if self.method == "minmax":
            self._scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        elif self.method == "zscore":
            self._scaler = StandardScaler()
        elif self.method == "none":
            self._scaler = None
        else:
            raise ValueError(f"Unknown normalize method {self.method!r}")
        if self._scaler is not None:
            self._scaler.fit(df[self.feature_names].values)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self._scaler is None:
            return df[self.feature_names].values.astype(np.float32)
        return self._scaler.transform(df[self.feature_names].values).astype(np.float32)

    def inverse_transform_close(self, scaled_close: np.ndarray) -> np.ndarray:
        """Inverse-transform only the ``close`` column. Useful for plotting."""
        if self._scaler is None:
            return scaled_close
        if "close" not in self.feature_names:
            raise KeyError("'close' is not a tracked feature")
        idx = self.feature_names.index("close")
        if self.method == "minmax":
            data_min = self._scaler.data_min_[idx]
            data_max = self._scaler.data_max_[idx]
            return scaled_close * (data_max - data_min) + data_min
        # zscore
        mean = self._scaler.mean_[idx]
        scale = self._scaler.scale_[idx]
        return scaled_close * scale + mean

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"method": self.method, "feature_names": self.feature_names, "scaler": self._scaler}, f)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureScaler":
        with Path(path).open("rb") as f:
            blob = pickle.load(f)
        obj = cls(method=blob["method"], feature_names=list(blob["feature_names"]))
        obj._scaler = blob["scaler"]
        return obj


# ---------------------------------------------------------------------------
# Sequence generation for the LSTM
# ---------------------------------------------------------------------------

def make_sequences(
    feature_array: np.ndarray,
    target_array: np.ndarray,
    sequence_length: int,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Slide a length-``sequence_length`` window over ``feature_array``.

    Each input X[i] is feature_array[i : i+seq_len], and the corresponding target
    y[i] = target_array[i + seq_len + horizon - 1]. Returns float32 arrays.
    """
    n = len(feature_array)
    last = n - sequence_length - horizon + 1
    if last <= 0:
        raise ValueError(
            f"Not enough rows ({n}) for sequence_length={sequence_length} and horizon={horizon}"
        )
    X = np.empty((last, sequence_length, feature_array.shape[1]), dtype=np.float32)
    y = np.empty((last,), dtype=np.float32)
    for i in range(last):
        X[i] = feature_array[i : i + sequence_length]
        y[i] = target_array[i + sequence_length + horizon - 1]
    return X, y


def build_target(df: pd.DataFrame, target_kind: str) -> pd.Series:
    """Build the LSTM regression/classification target column.

    For ``target_kind == "dual"``, returns the *regression* target (log_return).
    The companion direction target is derived from it by ``y_dir = (y_reg > 0)``;
    see ``make_sequences_dual`` for the dual-target sequence builder.
    """
    if target_kind == "log_return":
        return df["log_return"].astype(np.float32)
    if target_kind == "close":
        return df["close"].astype(np.float32)
    if target_kind == "direction":
        return (df["log_return"] > 0).astype(np.float32)
    if target_kind == "dual":
        # Dual-head consumes the regression target; classification target is
        # derived from sign() inside make_sequences_dual.
        return df["log_return"].astype(np.float32)
    raise ValueError(f"Unknown prediction_target {target_kind!r}")


def make_sequences_dual(
    feature_array: np.ndarray,
    target_array: np.ndarray,
    sequence_length: int,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Like ``make_sequences`` but also returns a direction target.

    Returns ``(X, y_reg, y_dir)`` where ``y_reg`` is the next log-return and
    ``y_dir`` is its sign as a float (1.0 = up, 0.0 = down/flat). Same window
    indexing as ``make_sequences`` so the two are pairwise aligned.
    """
    X, y_reg = make_sequences(feature_array, target_array, sequence_length, horizon)
    y_dir = (y_reg > 0).astype(np.float32)
    return X, y_reg, y_dir


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

@dataclass
class ProcessedData:
    raw: pd.DataFrame                 # full feature DataFrame (post-indicator, NaN-dropped)
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_names: List[str]
    target_name: str
    scaler: FeatureScaler
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    # Direction targets — only populated when target_kind == "dual"
    y_train_dir: np.ndarray = field(default=None)  # type: ignore[assignment]
    y_val_dir: np.ndarray = field(default=None)    # type: ignore[assignment]
    y_test_dir: np.ndarray = field(default=None)   # type: ignore[assignment]


def preprocess(
    df: pd.DataFrame,
    *,
    feature_list: List[str],
    target_kind: str = "log_return",
    sequence_length: int = 60,
    horizon: int = 1,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    normalize: str = "minmax",
    outlier_z: float = 8.0,
    interval_seconds: int | None = None,
    fill: str = "linear",
) -> ProcessedData:
    """Run the full preprocessing pipeline from raw OHLCV to LSTM tensors."""
    work = deduplicate_and_sort(df)
    if interval_seconds is not None:
        work = synchronise_grid(work, interval_seconds, fill=fill)
    work = add_all_indicators(work)
    work = work.dropna().copy()
    if work.empty:
        raise RuntimeError("After indicator warm-up, no rows remain")

    # Build target BEFORE normalisation so 'log_return' target is unscaled.
    target = build_target(work, target_kind)
    work["__target__"] = target

    # Outlier capping on features only
    feature_cols = [c for c in feature_list if c in work.columns]
    missing = set(feature_list) - set(feature_cols)
    if missing:
        log.warning("Dropping unknown features from config: %s", sorted(missing))
    work = winsorise_outliers(work, outlier_z, feature_cols)

    train_df, val_df, test_df = time_series_split(work, train_frac, val_frac)

    scaler = FeatureScaler(method=normalize, feature_names=feature_cols).fit(train_df)

    is_dual = (target_kind == "dual")

    def _to_xy(part: pd.DataFrame):
        feats = scaler.transform(part)
        tgt = part["__target__"].values.astype(np.float32)
        if is_dual:
            return make_sequences_dual(feats, tgt, sequence_length, horizon)
        return make_sequences(feats, tgt, sequence_length, horizon)

    if is_dual:
        X_train, y_train, y_train_dir = _to_xy(train_df)
        X_val, y_val, y_val_dir = _to_xy(val_df)
        X_test, y_test, y_test_dir = _to_xy(test_df)
    else:
        X_train, y_train = _to_xy(train_df)
        X_val, y_val = _to_xy(val_df)
        X_test, y_test = _to_xy(test_df)
        y_train_dir = y_val_dir = y_test_dir = None  # type: ignore[assignment]

    log.info(
        "Sequences -- train %s, val %s, test %s | features=%d | dual=%s",
        X_train.shape, X_val.shape, X_test.shape, len(feature_cols), is_dual,
    )

    return ProcessedData(
        raw=work,
        train_df=train_df.drop(columns="__target__"),
        val_df=val_df.drop(columns="__target__"),
        test_df=test_df.drop(columns="__target__"),
        feature_names=feature_cols,
        target_name=target_kind,
        scaler=scaler,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        y_train_dir=y_train_dir, y_val_dir=y_val_dir, y_test_dir=y_test_dir,
    )
