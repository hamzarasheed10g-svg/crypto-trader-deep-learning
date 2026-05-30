"""Tests for ``data.preprocess``."""
import numpy as np
import pandas as pd
import pytest

from data.preprocess import (
    FeatureScaler, deduplicate_and_sort, make_sequences, preprocess,
    synchronise_grid, time_series_split, winsorise_outliers,
)
from data.synthetic import make_synthetic_ohlcv


def test_deduplicate_and_sort():
    idx = pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-02", "2024-01-01"], utc=True)
    df = pd.DataFrame({"x": [1, 3, 2, 99]}, index=idx)
    out = deduplicate_and_sort(df)
    assert len(out) == 3
    assert list(out["x"]) == [1, 2, 3]


def test_synchronise_grid_fills_gaps():
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0]}, index=idx)
    # Remove the middle bar then re-grid
    df2 = df.drop(df.index[1])
    out = synchronise_grid(df2, interval_seconds=3600, fill="linear")
    assert len(out) == 3
    assert out["v"].iloc[1] == pytest.approx(2.0)


def test_winsorise_caps_outliers():
    df = pd.DataFrame({"a": [1, 1, 1, 1, 100.0]})
    out = winsorise_outliers(df, z_threshold=1.0, cols=["a"])
    assert out["a"].max() < 100.0


def test_time_series_split_no_overlap():
    df = pd.DataFrame({"x": np.arange(100)})
    tr, va, te = time_series_split(df, 0.6, 0.2)
    assert len(tr) == 60 and len(va) == 20 and len(te) == 20
    assert tr.index[-1] < va.index[0]
    assert va.index[-1] < te.index[0]


def test_make_sequences_shapes():
    n, f, seq = 100, 5, 10
    feats = np.random.RandomState(0).randn(n, f).astype(np.float32)
    tgt = np.arange(n, dtype=np.float32)
    X, y = make_sequences(feats, tgt, sequence_length=seq, horizon=1)
    assert X.shape == (n - seq, seq, f)
    assert y.shape == (n - seq,)
    # First window ends at index 9 and predicts index 10
    assert y[0] == 10.0


def test_make_sequences_too_short():
    with pytest.raises(ValueError):
        make_sequences(np.zeros((5, 3)), np.zeros(5), sequence_length=10)


def test_feature_scaler_roundtrip(tmp_path):
    df = pd.DataFrame({"a": np.arange(10, dtype=float), "b": np.arange(10, 20, dtype=float),
                       "close": np.linspace(100, 200, 10)})
    sc = FeatureScaler(method="minmax", feature_names=["a", "b", "close"]).fit(df)
    arr = sc.transform(df)
    assert arr.min() == pytest.approx(0.0)
    assert arr.max() == pytest.approx(1.0)
    # Save + load
    p = tmp_path / "scaler.pkl"
    sc.save(p)
    sc2 = FeatureScaler.load(p)
    np.testing.assert_allclose(sc.transform(df), sc2.transform(df))


def test_feature_scaler_inverse_close():
    df = pd.DataFrame({"close": np.linspace(100, 200, 10)})
    sc = FeatureScaler(method="minmax", feature_names=["close"]).fit(df)
    scaled = sc.transform(df)
    inv = sc.inverse_transform_close(scaled.reshape(-1))
    np.testing.assert_allclose(inv, df["close"].values, atol=1e-5)


def test_preprocess_end_to_end_synthetic():
    df = make_synthetic_ohlcv(n_bars=600, interval="1h", seed=1)
    pdata = preprocess(
        df,
        feature_list=["open", "high", "low", "close", "volume", "rsi_14", "macd",
                      "ema_12", "atr_14", "log_return"],
        target_kind="log_return",
        sequence_length=20,
        horizon=1,
        train_frac=0.7,
        val_frac=0.15,
        normalize="minmax",
        outlier_z=8.0,
    )
    assert pdata.X_train.shape[1] == 20
    assert pdata.X_train.shape[2] == len(pdata.feature_names)
    # Train min-max scaled into [0,1] (with slight float headroom)
    assert pdata.X_train.min() >= -1e-6
    assert pdata.X_train.max() <= 1 + 1e-6
