import pandas as pd
import pytest

from trading_models.features import add_features, compute_rsi


def test_compute_rsi_returns_series():
    s = pd.Series([1, 2, 3, 2, 4, 5, 4, 6, 7, 8, 7, 9, 10, 11, 10, 12])
    rsi = compute_rsi(s, 5)
    assert len(rsi) == len(s)


def test_compute_rsi_handles_all_gain_window_without_nan():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7])
    rsi = compute_rsi(s, 5)
    assert rsi.iloc[-1] == 100.0


def test_compute_rsi_handles_flat_window_as_neutral():
    s = pd.Series([5, 5, 5, 5, 5, 5, 5])
    rsi = compute_rsi(s, 5)
    assert rsi.iloc[-1] == 50.0


def test_add_features_includes_moving_average_spreads_without_extra_warmup():
    df = pd.DataFrame({"Close": range(1, 81)})

    features = add_features(df)
    last_row = features.iloc[-1]

    assert last_row["ma_10_vs_ma20"] == pytest.approx(last_row["ma_10"] / last_row["ma_20"] - 1)
    assert last_row["ma_20_vs_ma50"] == pytest.approx(last_row["ma_20"] / last_row["ma_50"] - 1)
    assert features["ma_10_vs_ma20"].first_valid_index() == features["price_vs_ma20"].first_valid_index()
    assert features["ma_20_vs_ma50"].first_valid_index() == features["price_vs_ma50"].first_valid_index()
