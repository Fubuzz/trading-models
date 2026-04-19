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


def test_add_features_includes_return_acceleration_without_extra_warmup():
    df = pd.DataFrame({"Close": range(1, 81)})

    features = add_features(df)
    last_row = features.iloc[-1]

    assert last_row["ret_5d_minus_ret_20d"] == pytest.approx(last_row["ret_5d"] - last_row["ret_20d"])
    assert features["ret_5d_minus_ret_20d"].first_valid_index() == features["ret_20d"].first_valid_index()


def test_add_features_includes_moving_average_spreads_without_extra_warmup():
    df = pd.DataFrame({"Close": range(1, 81)})

    features = add_features(df)
    last_row = features.iloc[-1]

    assert last_row["ma_10_vs_ma20"] == pytest.approx(last_row["ma_10"] / last_row["ma_20"] - 1)
    assert last_row["ma_20_vs_ma50"] == pytest.approx(last_row["ma_20"] / last_row["ma_50"] - 1)
    assert features["ma_10_vs_ma20"].first_valid_index() == features["price_vs_ma20"].first_valid_index()
    assert features["ma_20_vs_ma50"].first_valid_index() == features["price_vs_ma50"].first_valid_index()


def test_add_features_includes_20d_range_position_without_extra_warmup():
    df = pd.DataFrame({"Close": range(1, 81)})

    features = add_features(df)
    last_row = features.iloc[-1]

    assert last_row["range_position_20"] == pytest.approx(1.0)
    assert features["range_position_20"].first_valid_index() == features["price_vs_ma20"].first_valid_index()


def test_add_features_treats_flat_20d_range_as_neutral_position():
    df = pd.DataFrame({"Close": [5.0] * 30})

    features = add_features(df)

    assert features["range_position_20"].iloc[-1] == pytest.approx(0.5)


def test_add_features_includes_short_vs_long_volatility_ratio_without_extra_warmup():
    close = [100.0] * 30 + [105.0, 95.0, 110.0, 90.0, 115.0, 92.0, 118.0, 94.0, 120.0, 96.0]
    df = pd.DataFrame({"Close": close})

    features = add_features(df)
    last_row = features.iloc[-1]

    assert last_row["vol_ratio_5d_20d"] == pytest.approx(last_row["vol_5d"] / last_row["vol_20d"])
    assert features["vol_ratio_5d_20d"].first_valid_index() == features["vol_20d"].first_valid_index()


def test_add_features_treats_flat_volatility_window_as_neutral_ratio():
    df = pd.DataFrame({"Close": [5.0] * 30})

    features = add_features(df)

    assert features["vol_ratio_5d_20d"].iloc[-1] == pytest.approx(1.0)
