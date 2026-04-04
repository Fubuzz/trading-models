import pandas as pd

from trading_models.features import compute_rsi


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
