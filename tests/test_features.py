from trading_models.features import compute_rsi


def test_compute_rsi_returns_series():
    import pandas as pd

    s = pd.Series([1, 2, 3, 2, 4, 5, 4, 6, 7, 8, 7, 9, 10, 11, 10, 12])
    rsi = compute_rsi(s, 5)
    assert len(rsi) == len(s)
