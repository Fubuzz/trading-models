from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()

    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    no_losses = loss.eq(0) & gain.gt(0)
    no_gains = gain.eq(0) & loss.gt(0)
    flat_window = gain.eq(0) & loss.eq(0)

    rsi = rsi.mask(no_losses, 100.0)
    rsi = rsi.mask(no_gains, 0.0)
    rsi = rsi.mask(flat_window, 50.0)
    return rsi


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_1d"] = out["Close"].pct_change(1)
    out["ret_5d"] = out["Close"].pct_change(5)
    out["ret_20d"] = out["Close"].pct_change(20)
    out["ret_5d_minus_ret_20d"] = out["ret_5d"] - out["ret_20d"]
    out["ma_10"] = out["Close"].rolling(10).mean()
    out["ma_20"] = out["Close"].rolling(20).mean()
    out["ma_50"] = out["Close"].rolling(50).mean()
    out["range_high_20"] = out["Close"].rolling(20).max()
    out["range_low_20"] = out["Close"].rolling(20).min()
    out["drawdown_from_high_20"] = out["Close"] / out["range_high_20"] - 1
    out["rebound_from_low_20"] = out["Close"] / out["range_low_20"].replace(0, np.nan) - 1
    out["rebound_from_low_20"] = out["rebound_from_low_20"].mask(out["range_low_20"].eq(0), 0.0)
    out["range_width_pct_20"] = out["range_high_20"] / out["range_low_20"].replace(0, np.nan) - 1
    out["range_width_pct_20"] = out["range_width_pct_20"].mask(out["range_low_20"].eq(0), 0.0)
    out["price_vs_ma10"] = out["Close"] / out["ma_10"] - 1
    out["price_vs_ma20"] = out["Close"] / out["ma_20"] - 1
    out["price_vs_ma50"] = out["Close"] / out["ma_50"] - 1
    out["ma_10_vs_ma20"] = out["ma_10"] / out["ma_20"] - 1
    out["ma_10_vs_ma50"] = out["ma_10"] / out["ma_50"] - 1
    out["ma_20_vs_ma50"] = out["ma_20"] / out["ma_50"] - 1
    out["ma_10_slope_5d"] = out["ma_10"] / out["ma_10"].shift(5).replace(0, np.nan) - 1
    out["ma_10_slope_5d"] = out["ma_10_slope_5d"].mask(out["ma_10"].shift(5).eq(0), 0.0)
    out["ma_20_slope_5d"] = out["ma_20"] / out["ma_20"].shift(5).replace(0, np.nan) - 1
    out["ma_20_slope_5d"] = out["ma_20_slope_5d"].mask(out["ma_20"].shift(5).eq(0), 0.0)
    range_width_20 = out["range_high_20"] - out["range_low_20"]
    out["range_position_20"] = (out["Close"] - out["range_low_20"]) / range_width_20.replace(0, np.nan)
    out["range_position_20"] = out["range_position_20"].mask(range_width_20.eq(0), 0.5)
    daily_returns = out["Close"].pct_change()
    out["vol_5d"] = daily_returns.rolling(5).std()
    out["vol_20d"] = daily_returns.rolling(20).std()
    out["ret_5d_per_vol_5d"] = out["ret_5d"] / out["vol_5d"].replace(0, np.nan)
    out["ret_5d_per_vol_5d"] = out["ret_5d_per_vol_5d"].mask(out["vol_5d"].eq(0), 0.0)
    out["ret_20d_per_vol_20d"] = out["ret_20d"] / out["vol_20d"].replace(0, np.nan)
    out["ret_20d_per_vol_20d"] = out["ret_20d_per_vol_20d"].mask(out["vol_20d"].eq(0), 0.0)
    out["vol_ratio_5d_20d"] = out["vol_5d"] / out["vol_20d"].replace(0, np.nan)
    out["vol_ratio_5d_20d"] = out["vol_ratio_5d_20d"].mask(out["vol_20d"].eq(0), 1.0)
    out["rsi_14"] = compute_rsi(out["Close"], 14)
    return out
