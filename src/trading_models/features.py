from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_1d"] = out["Close"].pct_change(1)
    out["ret_5d"] = out["Close"].pct_change(5)
    out["ret_20d"] = out["Close"].pct_change(20)
    out["ma_10"] = out["Close"].rolling(10).mean()
    out["ma_20"] = out["Close"].rolling(20).mean()
    out["ma_50"] = out["Close"].rolling(50).mean()
    out["price_vs_ma10"] = out["Close"] / out["ma_10"] - 1
    out["price_vs_ma20"] = out["Close"] / out["ma_20"] - 1
    out["price_vs_ma50"] = out["Close"] / out["ma_50"] - 1
    out["vol_20d"] = out["Close"].pct_change().rolling(20).std()
    out["rsi_14"] = compute_rsi(out["Close"], 14)
    return out
