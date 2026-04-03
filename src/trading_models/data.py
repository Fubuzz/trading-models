from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from .config import LOOKBACK_PERIOD, TICKERS


def download_prices(tickers: list[str] | None = None, period: str = LOOKBACK_PERIOD) -> dict[str, pd.DataFrame]:
    tickers = tickers or TICKERS
    out: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        hist = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
        if hist is not None and not hist.empty:
            out[ticker] = hist.reset_index()
    return out


def save_raw_prices(price_map: dict[str, pd.DataFrame], raw_dir: str | Path) -> None:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for ticker, df in price_map.items():
        df.to_csv(raw_dir / f"{ticker}.csv", index=False)
