from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from trading_models.data import download_prices, save_raw_prices
from trading_models.model import train_for_ticker

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    prices = download_prices()
    save_raw_prices(prices, RAW_DIR)

    rows = []
    md_lines = [
        "# Baseline Model Results",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "| Ticker | Accuracy | Balanced Accuracy | Signal | Prob Up |",
        "|---|---:|---:|---|---:|",
    ]
    for ticker, df in prices.items():
        result = train_for_ticker(ticker, df)
        signal = "BUY" if result.latest_signal == 1 else "SELL"
        rows.append(
            {
                "ticker": ticker,
                "accuracy": result.accuracy,
                "balanced_accuracy": result.balanced_accuracy,
                "signal": signal,
                "prob_up": result.latest_probability_up,
            }
        )
        md_lines.append(
            f"| {ticker} | {result.accuracy:.3f} | {result.balanced_accuracy:.3f} | {signal} | {result.latest_probability_up:.3f} |"
        )
        (REPORTS_DIR / f"{ticker}_classification_report.txt").write_text(result.report)

    pd.DataFrame(rows).sort_values(["prob_up", "balanced_accuracy", "accuracy"], ascending=False).to_csv(
        REPORTS_DIR / "latest_predictions.csv", index=False
    )
    (REPORTS_DIR / "RESULTS.md").write_text("\n".join(md_lines) + "\n")
    print((REPORTS_DIR / "RESULTS.md").as_posix())


if __name__ == "__main__":
    main()
