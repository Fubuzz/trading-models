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


RESULTS_COLUMNS = ["ticker", "accuracy", "balanced_accuracy", "signal", "prob_up"]


def build_results_frame(rows: list[dict[str, float | str]]) -> pd.DataFrame:
    results = pd.DataFrame(rows, columns=RESULTS_COLUMNS)
    if results.empty:
        return results

    results["signal_confidence"] = results["prob_up"]
    sell_mask = results["signal"].eq("SELL")
    results.loc[sell_mask, "signal_confidence"] = 1 - results.loc[sell_mask, "prob_up"]
    results["signal_edge"] = results["signal_confidence"] - 0.5
    results["conviction_score"] = results["balanced_accuracy"] * results["signal_edge"]
    return results.sort_values(
        ["conviction_score", "balanced_accuracy", "signal_confidence", "accuracy"],
        ascending=False,
    ).reset_index(drop=True)


def render_results_markdown(results: pd.DataFrame) -> str:
    md_lines = [
        "# Baseline Model Results",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
    ]

    if not results.empty:
        strongest_fit = results.sort_values(["balanced_accuracy", "accuracy"], ascending=False).iloc[0]
        highest_upside = results.sort_values(["prob_up", "balanced_accuracy", "accuracy"], ascending=False).iloc[0]
        sell_results = results.loc[results["signal"].eq("SELL")]
        best_conviction = results.iloc[0]
        md_lines.extend(
            [
                "## Highlights",
                "",
                (
                    f"- Strongest historical fit: **{strongest_fit['ticker']}** "
                    f"(balanced accuracy {strongest_fit['balanced_accuracy']:.3f}, signal {strongest_fit['signal']})."
                ),
                (
                    f"- Highest upside probability: **{highest_upside['ticker']}** "
                    f"(`prob_up` {highest_upside['prob_up']:.3f}, balanced accuracy {highest_upside['balanced_accuracy']:.3f})."
                ),
            ]
        )
        if not sell_results.empty:
            highest_downside = sell_results.sort_values(
                ["signal_confidence", "balanced_accuracy", "accuracy"], ascending=False
            ).iloc[0]
            md_lines.append(
                f"- Highest downside probability: **{highest_downside['ticker']}** "
                f"(`prob_down` {highest_downside['signal_confidence']:.3f}, "
                f"balanced accuracy {highest_downside['balanced_accuracy']:.3f})."
            )
        md_lines.extend(
            [
                (
                    f"- Best conviction-adjusted signal: **{best_conviction['ticker']}** "
                    f"(conviction score {best_conviction['conviction_score']:.3f}, "
                    f"signal {best_conviction['signal']}, confidence {best_conviction['signal_confidence']:.3f})."
                ),
                "",
            ]
        )

    md_lines.extend(
        [
            "| Ticker | Accuracy | Balanced Accuracy | Signal | Prob Up | Signal Confidence | Conviction Score |",
            "|---|---:|---:|---|---:|---:|---:|",
        ]
    )
    for row in results.itertuples(index=False):
        md_lines.append(
            f"| {row.ticker} | {row.accuracy:.3f} | {row.balanced_accuracy:.3f} | {row.signal} | {row.prob_up:.3f} | {row.signal_confidence:.3f} | {row.conviction_score:.3f} |"
        )
    return "\n".join(md_lines) + "\n"


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    prices = download_prices()
    save_raw_prices(prices, RAW_DIR)

    rows = []
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
        (REPORTS_DIR / f"{ticker}_classification_report.txt").write_text(result.report)

    results = build_results_frame(rows)
    results.to_csv(REPORTS_DIR / "latest_predictions.csv", index=False)
    (REPORTS_DIR / "RESULTS.md").write_text(render_results_markdown(results))
    print((REPORTS_DIR / "RESULTS.md").as_posix())


if __name__ == "__main__":
    main()
