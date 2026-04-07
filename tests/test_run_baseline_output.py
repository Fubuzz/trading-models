from dataclasses import dataclass

import pandas as pd
import pytest

import scripts.run_baseline as run_baseline


@dataclass
class StubResult:
    accuracy: float
    balanced_accuracy: float
    latest_signal: int
    latest_probability_up: float
    report: str = "stub report"


STUB_RESULTS = {
    "SPY": StubResult(accuracy=0.61, balanced_accuracy=0.55, latest_signal=1, latest_probability_up=0.70),
    "BOTZ": StubResult(accuracy=0.58, balanced_accuracy=0.63, latest_signal=0, latest_probability_up=0.20),
    "COPX": StubResult(accuracy=0.67, balanced_accuracy=0.68, latest_signal=1, latest_probability_up=0.54),
}


def test_build_results_frame_adds_signal_confidence_from_predicted_side():
    results = run_baseline.build_results_frame(
        [
            {
                "ticker": "BUYME",
                "accuracy": 0.55,
                "balanced_accuracy": 0.60,
                "signal": "BUY",
                "prob_up": 0.70,
            },
            {
                "ticker": "SELLME",
                "accuracy": 0.58,
                "balanced_accuracy": 0.62,
                "signal": "SELL",
                "prob_up": 0.20,
            },
        ]
    )

    by_ticker = results.set_index("ticker")

    assert by_ticker.loc["BUYME", "signal_confidence"] == pytest.approx(0.70)
    assert by_ticker.loc["BUYME", "signal_edge"] == pytest.approx(0.20)
    assert by_ticker.loc["SELLME", "signal_confidence"] == pytest.approx(0.80)
    assert by_ticker.loc["SELLME", "signal_edge"] == pytest.approx(0.30)


def test_baseline_results_include_ranked_conviction_columns_and_highlights(monkeypatch, tmp_path):
    reports_dir = tmp_path / "reports"
    models_dir = tmp_path / "models"
    raw_dir = tmp_path / "data" / "raw"

    monkeypatch.setattr(run_baseline, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(run_baseline, "MODELS_DIR", models_dir)
    monkeypatch.setattr(run_baseline, "RAW_DIR", raw_dir)
    monkeypatch.setattr(
        run_baseline,
        "download_prices",
        lambda: {ticker: pd.DataFrame({"Close": [1, 2, 3]}) for ticker in STUB_RESULTS},
    )
    monkeypatch.setattr(run_baseline, "save_raw_prices", lambda prices, raw: None)
    monkeypatch.setattr(run_baseline, "train_for_ticker", lambda ticker, df: STUB_RESULTS[ticker])

    run_baseline.main()

    results = (reports_dir / "RESULTS.md").read_text()
    latest_predictions = pd.read_csv(reports_dir / "latest_predictions.csv")

    assert "| Ticker | Accuracy | Balanced Accuracy | Signal | Prob Up | Signal Confidence |" in results
    assert "## Highlights" in results
    assert "Best conviction-adjusted signal: **BOTZ**" in results
    assert latest_predictions.columns.tolist() == [
        "ticker",
        "accuracy",
        "balanced_accuracy",
        "signal",
        "prob_up",
        "signal_confidence",
        "signal_edge",
        "conviction_score",
    ]
    assert latest_predictions["ticker"].tolist() == ["BOTZ", "SPY", "COPX"]
    assert latest_predictions.loc[0, "conviction_score"] > latest_predictions.loc[1, "conviction_score"]
    assert latest_predictions.loc[0, "signal_confidence"] == 0.8
    assert results.index("| BOTZ |") < results.index("| SPY |") < results.index("| COPX |")
