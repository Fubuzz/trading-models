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
    latest_close: float
    latest_date: str
    train_rows: int
    test_rows: int
    train_positive_rate: float
    test_positive_rate: float
    report: str = "stub report"


STUB_RESULTS = {
    "SPY": StubResult(
        accuracy=0.61,
        balanced_accuracy=0.55,
        latest_signal=1,
        latest_probability_up=0.70,
        latest_close=500.25,
        latest_date="2026-04-08",
        train_rows=96,
        test_rows=24,
        train_positive_rate=0.55,
        test_positive_rate=0.50,
    ),
    "BOTZ": StubResult(
        accuracy=0.58,
        balanced_accuracy=0.63,
        latest_signal=0,
        latest_probability_up=0.20,
        latest_close=31.75,
        latest_date="2026-04-08",
        train_rows=80,
        test_rows=20,
        train_positive_rate=0.40,
        test_positive_rate=0.35,
    ),
    "COPX": StubResult(
        accuracy=0.67,
        balanced_accuracy=0.68,
        latest_signal=1,
        latest_probability_up=0.54,
        latest_close=42.10,
        latest_date="2026-04-08",
        train_rows=72,
        test_rows=18,
        train_positive_rate=0.62,
        test_positive_rate=0.56,
    ),
}


def test_build_results_frame_adds_signal_confidence_from_predicted_side():
    results = run_baseline.build_results_frame(
        [
            {
                "ticker": "BUYME",
                "latest_date": "2026-04-08",
                "latest_close": 100.5,
                "train_rows": 50,
                "test_rows": 12,
                "train_positive_rate": 0.64,
                "test_positive_rate": 0.58,
                "accuracy": 0.55,
                "balanced_accuracy": 0.60,
                "signal": "BUY",
                "prob_up": 0.70,
            },
            {
                "ticker": "SELLME",
                "latest_date": "2026-04-08",
                "latest_close": 88.25,
                "train_rows": 48,
                "test_rows": 10,
                "train_positive_rate": 0.42,
                "test_positive_rate": 0.30,
                "accuracy": 0.58,
                "balanced_accuracy": 0.62,
                "signal": "SELL",
                "prob_up": 0.20,
            },
        ]
    )

    by_ticker = results.set_index("ticker")

    assert by_ticker.loc["BUYME", "up_rate_delta"] == pytest.approx(-0.06)
    assert by_ticker.loc["BUYME", "signal_confidence"] == pytest.approx(0.70)
    assert by_ticker.loc["BUYME", "signal_edge"] == pytest.approx(0.20)
    assert by_ticker.loc["SELLME", "up_rate_delta"] == pytest.approx(-0.12)
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

    assert (
        "| Ticker | As Of | Last Close | Train Rows | Test Rows | Train Up Rate | Test Up Rate | Up Rate Delta | Accuracy | Balanced Accuracy | Signal | Prob Up | Signal Confidence | Conviction Score |"
        in results
    )
    assert "## Highlights" in results
    assert "Most upside-heavy test window: **COPX**" in results
    assert "Largest bullish label regime shift: **BOTZ**" in results
    assert "Largest bearish label regime shift: **COPX**" in results
    assert "Highest downside probability: **BOTZ**" in results
    assert "Best conviction-adjusted signal: **BOTZ**" in results
    assert latest_predictions.columns.tolist() == [
        "ticker",
        "latest_date",
        "latest_close",
        "train_rows",
        "test_rows",
        "train_positive_rate",
        "test_positive_rate",
        "accuracy",
        "balanced_accuracy",
        "signal",
        "prob_up",
        "up_rate_delta",
        "signal_confidence",
        "signal_edge",
        "conviction_score",
    ]
    assert latest_predictions["ticker"].tolist() == ["BOTZ", "SPY", "COPX"]
    assert latest_predictions.loc[0, "conviction_score"] > latest_predictions.loc[1, "conviction_score"]
    assert latest_predictions.loc[0, "signal_confidence"] == 0.8
    assert latest_predictions.loc[0, "up_rate_delta"] == pytest.approx(-0.05)
    assert latest_predictions.loc[0, "latest_date"] == "2026-04-08"
    assert latest_predictions.loc[0, "latest_close"] == pytest.approx(31.75)
    assert latest_predictions.loc[0, "train_rows"] == 80
    assert latest_predictions.loc[0, "test_rows"] == 20
    assert latest_predictions.loc[0, "train_positive_rate"] == pytest.approx(0.40)
    assert latest_predictions.loc[0, "test_positive_rate"] == pytest.approx(0.35)
    assert "| BOTZ | 2026-04-08 | 31.75 | 80 | 20 | 0.400 | 0.350 | -0.050 | 0.580 | 0.630 | SELL | 0.200 | 0.800 | 0.189 |" in results
    assert results.index("| BOTZ |") < results.index("| SPY |") < results.index("| COPX |")
