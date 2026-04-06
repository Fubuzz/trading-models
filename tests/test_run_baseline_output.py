from dataclasses import dataclass

import pandas as pd

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

    assert "| Ticker | Accuracy | Balanced Accuracy | Signal | Prob Up |" in results
    assert "## Highlights" in results
    assert "Best conviction-adjusted signal: **BOTZ**" in results
    assert latest_predictions.columns.tolist() == [
        "ticker",
        "accuracy",
        "balanced_accuracy",
        "signal",
        "prob_up",
        "probability_edge",
        "conviction_score",
    ]
    assert latest_predictions["ticker"].tolist() == ["BOTZ", "SPY", "COPX"]
    assert latest_predictions.loc[0, "conviction_score"] > latest_predictions.loc[1, "conviction_score"]
    assert results.index("| BOTZ |") < results.index("| SPY |") < results.index("| COPX |")
