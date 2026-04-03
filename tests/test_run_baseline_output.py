from dataclasses import dataclass
from pathlib import Path

import scripts.run_baseline as run_baseline


@dataclass
class StubResult:
    accuracy: float
    balanced_accuracy: float
    latest_signal: int
    latest_probability_up: float
    report: str = "stub report"


def test_baseline_results_include_balanced_accuracy_column(monkeypatch, tmp_path):
    reports_dir = tmp_path / "reports"
    models_dir = tmp_path / "models"
    raw_dir = tmp_path / "data" / "raw"

    monkeypatch.setattr(run_baseline, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(run_baseline, "MODELS_DIR", models_dir)
    monkeypatch.setattr(run_baseline, "RAW_DIR", raw_dir)
    monkeypatch.setattr(
        run_baseline,
        "download_prices",
        lambda: {"SPY": __import__("pandas").DataFrame({"Close": [1, 2, 3]})},
    )
    monkeypatch.setattr(run_baseline, "save_raw_prices", lambda prices, raw: None)
    monkeypatch.setattr(
        run_baseline,
        "train_for_ticker",
        lambda ticker, df: StubResult(
            accuracy=0.6,
            balanced_accuracy=0.55,
            latest_signal=1,
            latest_probability_up=0.7,
        ),
    )

    run_baseline.main()

    results = (reports_dir / "RESULTS.md").read_text()
    latest_predictions = (reports_dir / "latest_predictions.csv").read_text()

    assert "| Ticker | Accuracy | Balanced Accuracy | Signal | Prob Up |" in results
    assert "balanced_accuracy" in latest_predictions
