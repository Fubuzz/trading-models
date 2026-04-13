import numpy as np
import pandas as pd
import pytest

from trading_models.config import FORWARD_DAYS, TRAIN_TEST_SPLIT
from trading_models.model import (
    compute_classification_metrics,
    compute_probability_metrics,
    prepare_dataset,
    prepare_feature_frame,
    train_for_ticker,
)



def test_compute_classification_metrics_includes_balanced_accuracy_for_imbalanced_labels():
    y_true = [0, 0, 0, 0, 1, 1]
    y_pred = [0, 0, 0, 0, 0, 1]

    metrics = compute_classification_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 5 / 6
    assert metrics["balanced_accuracy"] == 0.75



def test_compute_probability_metrics_includes_brier_score_for_test_probabilities():
    y_true = [0, 1, 1, 0]
    y_prob = [0.10, 0.80, 0.60, 0.30]

    metrics = compute_probability_metrics(y_true, y_prob)

    assert metrics["brier_score"] == pytest.approx(0.075)



def test_prepare_dataset_drops_unlabeled_tail_rows_from_forward_target():
    df = pd.DataFrame({"Close": range(1, 81)})

    dataset = prepare_dataset(df)

    assert dataset["Close"].iloc[-1] == 80 - FORWARD_DAYS
    assert dataset["target"].eq(1).all()



def test_prepare_feature_frame_keeps_latest_row_for_live_inference():
    df = pd.DataFrame({"Close": range(1, 81)})

    feature_frame = prepare_feature_frame(df)

    assert feature_frame["Close"].iloc[-1] == 80



def test_train_for_ticker_reports_train_and_test_row_counts():
    close = 100 + np.sin(np.arange(160) / 4) * 5 + np.arange(160) * 0.1
    dates = pd.date_range("2024-01-01", periods=len(close), freq="D", tz="UTC")
    df = pd.DataFrame({"Date": dates, "Close": close})

    dataset = prepare_dataset(df)
    result = train_for_ticker("SYNTH", df)
    split_idx = max(20, int(len(dataset) * TRAIN_TEST_SPLIT))

    assert result.train_rows + result.test_rows == len(dataset)
    assert result.train_rows >= 20
    assert result.test_rows > 0
    assert result.train_positive_rate == dataset["target"].iloc[:split_idx].mean()
    assert result.test_positive_rate == dataset["target"].iloc[split_idx:].mean()
    assert 0.0 <= result.brier_score <= 1.0
    assert result.latest_close == close[-1]
    assert result.latest_date == dates[-1].date().isoformat()
