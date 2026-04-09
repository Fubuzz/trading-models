import numpy as np
import pandas as pd

from trading_models.config import FORWARD_DAYS
from trading_models.model import (
    compute_classification_metrics,
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
    df = pd.DataFrame({"Close": close})

    dataset = prepare_dataset(df)
    result = train_for_ticker("SYNTH", df)

    assert result.train_rows + result.test_rows == len(dataset)
    assert result.train_rows >= 20
    assert result.test_rows > 0
