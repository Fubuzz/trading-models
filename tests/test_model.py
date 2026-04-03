from trading_models.model import compute_classification_metrics


def test_compute_classification_metrics_includes_balanced_accuracy_for_imbalanced_labels():
    y_true = [0, 0, 0, 0, 1, 1]
    y_pred = [0, 0, 0, 0, 0, 1]

    metrics = compute_classification_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 5 / 6
    assert metrics["balanced_accuracy"] == 0.75
