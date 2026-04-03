from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

from .config import FORWARD_DAYS, RANDOM_STATE, TRAIN_TEST_SPLIT
from .features import add_features

FEATURE_COLUMNS = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "price_vs_ma10",
    "price_vs_ma20",
    "price_vs_ma50",
    "vol_20d",
    "rsi_14",
]


@dataclass
class ModelResult:
    ticker: str
    accuracy: float
    balanced_accuracy: float
    report: str
    latest_signal: int
    latest_probability_up: float



def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = add_features(df)
    out["target"] = (out["Close"].shift(-FORWARD_DAYS) > out["Close"]).astype(int)
    return out.dropna().reset_index(drop=True)



def compute_classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def train_for_ticker(ticker: str, df: pd.DataFrame) -> ModelResult:
    ds = prepare_dataset(df)
    split_idx = max(20, int(len(ds) * TRAIN_TEST_SPLIT))
    train = ds.iloc[:split_idx]
    test = ds.iloc[split_idx:]
    X_train = train[FEATURE_COLUMNS]
    y_train = train["target"]
    X_test = test[FEATURE_COLUMNS]
    y_test = test["target"]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    metrics = compute_classification_metrics(y_test, preds)
    report = classification_report(y_test, preds, digits=3)

    latest_features = ds[FEATURE_COLUMNS].iloc[[-1]]
    latest_signal = int(model.predict(latest_features)[0])
    latest_probability_up = float(model.predict_proba(latest_features)[0][1])

    return ModelResult(
        ticker=ticker,
        accuracy=metrics["accuracy"],
        balanced_accuracy=metrics["balanced_accuracy"],
        report=report,
        latest_signal=latest_signal,
        latest_probability_up=latest_probability_up,
    )
