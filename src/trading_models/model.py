from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, classification_report

from .config import FORWARD_DAYS, RANDOM_STATE, TRAIN_TEST_SPLIT
from .features import add_features

FEATURE_COLUMNS = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "ret_5d_minus_ret_20d",
    "ret_5d_per_vol_5d",
    "ret_20d_per_vol_20d",
    "price_vs_ma10",
    "price_vs_ma20",
    "price_vs_ma50",
    "ma_10_vs_ma20",
    "ma_10_vs_ma50",
    "ma_20_vs_ma50",
    "ma_stack_bullish_count",
    "ma_10_slope_5d",
    "ma_20_slope_5d",
    "range_position_20",
    "drawdown_from_high_20",
    "rebound_from_low_20",
    "range_width_pct_20",
    "vol_20d",
    "vol_ratio_5d_20d",
    "rsi_14",
    "rsi_14_change_5d",
]


@dataclass
class ModelResult:
    ticker: str
    accuracy: float
    balanced_accuracy: float
    brier_score: float
    report: str
    latest_signal: int
    latest_probability_up: float
    latest_close: float
    latest_date: str
    train_rows: int
    test_rows: int
    train_positive_rate: float
    test_positive_rate: float



def format_latest_date(latest_row: pd.Series) -> str:
    if "Date" not in latest_row.index:
        return "N/A"

    latest_date = pd.to_datetime(latest_row["Date"])
    if pd.isna(latest_date):
        return "N/A"
    return latest_date.date().isoformat()



def prepare_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    feature_frame = add_features(df)
    return feature_frame.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)



def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    feature_frame = prepare_feature_frame(df)
    future_close = feature_frame["Close"].shift(-FORWARD_DAYS)
    labeled = feature_frame.loc[future_close.notna()].copy()
    labeled["target"] = (future_close.loc[future_close.notna()] > labeled["Close"]).astype(int).to_numpy()
    return labeled.reset_index(drop=True)



def compute_classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def compute_positive_rate(labels: pd.Series) -> float:
    return float(labels.mean()) if len(labels) else 0.0


def compute_probability_metrics(y_true: pd.Series, y_prob: pd.Series) -> dict[str, float]:
    return {"brier_score": float(brier_score_loss(y_true, y_prob))}


def train_for_ticker(ticker: str, df: pd.DataFrame) -> ModelResult:
    feature_frame = prepare_feature_frame(df)
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
    metrics = {
        **compute_classification_metrics(y_test, preds),
        **compute_probability_metrics(y_test, probs),
    }
    report = classification_report(y_test, preds, digits=3)

    latest_row = feature_frame.iloc[-1]
    latest_features = feature_frame[FEATURE_COLUMNS].tail(1)
    latest_signal = int(model.predict(latest_features)[0])
    latest_probability_up = float(model.predict_proba(latest_features)[0][1])
    latest_close = float(latest_row["Close"])
    latest_date = format_latest_date(latest_row)

    return ModelResult(
        ticker=ticker,
        accuracy=metrics["accuracy"],
        balanced_accuracy=metrics["balanced_accuracy"],
        brier_score=metrics["brier_score"],
        report=report,
        latest_signal=latest_signal,
        latest_probability_up=latest_probability_up,
        latest_close=latest_close,
        latest_date=latest_date,
        train_rows=len(train),
        test_rows=len(test),
        train_positive_rate=compute_positive_rate(y_train),
        test_positive_rate=compute_positive_rate(y_test),
    )
