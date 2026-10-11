from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.preprocessing import prepare_features_and_target


def evaluate_model(model, preprocessor, df):
    """Evaluate a trained model using the dataset and return standard classification metrics."""
    X, y, _, _ = prepare_features_and_target(df)
    transformed = preprocessor.transform(X)
    predictions = model.predict(transformed)

    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(transformed)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
    }

    if probabilities is not None:
        metrics["roc_auc"] = float(roc_auc_score(y, probabilities))
    else:
        metrics["roc_auc"] = float("nan")

    return metrics
