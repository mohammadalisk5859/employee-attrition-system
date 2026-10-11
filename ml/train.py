import json
import os
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier

from config import (
    MODEL_FILE,
    MODEL_METADATA_FILE,
    PREPROCESSOR_FILE,
    RANDOM_STATE,
    TEST_SIZE,
)
from ml.evaluate import evaluate_model
from ml.optimize import optimize_models
from ml.preprocessing import build_preprocessor, get_feature_information, prepare_features_and_target


def train_model(df, model_type="stacked_ensemble"):
    """Train and persist a dataset-specific stacked ensemble using RF + XGBoost + logistic regression."""
    if model_type not in {"stacked_ensemble", "random_forest"}:
        raise ValueError(f"Unsupported model type: {model_type}")

    X, y, employee_column, attrition_column = prepare_features_and_target(df)
    if y.nunique() < 2:
        raise ValueError("The target column must contain both attrition classes.")

    preprocessor = build_preprocessor(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    optimization_trials = 1 if os.getenv("RENDER_MODE") == "true" else 5
    optimized_models = optimize_models(
        X_train_transformed,
        y_train,
        n_trials=optimization_trials,
    )
    rf_model = optimized_models["random_forest"]["model"]
    xgb_model = optimized_models["xgboost"]["model"]

    ensemble = StackingClassifier(
        estimators=[
            ("random_forest", rf_model),
            ("xgboost", xgb_model),
        ],
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        stack_method="predict_proba",
        passthrough=False,
    )
    ensemble.fit(X_train_transformed, y_train)

    predictions = ensemble.predict(X_test_transformed)
    probabilities = ensemble.predict_proba(X_test_transformed)[:, 1]

    # Compute metrics with the shared evaluator to keep the output stable and interpretable.
    evaluator_metrics = evaluate_model(ensemble, preprocessor, df)
    metrics = {
        "accuracy": float(evaluator_metrics["accuracy"]),
        "precision": float(evaluator_metrics["precision"]),
        "recall": float(evaluator_metrics["recall"]),
        "f1": float(evaluator_metrics["f1"]),
        "roc_auc": float(evaluator_metrics["roc_auc"]),
    }

    joblib.dump(ensemble, MODEL_FILE)
    joblib.dump(preprocessor, PREPROCESSOR_FILE)

    model_id = f"attrition-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    metadata = {
        "model_id": model_id,
        "model_type": "stacked_ensemble",
        "employee_column": employee_column,
        "attrition_column": attrition_column,
        "rows": int(len(df)),
        "features": get_feature_information(X),
        "metrics": metrics,
        "ready": True,
    }
    MODEL_METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "model": ensemble,
        "preprocessor": preprocessor,
        "metadata": metadata,
        "test_data": pd.DataFrame({"actual": y_test, "predicted": predictions, "probability": probabilities}),
    }
