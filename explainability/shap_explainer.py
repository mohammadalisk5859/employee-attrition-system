from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def _extract_feature_values(shap_values):
    """Normalize SHAP output across TreeExplainer and generic Explainer objects."""
    if hasattr(shap_values, "values"):
        values = shap_values.values
    else:
        values = shap_values

    if isinstance(values, list):
        if len(values) > 1:
            values = values[1]
        else:
            values = values[0]

    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1] if values.shape[-1] > 1 else values[:, :, 0]
    return values


def explain_with_shap(model, X: pd.DataFrame, row: pd.Series, preprocessor=None):
    """Return global and local SHAP-style feature contributions for a trained model."""
    transformed_X = X
    transformed_row = row.to_frame().T
    if preprocessor is not None:
        transformed_X = preprocessor.transform(X)
        transformed_row = preprocessor.transform(row.to_frame().T)

    feature_names = getattr(X, "columns", [f"feature_{idx}" for idx in range(X.shape[1])])
    if preprocessor is not None:
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            feature_names = [f"feature_{idx}" for idx in range(transformed_X.shape[1])]

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(transformed_X)
        row_values = explainer.shap_values(transformed_row)
    except (TypeError, ValueError, AttributeError):
        if hasattr(model, "predict_proba"):
            if preprocessor is not None:
                predict_fn = lambda values: model.predict_proba(preprocessor.transform(values))[:, 1]
            else:
                predict_fn = lambda values: model.predict_proba(values)[:, 1]
            explainer = shap.Explainer(predict_fn, X)
            shap_values = explainer(X)
            row_values = explainer(row.to_frame().T)
        else:
            raise

    feature_values = _extract_feature_values(shap_values)
    if feature_values.ndim == 2 and feature_values.shape[0] == len(X):
        global_importance = [
            {"feature": column, "importance": float(np.abs(feature_values[:, index]).mean())}
            for index, column in enumerate(feature_names)
        ]
    else:
        global_importance = [
            {"feature": column, "importance": 0.0}
            for column in feature_names
        ]

    row_feature_values = _extract_feature_values(row_values)
    if row_feature_values.ndim > 1:
        row_feature_values = row_feature_values.reshape(-1, len(feature_names))
    else:
        row_feature_values = row_feature_values.reshape(1, -1)

    local_contributions = [
        {"feature": column, "contribution": float(value)}
        for column, value in zip(feature_names, row_feature_values[0])
    ]

    return {
        "global_importance": sorted(global_importance, key=lambda item: item["importance"], reverse=True),
        "local_contributions": sorted(local_contributions, key=lambda item: abs(item["contribution"]), reverse=True),
    }
