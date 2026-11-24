from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def explain_instance(
    model_fn,
    feature_frame: pd.DataFrame,
    row: pd.Series,
    top_features: int = 5,
    num_samples: int = 256,
    kernel_width: float = 0.75,
):
    """Fit a weighted local surrogate and return LIME-style feature sensitivities."""
    if feature_frame.empty or top_features <= 0 or num_samples < 2:
        return {"top_features": [], "explanation": [], "feature_effects": []}

    columns = list(feature_frame.columns)
    base_row = row[columns].copy()
    neighborhood = pd.DataFrame(
        [base_row.copy() for _ in range(num_samples)],
        columns=columns,
    )
    interpretable = np.zeros((num_samples, len(columns)), dtype=float)
    distances = np.zeros(num_samples, dtype=float)
    random = np.random.default_rng(42)

    for column_index, column in enumerate(columns):
        source = feature_frame[column]
        original_value = base_row[column]
        if pd.api.types.is_numeric_dtype(source) and not pd.api.types.is_bool_dtype(source):
            series = pd.to_numeric(source, errors="coerce").dropna()
            if series.empty:
                continue
            center = pd.to_numeric(pd.Series([original_value]), errors="coerce").iloc[0]
            if pd.isna(center):
                center = float(series.median())
            spread = float(series.std(ddof=0)) if len(series) > 1 else 0.0
            spread = max(spread, 1.0)
            perturbations = random.normal(float(center), spread, num_samples - 1)
            perturbations = np.clip(perturbations, float(series.min()), float(series.max()))
            values = np.concatenate(([float(center)], perturbations))
            neighborhood[column] = values
            interpretable[:, column_index] = (values - float(center)) / spread
            distances += interpretable[:, column_index] ** 2
        else:
            values = source.dropna()
            if values.empty:
                continue
            candidates = values.astype(str).to_numpy()
            original_text = str(original_value)
            perturbations = random.choice(candidates, size=num_samples - 1, replace=True)
            sampled_values = np.concatenate(([original_value], perturbations))
            neighborhood[column] = sampled_values
            changed = np.asarray(
                [str(value) != original_text for value in sampled_values],
                dtype=float,
            )
            interpretable[:, column_index] = changed
            distances += changed

    distances = np.sqrt(distances / max(len(columns), 1))
    weights = np.exp(-((distances / max(kernel_width, 1e-6)) ** 2))
    predictions = np.asarray(model_fn(neighborhood), dtype=float).reshape(-1)
    if len(predictions) != num_samples:
        raise ValueError("The explanation model must return one probability per perturbed row.")

    surrogate = Ridge(alpha=1.0)
    surrogate.fit(interpretable, predictions, sample_weight=weights)
    importances = np.abs(surrogate.coef_)
    ranked_indices = [
        index for index in np.argsort(importances)[::-1]
        if importances[index] > 1e-8
    ][:top_features]
    top_features_list = [
        (columns[index], round(float(importances[index]), 4))
        for index in ranked_indices
    ]
    feature_effects = []
    for index in ranked_indices:
        coefficient = float(surrogate.coef_[index])
        is_numeric = (
            pd.api.types.is_numeric_dtype(feature_frame[columns[index]])
            and not pd.api.types.is_bool_dtype(feature_frame[columns[index]])
        )
        if is_numeric:
            direction = (
                "higher_values_increase_risk"
                if coefficient > 0
                else "higher_values_decrease_risk"
            )
        else:
            direction = (
                "alternative_categories_increase_risk"
                if coefficient > 0
                else "alternative_categories_decrease_risk"
            )
        feature_effects.append({
            "feature": columns[index],
            "coefficient": round(coefficient, 4),
            "kind": "numeric" if is_numeric else "categorical",
            "direction": direction,
        })
    return {
        "top_features": top_features_list,
        "explanation": top_features_list,
        "feature_effects": feature_effects,
    }
