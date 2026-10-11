from __future__ import annotations

import pandas as pd


def explain_instance(model_fn, feature_frame: pd.DataFrame, row: pd.Series, top_features: int = 5):
    """Return a stable local feature-importance summary for a single employee row."""
    if feature_frame.empty:
        return {"top_features": [], "explanation": []}

    base_row = row[feature_frame.columns].copy()
    base_df = pd.DataFrame([base_row])
    base_prediction = float(model_fn(base_df)[0])

    importance_scores = []

    for column in feature_frame.columns:
        original_value = row[column]
        candidate_values = []

        if pd.api.types.is_numeric_dtype(feature_frame[column]):
            series = feature_frame[column].dropna()
            if not series.empty:
                reference_spread = float(series.std(ddof=0)) if len(series) > 1 else max(float(series.median()), 1.0)
                candidate_values.extend([
                    float(original_value) * 0.8,
                    float(original_value) * 1.2,
                    float(original_value) + max(reference_spread, 1.0),
                    float(original_value) - max(reference_spread, 1.0),
                ])
        else:
            unique_values = [value for value in feature_frame[column].dropna().astype(str).unique() if str(value) != str(original_value)]
            if unique_values:
                candidate_values.extend(unique_values[:3])
            else:
                candidate_values.append(str(original_value))

        for candidate in candidate_values:
            candidate_row = base_row.copy()
            candidate_row[column] = candidate
            score = float(model_fn(pd.DataFrame([candidate_row]))[0])
            importance_scores.append((column, abs(score - base_prediction)))

    if not importance_scores:
        return {"top_features": [], "explanation": []}

    top_importance = sorted(importance_scores, key=lambda item: item[1], reverse=True)[:top_features]
    top_features_list = [(name, round(score, 4)) for name, score in top_importance]

    return {
        "top_features": top_features_list,
        "explanation": top_features_list,
    }
