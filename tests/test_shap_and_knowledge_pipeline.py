import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

from explainability.shap_explainer import explain_with_shap
from ml.preprocessing import build_preprocessor
from rag.generator import generate_retention_recommendations, summarize_recommendation_context


def test_shap_explainer_returns_feature_contributions():
    X = pd.DataFrame(
        {
            "MonthlyIncome": [2200, 2600, 3400, 4100, 4500, 5200],
            "YearsAtCompany": [1, 2, 3, 4, 5, 6],
            "OverTime": [1, 0, 1, 0, 1, 0],
        }
    )
    y = [1, 0, 1, 0, 1, 0]
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    explanation = explain_with_shap(model, X, X.iloc[0])

    assert "global_importance" in explanation
    assert "local_contributions" in explanation
    assert explanation["local_contributions"]


def test_stacked_classifier_explainer_uses_supported_base_model():
    X = pd.DataFrame(
        {
            "MonthlyIncome": [2200, 2600, 3400, 4100, 4500, 5200, 5600, 6100, 7000, 7800, 8200, 8700, 9100, 9600, 10100, 10500, 11000, 11500, 12000, 12500],
            "YearsAtCompany": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            "OverTime": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        }
    )
    y = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    model = StackingClassifier(
        estimators=[
            ("rf_a", RandomForestClassifier(random_state=42)),
            ("rf_b", RandomForestClassifier(random_state=7)),
        ],
        final_estimator=LogisticRegression(max_iter=2000),
        stack_method="predict_proba",
        cv=2,
    )
    model.fit(X, y)

    explanation = explain_with_shap(model, X, X.iloc[0])

    assert "global_importance" in explanation
    assert "local_contributions" in explanation
    assert explanation["local_contributions"]


def test_explainer_handles_preprocessed_categorical_inputs():
    X = pd.DataFrame(
        {
            "Department": ["HR", "Sales", "IT", "HR", "Sales", "IT"],
            "MonthlyIncome": [2200, 2600, 3400, 4100, 4500, 5200],
            "OverTime": ["Yes", "No", "Yes", "No", "Yes", "No"],
            "YearsAtCompany": [1, 2, 3, 4, 5, 6],
        }
    )
    y = [1, 0, 1, 0, 1, 0]
    preprocessor = build_preprocessor(X)
    model = RandomForestClassifier(random_state=42)
    model.fit(preprocessor.fit_transform(X), y)

    explanation = explain_with_shap(model, X, X.iloc[0], preprocessor=preprocessor)

    assert "global_importance" in explanation
    assert "local_contributions" in explanation
    assert explanation["local_contributions"]


def test_summary_uses_employee_context_and_knowledge():
    employee = {
        "MonthlyIncome": 2000,
        "OverTime": "Yes",
        "YearsAtCompany": 1,
        "JobSatisfaction": 2,
    }
    docs = [{"text": "Support workload balance and flexible schedules for new employees.", "source_name": "policy.txt"}]

    summary = summarize_recommendation_context(employee, 0.72, docs, ["OverTime", "JobSatisfaction"])

    assert "support" in summary.lower() or "workload" in summary.lower()
    assert "policy.txt" in summary


def test_retention_recommendations_include_context_input():
    employee = {"OverTime": "Yes", "JobSatisfaction": 1, "MonthlyIncome": 2500, "YearsAtCompany": 1}

    suggestions = generate_retention_recommendations(employee, 0.8, knowledge_documents=[{"text": "Encourage mentoring and schedule check-ins.", "source_name": "mentor_policy.txt"}])

    assert suggestions
    assert any("priority" in item.lower() or "check-in" in item.lower() for item in suggestions)
