import pandas as pd

from ml.train import train_model
from ml.evaluate import evaluate_model


def _sample_employee_data():
    rows = []
    for i in range(1, 41):
        rows.append(
            {
                "EmployeeID": f"E{i:03d}",
                "Department": "Sales" if i % 2 == 0 else "Engineering",
                "JobLevel": i % 5 + 1,
                "MonthlyIncome": 2500 + i * 120,
                "YearsAtCompany": (i % 7) + 1,
                "OverTime": "Yes" if i % 3 == 0 else "No",
                "Attrition": 1 if i % 4 == 0 else 0,
            }
        )
    return pd.DataFrame(rows)


def test_train_model_returns_stacked_ensemble_metadata():
    df = _sample_employee_data()

    result = train_model(df)

    assert result["metadata"]["model_type"] == "stacked_ensemble"
    assert "metrics" in result["metadata"]
    assert set(result["metadata"]["metrics"]).issuperset({"accuracy", "precision", "recall", "f1", "roc_auc"})
    assert result["metadata"]["ready"] is True


def test_evaluate_model_returns_required_metrics():
    df = _sample_employee_data()
    result = train_model(df)

    metrics = evaluate_model(result["model"], result["preprocessor"], df)

    assert set(metrics).issuperset({"accuracy", "precision", "recall", "f1", "roc_auc"})
