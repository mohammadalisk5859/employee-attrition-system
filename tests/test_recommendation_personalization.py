import pandas as pd

from ml.preprocessing import align_features_to_model, prepare_features_and_target
from rag.generator import generate_retention_recommendations, select_policy_cards
from rag.personalization import generate_personalized_retention_plan


def test_policy_cards_only_show_guidance_matched_to_employee_signals():
    documents = [
        {
            "text": "Review overtime and offer flexible work schedules where possible.",
            "source_name": "workload_policy.txt",
        },
        {
            "text": "Pair new employees with a mentor during onboarding.",
            "source_name": "onboarding_policy.txt",
        },
        {
            "text": "Our organization supports every employee.",
            "source_name": "general_policy.txt",
        },
    ]

    cards = select_policy_cards(
        {"OverTime": "Yes", "JobSatisfaction": 4, "YearsAtCompany": 8},
        documents,
    )

    assert cards == [
        (
            "Workload and work-life balance",
            "Review overtime and offer flexible work schedules where possible.",
            "workload_policy.txt",
        )
    ]


def test_policy_cards_follow_a_different_employee_signal():
    cards = select_policy_cards(
        {"OverTime": "No", "RelationshipSatisfaction": 1},
        [
            {
                "text": "Managers should help teams build healthy workplace relationships.",
                "source_name": "team_policy.txt",
            },
            {
                "text": "Review overtime and adjust employee schedules.",
                "source_name": "workload_policy.txt",
            },
        ],
    )

    assert cards == [
        (
            "Workplace relationships",
            "Managers should help teams build healthy workplace relationships.",
            "team_policy.txt",
        )
    ]


def test_policy_cards_do_not_infer_compensation_concerns_from_income_alone():
    documents = [
        {
            "text": "Employees can request a salary and benefits review.",
            "source_name": "compensation_policy.txt",
        }
    ]

    cards = select_policy_cards({"MonthlyIncome": 2500}, documents)

    assert cards == []


def test_retention_recommendations_change_with_employee_profile():
    overtime_recommendations = generate_retention_recommendations(
        {"OverTime": "Yes", "JobSatisfaction": 4, "YearsAtCompany": 8},
        0.1,
    )
    relationship_recommendations = generate_retention_recommendations(
        {"OverTime": "No", "RelationshipSatisfaction": 1, "YearsAtCompany": 8},
        0.1,
    )

    assert overtime_recommendations != relationship_recommendations
    assert any("workload" in item.lower() for item in overtime_recommendations)
    assert any("relationship" in item.lower() for item in relationship_recommendations)


def _employee_cohort():
    return pd.DataFrame(
        [
            {
                "EmployeeID": f"E{index:02d}",
                "Employee Name": f"Employee {index}",
                "Department": "Engineering" if index < 6 else "Sales",
                "JobRole": "Engineer" if index < 6 else "Sales Associate",
                "JobLevel": 2 if index < 6 else 1,
                "MonthlyIncome": 3000 + index * 500,
                "JobSatisfaction": [1, 2, 3, 4][index % 4],
                "EnvironmentSatisfaction": [2, 3, 4, 3][index % 4],
                "RelationshipSatisfaction": [3, 3, 2, 4][index % 4],
                "JobInvolvement": [2, 3, 4, 3][index % 4],
                "PerformanceRating": [2, 3, 4, 4][index % 4],
                "PercentSalaryHike": 10 + index,
                "YearsAtCompany": index + 1,
                "YearsSinceLastPromotion": index % 6,
                "YearsWithCurrentManager": index % 5 + 1,
                "WorkLifeBalance": [1, 2, 3, 4][index % 4],
                "OverTime": "Yes" if index % 2 == 0 else "No",
                "TrainingTimesLastYear": index % 4,
                "DistanceFromHome": index + 2,
                "BusinessTravel": "Travel_Frequently" if index % 3 == 0 else "Travel_Rarely",
                "Attrition": "No",
            }
            for index in range(12)
        ]
    )


def test_personalized_plan_uses_employee_data_risk_and_lime_with_hr_evidence():
    cohort = _employee_cohort()
    employee = cohort.loc[0].drop(labels=["Attrition"])
    explanation = {
        "top_features": [("OverTime", 0.18), ("JobSatisfaction", 0.12)]
    }

    plan = generate_personalized_retention_plan(
        employee,
        cohort.drop(columns=["Attrition"]),
        0.82,
        local_explanation=explanation,
        knowledge_documents=[
            {
                "text": "Managers should review overtime schedules and workload.",
                "source_name": "workload_policy.txt",
            }
        ],
    )

    assert 3 <= len(plan["recommendations"]) <= 5
    assert all(item["priority"] == "High" for item in plan["recommendations"])
    assert all(item["timeline"] == "Within 7 days" for item in plan["recommendations"])
    assert any("Overtime is recorded as Yes" in item["reason"] for item in plan["recommendations"])
    assert any(
        "LIME-style local surrogate ranked OverTime" in item["reason"]
        for item in plan["recommendations"]
    )
    assert any(item["policy_source"] == "workload_policy.txt" for item in plan["recommendations"])
    assert len({item["strategy_title"] for item in plan["recommendations"]}) == len(plan["recommendations"])


def test_personalized_plan_changes_for_high_performance_and_career_stagnation():
    cohort = _employee_cohort().drop(columns=["Attrition"])
    employee = cohort.loc[7].copy()
    employee["PerformanceRating"] = 4
    employee["JobInvolvement"] = 4
    employee["YearsSinceLastPromotion"] = 5

    plan = generate_personalized_retention_plan(employee, cohort, 0.2)
    titles = {item["strategy_title"] for item in plan["recommendations"]}

    assert "Recognize strong performance" in titles
    assert "Discuss career advancement" in titles
    assert all("overtime" not in item["strategy_title"].lower() for item in plan["recommendations"])


def test_personalized_plan_uses_low_involvement_and_training_opportunities():
    cohort = _employee_cohort().drop(columns=["Attrition"])
    employee = cohort.loc[5].copy()
    employee["JobInvolvement"] = 1
    employee["TrainingTimesLastYear"] = 0
    employee["PerformanceRating"] = 2
    employee["YearsSinceLastPromotion"] = 5
    employee["JobSatisfaction"] = 4
    employee["WorkLifeBalance"] = 4

    plan = generate_personalized_retention_plan(employee, cohort, 0.4)
    titles = {item["strategy_title"] for item in plan["recommendations"]}

    assert "Offer a meaningful project" in titles
    assert "Identify a relevant learning need" in titles
    assert "Discuss career direction" in titles


def test_compensation_recommendations_use_role_comparable_dataset_values():
    cohort = pd.DataFrame(
        {
            "Department": ["Finance"] * 8,
            "JobRole": ["Analyst"] * 8,
            "JobLevel": [2] * 8,
            "Monthly Income": ["$2,000", "$3,000", "$3,200", "$3,400", "$3,600", "$3,800", "$4,000", "$4,200"],
        }
    )

    plan = generate_personalized_retention_plan(cohort.loc[0], cohort, 0.2)
    recommendation = plan["recommendations"][0]

    assert recommendation["strategy_title"] == "Benchmark pay against comparable roles"
    assert "below the 25th percentile" in recommendation["reason"]
    assert "does not establish inequity" in recommendation["reason"]


def test_personalized_plan_requests_more_information_instead_of_inventing_issues():
    employee = pd.Series(
        {
            "JobSatisfaction": None,
            "OverTime": None,
            "PerformanceRating": None,
            "MonthlyIncome": None,
        }
    )
    cohort = pd.DataFrame([employee])

    plan = generate_personalized_retention_plan(employee, cohort, 0.4)

    assert plan["recommendations"] == []
    assert "insufficient information" in plan["insufficient_information"].lower()


def test_model_training_features_exclude_employee_ids_and_names():
    data = pd.DataFrame(
        {
            "EmployeeID": ["E1", "E2"],
            "Employee Name": ["Alex", "Jordan"],
            "Department": ["Sales", "Engineering"],
            "MonthlyIncome": [3000, 5000],
            "Attrition": ["No", "Yes"],
        }
    )

    features, _, employee_column, _ = prepare_features_and_target(data)
    aligned = align_features_to_model(
        features,
        ["EmployeeName", "Department", "MonthlyIncome"],
        numerical_features=["MonthlyIncome"],
    )

    assert employee_column == "EmployeeID"
    assert list(features.columns) == ["Department", "MonthlyIncome"]
    assert aligned["EmployeeName"].tolist() == ["__redacted_identity__"] * 2
    assert "Employee Name" not in aligned.columns
