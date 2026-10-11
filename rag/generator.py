import math


def _number(row, names):
    keys = row.keys() if isinstance(row, dict) else getattr(row, "index", [])
    for name in names:
        if name in keys:
            try:
                value = float(row[name])
                if not math.isnan(value):
                    return value
            except (TypeError, ValueError, KeyError):
                continue
    return None


def _text(row, names):
    keys = row.keys() if isinstance(row, dict) else getattr(row, "index", [])
    for name in names:
        if name in keys:
            value = row[name]
            text = str(value).strip().lower()
            if text:
                return text
    return None


def summarize_recommendation_context(employee, probability, knowledge_documents=None, risk_factors=None):
    """Create a short summary mixing prediction context, feature risk factors, and retrieved knowledge."""
    if knowledge_documents is None:
        knowledge_documents = []
    if risk_factors is None:
        risk_factors = []

    context_parts = []
    if probability >= 0.60:
        context_parts.append("high predicted attrition risk")
    elif probability >= 0.30:
        context_parts.append("moderate attrition risk")
    else:
        context_parts.append("low attrition risk")

    if risk_factors:
        context_parts.append("key factors: " + ", ".join(map(str, risk_factors)))

    overtime = _text(employee, ["OverTime", "Overtime", "overtime"])
    if overtime in {"yes", "true", "1"}:
        context_parts.append("overtime pattern")

    if knowledge_documents:
        knowledge_text = " ; ".join(
            f"{doc.get('source_name', 'source')} -> {doc.get('text', '')}"
            for doc in knowledge_documents[:2]
        )
        context_parts.append(f"supporting HR knowledge: {knowledge_text}")

    return ". ".join(context_parts)


def generate_retention_recommendations(employee, probability, knowledge_documents=None, risk_factors=None):
    """Generate targeted local retention actions for one employee, with optional knowledge context."""
    knowledge_documents = knowledge_documents or []
    risk_factors = risk_factors or []
    recommendations = []

    overtime = _text(employee, ["OverTime", "Overtime", "overtime"])
    if overtime in {"yes", "true", "1"}:
        recommendations.append(
            "Review workload and overtime with the employee; agree on a sustainable schedule."
        )

    satisfaction = _number(employee, [
        "JobSatisfaction", "Job Satisfaction", "EnvironmentSatisfaction",
        "Environment Satisfaction", "satisfaction",
    ])
    if satisfaction is not None and satisfaction <= 2:
        recommendations.append(
            "Schedule a confidential check-in to identify the main source of dissatisfaction and create a 30-day improvement plan."
        )

    income = _number(employee, ["MonthlyIncome", "Monthly Income", "Salary", "Income"])
    if income is not None and income <= 3000:
        recommendations.append(
            "Benchmark compensation and discuss a transparent pay or progression review."
        )

    tenure = _number(employee, ["YearsAtCompany", "Years At Company", "Tenure", "YearsInCurrentRole"])
    if tenure is not None and tenure <= 2:
        recommendations.append(
            "Assign a mentor and schedule a stay interview focused on onboarding, growth, and role clarity."
        )

    if knowledge_documents:
        recommendations.append(
            "Use the retrieved HR guidance to support coaching and retention planning for this employee."
        )

    if not recommendations:
        recommendations.append(
            "Hold a stay interview, clarify career goals, and agree on one measurable retention action."
        )

    if probability >= 0.60:
        recommendations.insert(
            0,
            "Prioritize this employee for an HR follow-up within 7 days because the predicted attrition risk is high.",
        )
    elif probability >= 0.30:
        recommendations.insert(
            0,
            "Schedule a manager check-in within 30 days and monitor the risk after the intervention.",
        )

    if risk_factors:
        recommendations.append(
            "Focus the review on the strongest risk drivers: " + ", ".join(map(str, risk_factors)) + "."
        )

    return recommendations
