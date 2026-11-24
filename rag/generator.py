import math
import re

import pandas as pd

from rag.personalization import generate_personalized_retention_plan


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


def _active_policy_topics(employee):
    """Return policy topics supported by explicit employee profile indicators."""
    topics = []
    overtime = _text(employee, ["OverTime", "Overtime", "overtime"])
    work_life_balance = _number(employee, ["WorkLifeBalance", "Work Life Balance"])
    if overtime in {"yes", "true", "1"} or (
        work_life_balance is not None and work_life_balance <= 2
    ):
        topics.append({
            "title": "Workload and work-life balance",
            "terms": (
                "overtime", "workload", "overwork", "schedule", "scheduling",
                "shift", "flexible hours", "flexible working", "flexible work",
                "work life balance", "working hours", "work from home", "remote",
                "hybrid", "time off", "leave",
            ),
            "recommendation": (
                "Discuss workload and scheduling preferences with the employee, "
                "then agree on a sustainable arrangement."
            ),
        })

    job_satisfaction = _number(employee, ["JobSatisfaction", "Job Satisfaction"])
    if job_satisfaction is not None and job_satisfaction <= 2:
        topics.append({
            "title": "Job satisfaction and recognition",
            "terms": (
                "job satisfaction", "recognition", "appreciation", "feedback",
                "engagement", "morale", "wellbeing", "well being", "check in",
                "stay interview", "employee voice", "listening",
            ),
            "recommendation": (
                "Hold a confidential stay conversation to learn what is affecting "
                "job satisfaction and agree on one employee-prioritized action."
            ),
        })

    environment_satisfaction = _number(
        employee,
        ["EnvironmentSatisfaction", "Environment Satisfaction"],
    )
    if environment_satisfaction is not None and environment_satisfaction <= 2:
        topics.append({
            "title": "Work environment",
            "terms": (
                "work environment", "workplace", "workspace", "equipment",
                "resources", "safety", "facilities", "tools", "work conditions",
            ),
            "recommendation": (
                "Ask which workplace conditions are affecting the employee and "
                "follow up on the specific issue they identify."
            ),
        })

    relationship_satisfaction = _number(
        employee,
        ["RelationshipSatisfaction", "Relationship Satisfaction"],
    )
    if relationship_satisfaction is not None and relationship_satisfaction <= 2:
        topics.append({
            "title": "Workplace relationships",
            "terms": (
                "relationship", "manager", "supervisor", "team", "colleague",
                "relationships", "managers", "teams", "peers", "communication",
                "conflict", "mediation",
            ),
            "recommendation": (
                "Offer a confidential conversation about workplace relationships "
                "and discuss appropriate manager or team support."
            ),
        })

    tenure = _number(
        employee,
        ["YearsAtCompany", "Years At Company", "Tenure", "YearsInCurrentRole"],
    )
    if tenure is not None and tenure <= 2:
        topics.append({
            "title": "Onboarding and career development",
            "terms": (
                "onboarding", "orientation", "mentor", "mentoring", "training",
                "development", "career", "progression", "growth", "role clarity",
                "expectations", "coaching", "learning",
            ),
            "recommendation": (
                "Check that role expectations and development goals are clear; "
                "offer a mentor or training if the employee wants one."
            ),
        })

    compensation_satisfaction = _number(
        employee,
        ["SalarySatisfaction", "CompensationSatisfaction", "PaySatisfaction"],
    )
    if compensation_satisfaction is not None and compensation_satisfaction <= 2:
        topics.append({
            "title": "Compensation and benefits",
            "terms": (
                "compensation", "salary", "pay", "benefits", "reward",
            ),
            "recommendation": (
                "Ask whether compensation or benefits are a concern and, if so, "
                "explain the relevant review process and options."
            ),
        })

    return topics


def select_policy_cards(employee, knowledge_documents, limit=3):
    """Select distinct HR guidance only when it matches an employee's risk signals."""
    topics = _active_policy_topics(employee)
    if not topics or not knowledge_documents:
        return []

    best_by_topic = {}
    for document_index, document in enumerate(knowledge_documents):
        text = re.sub(r"\s+", " ", str(document.get("text", ""))).strip()
        for sentence_index, sentence in enumerate(re.split(r"(?<=[.!?])\s+", text)):
            sentence = sentence.strip(" -\n\t")
            normalized = " " + " ".join(re.findall(r"[a-z0-9]+", sentence.lower())) + " "
            if len(normalized.split()) < 4:
                continue
            for topic_index, topic in enumerate(topics):
                matched_terms = [
                    term for term in topic["terms"]
                    if " " + term + " " in normalized
                    or " " + term + "s " in normalized
                ]
                if not matched_terms:
                    continue
                candidate = (
                    -len(matched_terms),
                    document_index,
                    sentence_index,
                    topic["title"],
                    sentence,
                    document.get("source_name", "HR policy"),
                )
                current = best_by_topic.get(topic_index)
                if current is None or candidate[:3] < current[:3]:
                    best_by_topic[topic_index] = candidate

    candidates = sorted(best_by_topic.values(), key=lambda item: item[:3])
    cards = []
    seen_sentences = set()
    for _, _, _, title, sentence, source in candidates:
        normalized_sentence = " ".join(re.findall(r"[a-z0-9]+", sentence.lower()))
        if normalized_sentence in seen_sentences:
            continue
        seen_sentences.add(normalized_sentence)
        cards.append((title, sentence, source))
        if len(cards) == limit:
            break
    return cards


def _document_recommendation(employee, knowledge_documents):
    """Combine applicable actions from employee-matched HR guidance."""
    cards = select_policy_cards(employee, knowledge_documents, limit=2)
    if not cards:
        return None
    guidance = " | ".join(sentence for _, sentence, _ in cards)
    if len(guidance) > 480:
        guidance = guidance[:477].rsplit(" ", 1)[0] + "..."
    return "Apply the organization HR guidance: " + guidance


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
    """Compatibility wrapper returning personalized actions as readable strings."""
    employee_values = employee if isinstance(employee, dict) else employee.to_dict()
    plan = generate_personalized_retention_plan(
        employee_values,
        pd.DataFrame([employee_values]),
        probability,
        knowledge_documents=knowledge_documents,
    )
    recommendations = [
        (
            f"{item['priority']} priority — {item['strategy_title']}: "
            f"{item['reason']} Recommended action: {item['action']}"
        )
        for item in plan["recommendations"]
    ]
    if not recommendations:
        recommendations.append(plan["insufficient_information"])
    if risk_factors:
        recommendations[0] += " Additional review factors: " + ", ".join(map(str, risk_factors)) + "."
    return recommendations
