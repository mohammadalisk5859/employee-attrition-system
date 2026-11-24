import math
import re

import pandas as pd


_FIELD_ALIASES = {
    "job_satisfaction": ("JobSatisfaction", "Job Satisfaction"),
    "environment_satisfaction": ("EnvironmentSatisfaction", "Environment Satisfaction"),
    "relationship_satisfaction": ("RelationshipSatisfaction", "Relationship Satisfaction"),
    "job_involvement": ("JobInvolvement", "Job Involvement"),
    "performance": ("PerformanceRating", "Performance Rating", "Performance"),
    "monthly_income": ("MonthlyIncome", "Monthly Income", "Salary", "Income"),
    "salary_hike": ("PercentSalaryHike", "Percent Salary Hike", "Salary Hike"),
    "tenure": ("YearsAtCompany", "Years At Company", "Tenure"),
    "promotion": ("YearsSinceLastPromotion", "Years Since Last Promotion"),
    "manager_tenure": ("YearsWithCurrentManager", "Years With Current Manager"),
    "work_life_balance": ("WorkLifeBalance", "Work Life Balance"),
    "overtime": ("OverTime", "Overtime"),
    "training": ("TrainingTimesLastYear", "Training Times Last Year"),
    "distance": ("DistanceFromHome", "Distance From Home", "CommuteDistance"),
    "job_level": ("JobLevel", "Job Level"),
    "department": ("Department",),
    "job_role": ("JobRole", "Job Role"),
    "business_travel": ("BusinessTravel", "Business Travel"),
    "career_growth": ("CareerGrowth", "Career Growth"),
    "manager_support": ("ManagerSupport", "Manager Support"),
}

_LOW_LABELS = {"low", "very low", "poor", "unsatisfied", "dissatisfied"}
_HIGH_LABELS = {"high", "very high", "excellent", "outstanding", "exceeds"}

_ACTIONS = {
    "job_satisfaction": [
        ("Confidential satisfaction discussion", "Offer a confidential, non-punitive discussion and ask which aspects of work the employee would like to improve.", ("job satisfaction", "work concerns", "employee voice", "listening", "stay interview")),
        ("Employee-led improvement action", "Agree on one practical change chosen with the employee, record the owner, and confirm what success would look like.", ("employee voice", "action plan", "feedback", "employee priorities")),
        ("Satisfaction follow-up", "Check back within 30 days to ask whether the agreed change helped; revise it with the employee if not.", ("check in", "follow up", "feedback", "wellbeing")),
    ],
    "environment_satisfaction": [
        ("Identify the workplace barrier", "Ask which work-environment condition is affecting the employee instead of assuming the cause.", ("work environment", "workplace", "workspace", "conditions")),
        ("Address the stated condition", "Route the specific issue the employee identifies to the appropriate facilities, safety, or team owner.", ("facilities", "safety", "equipment", "resources", "workspace")),
        ("Verify the improvement", "Follow up within 30 days to confirm whether the agreed workplace change resolved the stated concern.", ("follow up", "workplace", "work environment")),
    ],
    "relationship_satisfaction": [
        ("Confidential relationship check-in", "Offer a confidential conversation with HR or another trusted contact to understand the employee's perspective.", ("relationships", "manager", "supervisor", "team", "communication")),
        ("Agree on appropriate support", "With the employee's agreement, consider manager coaching, clearer communication, or facilitated support relevant to the concern they describe.", ("manager", "team", "communication", "conflict", "mediation")),
        ("Review the agreed support", "Check in within 30 days with the employee to see whether the agreed support improved their working relationships.", ("follow up", "relationships", "manager", "team")),
    ],
    "job_involvement": [
        ("Offer a meaningful project", "Ask which skills or work areas interest the employee, then offer a scoped project that uses those interests where operationally feasible.", ("meaningful work", "project", "job involvement", "skills")),
        ("Invite input into decisions", "Invite the employee to contribute to planning or decisions that affect their role and explain how their input will be used.", ("employee voice", "participation", "decisions", "planning")),
        ("Review engagement in the work", "Follow up within 30 days to ask whether the new responsibility is meaningful and appropriately supported.", ("engagement", "follow up", "project")),
    ],
    "performance_low": [
        ("Development-focused performance discussion", "Review expectations and obstacles with the employee, and distinguish skill or resource needs from other causes before setting actions.", ("performance support", "coaching", "expectations", "feedback")),
        ("Targeted skill development", "Select training or mentoring tied to a specific, observed skill gap and the employee's role.", ("training", "mentoring", "skills", "development")),
        ("Set supportive progress goals", "Agree on a small number of measurable near-term goals and schedule coaching check-ins; review progress before making conclusions.", ("goals", "coaching", "progress", "feedback")),
    ],
    "performance_high": [
        ("Recognize strong performance", "Recognize the recorded performance with specific, job-related feedback and ask which forms of recognition the employee values.", ("recognition", "performance", "appreciation")),
        ("Discuss career advancement", "Ask whether the employee wants advancement, then review the next-level criteria for their current role and job level.", ("career", "advancement", "promotion", "progression")),
        ("Offer a leadership opportunity", "If the employee is interested, offer a time-bounded stretch assignment or mentoring opportunity aligned with their role.", ("leadership", "stretch assignment", "mentoring", "development")),
    ],
    "monthly_income": [
        ("Benchmark pay against comparable roles", "Ask compensation staff to compare this employee's pay with the applicable internal range and genuinely comparable roles; dataset rank alone does not establish inequity.", ("compensation", "salary", "pay", "benchmark", "pay band")),
        ("Explain compensation pathways", "Explain the relevant pay-band, review, and progression process for the employee's role and level without promising an adjustment.", ("compensation", "salary", "pay", "progression", "pay band")),
        ("Close the loop on the review", "If the employee requests a review, communicate its owner and expected decision date, then follow up when due.", ("compensation", "pay review", "follow up")),
    ],
    "salary_hike": [
        ("Discuss pay progression", "Review the employee's recorded salary-hike history with the applicable role and level criteria; clarify the process and next review point.", ("salary", "pay", "compensation", "progression")),
        ("Agree on a growth plan", "If the employee wants progression, document the role-related skills or outcomes needed for the next compensation review.", ("career", "development", "salary", "progression")),
        ("Schedule a compensation follow-up", "Set a follow-up at the next relevant review date to revisit progress and explain any decision.", ("compensation", "follow up", "review")),
    ],
    "promotion": [
        ("Discuss career direction", "Ask whether the employee wants promotion or a different development path before proposing a next step.", ("career", "promotion", "progression", "development")),
        ("Clarify advancement criteria", "Review role-specific promotion requirements and identify any evidence or experience the employee wants help gaining.", ("promotion", "career", "criteria", "role")),
        ("Create a progression checkpoint", "Agree on a development milestone and review it with the employee within 60 days.", ("career", "development", "progression", "follow up")),
    ],
    "training": [
        ("Identify a relevant learning need", "Ask the employee which role-related skills they want to build and confirm the need with their manager.", ("training", "learning", "skills", "development")),
        ("Provide targeted training", "Offer a specific course, mentoring, or supervised practice that matches the agreed skill need and role.", ("training", "mentoring", "course", "skills")),
        ("Apply learning at work", "Pair the learning opportunity with a suitable task or project, then review whether it helped within 60 days.", ("project", "training", "learning", "follow up")),
    ],
    "work_life_balance": [
        ("Discuss work-life balance", "Ask the employee which parts of their schedule or workload are difficult and what adjustment would help.", ("work life balance", "flexible", "schedule", "workload")),
        ("Trial a workable schedule adjustment", "Where the role and operations permit, agree on a time-bounded schedule or flexibility trial with the employee.", ("flexible", "schedule", "remote", "hybrid", "working hours")),
        ("Review the schedule trial", "Check within 30 days whether the agreed schedule is sustainable and adjust it based on the employee's feedback.", ("schedule", "follow up", "work life balance")),
    ],
    "overtime": [
        ("Review overtime drivers", "Review recent overtime with the employee to identify workload, staffing, or scheduling drivers without assuming the cause.", ("overtime", "workload", "hours", "staffing")),
        ("Rebalance workload", "Agree with the manager which tasks can be redistributed, rescheduled, or deprioritized, then confirm operational ownership.", ("workload", "redistribute", "staffing", "schedule")),
        ("Trial a sustainable schedule", "Where feasible, trial a schedule adjustment and review recorded overtime again within 30 days.", ("schedule", "flexible", "working hours", "follow up")),
    ],
    "tenure": [
        ("Clarify role expectations", "Review role priorities and success measures with the employee and resolve any onboarding or role-clarity gaps they identify.", ("onboarding", "role clarity", "expectations", "orientation")),
        ("Offer a peer mentor", "Ask whether the employee would value a peer mentor familiar with the role or team.", ("mentor", "mentoring", "onboarding")),
        ("Set an early progress check-in", "Schedule a check-in within 30 days to review support needs and the employee's experience in the role.", ("check in", "onboarding", "support")),
    ],
    "distance": [
        ("Check commute impact", "Ask whether the recorded home-to-work distance creates a practical commute burden; do not infer that it does.", ("commute", "distance", "travel", "flexible")),
        ("Discuss feasible flexibility", "If the employee says the commute is a concern, discuss schedule or location flexibility that fits the role and policy.", ("commute", "flexible", "remote", "hybrid", "schedule")),
        ("Follow up on the commute plan", "If an adjustment is agreed, check within 30 days whether it is workable for the employee and team.", ("commute", "follow up", "flexible")),
    ],
    "business_travel": [
        ("Review travel demands", "Ask how the recorded travel pattern affects the employee and whether advance notice or a different cadence would help.", ("business travel", "travel", "schedule", "travel demands")),
        ("Plan travel recovery", "Where travel is required, review scheduling and recovery time with the employee and manager.", ("business travel", "travel", "recovery", "schedule")),
        ("Check travel sustainability", "Follow up within 30 days to see whether the agreed travel-planning changes are working.", ("travel", "follow up", "schedule")),
    ],
    "career_growth": [
        ("Discuss growth interests", "Ask which skills or career directions the employee wants to pursue before selecting a development activity.", ("career", "growth", "development", "progression")),
        ("Create a role-relevant development plan", "Agree on one development goal, a suitable learning opportunity, and how progress will be reviewed.", ("development", "career", "learning", "goal")),
        ("Review growth progress", "Set a 60-day checkpoint to review the employee's chosen development goal and remove barriers.", ("career", "development", "follow up")),
    ],
    "manager_support": [
        ("Ask what manager support would help", "Discuss the employee's recorded manager-support rating and ask what specific support they would find useful.", ("manager support", "manager", "feedback", "communication")),
        ("Agree on a manager action", "With the employee's input, agree on a concrete manager action such as clearer priorities or more regular feedback.", ("manager", "feedback", "priorities", "communication")),
        ("Review manager support", "Check within 30 days whether the agreed manager action is helping the employee.", ("manager", "follow up", "feedback")),
    ],
}


def _normalize(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def _column_name(row, aliases):
    if isinstance(row, pd.DataFrame):
        columns = row.columns
    elif hasattr(row, "index"):
        columns = row.index
    else:
        columns = row.keys()
    by_normalized = {_normalize(column): column for column in columns}
    for alias in aliases:
        column = by_normalized.get(_normalize(alias))
        if column is not None:
            return column
    return None


def _value(row, aliases):
    column = _column_name(row, aliases)
    if column is None:
        return None, None
    value = row[column]
    if pd.isna(value):
        return column, None
    return column, value


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        text = str(value).strip().replace(",", "")
        text = re.sub(r"^[^\d.+-]+", "", text)
        text = re.sub(r"[^\d.eE+-]+$", "", text)
        try:
            number = float(text)
        except (TypeError, ValueError):
            return None
    return number if math.isfinite(number) else None


def _cohort_numeric(cohort, column):
    cohort_column = _column_name(cohort, (column,)) if column is not None else None
    if cohort_column is None:
        return pd.Series(dtype=float)
    return cohort[cohort_column].map(_number).dropna().astype(float)


def _numeric_low(value, series, ordinal=False):
    if value is None:
        return False
    if ordinal and not series.empty and series.min() >= 1 and series.max() <= 5:
        return value <= 2
    if series.empty:
        return False
    if ordinal and series.min() >= 0 and series.max() <= 5:
        return value <= 2
    q25 = float(series.quantile(.25))
    median = float(series.median())
    return value <= q25 and value < median


def _numeric_high(value, series, ordinal=False):
    if value is None:
        return False
    if ordinal and not series.empty and series.min() >= 1 and series.max() <= 5:
        return value >= 4
    if series.empty:
        return False
    if ordinal and series.min() >= 0 and series.max() <= 5:
        return value >= 4
    q75 = float(series.quantile(.75))
    median = float(series.median())
    return value >= q75 and value > median


def _rating_signal(employee, cohort, key, label, ordinal=False):
    aliases = _FIELD_ALIASES[key]
    column, raw_value = _value(employee, aliases)
    if column is None or raw_value is None:
        return None
    number = _number(raw_value)
    series = _cohort_numeric(cohort, column)
    text_value = _text(raw_value).lower()
    if (number is not None and _numeric_low(number, series, ordinal=ordinal)) or text_value in _LOW_LABELS:
        return {
            "key": key,
            "label": label,
            "value": raw_value,
            "reason": f"{label} is recorded as {raw_value}.",
            "fields": {_normalize(column)},
            "strength": 1.0,
        }
    return None


def _relative_signal(employee, cohort, key, label, comparison="lower", zero_is_low=False):
    aliases = _FIELD_ALIASES[key]
    column, raw_value = _value(employee, aliases)
    number = _number(raw_value)
    if column is None or number is None:
        return None
    series = _cohort_numeric(cohort, column)
    if len(series) < 4:
        if not (zero_is_low and number == 0):
            return None
        threshold_description = "is 0 in the uploaded record"
    else:
        quantile = float(series.quantile(.25 if comparison == "lower" else .75))
        median = float(series.median())
        is_signal = (
            number <= quantile and number < median
            if comparison == "lower"
            else number >= quantile and number > median
        )
        if zero_is_low and number == 0:
            is_signal = True
        if not is_signal:
            return None
        band = "lowest" if comparison == "lower" else "highest"
        threshold_description = f"falls in the {band} quartile of the uploaded dataset"
    return {
        "key": key,
        "label": label,
        "value": raw_value,
        "reason": f"{label} is {raw_value} and {threshold_description}.",
        "fields": {_normalize(column)},
        "strength": 1.0,
    }


def _peer_income_signal(employee, cohort):
    aliases = _FIELD_ALIASES["monthly_income"]
    income_column, raw_income = _value(employee, aliases)
    income = _number(raw_income)
    if income_column is None or income is None:
        return None

    peer_frame = cohort
    comparison_label = "uploaded dataset"
    grouping = []
    for key in ("job_role", "job_level", "department"):
        column = _column_name(cohort, _FIELD_ALIASES[key])
        employee_column = _column_name(employee, _FIELD_ALIASES[key])
        if column is not None and employee_column is not None:
            value = _text(employee[employee_column])
            if value:
                grouping.append((column, value))
    for columns in (grouping, grouping[:1], grouping[1:]):
        if not columns:
            continue
        candidate = cohort
        for column, value in columns:
            candidate = candidate[
                candidate[column].astype(str).str.strip().str.casefold() == value.casefold()
            ]
        if len(candidate) >= 5:
            peer_frame = candidate
            comparison_label = "uploaded employees in the same " + ", ".join(
                column.replace("_", " ") for column, _ in columns
            )
            break

    series = _cohort_numeric(peer_frame, income_column)
    if len(series) < 4:
        return None
    q25 = float(series.quantile(.25))
    median = float(series.median())
    if income > q25 or income >= median:
        return None
    return {
        "key": "monthly_income",
        "label": "Monthly income",
        "value": raw_income,
        "reason": (
            f"Monthly income is {raw_income}, below the 25th percentile for the "
            f"{comparison_label}; this comparison alone does not establish inequity."
        ),
        "fields": {_normalize(income_column)},
        "strength": 1.0,
    }


def _collect_signals(employee, cohort):
    signals = []
    for key, label in (
        ("job_satisfaction", "Job satisfaction"),
        ("environment_satisfaction", "Environment satisfaction"),
        ("relationship_satisfaction", "Relationship satisfaction"),
        ("job_involvement", "Job involvement"),
        ("work_life_balance", "Work-life balance"),
        ("manager_support", "Manager support"),
    ):
        signal = _rating_signal(employee, cohort, key, label, ordinal=True)
        if signal:
            signals.append(signal)

    performance_column, performance_value = _value(employee, _FIELD_ALIASES["performance"])
    performance_number = _number(performance_value)
    if performance_column is not None and performance_value is not None:
        performance_series = _cohort_numeric(cohort, performance_column)
        performance_text = _text(performance_value).lower()
        if (
            performance_text in _HIGH_LABELS
            or _numeric_high(performance_number, performance_series, ordinal=True)
        ):
            signals.append({
                "key": "performance_high",
                "label": "Performance rating",
                "value": performance_value,
                "reason": f"Performance rating is recorded as {performance_value}.",
                "fields": {_normalize(performance_column)},
                "strength": 1.0,
            })
        elif (
            performance_text in _LOW_LABELS
            or _numeric_low(performance_number, performance_series, ordinal=True)
        ):
            signals.append({
                "key": "performance_low",
                "label": "Performance rating",
                "value": performance_value,
                "reason": f"Performance rating is recorded as {performance_value}.",
                "fields": {_normalize(performance_column)},
                "strength": 1.0,
            })

    overtime_column, overtime_value = _value(employee, _FIELD_ALIASES["overtime"])
    if overtime_column is not None and overtime_value is not None:
        overtime_text = _text(overtime_value).lower().replace("_", " ")
        overtime_number = _number(overtime_value)
        overtime_series = _cohort_numeric(cohort, overtime_column)
        if overtime_text in {"yes", "true", "1", "often", "frequently", "always"} or (
            overtime_number is not None
            and _numeric_high(overtime_number, overtime_series)
        ):
            signals.append({
                "key": "overtime",
                "label": "Overtime",
                "value": overtime_value,
                "reason": f"Overtime is recorded as {overtime_value}.",
                "fields": {_normalize(overtime_column)},
                "strength": 1.0,
            })

    for key, label in (
        ("salary_hike", "Percent salary hike"),
        ("training", "Training times last year"),
        ("career_growth", "Career growth"),
    ):
        signal = _relative_signal(
            employee,
            cohort,
            key,
            label,
            comparison="lower",
            zero_is_low=key == "training",
        )
        if signal:
            signals.append(signal)

    for key, label in (
        ("promotion", "Years since last promotion"),
        ("distance", "Distance from home"),
    ):
        signal = _relative_signal(employee, cohort, key, label, comparison="higher")
        if signal:
            if key == "promotion" and _number(signal["value"]) < 3:
                continue
            if key == "distance":
                signal["reason"] += " Distance alone does not establish a commute burden."
            signals.append(signal)

    relationship_signal = next(
        (signal for signal in signals if signal["key"] == "relationship_satisfaction"),
        None,
    )
    manager_tenure_column, manager_tenure_value = _value(
        employee,
        _FIELD_ALIASES["manager_tenure"],
    )
    if relationship_signal is not None and manager_tenure_column and manager_tenure_value is not None:
        relationship_signal["reason"] += (
            f" Years with the current manager is {manager_tenure_value}; this is context, "
            "not evidence that the manager caused the rating."
        )

    career_context = []
    for key in ("department", "job_role", "job_level"):
        column, value = _value(employee, _FIELD_ALIASES[key])
        if column is not None and value is not None:
            career_context.append(f"{column} is {value}")
    if career_context:
        for signal in signals:
            if signal["key"] in {"performance_high", "promotion"}:
                signal["reason"] += " Recorded career context: " + "; ".join(career_context) + "."

    income_signal = _peer_income_signal(employee, cohort)
    if income_signal:
        signals.append(income_signal)

    tenure_column, tenure_value = _value(employee, _FIELD_ALIASES["tenure"])
    tenure = _number(tenure_value)
    if tenure_column is not None and tenure is not None and tenure <= 2:
        signals.append({
            "key": "tenure",
            "label": "Years at company",
            "value": tenure_value,
            "reason": f"Tenure is {tenure_value} years, so onboarding and role clarity may still be relevant.",
            "fields": {_normalize(tenure_column)},
            "strength": 1.0,
        })

    travel_column, travel_value = _value(employee, _FIELD_ALIASES["business_travel"])
    if travel_column is not None and travel_value is not None:
        travel_text = _text(travel_value).lower().replace("_", " ")
        travel_number = _number(travel_value)
        travel_series = _cohort_numeric(cohort, travel_column)
        if travel_text in {"travel frequently", "frequent", "frequently", "high"} or (
            travel_number is not None and _numeric_high(travel_number, travel_series)
        ):
            signals.append({
                "key": "business_travel",
                "label": "Business travel",
                "value": travel_value,
                "reason": f"Business travel is recorded as {travel_value}; its impact should be confirmed with the employee.",
                "fields": {_normalize(travel_column)},
                "strength": 1.0,
            })

    return signals


def _explanation_importances(explanation):
    if not isinstance(explanation, dict):
        return {}
    items = explanation.get("top_features") or explanation.get("local_contributions") or []
    importances = {}
    for item in items:
        if isinstance(item, dict):
            name = item.get("feature")
            value = item.get("importance", item.get("sensitivity", item.get("contribution", 0)))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            name, value = item[0], item[1]
        else:
            continue
        score = _number(value)
        if name is not None and score is not None:
            key = _normalize(name)
            current = importances.get(key)
            if current is None or abs(score) > current[1]:
                importances[key] = (str(name), abs(score))
    return importances


def _best_feature_importance(signal, importances):
    matches = [
        (feature_name, sensitivity)
        for normalized_name, (feature_name, sensitivity) in importances.items()
        if any(
            normalized_name == field or field in normalized_name
            for field in signal["fields"]
        )
    ]
    return max(matches, key=lambda match: match[1]) if matches else (None, 0.0)


def _policy_reference(terms, knowledge_documents):
    normalized_terms = [
        " ".join(re.findall(r"[a-z0-9]+", term.lower()))
        for term in terms
    ]
    for document in knowledge_documents or []:
        text = re.sub(r"\s+", " ", str(document.get("text", ""))).strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            normalized = " " + " ".join(re.findall(r"[a-z0-9]+", sentence.lower())) + " "
            if any(term and f" {term} " in normalized for term in normalized_terms):
                return sentence.strip(), document.get("source_name", "Uploaded HR guidance")
    return None, None


def _signal_actions(signal):
    if signal["key"] in _ACTIONS:
        return _ACTIONS[signal["key"]]
    return []


def generate_personalized_retention_plan(
    employee,
    cohort,
    probability,
    local_explanation=None,
    knowledge_documents=None,
    limit=5,
):
    """Create employee-specific, evidence-linked actions without inferring causes."""
    signals = _collect_signals(employee, cohort)
    if not signals:
        return {
            "recommendations": [],
            "insufficient_information": (
                "Insufficient information is available to recommend a specific retention action. "
                "HR should verify data coverage and ask the employee about their priorities "
                "before proposing an intervention."
            ),
            "signals": [],
        }

    probability = _number(probability)
    probability = 0.0 if probability is None else min(max(probability, 0.0), 1.0)
    risk_category = "High" if probability >= .60 else "Medium" if probability >= .30 else "Low"
    importances = _explanation_importances(local_explanation)
    for signal in signals:
        signal["model_feature"], signal["model_sensitivity"] = _best_feature_importance(
            signal,
            importances,
        )
        signal["priority_score"] = (
            signal["model_sensitivity"] * 2
            + (1.0 if risk_category == "High" else .5 if risk_category == "Medium" else 0)
            + signal["strength"]
        )

    signals.sort(key=lambda signal: signal["priority_score"], reverse=True)
    candidates = []
    for signal in signals:
        actions = _signal_actions(signal)
        for action_index, (title, action, terms) in enumerate(actions):
            policy_text, policy_source = _policy_reference(terms, knowledge_documents or [])
            reason = signal["reason"]
            if risk_category != "Low":
                reason += f" The model estimates {risk_category.lower()} attrition risk ({probability:.0%})."
            if signal["model_feature"] is not None:
                reason += (
                    f" The LIME-style local surrogate ranked {signal['model_feature']} among "
                    f"this record's influential inputs (absolute weight "
                    f"{signal['model_sensitivity']:.3f}); this is not evidence of cause."
                )
            priority = (
                "High" if risk_category == "High"
                else "Medium" if risk_category == "Medium" or signal["model_feature"] is not None
                else "Low"
            )
            timeline = {
                "High": "Within 7 days",
                "Medium": "Within 30 days",
                "Low": "At the next 1:1, within 60 days",
            }[priority]
            candidates.append({
                "strategy_title": title,
                "reason": reason,
                "action": action,
                "priority": priority,
                "timeline": timeline,
                "policy_guidance": policy_text,
                "policy_source": policy_source,
                "signal_key": signal["key"],
                "signal_priority": signal["priority_score"],
                "action_index": action_index,
            })

    recommendations = []
    seen_titles = set()
    for signal in signals:
        for candidate in candidates:
            if candidate["signal_key"] != signal["key"] or candidate["action_index"] != 0:
                continue
            if candidate["strategy_title"] not in seen_titles:
                recommendations.append(candidate)
                seen_titles.add(candidate["strategy_title"])
            break
    for candidate in candidates:
        if candidate["strategy_title"] not in seen_titles:
            recommendations.append(candidate)
            seen_titles.add(candidate["strategy_title"])
        if len(recommendations) >= limit:
            break

    for recommendation in recommendations:
        recommendation.pop("signal_key", None)
        recommendation.pop("signal_priority", None)
        recommendation.pop("action_index", None)

    return {
        "recommendations": recommendations[:limit],
        "insufficient_information": None,
        "signals": signals,
    }
