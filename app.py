import io
import html
import json
import re
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st

from config import MODEL_METADATA_FILE
from ml.predict import load_artifacts, predict_attrition
from ml.preprocessing import clean_dataset, prepare_features_and_target, validate_dataset
from rag.document_loader import load_documents_from_files
from rag.generator import generate_retention_recommendations
from rag.vector_store import VectorStore


st.set_page_config(page_title="Employee Attrition", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --ink: #25313d;
        --muted: #68747c;
        --line: #d8ddd9;
        --paper: #f7f8f3;
        --blue: #dcebf5;
        --green: #1d594d;
    }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: #eef1ec; border-right: 1px solid var(--line); }
    .workspace-kicker { color: var(--green); font-size: .72rem; letter-spacing: .12em; font-weight: 700; }
    .workspace-title { color: var(--ink); font-size: clamp(2.25rem, 5vw, 4.2rem); line-height: .98; letter-spacing: -.04em; margin: .5rem 0 1rem; max-width: 760px; }
    .workspace-copy { color: var(--muted); font-size: 1.05rem; max-width: 700px; }
    .workspace-rule { border-top: 1px solid var(--line); margin: 2.5rem 0 1.5rem; }
    .workspace-note { background: var(--blue); border-radius: 8px; color: #31536c; padding: 1rem 1.25rem; margin-bottom: 1rem; }
    div[data-testid="stMetric"] { background: white; border: 1px solid var(--line); padding: 1rem; min-height: 112px; }
    div[data-testid="stMetricLabel"] p { color: var(--green); font-size: .72rem; letter-spacing: .1em; font-weight: 700; }
    .signal-card { background: #fff0ec; border: 1px solid #f0b5a8; padding: .85rem 1rem; margin: .55rem 0; }
    .signal-card strong { color: #c44b3c; }
    .signal-meta, .policy-source { color: #00635d; font-size: .68rem; letter-spacing: .1em; text-transform: uppercase; margin-top: .5rem; }
    .policy-card { background: #fbfcf9; border: 1px solid #cfd5d1; border-radius: 8px; padding: 1rem; margin: .65rem 0; }
    .policy-card strong { color: #1f2f3e; }
    .policy-card p { color: #29415a; margin: .7rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def read_employee_file(file_bytes, file_name):
    """Read CSV or Excel uploads into a dataframe."""
    if file_name.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes))
    return pd.read_excel(io.BytesIO(file_bytes))


def sample_employee_data():
    """Return a small, realistic dataset so the workspace is useful on first open."""
    records = [
        ("EMP-1042", "Maya Chen", 29, "Engineering", "Software Engineer", 4200, 2, "Yes", 2),
        ("EMP-1088", "Noah Williams", 41, "Finance", "Financial Analyst", 6500, 8, "No", 4),
        ("EMP-1120", "Amara Okafor", 34, "People", "HR Business Partner", 5100, 5, "No", 4),
        ("EMP-1156", "Liam Patel", 27, "Sales", "Account Executive", 3600, 1, "Yes", 2),
        ("EMP-1191", "Sofia Rossi", 46, "Operations", "Operations Lead", 7200, 11, "No", 3),
        ("EMP-1214", "Ethan Brooks", 31, "Engineering", "QA Engineer", 3900, 3, "No", 3),
        ("EMP-1247", "Priya Shah", 38, "Marketing", "Content Strategist", 4700, 6, "Yes", 2),
        ("EMP-1283", "Lucas Martin", 52, "Finance", "Finance Manager", 8300, 15, "No", 4),
        ("EMP-1319", "Hana Suzuki", 25, "Engineering", "Data Analyst", 3400, 1, "Yes", 1),
        ("EMP-1355", "Daniel Reed", 43, "Sales", "Sales Manager", 6900, 9, "No", 3),
        ("EMP-1398", "Zoe Mitchell", 36, "Operations", "Project Manager", 5600, 7, "No", 4),
        ("EMP-1431", "Mateo Garcia", 30, "Marketing", "Brand Specialist", 4100, 2, "Yes", 2),
        ("EMP-1462", "Ava Thompson", 40, "Engineering", "Product Manager", 7600, 10, "No", 4),
        ("EMP-1497", "Owen Kelly", 28, "Sales", "Sales Representative", 3200, 2, "Yes", 2),
        ("EMP-1524", "Nia Johnson", 45, "People", "Talent Manager", 6400, 12, "No", 4),
        ("EMP-1560", "Finn Murphy", 33, "Operations", "Process Analyst", 4300, 4, "No", 3),
        ("EMP-1593", "Leila Haddad", 26, "Engineering", "Frontend Developer", 3800, 1, "Yes", 2),
        ("EMP-1630", "Jack Wilson", 49, "Finance", "Senior Accountant", 7000, 14, "No", 3),
        ("EMP-1664", "Mina Park", 37, "Marketing", "Marketing Manager", 6100, 8, "No", 4),
        ("EMP-1699", "Theo Brown", 32, "Sales", "Account Manager", 4500, 4, "Yes", 2),
        ("EMP-1735", "Iris Nguyen", 42, "Operations", "Operations Manager", 6800, 10, "No", 4),
        ("EMP-1772", "Sam Carter", 24, "Engineering", "Junior Developer", 2900, 1, "Yes", 1),
        ("EMP-1806", "Elena Petrov", 39, "Finance", "Risk Analyst", 5900, 7, "No", 3),
        ("EMP-1843", "Caleb Adams", 35, "People", "Recruiter", 4000, 5, "No", 4),
    ]
    return pd.DataFrame(
        records,
        columns=[
            "Employee ID", "Employee Name", "Age", "Department", "Job Role",
            "Monthly Income", "Years At Company", "OverTime", "JobSatisfaction",
        ],
    ).assign(Attrition=["Yes", "No", "No", "Yes", "No", "No", "Yes", "No", "Yes", "No", "No", "Yes", "No", "Yes", "No", "No", "Yes", "No", "No", "Yes", "No", "Yes", "No", "No"])


def employee_risk_signals(employee):
    """Return the strongest visible profile signals for the employee detail view."""
    signals = []
    checks = [
        ("Job satisfaction", ["JobSatisfaction", "Job Satisfaction"]),
        ("Environment satisfaction", ["EnvironmentSatisfaction", "Environment Satisfaction"]),
        ("Relationship satisfaction", ["RelationshipSatisfaction", "Relationship Satisfaction"]),
        ("Work-life balance", ["WorkLifeBalance", "Work Life Balance"]),
        ("Flexible hours", ["FlexibleHours", "Flexible Hours"]),
        ("Overtime", ["OverTime", "Overtime"]),
        ("Monthly income", ["MonthlyIncome", "Monthly Income"]),
    ]
    for label, names in checks:
        column = next((name for name in names if name in employee.index), None)
        if column is None:
            continue
        value = employee[column]
        try:
            numeric_value = float(value)
            if numeric_value <= 2:
                signals.append((label, str(value), "RISK SIGNAL"))
        except (TypeError, ValueError):
            if str(value).strip().lower() in {"yes", "true", "low", "rarely"}:
                signals.append((label, str(value), "RISK SIGNAL"))
    if not signals:
        signals.append(("Profile indicators", "No elevated indicators", "PROTECTIVE / NEUTRAL"))
    return signals[:5]


def policy_cards(documents, employee):
    """Rank uploaded HR guidance against the selected employee's actual signals."""
    signals = set()
    for label, value, _ in employee_risk_signals(employee):
        signals.update(re.findall(r"[a-z0-9]+", f"{label} {value}".lower()))
    if "overtime" in signals:
        signals.update({"workload", "schedule", "hours", "flexible"})
    if "satisfaction" in " ".join(signals):
        signals.update({"listening", "conversation", "wellbeing", "support"})

    candidates = []
    seen = set()
    for document in documents:
        for sentence in re.split(r"(?<=[.!?])\s+", str(document.get("text", ""))):
            sentence = sentence.strip(" -\n\t")
            if len(sentence) < 24 or sentence.lower() in seen:
                continue
            seen.add(sentence.lower())
            title = " ".join(sentence.split()[:6]).strip(".,:;")
            words = set(re.findall(r"[a-z0-9]+", sentence.lower()))
            score = len(words & signals)
            candidates.append((score, title.title(), sentence, document.get("source_name", "HR policy")))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [(title, sentence, source) for _, title, sentence, source in candidates[:3]]


@st.cache_data
def fetch_web_strategy(query: str, max_chars: int = 800):
    """Fetch a short web summary for retention or attrition strategy guidance."""
    encoded_query = quote(query)
    url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_redirect=1&no_html=1&skip_disambig=1"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8", errors="ignore"))
    except Exception:
        return {
            "summary": "No live web strategy could be retrieved right now. Please try again shortly or use the HR document strategy panel.",
            "source": "offline",
        }

    summary = (payload.get("AbstractText") or payload.get("Abstract") or "").strip()
    if not summary:
        related = payload.get("RelatedTopics") or []
        for item in related:
            if isinstance(item, dict):
                text = (item.get("Text") or "").strip()
                if text:
                    summary = text
                    break
    if summary:
        summary = summary[:max_chars].strip()
        return {"summary": summary, "source": "duckduckgo"}
    return {
        "summary": "No live strategy summary was returned from the web. Try a broader keyword or use the document-based strategy view.",
        "source": "duckduckgo-empty",
    }


st.markdown('<div class="workspace-kicker">HR DECISION SUPPORT / WORKSPACE 01</div>', unsafe_allow_html=True)
st.markdown('<h1 class="workspace-title">See the signal<br>behind attrition.</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="workspace-copy">Turn employee data into a clearer conversation about who may need support, why, and what HR could consider next.</p>',
    unsafe_allow_html=True,
)
st.markdown('<div class="workspace-rule"></div>', unsafe_allow_html=True)

model_id = "unknown"
if MODEL_METADATA_FILE.exists():
    try:
        metadata = json.loads(MODEL_METADATA_FILE.read_text(encoding="utf-8"))
        model_id = metadata.get("model_id", "unknown")
    except Exception:
        model_id = "unknown"

st.sidebar.markdown(f"**Model ID:** {model_id}")

uploaded_file = st.sidebar.file_uploader(
    "Upload employee data",
    type=["csv", "xlsx", "xls"],
)

knowledge_files = st.sidebar.file_uploader(
    "Upload HR policy / knowledge documents",
    type=["txt", "pdf", "docx", "doc"],
    accept_multiple_files=True,
)

if uploaded_file is None:
    dataset = sample_employee_data()
    st.markdown(
        '<div class="workspace-note">Using the included sample dataset. Upload your own employee file in the sidebar, then select Train / refresh model.</div>',
        unsafe_allow_html=True,
    )
else:
    try:
        dataset = read_employee_file(uploaded_file.getvalue(), uploaded_file.name)
    except Exception as error:
        st.error(f"Unable to read the uploaded file: {error}")
        st.stop()

dataset = clean_dataset(dataset)
validation = validate_dataset(dataset)
st.sidebar.metric("Records", validation["rows"])
st.sidebar.metric("Columns", validation["columns"])

if validation["errors"]:
    for error in validation["errors"]:
        st.error(error)
    st.stop()

features, _, employee_column, attrition_column = prepare_features_and_target(dataset)

metric_columns = st.columns(3)
metric_columns[0].metric("Employees in view", validation["rows"])
metric_columns[1].metric("Features detected", validation["columns"] - 2)
metric_columns[2].metric("Workflow", "Upload -> Train -> Analyze")
st.subheader("Dataset preview")
st.dataframe(dataset.head(10), use_container_width=True)
st.caption(
    f"Identifier: {validation['employee_column']} | "
    f"Target: {validation['attrition_column']}"
)

training_result = None
if MODEL_METADATA_FILE.exists():
    try:
        metadata = json.loads(MODEL_METADATA_FILE.read_text(encoding="utf-8"))
        persisted_features = metadata.get("features", {})
        persisted_feature_names = (
            persisted_features.get("numerical_features", [])
            + persisted_features.get("categorical_features", [])
        )
        compatible_schema = (
            metadata.get("employee_column") == employee_column
            and metadata.get("attrition_column") == attrition_column
            and set(persisted_feature_names) == set(features.columns)
        )
        if metadata.get("ready") and compatible_schema:
            model, preprocessor, persisted_metadata = load_artifacts()
            training_result = {
                "model": model,
                "preprocessor": preprocessor,
                "metadata": persisted_metadata,
            }
            st.sidebar.success("Model READY: a compatible saved model was detected.")
        elif metadata.get("ready"):
            st.sidebar.info("Saved model schema differs from this dataset. Refresh the model to analyze it.")
    except Exception:
        training_result = None

if st.sidebar.button("Train / refresh model", type="primary", use_container_width=True):
    try:
        from ml.train import train_model

        with st.spinner("Training model..."):
            st.session_state["training_result"] = train_model(dataset)
        st.success("Model trained and saved.")
    except Exception as error:
        st.error(f"Training failed: {error}")

training_result = st.session_state.get("training_result", training_result)
if training_result is None:
    st.warning("Train the model from the sidebar to generate employee risk scores.")
    st.stop()

metrics = training_result["metadata"]["metrics"]
metric_columns = st.columns(5)
for column, (name, value) in zip(metric_columns, metrics.items()):
    column.metric(name.replace("_", " ").title(), f"{value:.1%}")

predictions = predict_attrition(
    features,
    model=training_result["model"],
    preprocessor=training_result["preprocessor"],
)

results = dataset.drop(columns=[attrition_column]).copy()
results["attrition_probability"] = predictions["probability"]
results["risk_level"] = predictions["risk"]
results = results.sort_values("attrition_probability", ascending=False)

knowledge_store = VectorStore()
if knowledge_files:
    extracted_docs = load_documents_from_files(knowledge_files)
    knowledge_store.add_documents(extracted_docs)
    st.sidebar.success(f"Indexed {len(extracted_docs)} HR knowledge document(s).")
else:
    st.sidebar.info("No HR knowledge documents indexed yet.")

st.subheader("Find an employee")
search_term = st.text_input(
    "Search by employee ID or name",
    placeholder="Enter an employee ID or name",
)

if search_term.strip():
    search_mask = results[employee_column].astype(str).str.contains(
        search_term.strip(), case=False, na=False
    )
    matching_employees = results[search_mask]

    if matching_employees.empty:
        st.warning("No employee matched that search.")
    else:
        selected_employee = st.selectbox(
            "Matching employees",
            matching_employees[employee_column].astype(str).tolist(),
        )
        selected_index = matching_employees[
            matching_employees[employee_column].astype(str) == selected_employee
        ].index[0]
        employee_result = results.loc[selected_index]
        employee_probability = float(employee_result["attrition_probability"])

        st.markdown(f"### Employee: {selected_employee}")
        metric_columns = st.columns(3)
        metric_columns[0].metric("Attrition probability", f"{employee_probability:.1%}")
        metric_columns[1].metric("Risk level", employee_result["risk_level"])
        metric_columns[2].metric(
            "Historical outcome",
            str(dataset.loc[selected_index, attrition_column]),
        )

        signal_column, support_column = st.columns([1, 1])
        with signal_column:
            st.markdown("#### Why this signal?")
            for label, value, signal_type in employee_risk_signals(employee_result):
                st.markdown(
                    f'<div class="signal-card"><strong>{html.escape(label)}: {html.escape(value)}</strong>'
                    f'<div class="signal-meta">{html.escape(signal_type)} · LOCAL PROFILE SIGNAL</div></div>',
                    unsafe_allow_html=True,
                )

        with support_column:
            st.markdown("#### Possible retention support")
            support_cards = policy_cards(knowledge_store.documents, employee_result)
            if not support_cards:
                support_cards = [
                    ("Personalized retention conversation", recommendation, "LOCAL STRATEGY")
                    for recommendation in generate_retention_recommendations(
                        employee_result,
                        employee_probability,
                        risk_factors=[],
                    )[:3]
                ]
            for title, description, source in support_cards:
                st.markdown(
                    f'<div class="policy-card"><strong>{html.escape(title)}</strong>'
                    f'<p>{html.escape(description)}</p>'
                    f'<div class="policy-source">SOURCE: {html.escape(str(source))}</div></div>',
                    unsafe_allow_html=True,
                )

        profile_column, action_column = st.columns([1, 1])
        with profile_column:
            st.markdown("#### Employee profile")
            profile = employee_result.drop(
                labels=["attrition_probability", "risk_level"]
            ).to_frame("Value")
            st.dataframe(profile, use_container_width=True)

        with action_column:
            st.markdown("#### Retention / attrition strategy")
            strategy_source = st.radio(
                "Strategy source",
                ["HR document", "Web search"],
                horizontal=True,
                key=f"strategy_source_{selected_employee}",
            )

            recommended_docs = knowledge_store.documents if knowledge_store.documents else []
            risk_factors = []
            if str(employee_result.get("OverTime", "")).strip().lower() in {"yes", "true", "1"}:
                risk_factors.append("OverTime")
            try:
                if float(employee_result.get("JobSatisfaction", 99)) <= 2:
                    risk_factors.append("JobSatisfaction")
            except (TypeError, ValueError):
                pass
            try:
                if float(employee_result.get("Years At Company", 99)) <= 2:
                    risk_factors.append("Years At Company")
            except (TypeError, ValueError):
                pass
            if not risk_factors:
                risk_factors.append("overall risk level")

            if st.button("Generate strategy", key=f"generate_strategy_{selected_employee}", type="primary"):
                if strategy_source == "HR document":
                    st.caption("Top HR document strategies")
                    recommendations = generate_retention_recommendations(
                        employee_result,
                        employee_probability,
                        knowledge_documents=recommended_docs,
                        risk_factors=risk_factors,
                    )[:3]
                    for recommendation in recommendations:
                        st.write(f"- {recommendation}")
                else:
                    query = (
                        f"employee retention strategy attrition reduction for {selected_employee} "
                        f"risk={employee_result['risk_level']} "
                        f"overtime={employee_result.get('OverTime', '')} "
                        f"job satisfaction={employee_result.get('JobSatisfaction', '')}"
                    )
                    web_result = fetch_web_strategy(query)
                    st.caption("Live web strategy summary")
                    st.info(web_result["summary"])
                    if web_result["source"] != "offline":
                        search_url = "https://duckduckgo.com/?q=" + quote(query)
                        st.link_button("Open full web search", search_url, use_container_width=True)

st.subheader("Employee risk ranking")
st.dataframe(
    results.style.format({"attrition_probability": "{:.1%}"}),
    use_container_width=True,
)
