import io
import html
import json
import re
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st

from config import MODEL_METADATA_FILE
from explainability.lime_explainer import explain_instance
from ml.predict import load_artifacts, predict_attrition
from ml.preprocessing import (
    align_features_to_model,
    clean_dataset,
    is_employee_identity_column,
    prepare_features_and_target,
    validate_dataset,
)
from rag.document_loader import load_documents_from_files
from rag.personalization import generate_personalized_retention_plan
from rag.vector_store import VectorStore


st.set_page_config(page_title="Employee Attrition", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --ink: #172b3a;
        --muted: #647582;
        --line: #dce5e7;
        --paper: #f3f6f6;
        --navy: #102b3f;
        --navy-soft: #183d52;
        --teal: #168b79;
        --mint: #b7ead9;
        --gold: #e6c27a;
        --surface: #ffffff;
        --shadow: 0 14px 36px rgba(16, 43, 63, .07);
    }
    @keyframes portal-enter {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stApp {
        background:
            radial-gradient(ellipse at 82% 0%, rgba(22, 139, 121, .055), transparent 34rem),
            var(--paper);
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: rgba(243, 246, 246, .88); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8faf9 0%, #edf3f2 100%);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
    .portal-hero {
        position: relative;
        isolation: isolate;
        overflow: hidden;
        padding: clamp(2rem, 5vw, 3.6rem);
        margin: .75rem 0 1.6rem;
        border: 1px solid rgba(255, 255, 255, .12);
        border-radius: 22px;
        background: linear-gradient(122deg, var(--navy) 0%, #153b50 62%, #176253 115%);
        box-shadow: 0 22px 48px rgba(16, 43, 63, .16);
        animation: portal-enter .65s cubic-bezier(.2, .75, .25, 1) both;
    }
    .portal-hero::before, .portal-hero::after {
        position: absolute;
        z-index: -1;
        content: "";
        border: 1px solid rgba(183, 234, 217, .13);
        border-radius: 50%;
        pointer-events: none;
    }
    .portal-hero::before {
        width: 390px;
        height: 390px;
        top: -225px;
        right: -65px;
        box-shadow: 0 0 0 34px rgba(183, 234, 217, .035), 0 0 0 70px rgba(183, 234, 217, .025);
    }
    .portal-hero::after {
        width: 210px;
        height: 210px;
        right: 180px;
        bottom: -180px;
        border-color: rgba(230, 194, 122, .22);
    }
    .portal-hero-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    .workspace-kicker {
        color: var(--mint);
        font-size: .7rem;
        letter-spacing: .16em;
        font-weight: 750;
        text-transform: uppercase;
    }
    .portal-badge {
        padding: .48rem .75rem;
        border: 1px solid rgba(230, 194, 122, .42);
        border-radius: 999px;
        color: #f1d89f;
        background: rgba(255, 255, 255, .045);
        font-size: .64rem;
        letter-spacing: .12em;
        font-weight: 700;
        text-transform: uppercase;
        white-space: nowrap;
    }
    .workspace-title {
        color: #fff;
        font-size: clamp(2.25rem, 5vw, 4.15rem);
        line-height: 1;
        letter-spacing: -.045em;
        margin: 0 0 1rem;
        max-width: 760px;
    }
    .portal-hero h1.workspace-title { color: #fff !important; }
    .workspace-copy {
        color: rgba(238, 246, 245, .78);
        font-size: 1.04rem;
        line-height: 1.75;
        max-width: 700px;
        margin: 0;
    }
    .workspace-rule { border-top: 1px solid var(--line); margin: 1.7rem 0 1.5rem; }
    .workspace-note {
        background: linear-gradient(110deg, #e5f3ef, #eaf1f7);
        border: 1px solid #d5e8e2;
        border-left: 3px solid var(--teal);
        border-radius: 10px;
        color: #294958;
        padding: 1rem 1.25rem;
        margin-bottom: 1.2rem;
    }
    div[data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1.15rem 1.2rem;
        min-height: 112px;
        box-shadow: 0 5px 16px rgba(16, 43, 63, .035);
        animation: portal-enter .55s cubic-bezier(.2, .75, .25, 1) both;
        transition: transform .22s ease, box-shadow .22s ease, border-color .22s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        border-color: rgba(22, 139, 121, .38);
        box-shadow: var(--shadow);
    }
    div[data-testid="stMetricLabel"] p {
        color: var(--teal);
        font-size: .68rem;
        letter-spacing: .11em;
        font-weight: 750;
        text-transform: uppercase;
    }
    div[data-testid="stMetricValue"] { color: var(--navy); font-weight: 650; }
    div[data-testid="stHorizontalBlock"] > div:nth-child(3) [data-testid="stMetricValue"] {
        font-size: 1.15rem;
    }
    [data-testid="stSidebar"] div[data-testid="stMetric"] { min-height: 0; }
    [data-testid="stFileUploader"] section {
        border: 1px dashed #b9cbc9;
        border-radius: 12px;
        background: rgba(255, 255, 255, .76);
        transition: border-color .2s ease, background .2s ease;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: var(--teal);
        background: #fff;
    }
    div.stButton > button {
        border-radius: 10px;
        font-weight: 650;
        transition: transform .18s ease, box-shadow .18s ease, background .18s ease;
    }
    div.stButton > button[kind="primary"] {
        color: #fff;
        border: 1px solid var(--navy);
        background: linear-gradient(105deg, var(--navy), #176253);
        box-shadow: 0 7px 17px rgba(16, 43, 63, .16);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 22px rgba(16, 43, 63, .15);
    }
    div[data-testid="stDataFrame"] {
        overflow: hidden;
        border: 1px solid var(--line);
        border-radius: 12px;
        box-shadow: 0 5px 18px rgba(16, 43, 63, .04);
    }
    .signal-card {
        background: linear-gradient(110deg, #fff5f1, #fffaf7);
        border: 1px solid #f0d7ce;
        border-left: 3px solid #d9775d;
        border-radius: 10px;
        padding: .85rem 1rem;
        margin: .55rem 0;
    }
    .signal-card strong { color: #a64c3c; }
    .signal-meta, .policy-source {
        color: var(--teal);
        font-size: .66rem;
        letter-spacing: .1em;
        text-transform: uppercase;
        margin-top: .5rem;
    }
    .policy-card {
        background: linear-gradient(145deg, #fff 0%, #f7fbf9 100%);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 1rem;
        margin: .65rem 0;
        box-shadow: 0 5px 16px rgba(16, 43, 63, .035);
        transition: transform .2s ease, box-shadow .2s ease;
    }
    .policy-card:hover { transform: translateY(-2px); box-shadow: var(--shadow); }
    .policy-card strong { color: var(--navy); }
    .policy-card p { color: #425968; margin: .7rem 0; line-height: 1.65; }
    .recommendation-card {
        background: linear-gradient(145deg, #fff 0%, #f7fbf9 100%);
        border: 1px solid var(--line);
        border-left: 4px solid var(--teal);
        border-radius: 12px;
        padding: 1rem 1.15rem;
        margin: .8rem 0;
        box-shadow: 0 5px 16px rgba(16, 43, 63, .04);
    }
    .recommendation-card.priority-high { border-left-color: #bd5548; }
    .recommendation-card.priority-medium { border-left-color: #c28a2d; }
    .recommendation-meta {
        display: flex;
        justify-content: space-between;
        gap: .75rem;
        color: var(--muted);
        font-size: .68rem;
        letter-spacing: .08em;
        text-transform: uppercase;
    }
    .recommendation-priority { color: var(--teal); font-weight: 750; }
    .priority-high .recommendation-priority { color: #a63e35; }
    .priority-medium .recommendation-priority { color: #9a671b; }
    .recommendation-card h4 { color: var(--navy); margin: .65rem 0; }
    .recommendation-card p { color: #425968; line-height: 1.6; margin: .45rem 0; }
    .recommendation-guidance {
        padding-top: .55rem;
        border-top: 1px solid var(--line);
        font-size: .85rem;
    }
    h2, h3 { color: var(--navy); letter-spacing: -.025em; }
    [data-testid="stAlert"] { border-radius: 10px; }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: .01ms !important;
            animation-iteration-count: 1 !important;
            scroll-behavior: auto !important;
            transition-duration: .01ms !important;
        }
    }
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


def employee_value(employee, aliases):
    """Return a display-safe value for the first available dataset column."""
    normalized_aliases = {
        re.sub(r"[^a-z0-9]+", "", alias.lower())
        for alias in aliases
    }
    for column in employee.index:
        normalized_column = re.sub(r"[^a-z0-9]+", "", str(column).lower())
        if normalized_column in normalized_aliases:
            value = employee[column]
            return "Not available" if pd.isna(value) else str(value)
    return "Not available"


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


st.markdown(
    '<section class="portal-hero">'
    '<div class="portal-hero-top">'
    '<div class="workspace-kicker">People analytics / Workforce intelligence</div>'
    '<div class="portal-badge">Employee insights</div>'
    '</div>'
    '<h1 class="workspace-title">See the signal<br>behind attrition.</h1>'
    '<p class="workspace-copy">Turn employee data into a clearer conversation about who may need support, why, and what HR could consider next.</p>'
    '</section>',
    unsafe_allow_html=True,
)

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
        persisted_attributes = {
            name for name in persisted_feature_names
            if not is_employee_identity_column(name)
        }
        compatible_schema = (
            metadata.get("employee_column") == employee_column
            and metadata.get("attrition_column") == attrition_column
            and persisted_attributes == set(features.columns)
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

model_metadata = training_result["metadata"]
if (
    model_metadata.get("employee_column") != employee_column
    or model_metadata.get("attrition_column") != attrition_column
):
    st.error("The loaded model was trained for a different employee schema. Refresh the model before analyzing.")
    st.stop()

persisted_features = model_metadata.get("features", {})
model_feature_names = (
    persisted_features.get("numerical_features", [])
    + persisted_features.get("categorical_features", [])
)
if not model_feature_names:
    model_feature_names = list(features.columns)
try:
    model_features = align_features_to_model(
        features,
        model_feature_names,
        numerical_features=persisted_features.get("numerical_features", []),
    )
except ValueError as error:
    st.error(f"The loaded model cannot analyze this dataset: {error}")
    st.stop()

metrics = training_result["metadata"]["metrics"]
metric_columns = st.columns(5)
for column, (name, value) in zip(metric_columns, metrics.items()):
    column.metric(name.replace("_", " ").title(), f"{value:.1%}")

predictions = predict_attrition(
    model_features,
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
        employee_record = dataset.loc[selected_index].drop(labels=[attrition_column])
        employee_probability = float(employee_result["attrition_probability"])

        st.markdown(f"### Employee: {html.escape(str(selected_employee))}")
        st.markdown("#### Employee profile")
        identifier_label = (
            "Employee name"
            if "name" in re.sub(r"[^a-z0-9]+", "", str(employee_column).lower())
            else "Employee ID"
        )
        profile_fields = [
            (identifier_label, [employee_column]),
            ("Department", ["Department"]),
            ("Job role", ["JobRole", "Job Role"]),
            ("Performance rating", ["PerformanceRating", "Performance Rating", "Performance"]),
            ("Job satisfaction", ["JobSatisfaction", "Job Satisfaction"]),
        ]
        profile_metrics = st.columns(len(profile_fields))
        for column, (label, aliases) in zip(profile_metrics, profile_fields):
            column.metric(label, employee_value(employee_record, aliases))

        st.markdown("#### Attrition analysis")
        analysis_metrics = st.columns(3)
        analysis_metrics[0].metric("Predicted attrition probability", f"{employee_probability:.1%}")
        analysis_metrics[1].metric("Risk category", employee_result["risk_level"])
        analysis_metrics[2].metric(
            "Historical outcome",
            employee_value(dataset.loc[selected_index], [attrition_column]),
        )

        try:
            local_explanation = explain_instance(
                lambda rows: predict_attrition(
                    rows,
                    model=training_result["model"],
                    preprocessor=training_result["preprocessor"],
                )["probability"],
                model_features,
                model_features.loc[selected_index],
                top_features=8,
            )
        except Exception as error:
            local_explanation = {"top_features": []}
            st.warning(
                "Local LIME sensitivity could not be calculated for this record. "
                f"Profile-based recommendations remain available. Details: {error}"
            )

        plan = generate_personalized_retention_plan(
            employee_record,
            dataset,
            employee_probability,
            local_explanation=local_explanation,
            knowledge_documents=knowledge_store.documents,
        )

        st.markdown("#### Potential reasons behind this risk estimate")
        st.caption(
            "These are observed profile signals that may warrant a conversation, alongside model sensitivity. "
            "They are not confirmed reasons the employee may leave or proof of cause."
        )
        if plan["signals"]:
            for signal in plan["signals"][:5]:
                reason = signal["reason"]
                if signal.get("model_feature"):
                    reason += (
                        f" The model also ranked {signal['model_feature']} as influential "
                        f"for this record (local sensitivity {signal['model_sensitivity']:.3f})."
                    )
                st.markdown(
                    f"- **{html.escape(signal['label'])}:** {html.escape(reason)}"
                )
        else:
            st.info(
                plan["insufficient_information"]
                or "No specific employee-profile signals were found; HR should ask the employee what support would help."
            )

        drivers_column, record_column = st.columns([1, 1])
        with drivers_column:
            st.markdown("#### How the model responds to these features")
            st.caption(
                "A LIME-style local surrogate estimates associations near this record. "
                "Numeric direction describes how higher values relate to predicted risk locally. "
                "Category direction averages sampled alternatives to the recorded category."
            )
            effects_by_feature = {
                effect["feature"]: effect
                for effect in local_explanation.get("feature_effects", [])
            }
            displayed_drivers = 0
            for feature, sensitivity in local_explanation.get("top_features", []):
                if is_employee_identity_column(feature):
                    continue
                feature_value = employee_value(employee_record, [feature])
                effect = effects_by_feature.get(feature)
                direction = "Direction unavailable."
                if effect and effect["kind"] == "numeric":
                    direction = (
                        "Higher values are associated with higher predicted risk locally."
                        if effect["coefficient"] > 0
                        else "Higher values are associated with lower predicted risk locally."
                    )
                elif effect:
                    direction = (
                        "Changing from the recorded category tends to increase predicted risk locally."
                        if effect["coefficient"] > 0
                        else "Changing from the recorded category tends to decrease predicted risk locally."
                    )
                st.markdown(
                    f"**{html.escape(str(feature))}:** {html.escape(feature_value)}  \n"
                    f"{direction} Local sensitivity magnitude: {float(sensitivity):.3f}.",
                    unsafe_allow_html=True,
                )
                displayed_drivers += 1
            if not displayed_drivers:
                st.info("No reliable local feature sensitivity was available for display.")

        with record_column:
            st.markdown("#### Complete employee record")
            st.dataframe(
                employee_record.astype(str).to_frame("Value"),
                use_container_width=True,
            )

        st.markdown("#### Personalized retention strategies")
        st.caption(
            "Recommendations use observed profile values and local model sensitivity. "
            "HR review is required; these are not employment decisions."
        )
        if plan["insufficient_information"]:
            st.info(plan["insufficient_information"])
        else:
            for recommendation in plan["recommendations"]:
                priority_class = recommendation["priority"].lower()
                card = (
                    f'<article class="recommendation-card priority-{priority_class}">'
                    '<div class="recommendation-meta">'
                    f'<span class="recommendation-priority">{html.escape(recommendation["priority"])} priority</span>'
                    f'<span>Follow up: {html.escape(recommendation["timeline"])}</span>'
                    '</div>'
                    f'<h4>{html.escape(recommendation["strategy_title"])}</h4>'
                    f'<p><strong>Why this applies:</strong> {html.escape(recommendation["reason"])}</p>'
                    f'<p><strong>Recommended action:</strong> {html.escape(recommendation["action"])}</p>'
                )
                if recommendation["policy_guidance"]:
                    card += (
                        '<p class="recommendation-guidance"><strong>Related uploaded HR guidance:</strong> '
                        f'{html.escape(recommendation["policy_guidance"])}'
                        f' <em>Source: {html.escape(str(recommendation["policy_source"]))}</em></p>'
                    )
                card += "</article>"
                st.markdown(card, unsafe_allow_html=True)

        with st.expander("Optional general web context"):
            st.caption(
                "This search is general and does not include employee identifiers or profile data."
            )
            if st.button(
                "Look up general retention guidance",
                key=f"general_web_strategy_{selected_employee}",
            ):
                query = "evidence-based employee retention strategies"
                web_result = fetch_web_strategy(query)
                st.info(web_result["summary"])
                if web_result["source"] != "offline":
                    search_url = "https://duckduckgo.com/?q=" + quote(query)
                    st.link_button("Open full web search", search_url, use_container_width=True)

st.subheader("Employee risk ranking")
st.dataframe(
    results.style.format({"attrition_probability": "{:.1%}"}),
    use_container_width=True,
)
