import io
import json
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st

from config import MODEL_METADATA_FILE
from explainability.shap_explainer import explain_with_shap
from ml.predict import load_artifacts, predict_attrition
from ml.preprocessing import clean_dataset, prepare_features_and_target, validate_dataset
from ml.train import train_model
from rag.document_loader import load_documents_from_files
from rag.generator import generate_retention_recommendations, summarize_recommendation_context
from rag.vector_store import VectorStore


st.set_page_config(page_title="Employee Attrition", page_icon="📊", layout="wide")


@st.cache_data
def read_employee_file(file_bytes, file_name):
    """Read CSV or Excel uploads into a dataframe."""
    if file_name.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes))
    return pd.read_excel(io.BytesIO(file_bytes))


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


st.title("Employee Attrition Intelligence")
st.caption("Train a local model, identify workforce risk, and review employees requiring attention.")

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
    st.info("Upload a CSV or Excel file containing an employee identifier and an attrition target.")
    st.stop()

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
        if metadata.get("ready"):
            model, preprocessor, persisted_metadata = load_artifacts()
            training_result = {
                "model": model,
                "preprocessor": preprocessor,
                "metadata": persisted_metadata,
            }
            st.sidebar.success("Model READY: a compatible saved model was detected.")
    except Exception:
        training_result = None

if st.sidebar.button("Train / refresh model", type="primary", use_container_width=True):
    try:
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

features, _, employee_column, attrition_column = prepare_features_and_target(dataset)
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

            recommended_docs = knowledge_store.search(
                str(dataset.loc[selected_index, :].to_dict()),
                top_k=2,
            ) if knowledge_store.documents else []
            risk_factors = ["OverTime", "JobSatisfaction"]

            if st.button("Generate strategy", key=f"generate_strategy_{selected_employee}", type="primary"):
                if strategy_source == "HR document":
                    summary = summarize_recommendation_context(
                        employee_result,
                        employee_probability,
                        knowledge_documents=recommended_docs,
                        risk_factors=risk_factors,
                    )
                    st.write(summary)
                    if recommended_docs:
                        st.caption("Relevant document guidance:")
                        for doc in recommended_docs:
                            st.write(f"- {doc.get('source_name', 'document')}: {doc.get('text', '')[:240]}")
                    for recommendation in generate_retention_recommendations(
                        employee_result,
                        employee_probability,
                        knowledge_documents=recommended_docs,
                        risk_factors=risk_factors,
                    ):
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

        employee_X = features.loc[[selected_index]].copy()
        employee_row = features.loc[selected_index]
        explanation = explain_with_shap(
            training_result["model"],
            employee_X,
            employee_row,
            preprocessor=training_result["preprocessor"],
        )
        st.markdown("#### SHAP-style feature contributions")
        st.json(explanation["local_contributions"][:5])

st.subheader("Employee risk ranking")
st.dataframe(
    results.style.format({"attrition_probability": "{:.1%}"}),
    use_container_width=True,
)
