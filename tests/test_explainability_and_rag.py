from pathlib import Path

import pandas as pd

from explainability.lime_explainer import explain_instance
from rag.document_loader import load_documents_from_files
from rag.vector_store import VectorStore


def test_lime_explainer_returns_features():
    data = pd.DataFrame(
        {
            "MonthlyIncome": [2500, 3500, 4200, 4600],
            "YearsAtCompany": [1, 2, 4, 5],
            "JobSatisfaction": [1, 3, 3, 4],
            "OverTime": ["Yes", "No", "No", "No"],
        }
    )
    model = lambda values: [0.05, 0.2, 0.8, 0.7]

    explanation = explain_instance(model, data, data.iloc[0], top_features=3)

    assert explanation["top_features"]
    assert all(len(item) == 2 for item in explanation["top_features"])


def test_document_loader_reads_text_file(tmp_path):
    file_path = tmp_path / "policy.txt"
    file_path.write_text("Work-life balance and mentorship are encouraged.", encoding="utf-8")

    documents = load_documents_from_files([file_path])

    assert documents[0]["text"]
    assert documents[0]["source_name"] == "policy.txt"


def test_vector_store_retrieves_relevant_documents():
    store = VectorStore()
    store.add_documents([
        {"text": "Flexible scheduling and leave policy support retention.", "source_name": "policy_1"},
        {"text": "Performance reviews are conducted quarterly.", "source_name": "policy_2"},
    ])

    results = store.search("flexible scheduling leave", top_k=1)

    assert results
    assert results[0]["source_name"] == "policy_1"
