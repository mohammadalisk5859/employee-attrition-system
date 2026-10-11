from __future__ import annotations

from typing import List

import numpy as np


class VectorStore:
    """A lightweight in-memory vector store for HR knowledge retrieval."""

    def __init__(self):
        self.documents = []
        self.embeddings = []

    def add_documents(self, documents):
        for document in documents:
            text = (document.get("text") or "").strip()
            if not text:
                continue
            self.documents.append(document)
            self.embeddings.append(self._simple_embedding(text))

    @staticmethod
    def _simple_embedding(text: str):
        tokens = text.lower().split()
        vector = np.zeros(16, dtype=float)
        for index, token in enumerate(tokens[:16]):
            vector[index] = sum(ord(char) for char in token) % 97
        return vector

    def search(self, query: str, top_k: int = 3):
        if not self.documents:
            return []

        query_vector = self._simple_embedding(query)
        similarities = []
        for index, document in enumerate(self.documents):
            embedding = self.embeddings[index]
            similarity = float(np.dot(query_vector, embedding) / (np.linalg.norm(query_vector) * np.linalg.norm(embedding) + 1e-8))
            similarities.append((similarity, document))

        similarities.sort(key=lambda item: item[0], reverse=True)
        return [document for _, document in similarities[:top_k]]
