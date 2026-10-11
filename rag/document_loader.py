from __future__ import annotations

from pathlib import Path
from typing import Iterable


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def _read_pdf_file(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages).strip()


def _read_docx_file(path: Path) -> str:
    try:
        import docx
    except ImportError:
        return ""

    document = docx.Document(str(path))
    paragraphs = [paragraph.text for paragraph in document.paragraphs]
    return "\n".join(paragraphs).strip()


def extract_text_from_file(file_path: str | Path) -> str:
    path = Path(file_path)
    if not path.exists():
        return ""

    suffix = path.suffix.lower()
    if suffix == ".txt":
        return _read_text_file(path)
    if suffix == ".pdf":
        return _read_pdf_file(path)
    if suffix in {".docx", ".doc"}:
        return _read_docx_file(path)
    if suffix == ".csv":
        return _read_text_file(path)
    return _read_text_file(path)


def load_documents_from_files(files: Iterable[str | Path]):
    """Load text documents from files and return a normalized list of document records."""
    documents = []
    for file_path in files:
        path = Path(file_path)
        if not path.exists():
            continue

        source_name = path.name
        text = extract_text_from_file(path)
        if text:
            documents.append({
                "source_name": source_name,
                "source_path": str(path),
                "text": text,
            })
    return documents
