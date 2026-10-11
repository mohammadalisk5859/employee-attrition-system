from __future__ import annotations

import io
import tempfile
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


def _read_uploaded_file(uploaded_file) -> str:
    """Read a Streamlit UploadedFile without converting it to a filesystem path."""
    name_value = getattr(uploaded_file, "name", "")
    while not isinstance(name_value, (str, bytes)) and hasattr(name_value, "name"):
        name_value = name_value.name
    name = str(name_value, errors="ignore") if isinstance(name_value, bytes) else str(name_value)
    suffix = Path(name).suffix.lower()
    content = uploaded_file.getvalue()
    seen_objects = set()
    while not isinstance(content, (bytes, bytearray, memoryview)):
        object_id = id(content)
        if object_id in seen_objects:
            raise TypeError("Uploaded file did not provide byte content.")
        seen_objects.add(object_id)
        if hasattr(content, "getvalue"):
            content = content.getvalue()
        elif hasattr(content, "read"):
            content = content.read()
        else:
            raise TypeError("Uploaded file did not provide byte content.")
    content = bytes(content)

    if suffix in {".txt", ".csv"}:
        return content.decode("utf-8", errors="ignore").strip()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            return ""
        reader = PdfReader(io.BytesIO(content))
        return "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    if suffix in {".docx", ".doc"}:
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
                temporary_file.write(content)
                temporary_path = Path(temporary_file.name)
            return _read_docx_file(temporary_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    return content.decode("utf-8", errors="ignore").strip()


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
        if hasattr(file_path, "getvalue") and hasattr(file_path, "name"):
            text = _read_uploaded_file(file_path)
            if text:
                documents.append({
                    "source_name": str(file_path.name),
                    "source_path": str(file_path.name),
                    "text": text,
                })
            continue

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
