# What this file does: reads an uploaded document into pages of plain text. Text and
# markdown files are one page each (markdown headings still start new chunks). A PDF is
# read page by page with pypdf; a PDF with no text layer (a scan) is refused with a
# reason, never turned into a blank answer. Nothing here fetches anything.
"""ingest.py: bytes -> pages of text, or a RefusedDocument error."""

from __future__ import annotations

MAX_BYTES = 5 * 1024 * 1024


class RefusedDocument(ValueError):
    """The document cannot be used; the message says why."""


def pages_from_bytes(data: bytes, name: str) -> list[str]:
    if not data:
        raise RefusedDocument("the file is empty")
    if len(data) > MAX_BYTES:
        raise RefusedDocument("the file is larger than five megabytes")
    lower = name.lower()
    if lower.endswith(".pdf") or data[:5] == b"%PDF-":
        return _pdf_pages(data)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RefusedDocument("the file is not UTF-8 text, markdown or a PDF") from exc
    if not text.strip():
        raise RefusedDocument("the file has no text")
    return [text]


def _pdf_pages(data: bytes) -> list[str]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise RefusedDocument("PDF reading needs the pypdf package") from exc
    import io

    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [(p.extract_text() or "") for p in reader.pages]
    except Exception as exc:  # noqa: BLE001  (pypdf raises many kinds; all mean "unreadable")
        raise RefusedDocument(f"the PDF could not be read ({type(exc).__name__})") from exc
    if not any(p.strip() for p in pages):
        raise RefusedDocument("the PDF has no text layer (a scan?); OCR it first")
    return pages
