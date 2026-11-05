# What this file does: shared pieces for the tests: the made-up handbook as text, a tiny
# hand-written PDF writer (two pages of real text, no library needed), and a "scanned"
# PDF with no text layer. Everything here is invented data.
"""Fixtures for grounded-qa's tests."""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

HANDBOOK = os.path.join(ROOT, "data", "handbook.md")


@pytest.fixture
def handbook() -> bytes:
    with open(HANDBOOK, "rb") as fh:
        return fh.read()


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


# In plain words: a minimal PDF with one line of text per page, written by hand so the
# tests need no PDF-writing library. pypdf reads it back as ordinary page text.
def make_pdf(pages: list[str]) -> bytes:
    objs: list[bytes] = []
    n_pages = len(pages)
    # 1 catalog, 2 pages, 3 font, then per page: page object and content stream
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n_pages))
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i, text in enumerate(pages):
        page_obj = 4 + 2 * i
        content_obj = page_obj + 1
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents {content_obj} 0 R >>".encode()
        )
        stream = f"BT /F1 12 Tf 72 720 Td ({_pdf_escape(text)}) Tj ET".encode()
        objs.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def make_scanned_pdf() -> bytes:
    """A one-page PDF with no text at all (what a scan looks like to a reader)."""
    return make_pdf([""])
