# What this file does: tests reading documents (text, markdown, a two-page PDF written
# by hand, a scanned PDF refused with a reason, an empty or oversized file refused), the
# in-memory store, the FastAPI round trip (upload → ask → answer with receipts; bad
# document → a 422 with the reason; unknown id → 404) and the command line.
"""ingest.py, store.py, api.py, __main__.py."""

import json
import subprocess
import sys

import pytest
from conftest import ROOT, make_pdf, make_scanned_pdf
from fastapi.testclient import TestClient

from groundedqa import ingest
from groundedqa.api import app, store
from groundedqa.store import Store


def test_text_and_markdown_are_one_page_each(handbook):
    assert len(ingest.pages_from_bytes(b"plain text", "notes.txt")) == 1
    assert len(ingest.pages_from_bytes(handbook, "handbook.md")) == 1


def test_a_two_page_pdf_reads_page_by_page_and_answers_cite_the_page():
    pdf = make_pdf(["The rinse tanks are emptied every evening by the last person out.",
                    "Lost gear is kept for thirty days and then donated."])
    pages = ingest.pages_from_bytes(pdf, "handbook.pdf")
    assert len(pages) == 2 and "rinse" in pages[0] and "donated" in pages[1]
    s = Store()
    doc = s.add(pdf, "handbook.pdf")
    out = s.ask(doc.id, "what happens to lost gear after thirty days?")
    assert not out.refused and out.receipts[0].page == 2


def test_a_scanned_pdf_is_refused_with_a_reason_never_a_blank_answer():
    with pytest.raises(ingest.RefusedDocument, match="no text layer"):
        ingest.pages_from_bytes(make_scanned_pdf(), "scan.pdf")


def test_empty_binary_and_oversized_files_are_refused():
    with pytest.raises(ingest.RefusedDocument, match="empty"):
        ingest.pages_from_bytes(b"", "x.txt")
    with pytest.raises(ingest.RefusedDocument, match="not UTF-8"):
        ingest.pages_from_bytes(b"\xff\xfe\x00binary", "x.bin")
    with pytest.raises(ingest.RefusedDocument, match="five megabytes"):
        ingest.pages_from_bytes(b"a" * (ingest.MAX_BYTES + 1), "big.txt")
    with pytest.raises(ingest.RefusedDocument, match="no text"):
        ingest.pages_from_bytes(b"   \n  ", "blank.txt")


def test_the_store_keeps_documents_apart(handbook):
    s = Store()
    a = s.add(handbook, "handbook.md")
    b = s.add(b"The moon is made of green cheese, says the old song.", "song.txt")
    assert not s.ask(a.id, "what happens to lost gear?").refused
    assert s.ask(b.id, "what happens to lost gear?").refused
    assert "cheese" in s.ask(b.id, "what is the moon made of?").answer


def test_the_api_round_trip_upload_ask_receipts(handbook):
    client = TestClient(app)
    store.docs.clear()
    r = client.post("/documents", files={"file": ("handbook.md", handbook, "text/markdown")})
    assert r.status_code == 200, r.text
    doc_id = r.json()["id"]
    assert r.json()["pages"] == 1 and r.json()["chunks"] >= 30
    assert client.get("/documents").json()[0]["id"] == doc_id
    r = client.post(f"/documents/{doc_id}/ask", json={"question": "where is the emergency oxygen kit kept?"})
    body = r.json()
    assert r.status_code == 200 and not body["refused"]
    assert "orange case" in body["answer"]
    assert body["receipts"][0]["id"].startswith("p1-c") and body["answer"] in body["receipts"][0]["text"]
    r = client.post(f"/documents/{doc_id}/ask", json={"question": "what is the capital of France?"})
    assert r.json()["refused"] and r.json()["answer"] == "not in the document"


def test_the_api_refuses_an_oversized_upload_before_reading_it_and_clamps_k(handbook):
    from groundedqa.ingest import MAX_BYTES
    client = TestClient(app)
    r = client.post("/documents", files={"file": ("big.txt", b"a" * (MAX_BYTES + 1), "text/plain")})
    assert r.status_code == 422 and "five megabytes" in r.json()["detail"]
    r = client.post("/documents", files={"file": ("handbook.md", handbook, "text/markdown")})
    doc_id = r.json()["id"]
    r = client.post(f"/documents/{doc_id}/ask", json={"question": "where is the emergency oxygen kit kept?", "k": 999})
    assert r.status_code == 200 and len(r.json()["receipts"]) <= 10


def test_the_api_refuses_bad_documents_and_unknown_ids():
    client = TestClient(app)
    r = client.post("/documents", files={"file": ("scan.pdf", make_scanned_pdf(), "application/pdf")})
    assert r.status_code == 422 and "no text layer" in r.json()["detail"]
    r = client.post("/documents/nope/ask", json={"question": "anything at all?"})
    assert r.status_code == 404


def test_the_command_line_answers_with_receipts_and_refuses_bad_files(tmp_path):
    run = subprocess.run([sys.executable, "-m", "groundedqa", "ask", f"{ROOT}/data/handbook.md", "what happens to lost gear after thirty days?"],
                         capture_output=True, text=True, cwd=ROOT)
    assert run.returncode == 0, run.stderr
    out = json.loads(run.stdout)
    assert "donated" in out["answer"] and out["receipts"]
    bad = tmp_path / "scan.pdf"
    bad.write_bytes(make_scanned_pdf())
    run = subprocess.run([sys.executable, "-m", "groundedqa", "ask", str(bad), "anything?"], capture_output=True, text=True, cwd=ROOT)
    assert run.returncode == 1 and json.loads(run.stdout)["refused"]
    run = subprocess.run([sys.executable, "-m", "groundedqa"], capture_output=True, text=True, cwd=ROOT)
    assert run.returncode == 2 and "usage" in run.stderr


def test_the_corpus_is_made_up_and_names_nobody(handbook):
    text = handbook.decode()
    assert "made up" in text.lower()
    assert "@" not in text and "+27" not in text and "/Users/" not in text
    assert "Grey Seal Dive Centre, Nowhere Bay" in text  # an obviously invented name and place
