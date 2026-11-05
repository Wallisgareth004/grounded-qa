# What this file does: the small web API around the store: upload a document, ask it a
# question, get the answer with its receipts. FastAPI, three routes, JSON in and out. No
# key, no network call; the store is memory only (an upload over one megabyte passes
# through an OS temp file while the web framework parses it). At most fifty documents
# are held. Not for the open internet as it is: no login. `uvicorn groundedqa.api:app`.
"""api.py: POST /documents (upload), POST /documents/{id}/ask, GET /documents."""

from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from .ingest import MAX_BYTES, RefusedDocument
from .store import Store

app = FastAPI(title="grounded-qa", version="0.1.0")
store = Store()
MAX_DOCUMENTS = 50


class Question(BaseModel):
    question: str
    k: int = 3


@app.get("/documents")
def documents() -> list[dict]:
    return [{"id": d.id, "name": d.name, "pages": d.pages, "chunks": len(d.chunks)} for d in store.docs.values()]


@app.post("/documents")
async def upload(file: UploadFile = File(...)) -> dict:
    # In plain words: the size is checked before the file is read into memory, and the
    # read itself stops one byte past the limit.
    if file.size is not None and file.size > MAX_BYTES:
        raise HTTPException(status_code=422, detail="the file is larger than five megabytes")
    if len(store.docs) >= MAX_DOCUMENTS:
        raise HTTPException(status_code=422, detail="the store is full; restart the server")
    data = await file.read(MAX_BYTES + 1)
    try:
        doc = store.add(data, file.filename or "upload")
    except RefusedDocument as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"id": doc.id, "name": doc.name, "pages": doc.pages, "chunks": len(doc.chunks)}


@app.post("/documents/{doc_id}/ask")
def ask(doc_id: str, body: Question) -> dict:
    if doc_id not in store.docs:
        raise HTTPException(status_code=404, detail="no such document")
    return store.ask(doc_id, body.question, k=max(1, min(body.k, 10))).to_dict()
