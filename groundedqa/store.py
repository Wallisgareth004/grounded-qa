# What this file does: holds the documents a session has uploaded, in memory, each with
# its chunks and its own retriever. Nothing is written to disk; restarting the server
# forgets everything, which is what a prototype on someone else's data should do.
"""store.py: Document and Store."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from .answer import Answer, answer
from .chunk import Chunk, chunk_pages
from .ingest import pages_from_bytes
from .retrieve import BM25


@dataclass
class Document:
    id: str
    name: str
    pages: int
    chunks: list[Chunk] = field(default_factory=list)
    retriever: BM25 = field(default_factory=BM25)


class Store:
    def __init__(self, chunk_size: int = 600) -> None:
        self.docs: dict[str, Document] = {}
        self.chunk_size = chunk_size

    def add(self, data: bytes, name: str) -> Document:
        pages = pages_from_bytes(data, name)
        chunks = chunk_pages(pages, self.chunk_size)
        doc = Document(uuid.uuid4().hex[:12], name, len(pages), chunks)
        doc.retriever.index(chunks)
        self.docs[doc.id] = doc
        return doc

    def ask(self, doc_id: str, question: str, k: int = 3, model=None) -> Answer:
        doc = self.docs[doc_id]
        return answer(question, doc.retriever, k=k, model=model)
