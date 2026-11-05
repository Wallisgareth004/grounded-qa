# What this file does: cuts a document into passages ("chunks") the retriever can score,
# and remembers where each one came from (page and character offset) so an answer can
# point back at it. Deterministic: the same text always gives the same chunks. A heading
# line in markdown starts a new chunk, so a passage never straddles two sections.
"""chunk.py: text -> list of Chunk."""

from __future__ import annotations

import re
from dataclasses import dataclass

HEADING = re.compile(r"^#{1,6}\s+")


@dataclass(frozen=True)
class Chunk:
    id: str  # "p3-c2": page 3, chunk 2
    page: int
    offset: int  # character offset of the chunk within its page
    text: str


# In plain words: splits one page's text into chunks of about `size` characters, cutting
# at paragraph breaks or headings, never in the middle of a sentence when it can help it.
def chunk_page(text: str, page: int, size: int = 600) -> list[Chunk]:
    chunks: list[Chunk] = []
    buf, buf_start, pos = [], 0, 0
    paragraphs = re.split(r"(\n\s*\n)", text)

    def flush() -> None:
        nonlocal buf, buf_start
        body = "".join(buf).strip()
        if body:
            chunks.append(Chunk(f"p{page}-c{len(chunks) + 1}", page, buf_start, body))
        buf, buf_start = [], pos

    for part in paragraphs:
        if HEADING.match(part.lstrip()) and buf:
            flush()
        if sum(len(b) for b in buf) + len(part) > size and buf:
            flush()
        if not buf:
            buf_start = pos
        buf.append(part)
        pos += len(part)
    flush()
    return chunks


def chunk_pages(pages: list[str], size: int = 600) -> list[Chunk]:
    out: list[Chunk] = []
    for n, text in enumerate(pages, start=1):
        out += chunk_page(text, n, size)
    return out
