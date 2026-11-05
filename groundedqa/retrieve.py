# What this file does: finds the passages most like the question. The built-in retriever
# is BM25, a classic word-overlap score that needs no model, no key and no network, so the
# demo and the tests run on any machine. Anything else (an embedding model, a vector
# store) plugs in by giving the same two methods: index(chunks) and search(question, k).
"""retrieve.py: the Retriever protocol and the BM25 retriever."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol

from .chunk import Chunk

TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = set(
    "a an and are as at be by for from how in is it of on or that the this to was what when where which who why will with do does did can could should would there their they them its into if then than our your you we i not no".split()
)


# In plain words: a light stemmer so "runs" meets "run" and "donated" meets "donate":
# common English endings are trimmed from longer words. Crude by design and the same on
# both the question and the passages, which is all that matters for matching.
def stem(word: str) -> str:
    if len(word) <= 3:
        return word
    for ending in ("ing", "ies", "ed", "es", "s"):
        if word.endswith(ending) and not word.endswith("ss"):
            base = word[: -len(ending)]
            if ending == "ies":
                base += "y"
            return base if len(base) >= 3 else word
    return word


def tokens(text: str) -> list[str]:
    return [stem(t) for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


class Retriever(Protocol):
    def index(self, chunks: list[Chunk]) -> None: ...
    def search(self, question: str, k: int = 3) -> list[tuple[Chunk, float]]: ...


class BM25:
    """Okapi BM25 over the chunks' words; scores are 0 when no question word appears."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.chunks: list[Chunk] = []
        self.doc_tokens: list[list[str]] = []
        self.df: Counter = Counter()
        self.avg_len = 0.0

    def index(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)
        self.doc_tokens = [tokens(c.text) for c in self.chunks]
        self.df = Counter()
        for toks in self.doc_tokens:
            self.df.update(set(toks))
        n = len(self.doc_tokens)
        self.avg_len = (sum(len(t) for t in self.doc_tokens) / n) if n else 0.0

    def _idf(self, term: str) -> float:
        n, df = len(self.doc_tokens), self.df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def score(self, question_tokens: list[str], i: int) -> float:
        toks = self.doc_tokens[i]
        if not toks:
            return 0.0
        counts = Counter(toks)
        total = 0.0
        for term in question_tokens:
            tf = counts.get(term, 0)
            if not tf:
                continue
            denom = tf + self.k1 * (1 - self.b + self.b * len(toks) / self.avg_len)
            total += self._idf(term) * tf * (self.k1 + 1) / denom
        return total

    def search(self, question: str, k: int = 3) -> list[tuple[Chunk, float]]:
        q = tokens(question)
        if not q or not self.chunks:
            return []
        scored = [(c, self.score(q, i)) for i, c in enumerate(self.chunks)]
        scored = [(c, s) for c, s in scored if s > 0]
        scored.sort(key=lambda cs: (-cs[1], cs[0].page, cs[0].offset))
        return scored[:k]
