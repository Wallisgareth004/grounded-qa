# What this file does: tests chunking (deterministic, keeps page and offset, a heading
# starts a new chunk) and the BM25 retriever (a known question finds its known passage
# first; a question with no words in common scores nothing; stopwords are ignored).
"""chunk.py and retrieve.py."""

from groundedqa.chunk import chunk_page, chunk_pages
from groundedqa.retrieve import BM25, tokens


def test_chunking_is_deterministic_and_keeps_page_and_offset(handbook):
    text = handbook.decode()
    a = chunk_pages([text])
    b = chunk_pages([text])
    assert [c.id for c in a] == [c.id for c in b]
    assert all(c.page == 1 for c in a)
    for c in a:
        assert text[c.offset:c.offset + len(c.text)].strip().startswith(c.text[:20])
    assert len(a) >= 30  # one chunk per section at least


def test_a_heading_starts_a_new_chunk_so_sections_never_straddle():
    text = "## Section 1: A\n\nFirst body.\n\n## Section 2: B\n\nSecond body.\n"
    chunks = chunk_page(text, 1, size=10_000)
    assert len(chunks) == 2
    assert chunks[0].text.startswith("## Section 1") and "Second" not in chunks[0].text
    assert chunks[1].id == "p1-c2" and chunks[1].offset == text.index("## Section 2")


def test_long_pages_split_by_size_and_multiple_pages_number_from_one():
    long = "\n\n".join(f"Paragraph number {i} talks about rinsing fins." for i in range(40))
    chunks = chunk_pages([long, "Second page text."], size=300)
    assert all(len(c.text) <= 400 for c in chunks)
    assert chunks[-1].page == 2 and chunks[-1].id == "p2-c1"


def test_tokens_drop_stopwords_lowercase_and_trim_common_endings():
    assert tokens("What is the Rule for LOST gear?") == ["rule", "lost", "gear"]
    assert tokens("runs refuelled donated bodies glass") == ["run", "refuell", "donat", "body", "glass"]
    assert tokens("who runs the compressor?") == ["run", "compressor"]


def test_a_known_question_finds_its_known_passage_first(handbook):
    bm = BM25()
    bm.index(chunk_pages([handbook.decode()]))
    hits = bm.search("what happens to lost gear after thirty days?")
    assert hits and "Lost gear" in hits[0][0].text and "donated" in hits[0][0].text
    hits = bm.search("where is the emergency oxygen kit kept?")
    assert "orange case" in hits[0][0].text


def test_no_shared_word_means_no_hit(handbook):
    bm = BM25()
    bm.index(chunk_pages([handbook.decode()]))
    assert bm.search("quantum chromodynamics lattice") == []
    assert bm.search("the of and") == []
    assert BM25().search("anything") == []
