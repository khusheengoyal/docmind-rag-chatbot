import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest import chunk_text, CHUNK_SIZE, CHUNK_OVERLAP


def test_single_chunk_for_short_text():
    text = "word " * 100  # 100 words — well under CHUNK_SIZE
    chunks = chunk_text(text.strip(), source="test.txt")
    assert len(chunks) == 1
    assert chunks[0]["source"] == "test.txt"
    assert chunks[0]["chunk_index"] == 0


def test_multiple_chunks_for_long_text():
    text = " ".join(f"word{i}" for i in range(CHUNK_SIZE * 3))
    chunks = chunk_text(text, source="big.txt")
    assert len(chunks) > 1


def test_overlap_is_present():
    # Build text with uniquely labelled words so we can find them
    words = [f"w{i}" for i in range(CHUNK_SIZE + 10)]
    text = " ".join(words)
    chunks = chunk_text(text, source="overlap.txt")

    tail_of_first = chunks[0]["text"].split()[-(CHUNK_OVERLAP):]
    head_of_second = chunks[1]["text"].split()[:CHUNK_OVERLAP]
    assert tail_of_first == head_of_second, "Overlap words must repeat across chunk boundary"


def test_chunk_index_is_sequential():
    text = " ".join(f"w{i}" for i in range(CHUNK_SIZE * 2))
    chunks = chunk_text(text, source="seq.txt")
    indices = [c["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))


def test_empty_file_returns_no_chunks():
    chunks = chunk_text("", source="empty.txt")
    assert chunks == []
