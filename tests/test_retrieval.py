import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest import load_embedding_model, get_chroma_client, get_collection
from retrieval import retrieve

# These tests require the sample docs to be ingested (run the smoke-test first).
# They hit the real ChromaDB store — no mocking.

model = load_embedding_model()
client = get_chroma_client()
collection = get_collection(client)


def test_retrieval_returns_results():
    results = retrieve("What is machine learning?", model, collection)
    assert len(results) > 0


def test_top_result_is_relevant_to_query():
    results = retrieve("What is the Paris Agreement?", model, collection)
    assert len(results) > 0
    top = results[0]
    # The most relevant chunk should come from the climate document
    assert top["source"] == "sample_climate.txt"


def test_space_query_hits_space_doc():
    results = retrieve("Who was the first human in space?", model, collection)
    assert results[0]["source"] == "sample_space.txt"


def test_result_has_expected_keys():
    results = retrieve("What are embeddings?", model, collection)
    assert len(results) > 0
    keys = results[0].keys()
    assert {"text", "source", "chunk_index", "distance"} <= set(keys)


def test_empty_collection_returns_empty_list(tmp_path):
    import chromadb
    empty_client = chromadb.PersistentClient(path=str(tmp_path / "empty_store"))
    empty_col = empty_client.get_or_create_collection("empty")
    results = retrieve("anything", model, empty_col)
    assert results == []
