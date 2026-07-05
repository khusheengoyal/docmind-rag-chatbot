import io
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

CHROMA_DIR = str(Path(__file__).parent / "chroma_store")
COLLECTION_NAME = "docmind"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 400   # words
CHUNK_OVERLAP = 50  # words repeated at the start of the next chunk


def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)


def get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=CHROMA_DIR)


def get_collection(client: chromadb.PersistentClient):
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def extract_text(file_bytes: bytes, filename: str) -> tuple[str, str]:
    """Return (text, error). One is always empty."""
    suffix = Path(filename).suffix.lower()
    try:
        if suffix == ".txt":
            text = file_bytes.decode("utf-8", errors="replace")
        elif suffix == ".pdf":
            reader = PdfReader(io.BytesIO(file_bytes))
            pages = [page.extract_text() or "" for page in reader.pages]
            text = "\n".join(pages)
        else:
            return "", f"Unsupported type '{suffix}'. Upload PDF or TXT files only."

        text = text.strip()
        if not text:
            return "", f"'{filename}' is empty or contains no extractable text — skipped."
        return text, ""
    except Exception as exc:
        return "", f"Could not read '{filename}': {exc}"


def chunk_text(text: str, source: str) -> list[dict]:
    words = text.split()
    chunks, start, idx = [], 0, 0
    while start < len(words):
        chunk = " ".join(words[start : start + CHUNK_SIZE])
        chunks.append({"text": chunk, "source": source, "chunk_index": idx})
        idx += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def ingest_file(
    file_bytes: bytes,
    filename: str,
    model: SentenceTransformer,
    collection,
) -> tuple[int, str]:
    """
    Extract, chunk, embed, and store one file.
    Returns (chunks_added, error_message). chunks_added is 0 on skip or error.
    """
    existing = collection.get(where={"source": {"$eq": filename}}, limit=1)
    if existing["ids"]:
        return 0, f"'{filename}' is already in the knowledge base — skipped."

    text, error = extract_text(file_bytes, filename)
    if error:
        return 0, error

    chunks = chunk_text(text, filename)
    texts = [c["text"] for c in chunks]

    # Normalise so dot-product == cosine similarity (required by the cosine collection)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    collection.add(
        ids=[f"{filename}::chunk::{c['chunk_index']}" for c in chunks],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[
            {"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks
        ],
    )
    return len(chunks), ""


def list_ingested_files(collection) -> list[str]:
    result = collection.get(include=["metadatas"])
    if not result["metadatas"]:
        return []
    return sorted({m["source"] for m in result["metadatas"]})


def clear_collection(collection) -> int:
    """Delete every chunk. Returns the count removed."""
    all_ids = collection.get()["ids"]
    if all_ids:
        collection.delete(ids=all_ids)
    return len(all_ids)
