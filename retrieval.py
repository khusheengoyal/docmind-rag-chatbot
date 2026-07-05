from sentence_transformers import SentenceTransformer

# BGE models require this prefix on queries (not on stored documents) for best retrieval quality
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

TOP_K = 4  # chunks returned per query


def retrieve(
    query: str,
    model: SentenceTransformer,
    collection,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Embed the query with the BGE prefix and return the top_k most relevant chunks.
    Each result dict has keys: 'text', 'source', 'chunk_index', 'distance'.
    Returns an empty list when the collection is empty.
    """
    if collection.count() == 0:
        return []

    prefixed = BGE_QUERY_PREFIX + query
    query_embedding = model.encode(prefixed, normalize_embeddings=True)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append(
            {
                "text": doc,
                "source": meta["source"],
                "chunk_index": meta["chunk_index"],
                "distance": dist,
            }
        )
    return chunks
