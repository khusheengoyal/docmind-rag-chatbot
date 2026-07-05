import json
import os
import random
import re

from dotenv import load_dotenv
from groq import Groq

from retrieval import retrieve

load_dotenv()

MODEL = "openai/gpt-oss-20b"
SAMPLE_SIZE = 8  # chunks fetched when no topic is provided

SYSTEM_PROMPT = (
    "You are a study-aid generator. Create concise flashcard question/answer pairs "
    "based ONLY on the provided document chunks. Do not use any outside knowledge. "
    "If the content is insufficient for the requested number of cards, generate fewer "
    "pairs rather than inventing information."
)


def sample_chunks(collection, n: int = SAMPLE_SIZE) -> list[dict]:
    """Return a random sample of chunks from the full collection for broad coverage."""
    result = collection.get(include=["documents", "metadatas"])
    if not result["documents"]:
        return []
    pairs = list(zip(result["documents"], result["metadatas"]))
    sampled = random.sample(pairs, min(n, len(pairs)))
    return [
        {"text": doc, "source": meta["source"], "chunk_index": meta["chunk_index"]}
        for doc, meta in sampled
    ]


def _format_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[Chunk {i} — {c['source']}]\n{c['text']}" for i, c in enumerate(chunks, 1)
    )


def _parse_cards(raw: str) -> list[dict]:
    """
    Strip markdown fences then JSON-parse the response.
    Returns a validated list of {question, answer} dicts, or [] on any failure.
    We do this in a try/except because the model may add preamble text, wrap the
    JSON in ```json fences, or produce subtly malformed output — none of which
    should crash the app.
    """
    # Remove ```json ... ``` or ``` ... ``` fences
    text = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    # gpt-oss-20b may emit reasoning before the final JSON; find the first '['
    start = text.find("[")
    if start != -1:
        text = text[start:]

    try:
        cards = json.loads(text)
        if isinstance(cards, list):
            return [
                c for c in cards
                if isinstance(c, dict) and "question" in c and "answer" in c
            ]
    except json.JSONDecodeError:
        pass
    return []


def generate_flashcards(chunks: list[dict], n: int) -> list[dict]:
    """
    Make one Groq call and return up to n parsed flashcard dicts.
    Raises ValueError if the API key is missing.
    Returns [] if parsing fails (caller shows a friendly retry message).
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")

    context = _format_context(chunks)
    user_msg = (
        f"Document chunks:\n\n{context}\n\n"
        f"Generate exactly {n} flashcard question/answer pairs using only the chunks above.\n"
        f"Return ONLY a JSON array — no preamble, no markdown fences, nothing else:\n"
        f'[{{"question": "...", "answer": "..."}}, ...]'
    )

    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=3000,
        stream=False,
    )

    raw = resp.choices[0].message.content or ""
    return _parse_cards(raw)
