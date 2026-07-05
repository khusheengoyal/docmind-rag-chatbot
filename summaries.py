import json
import os
import re

from dotenv import load_dotenv
from groq import Groq

from flashcards import sample_chunks  # reuse existing chunk sampler

load_dotenv()

MODEL = "openai/gpt-oss-20b"

_SUMMARY_SYSTEM = (
    "You are a document analyst. Write a concise, grounded overview of the provided "
    "content using ONLY the information in the chunks. Do not use outside knowledge. "
    "If content is sparse, summarize what's there rather than padding."
)

_QUESTIONS_SYSTEM = (
    "You are a study-aid generator. Generate specific, answerable questions based "
    "ONLY on the provided document chunks. Do not ask about topics not covered."
)


def _groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")
    return Groq(api_key=api_key)


def _format_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[Chunk {i} — {c['source']}]\n{c['text']}" for i, c in enumerate(chunks, 1)
    )


def generate_summary(chunks: list[dict]) -> str:
    """Return a grounded 4-6 sentence overview of the provided chunks."""
    context = _format_context(chunks)
    resp = _groq_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _SUMMARY_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Document chunks:\n\n{context}\n\n"
                    "Write a concise overview (4-6 sentences or up to 5 bullet points) "
                    "of what these documents cover, using only the content above."
                ),
            },
        ],
        temperature=0.2,
        max_tokens=2000,
        stream=False,
    )
    return (resp.choices[0].message.content or "").strip()


def _parse_questions(raw: str) -> list[str]:
    """Strip markdown fences, locate the JSON array, parse safely."""
    text = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    start = text.find("[")
    if start != -1:
        text = text[start:]
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [q for q in parsed if isinstance(q, str) and q.strip()]
    except json.JSONDecodeError:
        pass
    return []


def generate_questions(chunks: list[dict], n: int = 4) -> list[str]:
    """Return n grounded suggested questions the documents can actually answer."""
    context = _format_context(chunks)
    resp = _groq_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _QUESTIONS_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Document chunks:\n\n{context}\n\n"
                    f"Generate exactly {n} specific questions these documents can answer.\n"
                    f"Return ONLY a JSON array of strings — no preamble, no markdown fences:\n"
                    f'["question 1", "question 2", ...]'
                ),
            },
        ],
        temperature=0.4,
        max_tokens=2000,
        stream=False,
    )
    return _parse_questions(resp.choices[0].message.content or "")
