import os
from typing import Generator

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = "openai/gpt-oss-20b"
TEMPERATURE = 0.1
MAX_TOKENS = 1024

SYSTEM_PROMPT = """You are a precise document assistant. Answer the user's question using ONLY the retrieved document chunks provided below.

Rules you must follow without exception:
- Base every statement on the provided chunks. Do not use outside knowledge.
- If the answer is not present in the chunks, respond with exactly: "The information you're looking for isn't in the uploaded documents."
- Never guess, infer beyond what is written, or fabricate details.
- Be concise and direct. Quote or closely paraphrase the source text where helpful."""


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[Chunk {i} — source: {chunk['source']}]\n{chunk['text']}")
    return "\n\n".join(parts)


def stream_answer(query: str, chunks: list[dict]) -> Generator[str, None, None]:
    """
    Yield answer tokens one at a time.
    Raises ValueError for a missing API key so the UI can show a friendly message.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it to your .env file and restart the app."
        )

    client = Groq(api_key=api_key)
    context = _format_context(chunks)
    user_message = f"Retrieved chunks:\n\n{context}\n\nQuestion: {query}"

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
