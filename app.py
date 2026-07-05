import streamlit as st
from dotenv import load_dotenv

from ingest import (
    clear_collection,
    get_chroma_client,
    get_collection,
    ingest_file,
    list_ingested_files,
    load_embedding_model,
)
from retrieval import retrieve
from generation import stream_answer
from flashcards import generate_flashcards, sample_chunks
from summaries import generate_summary, generate_questions

load_dotenv()

# ── 1. Page config ────────────────────────────────────────────────────────────
# Must be the first Streamlit call — anything before it raises an error.
st.set_page_config(
    page_title="DocMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 2. Custom CSS ─────────────────────────────────────────────────────────────
# @import pulls Inter from Google Fonts. Everything else overrides Streamlit's
# defaults using data-testid selectors (stable across Streamlit versions) and
# custom class names we inject ourselves via st.markdown HTML.
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
/* Keep header in the DOM so the sidebar toggle stays clickable;
   just make it transparent so branding/deploy button don't show. */
header { background: transparent !important; border: none !important; }
[data-testid="stDeployButton"] { display: none !important; }
[data-testid="stToolbarActions"] { display: none !important; }

/* DEPTH TECHNIQUE 1 — Background gradient.
   linear-gradient(160deg, ...) angles across the whole canvas so the page
   darkens from top-left navy into a blue-tinted mid-tone and settles into
   near-black at the bottom-right. The 160deg angle avoids a banded look. */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #0D1117 0%, #101828 50%, #0B0F1E 100%) !important;
    min-height: 100vh;
}
[data-testid="stMain"] { background: transparent !important; }

/* DEPTH TECHNIQUE 2 — Sidebar as a distinct panel.
   Cooler/darker gradient than the canvas + a right-side shadow (box-shadow on
   the sidebar element itself) makes it feel like a panel sitting slightly in
   front of the background, not merged with it. */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #090C18 0%, #0C0F1D 100%);
    border-right: 1px solid rgba(79,139,249,0.13);
    box-shadow: 4px 0 32px rgba(0,0,0,0.55);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0; }

/* ACCENT TECHNIQUE — Gradient text.
   Set a blue→purple gradient as the element's background, clip it to the
   text shape with background-clip:text, then set color:transparent so only
   the gradient shows through. Used consistently on the brand, header, and
   source filenames to tie the accent colour across the whole layout. */
.dm-brand { padding: 1.4rem 0 1.2rem; margin-bottom: 1.1rem;
            border-bottom: 1px solid rgba(255,255,255,0.06); }
.dm-brand .name {
    font-size: 1.3rem; font-weight: 700; letter-spacing: -0.01em; margin: 0;
    background: linear-gradient(135deg, #4F8BF9 0%, #8B5CF6 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.dm-brand .tagline { font-size: 0.76rem; color: rgba(250,250,250,0.36); margin: 0.28rem 0 0; }

.dm-label { font-size: 0.67rem; font-weight: 600; letter-spacing: 0.09em;
            text-transform: uppercase; color: rgba(250,250,250,0.3);
            margin: 1.2rem 0 0.5rem; }

/* Stats chip: accent numbers draw the eye to the key metrics */
.dm-stats { background: rgba(79,139,249,0.07);
            border: 1px solid rgba(110,70,220,0.22);
            border-radius: 8px; padding: 0.55rem 0.8rem;
            font-size: 0.8rem; color: rgba(250,250,250,0.56);
            margin-bottom: 0.75rem; line-height: 1.55; }
.dm-stats strong {
    background: linear-gradient(135deg, #4F8BF9, #8B5CF6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ACCENT TECHNIQUE — Left bar on file items.
   border-left with a semi-transparent accent colour acts as a tiny visual
   indicator that these files are "active" in the knowledge base. */
.dm-file { display: flex; align-items: center; gap: 0.45rem;
           padding: 0.36rem 0.55rem 0.36rem 0.65rem;
           background: rgba(255,255,255,0.03);
           border: 1px solid rgba(255,255,255,0.06);
           border-left: 2px solid rgba(79,139,249,0.5);
           border-radius: 6px; margin-bottom: 0.28rem;
           font-size: 0.79rem; color: rgba(250,250,250,0.56);
           word-break: break-all; }

/* ACCENT TECHNIQUE — Gradient buttons.
   Same blue→purple gradient used on text is now the button fill, so pressing
   it feels connected to the rest of the accent language. The hover state
   deepens the colours and adds a coloured glow (box-shadow with spread). */
.stButton > button {
    border-radius: 8px; font-family: 'Inter', sans-serif;
    font-size: 0.87rem; font-weight: 500;
    transition: all 0.16s ease; border: none; }
button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #4F8BF9 0%, #8B5CF6 100%) !important;
    color: #fff !important;
    box-shadow: 0 4px 16px rgba(79,139,249,0.28);
}
button[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(135deg, #3a74e0 0%, #7440d4 100%) !important;
    box-shadow: 0 6px 24px rgba(100,60,220,0.42);
    transform: translateY(-1px);
}
button[data-testid="baseButton-secondary"] {
    background: transparent !important;
    color: rgba(250,250,250,0.4) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}
button[data-testid="baseButton-secondary"]:hover {
    border-color: #ff4b4b !important; color: #ff4b4b !important;
    background: rgba(255,75,75,0.07) !important;
}

[data-testid="stFileUploadDropzone"] {
    border: 1.5px dashed rgba(79,139,249,0.32) !important;
    border-radius: 10px !important;
    background: rgba(79,139,249,0.03) !important;
}

/* Header title uses the same gradient-text technique as the brand name */
.dm-header { padding: 1.5rem 0 0; }
.dm-header h1 {
    font-size: 1.7rem; font-weight: 700; margin: 0; letter-spacing: -0.02em;
    background: linear-gradient(135deg, #4F8BF9 0%, #8B5CF6 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.dm-header .sub { font-size: 0.88rem; color: rgba(250,250,250,0.38); margin: 0.3rem 0 1rem; }
.dm-divider { border: none; border-top: 1px solid rgba(255,255,255,0.07); margin: 0 0 1.4rem; }

.dm-empty { display: flex; flex-direction: column; align-items: center;
            justify-content: center; padding: 5rem 2rem; text-align: center; }
.dm-empty .icon { font-size: 3rem; margin-bottom: 1.2rem; opacity: 0.5; }
.dm-empty h3 { font-size: 1.1rem; font-weight: 500;
               color: rgba(250,250,250,0.6); margin: 0 0 0.45rem; }
.dm-empty p { font-size: 0.86rem; color: rgba(250,250,250,0.34);
              max-width: 330px; line-height: 1.65; margin: 0; }

/* DEPTH TECHNIQUE 3 — Elevated chat bubbles.
   box-shadow creates the illusion of the card floating above the gradient
   background. Two layers: a wider soft shadow (depth) and a tight near-black
   shadow (contact). User bubbles get a blue-tinted glow; assistant stays neutral. */
[data-testid="stChatMessage"] {
    background: rgba(14, 18, 36, 0.7) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 20px rgba(0,0,0,0.45), 0 1px 3px rgba(0,0,0,0.6);
    margin-bottom: 0.6rem !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(79,139,249,0.1) !important;
    border-color: rgba(79,139,249,0.22) !important;
    box-shadow: 0 2px 20px rgba(79,139,249,0.15), 0 1px 3px rgba(0,0,0,0.5);
}

/* ACCENT TECHNIQUE — Inset left glow on the Sources expander.
   box-shadow: inset 3px 0 0 0 colour draws a coloured bar on the inner-left
   edge. Unlike border-left, inset shadows respect border-radius so corners
   stay rounded. The purple end of the gradient distinguishes it from the
   blue-accented file list items. */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 8px !important;
    box-shadow: inset 3px 0 0 0 rgba(139,92,246,0.55), 0 2px 12px rgba(0,0,0,0.3);
    margin-top: 0.6rem;
}
[data-testid="stExpander"] summary { font-size: 0.81rem; color: rgba(250,250,250,0.46); }

.dm-src { background: rgba(255,255,255,0.025);
          border: 1px solid rgba(255,255,255,0.07);
          border-radius: 8px; padding: 0.65rem 0.85rem;
          margin-bottom: 0.45rem; }
.dm-src .src-name {
    font-size: 0.76rem; font-weight: 600; margin-bottom: 0.3rem;
    background: linear-gradient(135deg, #4F8BF9, #8B5CF6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.dm-src .src-text { font-size: 0.79rem; color: rgba(250,250,250,0.45);
                    line-height: 1.55; white-space: pre-wrap; }

[data-testid="stChatInputContainer"] {
    border-top: 1px solid rgba(255,255,255,0.07);
    padding-top: 0.4rem;
}

/* ── Knowledge base summary card ──
   Purple inset top line matches the Sources expander accent,
   distinguishing it from the blue-accented chat bubbles. */
.dm-summary { background: rgba(14,18,36,0.7);
              border: 1px solid rgba(255,255,255,0.08); border-radius: 10px;
              padding: 1rem 1.2rem; font-size: 0.87rem;
              color: rgba(250,250,250,0.72); line-height: 1.68;
              box-shadow: 0 2px 14px rgba(0,0,0,0.35), inset 0 1px 0 0 rgba(139,92,246,0.22); }
.dm-summary p { margin: 0 0 0.3rem; }
.dm-summary p:last-child { margin-bottom: 0; }
.dm-summary ul, .dm-summary ol { padding-left: 1.2rem; margin: 0.1rem 0 0.3rem; }
.dm-summary li { margin-bottom: 0.25rem; }

/* ── Flashcard cards ──
   inset 0 1px 0 0 ... draws a 1px blue line along the inner top edge —
   same accent language as the rest of the UI, without a gradient border. */
.fc-card { background: rgba(14,18,36,0.75);
           border: 1px solid rgba(255,255,255,0.08); border-radius: 12px;
           padding: 1.1rem 1.2rem 0.7rem; margin-bottom: 0.75rem;
           box-shadow: 0 2px 18px rgba(0,0,0,0.4), inset 0 1px 0 0 rgba(79,139,249,0.28); }
.fc-q { font-size: 0.9rem; font-weight: 500; color: rgba(250,250,250,0.85);
        line-height: 1.55; margin-bottom: 0.4rem; }
.fc-answer { background: rgba(139,92,246,0.09); border: 1px solid rgba(139,92,246,0.22);
             border-radius: 8px; padding: 0.65rem 0.85rem;
             font-size: 0.84rem; color: rgba(250,250,250,0.75);
             line-height: 1.6; margin-top: 0.5rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ── 3. Cached resources ───────────────────────────────────────────────────────
# @st.cache_resource keeps a single shared instance alive for the lifetime of
# the server process. Without it, every user interaction reloads the 130 MB
# model from disk — fatal on a weak laptop.

@st.cache_resource(show_spinner="Loading embedding model…")
def _load_model():
    return load_embedding_model()

@st.cache_resource(show_spinner="Opening knowledge base…")
def _load_collection():
    return get_collection(get_chroma_client())


model = _load_model()
collection = _load_collection()

# ── 4. Session state ──────────────────────────────────────────────────────────
# messages: list of {role, content} dicts — the visible chat history.
# sources:  dict of {message_index: [chunk, ...]} — stored separately so we
#           can render source cards under each assistant reply, both on the
#           current run and when replaying history on subsequent reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources" not in st.session_state:
    st.session_state.sources = {}
if "flashcards" not in st.session_state:
    st.session_state.flashcards = []
if "fc_flipped" not in st.session_state:
    st.session_state.fc_flipped = set()
# None = not yet generated; [] = generated but empty/failed
if "kb_summary" not in st.session_state:
    st.session_state.kb_summary = None
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = None
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# ── 5. Helper: render source cards ───────────────────────────────────────────
def render_sources(chunks: list[dict]) -> None:
    unique = sorted({c["source"] for c in chunks})
    label = f"📎 {len(unique)} source{'s' if len(unique) != 1 else ''} used"
    with st.expander(label, expanded=False):
        for chunk in chunks:
            snippet = (
                chunk["text"][:280] + "…"
                if len(chunk["text"]) > 280
                else chunk["text"]
            )
            st.markdown(
                f'<div class="dm-src">'
                f'<div class="src-name">📄 {chunk["source"]} &nbsp;·&nbsp; chunk {chunk["chunk_index"] + 1}</div>'
                f'<div class="src-text">{snippet}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ── 6. Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="dm-brand">'
        '<p class="name">🧠 DocMind</p>'
        '<p class="tagline">Grounded answers from your documents.</p>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<p class="dm-label">📁 Upload Documents</p>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button(
        "⚡ Process Documents",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    ):
        with st.status("Processing…", expanded=True) as status:
            total, errors = 0, []
            for uf in uploaded_files:
                st.write(f"📄 Reading **{uf.name}**…")
                raw = uf.read()
                st.write("🔢 Embedding chunks…")
                count, err = ingest_file(raw, uf.name, model, collection)
                if err:
                    errors.append(err)
                    st.write(f"⚠️ {err}")
                else:
                    total += count
                    st.write(f"✅ **{uf.name}** — {count} chunks stored")
            if errors and total == 0:
                status.update(label="All files skipped.", state="error")
            else:
                label = f"Done — {total} chunks added"
                if errors:
                    label += f" ({len(errors)} file(s) skipped)"
                status.update(label=label + ".", state="complete")
        st.rerun()

    # Stats + file list
    ingested = list_ingested_files(collection)
    chunk_count = len(collection.get()["ids"])

    if ingested:
        st.markdown('<p class="dm-label">📚 Knowledge Base</p>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="dm-stats">'
            f'<strong>{len(ingested)}</strong> document{"s" if len(ingested) != 1 else ""}'
            f"&nbsp;·&nbsp;"
            f"<strong>{chunk_count}</strong> chunks indexed"
            f"</div>",
            unsafe_allow_html=True,
        )
        for name in ingested:
            st.markdown(f'<div class="dm-file">📄 {name}</div>', unsafe_allow_html=True)

        st.markdown('<p class="dm-label">⚙️ Actions</p>', unsafe_allow_html=True)
        if st.button("📋 Summarize Knowledge Base", type="secondary", use_container_width=True):
            with st.spinner("Summarizing…"):
                try:
                    chunks = sample_chunks(collection, n=8)
                    result = generate_summary(chunks)
                    if result:
                        st.session_state.kb_summary = result
                        st.toast("Summary ready — see the Chat tab.", icon="📋")
                    else:
                        st.error("Groq returned an empty response. Try again.")
                except ValueError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Summary error: {exc}")

        if st.button("🗑 Clear Knowledge Base", type="secondary", use_container_width=True):
            removed = clear_collection(collection)
            st.session_state.messages = []
            st.session_state.sources = {}
            st.session_state.flashcards = []
            st.session_state.fc_flipped = set()
            st.session_state.kb_summary = None
            st.session_state.suggested_questions = None
            st.session_state.pending_prompt = None
            st.success(f"Cleared {removed} chunks.")
            st.rerun()


# ── 7. Header + tabs ──────────────────────────────────────────────────────────
st.markdown(
    '<div class="dm-header">'
    "<h1>🧠 DocMind</h1>"
    '<p class="sub">Upload documents in the sidebar — then chat or generate flashcards.</p>'
    "</div>"
    '<hr class="dm-divider">',
    unsafe_allow_html=True,
)

ingested = list_ingested_files(collection)
tab_chat, tab_fc = st.tabs(["💬 Chat", "🃏 Flashcards"])

# ── Tab 1: Chat ───────────────────────────────────────────────────────────────
with tab_chat:
    if not ingested:
        st.markdown(
            '<div class="dm-empty">'
            '<div class="icon">📂</div>'
            "<h3>No documents loaded yet</h3>"
            "<p>Upload PDF or TXT files using the sidebar, then click "
            "<strong>Process Documents</strong> to build your knowledge base.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        # Summary expander — shown when the user has clicked Summarize
        if st.session_state.kb_summary:
            with st.expander("📋 Knowledge Base Summary", expanded=False):
                st.markdown(st.session_state.kb_summary)

        # Lazy-generate suggested questions on the first render after ingestion.
        # None = not yet attempted; [] = tried but failed or empty.
        if st.session_state.suggested_questions is None:
            with st.spinner("Preparing suggested questions…"):
                try:
                    sq_chunks = sample_chunks(collection, n=6)
                    st.session_state.suggested_questions = generate_questions(sq_chunks, n=4)
                except Exception:
                    st.session_state.suggested_questions = []

        # Show suggestions only before the user has sent any message
        if not st.session_state.messages and st.session_state.suggested_questions:
            st.markdown('<p class="dm-label">💡 Try asking</p>', unsafe_allow_html=True)
            sq_cols = st.columns(2)
            for i, q in enumerate(st.session_state.suggested_questions):
                with sq_cols[i % 2]:
                    if st.button(q, key=f"sq_{i}", use_container_width=True):
                        st.session_state.pending_prompt = q

        # Chat history
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and i in st.session_state.sources:
                    render_sources(st.session_state.sources[i])

    # Prompt bridge: button click OR typed input, one unified pipeline.
    # st.chat_input is always called so it remains visible.
    chat_typed = st.chat_input(
        placeholder="Ask a question about your documents…",
        disabled=not ingested,
    )
    pending = st.session_state.pending_prompt
    if pending:
        st.session_state.pending_prompt = None   # consume the one-shot signal
        prompt = pending
    else:
        prompt = chat_typed

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents…"):
                try:
                    chunks = retrieve(prompt, model, collection)
                except Exception as exc:
                    st.error(f"Retrieval failed: {exc}")
                    st.stop()

            if not chunks:
                answer = "The information you're looking for isn't in the uploaded documents."
                st.markdown(answer)
                chunks = []
            else:
                try:
                    answer = st.write_stream(stream_answer(prompt, chunks))
                except ValueError as exc:
                    st.error(str(exc))
                    st.stop()
                except Exception as exc:
                    st.error(f"Generation error: {exc}")
                    st.stop()
                render_sources(chunks)

        msg_index = len(st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        if chunks:
            st.session_state.sources[msg_index] = chunks

# ── Tab 2: Flashcards ─────────────────────────────────────────────────────────
with tab_fc:
    if not ingested:
        st.markdown(
            '<div class="dm-empty">'
            '<div class="icon">🃏</div>'
            "<h3>No documents loaded yet</h3>"
            "<p>Upload and process documents first, then generate flashcards.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        col_a, col_b, col_c = st.columns([4, 1, 1])
        with col_a:
            topic = st.text_input(
                "Topic",
                placeholder="Topic to focus on — leave blank to sample across all documents",
                label_visibility="collapsed",
            )
        with col_b:
            n_cards = st.number_input("Cards", min_value=3, max_value=10, value=6, step=1)
        with col_c:
            st.markdown("<br>", unsafe_allow_html=True)
            generate = st.button("🃏 Generate", type="primary", use_container_width=True)

        if generate:
            with st.spinner("Generating flashcards…"):
                try:
                    if topic.strip():
                        fc_chunks = retrieve(topic.strip(), model, collection)
                    else:
                        fc_chunks = sample_chunks(collection)

                    if not fc_chunks:
                        st.error("No content found. Try processing documents first.")
                    else:
                        cards = generate_flashcards(fc_chunks, n=int(n_cards))
                        if cards:
                            st.session_state.flashcards = cards
                            st.session_state.fc_flipped = set()
                        else:
                            st.warning(
                                "Couldn't parse the response — please try again "
                                "or use a more specific topic."
                            )
                except ValueError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Generation error: {exc}")

        # Render stored cards (persists across reruns via session_state —
        # clicking a card flip doesn't regenerate cards or call the API again)
        if st.session_state.flashcards:
            cards = st.session_state.flashcards
            st.caption(f"{len(cards)} card{'s' if len(cards) != 1 else ''} · click to reveal answers")
            cols = st.columns(2)
            for i, card in enumerate(cards):
                with cols[i % 2]:
                    is_flipped = i in st.session_state.fc_flipped
                    answer_html = (
                        f'<div class="fc-answer">💡 {card["answer"]}</div>'
                        if is_flipped else ""
                    )
                    st.markdown(
                        f'<div class="fc-card">'
                        f'<div class="fc-q">❓ {card["question"]}</div>'
                        f"{answer_html}</div>",
                        unsafe_allow_html=True,
                    )
                    btn_label = "Hide answer ▲" if is_flipped else "Show answer ▼"
                    if st.button(btn_label, key=f"fc_flip_{i}", use_container_width=True):
                        flipped = st.session_state.fc_flipped
                        if i in flipped:
                            flipped.discard(i)
                        else:
                            flipped.add(i)
                        st.session_state.fc_flipped = flipped
