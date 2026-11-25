import time

import streamlit as st

from src.document_loader import extract_text
from src.embeddings import EMBEDDING_DIMENSION, MODEL_NAME as EMBEDDING_MODEL
from src.generator import (
    MODEL_NAME as PRIMARY_MODEL,
    BACKUP_MODEL,
    NO_ANSWER_MESSAGE,
    extract_cited_numbers,
    generate_answer_with_usage,
)
from src.metrics import estimate_cost
from src.reranker import MIN_RELEVANCE_SCORE, MODEL_NAME as RERANK_MODEL, rerank_chunks
from src.retriever import retrieve_chunks
from src.text_chunker import chunk_text
from src.vector_store import (
    delete_document,
    list_documents,
    make_document_id,
    store_chunks,
)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# ==========================================
# PAGE SETUP AND STYLE
# ==========================================

st.set_page_config(
    page_title="Mini RAG",
    page_icon="📚",
    layout="wide",
)

CUSTOM_CSS = """
<style>
.block-container { padding-top: 2rem; max-width: 1100px; }

.hero {
    padding: 2rem 2.2rem;
    border-radius: 20px;
    background: linear-gradient(135deg, #5b3df5 0%, #8b5cf6 45%, #06b6d4 100%);
    color: white;
    margin-bottom: 1.2rem;
    box-shadow: 0 12px 40px rgba(124, 92, 255, 0.25);
}
.hero h1 { margin: 0; font-size: 2.2rem; font-weight: 800; color: white; }
.hero p { margin: 0.4rem 0 0 0; font-size: 1.05rem; opacity: 0.92; }

.pipeline { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 1rem; }
.pipeline span {
    background: rgba(255, 255, 255, 0.18);
    padding: 0.3rem 0.8rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
}

.chips { display: flex; gap: 0.45rem; flex-wrap: wrap; margin: 0.6rem 0 0.2rem 0; }
.chip {
    background: #1d2433;
    border: 1px solid #2c3550;
    color: #c9d1e6;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    font-size: 0.78rem;
}

div[data-testid="stMetric"] {
    background: #161B26;
    border: 1px solid #263049;
    padding: 0.9rem 1rem;
    border-radius: 14px;
}

.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    border: 2px dashed #2c3550;
    border-radius: 18px;
    color: #9aa6c4;
}
.empty-state h3 { color: #e6e9f0; margin-bottom: 0.3rem; }

section[data-testid="stSidebar"] { border-right: 1px solid #1f2740; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==========================================
# SESSION STATE
# ==========================================

DEFAULTS = {
    "active_document_id": None,
    "messages": [],
    "last_ingest": None,
    "flash": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==========================================
# HELPERS
# ==========================================

def render_sources(sources, answer):
    """Show the snippets that the [1], [2] numbers point to."""

    if not sources or answer.strip() == NO_ANSWER_MESSAGE:
        return

    cited = extract_cited_numbers(answer, len(sources))

    st.markdown("**📎 Sources**")

    for number, source in enumerate(sources, start=1):
        tag = " ✅ cited" if number in cited else ""
        title = (
            f"[{number}] {source['document_name']} · "
            f"chunk {source['chunk_index']}{tag}"
        )

        with st.expander(title, expanded=False):
            st.write(source["text"])
            st.caption(f"Relevance score: {source.get('rerank_score', 0):.3f}")


def render_stats(stats):
    """Show timing, tokens and estimated cost as small chips."""

    chips = [
        f"⏱️ {stats['total_s']:.1f}s total",
        f"🔎 retrieve {stats['retrieval_s']:.2f}s",
        f"🎯 rerank {stats['rerank_s']:.2f}s",
        f"✍️ generate {stats['generation_s']:.2f}s",
        f"🔢 {stats['prompt_tokens']:,} in / {stats['output_tokens']:,} out tokens",
        f"💲 ~${stats['cost_usd']:.5f} (estimate)",
    ]

    html = "".join(f'<span class="chip">{chip}</span>' for chip in chips)

    st.markdown(f'<div class="chips">{html}</div>', unsafe_allow_html=True)


def ingest(get_text, name):
    """Read text, split it, embed it and store it. Returns a summary."""

    started = time.perf_counter()

    with st.status("Processing your document...", expanded=True) as status:

        st.write("📖 Reading the text...")
        text = get_text()

        if not text or not text.strip():
            raise ValueError("No text could be found.")

        st.write("✂️ Splitting into chunks...")
        chunks = chunk_text(
            text,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        if not chunks:
            raise ValueError("No chunks were generated.")

        st.write(f"🧮 Embedding and storing {len(chunks)} chunks...")
        stored = store_chunks(chunks=chunks, document_name=name)

        if stored != len(chunks):
            raise ValueError("Stored vector count does not match chunk count.")

        elapsed = time.perf_counter() - started
        status.update(
            label=f"Done in {elapsed:.1f}s",
            state="complete",
            expanded=False,
        )

    return {
        "document_id": make_document_id(name),
        "name": name,
        "characters": len(text),
        "chunks": len(chunks),
        "stored": stored,
        "seconds": elapsed,
    }


def run_ingest(get_text, name):
    try:
        summary = ingest(get_text, name)
    except Exception as error:
        st.error(f"Processing failed: {error}")
        return

    st.session_state.active_document_id = summary["document_id"]
    st.session_state.messages = []
    st.session_state.last_ingest = summary
    st.session_state.flash = (
        f"“{name}” is ready: {summary['chunks']} chunks stored."
    )
    st.rerun()


# ==========================================
# LOAD STORED DOCUMENTS
# ==========================================

try:
    stored_documents = list_documents()
except Exception as error:
    stored_documents = []
    st.error(f"Could not load stored documents: {error}")

document_ids = [d["document_id"] for d in stored_documents]
document_names = {d["document_id"]: d["document_name"] for d in stored_documents}
document_labels = {
    d["document_id"]: f"{d['document_name']} ({d['total_chunks']} chunks)"
    for d in stored_documents
}

if st.session_state.active_document_id not in document_ids:
    st.session_state.active_document_id = document_ids[0] if document_ids else None


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("## 📚 Mini RAG")
    st.caption("Retrieve → Rerank → Answer, with citations")

    st.markdown("### 🗂️ Your documents")

    if document_ids:
        selected_id = st.selectbox(
            "Ask about",
            options=document_ids,
            index=document_ids.index(st.session_state.active_document_id),
            format_func=lambda doc_id: document_labels[doc_id],
        )

        if selected_id != st.session_state.active_document_id:
            st.session_state.active_document_id = selected_id
            st.session_state.messages = []
            st.rerun()

        confirm_delete = st.checkbox("I want to delete this document")

        if st.button("🗑️ Delete document", disabled=not confirm_delete):
            try:
                delete_document(st.session_state.active_document_id)
                st.session_state.active_document_id = None
                st.session_state.messages = []
                st.session_state.last_ingest = None
                st.rerun()
            except Exception as error:
                st.error(f"Delete failed: {error}")
    else:
        st.caption("No documents yet.")

    st.markdown("### ⚙️ Retrieval settings")

    top_k = st.slider("Chunks to retrieve (top-k)", 3, 10, 5)
    top_n = st.slider("Chunks kept after rerank", 1, 5, 3)

    st.markdown("### 🤖 AI Model Configuration")

    st.success(f"**Primary Model:** {PRIMARY_MODEL}")

    st.info(f"**Backup Model:** {BACKUP_MODEL}")

    st.caption(
        "The primary model is used first. "
        "The backup model is attempted if the primary "
        "model encounters temporary errors."
    )

    with st.expander("🔧 Pipeline details"):
        st.write(f"**Chunking:** {CHUNK_SIZE} chars, {CHUNK_OVERLAP} overlap")
        st.write(f"**Embeddings:** {EMBEDDING_MODEL} ({EMBEDDING_DIMENSION} dims)")
        st.write("**Vector DB:** Qdrant Cloud (cosine)")
        st.write(f"**Reranker:** {RERANK_MODEL}")
        st.write(f"**Min relevance:** {MIN_RELEVANCE_SCORE}")
        st.write(f"**Primary LLM:** {PRIMARY_MODEL}")
        st.write(f"**Backup LLM:** {BACKUP_MODEL}")
        st.write(f"**Fallback order:** {PRIMARY_MODEL} → {BACKUP_MODEL}")

    st.caption("Costs shown are rough estimates.")


# ==========================================
# HEADER
# ==========================================

if st.session_state.flash:
    st.toast(st.session_state.flash, icon="✅")
    st.session_state.flash = None

st.markdown(
    """
    <div class="hero">
        <h1>📚 Mini RAG</h1>
        <p>Paste text or upload a file, then ask questions and get answers
        with citations you can check.</p>
        <div class="pipeline">
            <span>1 · Chunk</span>
            <span>2 · Embed</span>
            <span>3 · Qdrant search</span>
            <span>4 · Cohere rerank</span>
            <span>5 · Gemini answer</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# ADD CONTENT
# ==========================================

with st.expander("➕ Add content", expanded=not stored_documents):

    tab_paste, tab_upload = st.tabs(["📝 Paste text", "📄 Upload PDF / TXT"])

    with tab_paste:
        paste_title = st.text_input("Title", value="My notes")
        pasted_text = st.text_area(
            "Paste your text here",
            height=220,
            placeholder="Paste an article, notes, or any text you want to ask about...",
        )

        if st.button("Process pasted text", type="primary", key="paste_button"):
            if not pasted_text.strip():
                st.warning("Please paste some text first.")
            else:
                run_ingest(
                    lambda: pasted_text,
                    paste_title.strip() or "Pasted text",
                )

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Choose a PDF or TXT file",
            type=["pdf", "txt"],
        )

        if uploaded_file is not None:
            if st.button("Process file", type="primary", key="file_button"):
                run_ingest(
                    lambda: extract_text(uploaded_file),
                    uploaded_file.name,
                )


# ==========================================
# DOCUMENT STATS
# ==========================================

last = st.session_state.last_ingest

if last and last["document_id"] == st.session_state.active_document_id:

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Characters", f"{last['characters']:,}")
    col2.metric("Chunks", last["chunks"])
    col3.metric("Vectors stored", last["stored"])
    col4.metric("Ingestion time", f"{last['seconds']:.1f}s")


# ==========================================
# CHAT
# ==========================================

st.markdown("")

active_id = st.session_state.active_document_id

if not active_id:

    st.markdown(
        """
        <div class="empty-state">
            <h3>Nothing to ask about yet</h3>
            <p>Open <b>➕ Add content</b> above, paste some text or upload a
            file, and your questions will appear here.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

else:

    st.markdown(f"### 💬 Chat with **{document_names[active_id]}**")

    for message in st.session_state.messages:

        avatar = "🧑" if message["role"] == "user" else "🤖"

        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                render_sources(message.get("sources", []), message["content"])

                if "stats" in message:
                    render_stats(message["stats"])

    user_query = st.chat_input("Ask a question about your document...")

    if user_query:

        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_query)

        with st.chat_message("assistant", avatar="🤖"):

            try:
                with st.status("Thinking...", expanded=True) as status:

                    total_start = time.perf_counter()

                    st.write("🔎 Searching the vector database...")
                    start = time.perf_counter()
                    retrieved_chunks = retrieve_chunks(
                        query=user_query,
                        top_k=top_k,
                        document_id=active_id,
                    )
                    retrieval_s = time.perf_counter() - start

                    st.write("🎯 Reranking the best matches...")
                    start = time.perf_counter()
                    reranked_chunks = rerank_chunks(
                        query=user_query,
                        retrieved_chunks=retrieved_chunks,
                        top_n=min(top_n, top_k),
                        min_score=MIN_RELEVANCE_SCORE,
                    )
                    rerank_s = time.perf_counter() - start

                    st.write("✍️ Writing the answer...")
                    start = time.perf_counter()
                    answer, usage = generate_answer_with_usage(
                        query=user_query,
                        retrieved_chunks=reranked_chunks,
                    )
                    generation_s = time.perf_counter() - start

                    total_s = time.perf_counter() - total_start

                    status.update(
                        label=f"Answered in {total_s:.1f}s",
                        state="complete",
                        expanded=False,
                    )

                stats = {
                    "retrieval_s": retrieval_s,
                    "rerank_s": rerank_s,
                    "generation_s": generation_s,
                    "total_s": total_s,
                    "prompt_tokens": usage["prompt_tokens"],
                    "output_tokens": usage["output_tokens"],
                    "cost_usd": estimate_cost(
                        usage["prompt_tokens"],
                        usage["output_tokens"],
                    ),
                }

                st.markdown(answer)
                render_sources(reranked_chunks, answer)
                render_stats(stats)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": reranked_chunks,
                    "stats": stats,
                })

            except Exception as error:
                st.error(f"An error occurred: {error}")