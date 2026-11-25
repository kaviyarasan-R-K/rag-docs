
import sys
import tempfile
from pathlib import Path

# Lets the app run on hosts that don't `pip install` this project (e.g. Streamlit Cloud).
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st

from ragdocs.config import load_config
from ragdocs.embedder import GeminiEmbedder
from ragdocs.ingest import ingest
from ragdocs.llm import generate, make_client
from ragdocs.rag import EmptyIndexError, answer_question
from ragdocs.store import IndexMisMatch, VectorStore

st.set_page_config(page_title="Chat with your docs", page_icon="📄")
cfg = load_config()


@st.cache_resource
def get_client():
    return make_client()


try:
    client = get_client()
except RuntimeError as exc:
    st.error(f"{exc} Set it in your terminal, then restart the app.")
    st.stop()

if "store" not in st.session_state:
    try:
        st.session_state.store = VectorStore.load(cfg.index_path, cfg.embed_model, cfg.embed_dim)
    except IndexMisMatch as exc:
        st.error(str(exc))
        st.stop()
    st.session_state.messages = []

store = st.session_state.store
embedder = GeminiEmbedder(client, cfg.embed_model, cfg.embed_dim)


def _render_sources(sources: list[dict]) -> None:
    for s in sources:
        label = f"[{s['n']}] {s['label']}" + ("  (cited)" if s["cited"] else "")
        with st.expander(label):
            st.caption(f"similarity {s['score']:.3f}")
            st.write(s["text"])


# ---- sidebar: upload + index status ----------------------------------------
with st.sidebar:
    st.header("Your documents")
    st.caption("Files are sent to Google's Gemini API to be embedded and answered from.")
    uploads = st.file_uploader("Add files", type=["pdf", "md", "txt"], accept_multiple_files=True)
    if uploads and st.button("Index uploaded files"):
        with tempfile.TemporaryDirectory() as tmp:
            for up in uploads:
                (Path(tmp) / up.name).write_bytes(up.getvalue())
            with st.spinner("Embedding..."):
                try:
                    rep = ingest(
                        tmp,
                        store,
                        embedder,
                        cfg.chunk_size,
                        cfg.chunk_overlap,
                        on_batch=lambda: store.save(cfg.index_path),
                    )
                    store.save(cfg.index_path)
                    st.success(f"{rep.chunks_new} new chunks indexed ({rep.chunks_skipped} already there).")
                except Exception as exc:
                    st.error(f"Indexing stopped: {exc}. Progress was saved; click again to resume.")

    st.divider()
    st.write(f"**{len(store)} chunks indexed**")
    for name, n in store.sources().items():
        st.caption(f"{name} ({n})")

    if st.session_state.messages and st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---- chat ------------------------------------------------------------------
st.title("Chat with your docs")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        _render_sources(m.get("sources", []))

if question := st.chat_input("Ask something about your documents"):
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                ans = answer_question(
                    question,
                    store,
                    embedder,
                    lambda p: generate(client, cfg.chat_model, p),
                    k=cfg.top_k,
                    min_score=cfg.min_score,
                )
            st.markdown(ans.text)
            sources = [
                {"n": s.n, "label": s.label, "score": s.score, "text": s.text, "cited": s.cited}
                for s in ans.sources
            ]
            _render_sources(sources)

            st.session_state.messages.append({"role": "user", "content": question})
            st.session_state.messages.append(
                {"role": "assistant", "content": ans.text, "sources": sources}
            )
        except EmptyIndexError:
            st.info("Upload and index some documents first (sidebar).")
        except Exception as exc:
            st.error(f"Gemini API call failed: {exc}")