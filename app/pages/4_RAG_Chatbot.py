import sys, os, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from utils.rag_pipeline import (
    extract_text_from_pdfs, build_faiss_index,
    load_faiss_index, query_rag,
    DOCS_DIR, FAISS_INDEX_PATH
)

st.set_page_config(page_title="Medical RAG Chatbot", page_icon="🤖")
st.title("🤖 Medical Document Chatbot (RAG)")
st.markdown(
    "Upload medical PDFs, build the FAISS index, then ask questions grounded in your documents. "
    "Generation via **Llama-3.1-8B** on Groq API."
)
st.caption("LLM: llama-3.1-8b-instant (Groq) | Embeddings: sentence-transformers/all-MiniLM-L6-v2 (HuggingFace)")

DOCS_ABS  = os.path.abspath(DOCS_DIR)
FAISS_ABS = os.path.abspath(FAISS_INDEX_PATH)

with st.sidebar:
    st.header("📂 Documents")
    uploaded = st.file_uploader("Upload Medical PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        os.makedirs(DOCS_ABS, exist_ok=True)
        for f in uploaded:
            with open(os.path.join(DOCS_ABS, f.name), "wb") as out:
                out.write(f.getbuffer())
        st.success(f"Uploaded {len(uploaded)} file(s)")

    if os.path.exists(DOCS_ABS):
        pdfs = [f for f in os.listdir(DOCS_ABS) if f.endswith(".pdf")]
        if pdfs:
            st.markdown("**In rag_docs/:**")
            for f in pdfs:
                st.caption(f"📄 {f}")

    st.markdown("---")
    if st.button("🔄 Build / Rebuild Index", type="primary"):
        try:
            with st.spinner("Extracting text and calling HF Embedding API..."):
                chunks = extract_text_from_pdfs()
                build_faiss_index(chunks)
            st.success("✅ Index built!")
            st.cache_resource.clear()
            st.rerun()
        except FileNotFoundError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Error: {e}")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()


@st.cache_resource(show_spinner="Loading FAISS index...")
def get_index():
    if not os.path.exists(FAISS_ABS):
        return None, None
    try:
        return load_faiss_index()
    except Exception as e:
        st.error(f"Failed to load index: {e}")
        return None, None

index, chunks = get_index()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources"):
                for s in msg["sources"]:
                    st.caption(f"• {s}")

if index is None:
    st.warning(
        "⚠️ No index found.\n\n"
        "1. Upload a PDF in the sidebar\n"
        "2. Click **Build / Rebuild Index**\n"
        "3. Start chatting!"
    )
else:
    if prompt := st.chat_input("Ask a question about your medical documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer via Groq..."):
                try:
                    resp    = query_rag(prompt, index, chunks)
                    answer  = resp["answer"]
                    sources = resp["sources"]
                except Exception as e:
                    answer  = f"❌ **Error:**\n```\n{traceback.format_exc()}\n```"
                    sources = []
            st.markdown(answer)
            if sources:
                with st.expander("📚 Sources"):
                    for s in sources:
                        st.caption(f"• {s}")

        st.session_state.messages.append({
            "role": "assistant", "content": answer, "sources": sources
        })
