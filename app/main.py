import streamlit as st

st.set_page_config(
    page_title="MediAssist AI",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 MediAssist AI")
st.subheader("API-Driven NLP Medical Document Assistant")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    Welcome to **MediAssist AI** — an API-driven NLP solution for the Healthcare domain
    powered by the **HuggingFace Inference API**.

    ### 🔍 Available Tools

    | Page | Tool | Model |
    |---|---|---|
    | 📋 Page 1 | **Text Summarization** | `facebook/bart-large-cnn` |
    | 💬 Page 2 | **Sentiment Analysis** | `distilbert-base-uncased-finetuned-sst-2-english` |
    | ❓ Page 3 | **Question Answering** | `deepset/roberta-base-squad2` |
    | 🤖 Page 4 | **RAG Chatbot** | `mistralai/Mistral-7B-Instruct-v0.1` |

    👈 **Use the sidebar to navigate between tools.**
    """)

with col2:
    st.info("""
    **🌐 API-Driven**

    All models run on
    HuggingFace's servers.

    No GPU. No large downloads.
    Pure API calls.
    """)

st.markdown("---")
st.caption("CCZG506 — API-Driven Cloud Native Solutions | Assignment II")
