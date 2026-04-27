# 🏥 MediAssist AI — NLP-Powered Medical Document Assistant
### CCZG506 · API-Driven Cloud Native Solutions · Assignment II

> An end-to-end **API-driven** NLP application covering **Text Summarization**,
> **Sentiment Analysis**, and **Question Answering**, with **Fine-Tuning** and a
> **RAG-based Chatbot** — powered by the **HuggingFace Inference API** and **Groq API**.
>
> ✅ No large model downloads. No GPU needed. ~300 MB total disk usage.

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Before You Start — API Setup](#3-before-you-start--api-setup)
4. [Step 1 — Connect to the BITS Virtual Lab VM](#4-step-1--connect-to-the-bits-virtual-lab-vm)
5. [Step 2 — Create Project Folder Structure](#5-step-2--create-project-folder-structure)
6. [Step 3 — Create All Project Files](#6-step-3--create-all-project-files)
7. [Step 4 — Install Python Dependencies](#7-step-4--install-python-dependencies)
8. [Step 5 — Test the API Connections](#8-step-5--test-the-api-connections)
9. [Step 6 — Part II: Fine-Tuning via HuggingFace AutoTrain](#9-step-6--part-ii-fine-tuning-via-huggingface-autotrain)
10. [Step 7 — Part III: Add a PDF for RAG](#10-step-7--part-iii-add-a-pdf-for-rag)
11. [Step 8 — Run the App](#11-step-8--run-the-app)
12. [Step 9 — Open the App in Your Browser](#12-step-9--open-the-app-in-your-browser)
13. [Step 10 — Keep the App Running](#13-step-10--keep-the-app-running)
14. [Step 11 — Demo Guide & What to Show in Streamlit UI](#14-step-11--demo-guide--what-to-show-in-streamlit-ui)
15. [Step 12 — Screenshots Checklist](#15-step-12--screenshots-checklist)
16. [Troubleshooting](#16-troubleshooting)
17. [Group Contributions](#17-group-contributions)
18. [References](#18-references)

---

## 1. Project Overview

| Field | Details |
|---|---|
| **Domain** | Healthcare |
| **Category** | Natural Language Processing (NLP) |
| **Sub-task 1** | Text Summarization → `facebook/bart-large-cnn` |
| **Sub-task 2** | Sentiment Analysis → `distilbert/distilbert-base-uncased-finetuned-sst-2-english` |
| **Sub-task 3** | Question Answering → `deepset/roberta-base-squad2` |
| **RAG Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| **RAG Generation** | `llama-3.1-8b-instant` via **Groq API** (free tier) |
| **RAG Fallback** | BART (summaries) + RoBERTa (Q&A) if Groq unavailable |
| **HF API Client** | `huggingface_hub` InferenceClient + direct `requests` |
| **Groq API Client** | `groq` Python SDK |
| **Fine-Tuning** | HuggingFace AutoTrain (free, web UI) |
| **Vector Store** | FAISS-CPU (local, ~10 MB) |
| **UI** | Streamlit multi-page app |
| **VM** | BITS Virtual Lab — Amazon Linux EC2 |
| **Total VM Disk Used** | ~300 MB |

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│            BITS Virtual Lab — Amazon Linux EC2                    │
│                                                                  │
│   Streamlit App (port 8501)                                      │
│   ┌──────────────┐  ┌─────────────┐  ┌──────────────────────┐   │
│   │ Summarizer   │  │  Sentiment  │  │  Question Answering  │   │
│   │  HF API call │  │ HF API call │  │    HF API call       │   │
│   └──────────────┘  └─────────────┘  └──────────────────────┘   │
│                                                                  │
│   ┌────────────────────────────────────────────────────────┐     │
│   │  RAG Chatbot                                           │     │
│   │  PDF → Chunks → FAISS (local) → HF Embed API          │     │
│   │  Query → Retrieved Chunks → Groq Llama-3.1-8B → Answer│     │
│   │  (Fallback: BART for summaries / RoBERTa for Q&A)     │     │
│   └────────────────────────────────────────────────────────┘     │
└──────────────────────┬───────────────────────────────────────────┘
                       │ HTTPS
           ┌───────────┴────────────┐
           │                        │
           ▼                        ▼
┌─────────────────────┐   ┌──────────────────────┐
│  HuggingFace API    │   │     Groq API          │
│  router.hugging     │   │  (Free tier)          │
│  face.co            │   │                       │
│                     │   │  Llama-3.1-8B-Instant │
│  BART  Summarize    │   │  RAG Generation       │
│  DistilBERT Sentiment│  └──────────────────────┘
│  RoBERTa  Q&A       │
│  MiniLM   Embeddings│
└─────────────────────┘
```

---

## 3. Before You Start — API Setup

Do this on your **local laptop** before touching the VM.

### 3.1 HuggingFace — Create Account and Token

1. Go to [https://huggingface.co/join](https://huggingface.co/join) → sign up, verify email
2. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Click **"New token"** → Name: `mediassist` → Role: **Read**
4. Click **"Generate a token"** → copy the token (starts with `hf_...`)

### 3.2 Groq — Create Free Account and API Key

> Groq provides a **completely free** API tier with no credit card required.
> It runs Llama-3.1-8B with up to 30 requests/minute — ideal for this project.

1. Go to [https://console.groq.com](https://console.groq.com) → sign up (free, no credit card)
2. Click **API Keys** in the left sidebar
3. Click **"Create API key"** → name it `mediassist`
4. Copy the key — it starts with `gsk_...`

---

## 4. Step 1 — Connect to the BITS Virtual Lab VM

1. Go to your BITS Virtual Lab portal
2. Under **Course (Virtual Lab)** → find **25S2NSP3-API(Virtual lab)**
3. If State shows `stopped` → click **Launch**, wait ~2 minutes
4. Once State shows `started` → click **Connect**
5. A browser-based terminal opens

---

## 5. Step 2 — Create Project Folder Structure

```bash
mkdir -p ~/mediassist-ai/app/pages \
          ~/mediassist-ai/app/utils \
          ~/mediassist-ai/finetune \
          ~/mediassist-ai/rag_docs \
          ~/mediassist-ai/faiss_index
```

Verify:

```bash
find ~/mediassist-ai -type d
```

Expected:
```
/home/cloud/mediassist-ai
/home/cloud/mediassist-ai/app
/home/cloud/mediassist-ai/app/pages
/home/cloud/mediassist-ai/app/utils
/home/cloud/mediassist-ai/finetune
/home/cloud/mediassist-ai/rag_docs
/home/cloud/mediassist-ai/faiss_index
```

---

## 6. Step 3 — Create All Project Files

Copy and paste each block **completely** into the terminal. Each block ends with `EOF` on its own line.

### 6.1 `requirements.txt`

```bash
cat > ~/mediassist-ai/requirements.txt << 'EOF'
huggingface_hub==0.23.0
requests==2.31.0
streamlit==1.34.0
python-dotenv==1.0.1
faiss-cpu==1.8.0
pypdf==4.2.0
numpy==1.26.4
scikit-learn==1.4.2
fpdf2==2.8.4
groq
EOF
```

### 6.2 `.env`

```bash
cat > ~/mediassist-ai/.env << 'EOF'
HF_TOKEN=PASTE_YOUR_HF_TOKEN_HERE
GROQ_API_KEY=PASTE_YOUR_GROQ_KEY_HERE
EOF
```

Replace both placeholders:

```bash
nano ~/mediassist-ai/.env
```

- Replace `PASTE_YOUR_HF_TOKEN_HERE` with your `hf_...` token
- Replace `PASTE_YOUR_GROQ_KEY_HERE` with your `gsk_...` key
- Press `Ctrl+O` → `Enter` → `Ctrl+X`

Verify:

```bash
cat ~/mediassist-ai/.env
```

Expected:
```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx
```

### 6.3 `app/utils/__init__.py`

```bash
touch ~/mediassist-ai/app/utils/__init__.py
```

### 6.4 `app/utils/hf_api.py`

> ✅ **Key design decisions:**
> - HuggingFace `hf-inference` provider forced explicitly — prevents random routing to paid providers
> - Summarization uses direct `requests` to `router.huggingface.co` — bypasses `huggingface_hub==0.23.0` parameter bugs
> - `wait_for_model` placed in `options` key (top-level), NOT inside `parameters`
> - 3D embedding tensors `(batch, tokens, dim)` handled via mean-pooling automatically
> - Groq client initialized for RAG generation — Llama-3.1-8B on free tier
> - `generate_answer()` takes explicit `context` and `question` for true RAG generation

```bash
cat > ~/mediassist-ai/app/utils/hf_api.py << 'EOF'
import os
import numpy as np
import requests
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from groq import Groq

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

HF_TOKEN  = os.getenv("HF_TOKEN", "")
GROQ_KEY  = os.getenv("GROQ_API_KEY", "")

# HuggingFace client — force hf-inference to avoid paid provider routing
client      = InferenceClient(token=HF_TOKEN, provider="hf-inference")

# Groq client — free tier, no credit card needed
groq_client = Groq(api_key=GROQ_KEY)

HF_HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}
HF_BASE_URL = "https://router.huggingface.co/hf-inference/models"

MODELS = {
    "summarization": "facebook/bart-large-cnn",
    "sentiment":     "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    "qa":            "deepset/roberta-base-squad2",
    "embedding":     "sentence-transformers/all-MiniLM-L6-v2",
    "generation":    "llama-3.1-8b-instant",   # Groq free tier
}


def summarize(text: str, max_length: int = 150, min_length: int = 50) -> str:
    """
    Uses direct requests to router.huggingface.co.
    wait_for_model is in options{} (top-level), NOT inside parameters{}.
    """
    if len(text.split()) < 50:
        return "Input is too short. Please provide at least 50 words."
    url     = f"{HF_BASE_URL}/{MODELS['summarization']}"
    payload = {
        "inputs": text[:1024],
        "parameters": {
            "max_length": max_length,
            "min_length": min_length,
        },
        "options": {
            "wait_for_model": True
        }
    }
    response = requests.post(url, headers=HF_HEADERS, json=payload, timeout=60)
    if response.status_code == 200:
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            return data[0].get("summary_text", "No summary returned.")
    raise RuntimeError(
        f"Summarization failed: HTTP {response.status_code} — {response.text[:200]}"
    )


def analyze_sentiment(text: str) -> dict:
    result = client.text_classification(
        text[:512],
        model=MODELS["sentiment"]
    )
    best = max(result, key=lambda x: x.score)
    return {
        "label": best.label,
        "score": round(best.score * 100, 2)
    }


def answer_question(question: str, context: str) -> dict:
    result = client.question_answering(
        question=question,
        context=context[:2000],
        model=MODELS["qa"]
    )
    return {
        "answer": result.answer,
        "score":  round(result.score * 100, 2),
        "start":  result.start,
        "end":    result.end,
    }


def get_embeddings(texts: list) -> list:
    """
    Returns 1-D float vectors per text.
    Handles 3-D (batch, tokens, dim) API responses via mean-pooling.
    """
    result = client.feature_extraction(
        texts,
        model=MODELS["embedding"]
    )
    arr = np.array(result, dtype="float32")
    if arr.ndim == 3:
        arr = arr.mean(axis=1)
    elif arr.ndim == 1:
        arr = arr[np.newaxis, :]
    return arr.tolist()


def generate_answer(prompt: str, context: str = "", question: str = "") -> str:
    """
    True RAG generation via Groq free API (Llama-3.1-8B-Instant).
    Synthesizes a grounded answer from retrieved context.
    Raises exception on failure — caller handles fallback.
    """
    if not GROQ_KEY:
        raise RuntimeError("GROQ_API_KEY not set in .env")

    system_msg = (
        "You are a helpful medical assistant. "
        "Answer the question using ONLY the context provided below. "
        "Be concise and accurate. "
        "If the answer is not in the context, say: "
        "'I could not find that information in the provided documents.'"
    )
    user_msg = f"Context:\n{context[:2000]}\n\nQuestion: {question or prompt}"

    response = groq_client.chat.completions.create(
        model=MODELS["generation"],
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=400,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()
EOF
```

### 6.5 `app/utils/rag_pipeline.py`

> ✅ **Key design decisions:**
> - Full RAG pipeline: Retrieve (MiniLM + FAISS) → Augment (top-k chunks) → Generate (Llama-3.1-8B via Groq)
> - Batch size 2 for HF free-tier embedding stability
> - Per-batch try/except — one failed batch does not crash the full index build
> - FAISS shape validation before indexing
> - Graceful fallback: BART for summaries, RoBERTa for Q&A if Groq is unavailable

```bash
cat > ~/mediassist-ai/app/utils/rag_pipeline.py << 'EOF'
import os
import pickle
import numpy as np
import faiss
from pypdf import PdfReader
from utils.hf_api import get_embeddings, answer_question, summarize, generate_answer

DOCS_DIR         = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../rag_docs"))
FAISS_INDEX_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../faiss_index/index.bin"))
CHUNKS_PATH      = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../faiss_index/chunks.pkl"))

CHUNK_SIZE    = 400
CHUNK_OVERLAP = 80


def extract_text_from_pdfs(docs_dir: str = DOCS_DIR) -> list:
    docs_dir = os.path.abspath(docs_dir)
    if not os.path.exists(docs_dir):
        raise FileNotFoundError(f"rag_docs folder not found: {docs_dir}")
    pdfs = [f for f in os.listdir(docs_dir) if f.endswith(".pdf")]
    if not pdfs:
        raise FileNotFoundError("No PDFs found in rag_docs/. Please upload at least one PDF.")

    all_chunks = []
    for pdf_file in pdfs:
        path      = os.path.join(docs_dir, pdf_file)
        reader    = PdfReader(path)
        full_text = ""
        for page in reader.pages:
            full_text += (page.extract_text() or "")
        i = 0
        while i < len(full_text):
            chunk = full_text[i:i + CHUNK_SIZE].strip()
            if chunk:
                all_chunks.append({"text": chunk, "source": pdf_file})
            i += CHUNK_SIZE - CHUNK_OVERLAP

    print(f"Extracted {len(all_chunks)} chunks from {len(pdfs)} PDF(s).")
    return all_chunks


def build_faiss_index(chunks: list):
    texts = [c["text"] for c in chunks]
    print("Getting embeddings via HuggingFace API...")

    all_embeddings = []
    batch_size = 2          # Free-tier HF API is sensitive to large batches
    failed     = 0

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            embs = get_embeddings(batch)
            arr  = np.array(embs, dtype="float32")
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            all_embeddings.extend(arr.tolist())
        except Exception as e:
            print(f"  Warning: batch {i}-{i+batch_size} failed ({e}), skipping.")
            failed += len(batch)
            for _ in batch:
                all_embeddings.append([0.0] * 384)

        print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks...")

    if failed:
        print(f"  {failed} chunks used zero vectors due to API errors.")

    vectors = np.array(all_embeddings, dtype="float32")
    if vectors.ndim != 2:
        raise ValueError(f"Unexpected embedding shape {vectors.shape}.")

    dim   = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"FAISS index saved — {index.ntotal} vectors, dim={dim}.")
    return index, chunks


def load_faiss_index():
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError("No FAISS index found. Build the index first.")
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def retrieve_chunks(query: str, index, chunks: list, top_k: int = 4) -> list:
    query_emb = np.array(get_embeddings([query]), dtype="float32")
    if query_emb.ndim == 1:
        query_emb = query_emb[np.newaxis, :]
    _, indices = index.search(query_emb, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


def query_rag(query: str, index, chunks: list) -> dict:
    """
    Full RAG pipeline:
      [R]etrieve  — MiniLM embeddings + FAISS vector search (HuggingFace API)
      [A]ugment   — Top-k chunks assembled as context
      [G]enerate  — Llama-3.1-8B via Groq free API (true generative answer)
      Fallback    — BART (summary queries) or RoBERTa (Q&A) if Groq fails
    """
    relevant = retrieve_chunks(query, index, chunks)
    sources  = list(set([c["source"] for c in relevant]))
    context  = "\n\n".join([c["text"] for c in relevant])

    # Primary: true LLM generation via Groq (Llama-3.1-8B)
    try:
        answer = generate_answer(prompt=query, context=context, question=query)
        return {"answer": answer, "sources": sources}
    except Exception as groq_err:
        print(f"Groq generation failed ({groq_err}), falling back...")

    # Fallback: BART for summaries, RoBERTa for Q&A
    q_lower    = query.lower()
    is_summary = any(w in q_lower for w in [
        "summarize", "summary", "summarise", "overview",
        "what is this", "what does this", "describe the document",
        "tell me about the document"
    ])

    if is_summary:
        words         = context.split()
        context_input = " ".join(words[:300])
        if len(context_input.split()) < 50:
            context_input = context_input + " " + context_input
        try:
            answer = summarize(context_input, max_length=150, min_length=40)
        except Exception as e:
            answer = f"Could not summarize: {e}"
    else:
        try:
            result = answer_question(query, context)
            answer = result["answer"]
            score  = result["score"]
            if score < 5:
                answer += f"\n\n_(Low confidence: {score:.1f}% — try rephrasing)_"
        except Exception as e:
            answer = f"Could not find an answer: {e}"

    return {"answer": answer, "sources": sources}
EOF
```

### 6.6 `app/main.py`

```bash
cat > ~/mediassist-ai/app/main.py << 'EOF'
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
    powered by the **HuggingFace Inference API** and **Groq API**.

    ### 🔍 Available Tools

    | Page | Tool | Model | API |
    |---|---|---|---|
    | 📋 Page 1 | **Text Summarization** | `facebook/bart-large-cnn` | HuggingFace |
    | 💬 Page 2 | **Sentiment Analysis** | `distilbert-base-uncased-finetuned-sst-2-english` | HuggingFace |
    | ❓ Page 3 | **Question Answering** | `deepset/roberta-base-squad2` | HuggingFace |
    | 🤖 Page 4 | **RAG Chatbot** | `llama-3.1-8b-instant` + MiniLM + FAISS | Groq + HuggingFace |

    👈 **Use the sidebar to navigate between tools.**
    """)

with col2:
    st.info("""
    **🌐 API-Driven**

    All models run in the cloud.
    HuggingFace: NLP tasks.
    Groq: RAG generation.

    No GPU. No large downloads.
    Pure API calls.
    """)

st.markdown("---")
st.caption("CCZG506 — API-Driven Cloud Native Solutions | Assignment II")
EOF
```

### 6.7 `app/pages/1_Summarization.py`

```bash
cat > ~/mediassist-ai/app/pages/1_Summarization.py << 'EOF'
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from utils.hf_api import summarize

st.set_page_config(page_title="Medical Text Summarizer", page_icon="📋")
st.title("📋 Medical Text Summarizer")
st.markdown("Summarize clinical notes or research abstracts via **BART** on HuggingFace API.")
st.caption("Model: `facebook/bart-large-cnn`")

user_input = st.text_area("📝 Input Text", height=300,
    placeholder="Paste your medical text here (minimum 50 words)...")

col1, col2 = st.columns(2)
max_len = col1.slider("Max Summary Length (words)", 50, 300, 150)
min_len = col2.slider("Min Summary Length (words)", 20, 100, 50)

if st.button("Summarize", type="primary"):
    if user_input.strip():
        with st.spinner("Calling HuggingFace API... (first call may take ~20s)"):
            try:
                summary = summarize(user_input, max_len, min_len)
                st.success("Done!")
                st.markdown("### 📄 Summary")
                st.info(summary)
                orig = len(user_input.split())
                summ = len(summary.split())
                st.caption(
                    f"Original: {orig} words → Summary: {summ} words "
                    f"({round((1 - summ/orig)*100)}% reduction)"
                )
            except Exception as e:
                st.error(f"❌ {e}")
    else:
        st.warning("Please enter some text first.")
EOF
```

### 6.8 `app/pages/2_Sentiment_Analysis.py`

```bash
cat > ~/mediassist-ai/app/pages/2_Sentiment_Analysis.py << 'EOF'
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from utils.hf_api import analyze_sentiment

st.set_page_config(page_title="Patient Feedback Analyzer", page_icon="💬")
st.title("💬 Patient Feedback Sentiment Analyzer")
st.markdown("Classify patient feedback as **Positive** or **Negative** via **DistilBERT** on HuggingFace API.")
st.caption("Model: `distilbert/distilbert-base-uncased-finetuned-sst-2-english`")

EXAMPLES = [
    "The doctor was very attentive and explained everything clearly. I felt well cared for.",
    "I waited 3 hours and the staff was rude. The diagnosis was wrong and I had to visit again.",
    "Average experience. Nothing exceptional but nothing terrible either."
]

with st.expander("💡 Try an example"):
    for ex in EXAMPLES:
        if st.button(ex[:65] + "...", key=ex):
            st.session_state["sentiment_input"] = ex

feedback = st.text_area("🗣️ Patient Feedback", height=150,
    value=st.session_state.get("sentiment_input", ""),
    placeholder="e.g., The doctor was very attentive and the staff was helpful...")

if st.button("Analyze Sentiment", type="primary"):
    if feedback.strip():
        with st.spinner("Calling HuggingFace API..."):
            try:
                result = analyze_sentiment(feedback)
                label  = result["label"]
                score  = result["score"]
                col1, col2 = st.columns(2)
                with col1:
                    if "POSITIVE" in label:
                        st.success(f"✅ **{label}**")
                    else:
                        st.error(f"❌ **{label}**")
                with col2:
                    st.metric("Confidence", f"{score}%")
                st.progress(int(score) / 100)
            except Exception as e:
                st.error(f"❌ {e}")
    else:
        st.warning("Please enter some feedback text.")
EOF
```

### 6.9 `app/pages/3_Question_Answering.py`

```bash
cat > ~/mediassist-ai/app/pages/3_Question_Answering.py << 'EOF'
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from utils.hf_api import answer_question

st.set_page_config(page_title="Medical Q&A", page_icon="❓")
st.title("❓ Medical Question Answering")
st.markdown("Extract answers from medical passages via **RoBERTa** on HuggingFace API.")
st.caption("Model: `deepset/roberta-base-squad2`")

SAMPLE_CONTEXT = (
    "Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID) used to treat pain, "
    "fever, and inflammation. The typical adult dose is 200-400 mg every 4-6 hours, "
    "not exceeding 1200 mg per day for over-the-counter use. It should be taken with "
    "food or milk to reduce stomach upset. Ibuprofen is contraindicated in patients "
    "with peptic ulcer disease and should be used with caution in patients with "
    "cardiovascular disease or kidney problems."
)

with st.expander("💡 Try this example"):
    if st.button("Load example"):
        st.session_state["qa_context"]  = SAMPLE_CONTEXT
        st.session_state["qa_question"] = "What is the maximum daily dose of ibuprofen?"

context = st.text_area("📖 Medical Passage / Context", height=250,
    value=st.session_state.get("qa_context", ""),
    placeholder="Paste a paragraph from a medical document...")

question = st.text_input("🤔 Your Question",
    value=st.session_state.get("qa_question", ""),
    placeholder="e.g., What is the recommended dosage?")

if st.button("Get Answer", type="primary"):
    if context.strip() and question.strip():
        with st.spinner("Calling HuggingFace API..."):
            try:
                result = answer_question(question, context)
                st.success("Answer found!")
                st.markdown(f"### 💡 Answer: **{result['answer']}**")
                st.metric("Confidence", f"{result['score']}%")
                st.progress(int(result["score"]) / 100)
                s, e = result.get("start", 0), result.get("end", 0)
                if s < e:
                    highlighted = (
                        context[:s]
                        + f"**`{context[s:e]}`**"
                        + context[e:]
                    )
                    with st.expander("📌 Answer highlighted in context"):
                        st.markdown(highlighted)
            except Exception as e:
                st.error(f"❌ {e}")
    else:
        st.warning("Please provide both a passage and a question.")
EOF
```

### 6.10 `app/pages/4_RAG_Chatbot.py`

```bash
cat > ~/mediassist-ai/app/pages/4_RAG_Chatbot.py << 'EOF'
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
EOF
```

### 6.11 Fine-Tuning Dataset

```bash
cat > ~/mediassist-ai/finetune/medical_sentiment_dataset.csv << 'EOF'
text,label
"The doctor was very attentive and explained my condition thoroughly.",POSITIVE
"Excellent care from the nursing staff. I felt safe and well-looked after.",POSITIVE
"The physician was knowledgeable and addressed all my concerns.",POSITIVE
"Waited only 10 minutes. The consultation was thorough and reassuring.",POSITIVE
"The specialist was compassionate and the treatment was effective.",POSITIVE
"Great hospital. Clean, organized, and staff were very professional.",POSITIVE
"The doctor listened carefully and prescribed the right medication.",POSITIVE
"Outstanding service. My recovery was faster than expected.",POSITIVE
"The physiotherapy team was brilliant and very encouraging.",POSITIVE
"Very happy with the diagnosis and the follow-up care provided.",POSITIVE
"I waited 4 hours in the emergency room with no updates.",NEGATIVE
"The doctor dismissed my symptoms without a proper examination.",NEGATIVE
"Terrible experience. Staff were rude and unhelpful.",NEGATIVE
"Wrong diagnosis led to unnecessary medication and side effects.",NEGATIVE
"The facility was dirty and the wait times were unacceptable.",NEGATIVE
"My prescription was lost and I had to wait another week.",NEGATIVE
"The nurse was dismissive when I reported severe pain.",NEGATIVE
"Billing errors and nobody helped resolve them for weeks.",NEGATIVE
"Poor communication between departments caused serious delays.",NEGATIVE
"I left feeling worse than when I arrived.",NEGATIVE
"The visit was okay. Nothing stood out positively or negatively.",NEUTRAL
"Average hospital experience. Got what I needed eventually.",NEUTRAL
"The consultation was brief but covered the basics.",NEUTRAL
"Standard care. Adequate but nothing exceptional.",NEUTRAL
"The wait was long but the treatment was satisfactory.",NEUTRAL
"Routine checkup went as expected.",NEUTRAL
"Decent experience overall. Room for improvement.",NEUTRAL
"The facility was average. Staff were professional but rushed.",NEUTRAL
"Neither impressed nor disappointed with the service.",NEUTRAL
"Met expectations but nothing beyond that.",NEUTRAL
EOF
```

### 6.12 Verify All Files

```bash
find ~/mediassist-ai -type f | sort
```

Expected:
```
/home/cloud/mediassist-ai/.env
/home/cloud/mediassist-ai/app/main.py
/home/cloud/mediassist-ai/app/pages/1_Summarization.py
/home/cloud/mediassist-ai/app/pages/2_Sentiment_Analysis.py
/home/cloud/mediassist-ai/app/pages/3_Question_Answering.py
/home/cloud/mediassist-ai/app/pages/4_RAG_Chatbot.py
/home/cloud/mediassist-ai/app/utils/__init__.py
/home/cloud/mediassist-ai/app/utils/hf_api.py
/home/cloud/mediassist-ai/app/utils/rag_pipeline.py
/home/cloud/mediassist-ai/finetune/medical_sentiment_dataset.csv
/home/cloud/mediassist-ai/requirements.txt
```

---

## 7. Step 4 — Install Python Dependencies

```bash
cd ~/mediassist-ai
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
```

Verify:

```bash
pip show streamlit huggingface_hub faiss-cpu pypdf fpdf2 groq
```

All six should show version info without errors.

> Every time you open a new terminal: `source ~/mediassist-ai/venv/bin/activate`

---

## 8. Step 5 — Test the API Connections

### Test HuggingFace

```bash
cd ~/mediassist-ai
python3 -c "
from app.utils.hf_api import analyze_sentiment
result = analyze_sentiment('The doctor was excellent and very helpful.')
print('HuggingFace Token OK!')
print(result)
"
```

Expected: `HuggingFace Token OK!` + `{'label': 'POSITIVE', 'score': 99.x}`

### Test Groq

```bash
python3 -c "
from app.utils.hf_api import generate_answer
result = generate_answer('What is ibuprofen?', context='Ibuprofen is a painkiller used for fever and inflammation.', question='What is ibuprofen?')
print('Groq API OK!')
print(result)
"
```

Expected: `Groq API OK!` + a generated answer about ibuprofen.

---

## 9. Step 6 — Part II: Fine-Tuning via HuggingFace AutoTrain

Fine-tuning runs entirely on HuggingFace's cloud — zero disk used on VM.

### 9.1 Get Dataset onto Your Laptop

```bash
cat ~/mediassist-ai/finetune/medical_sentiment_dataset.csv
```

Copy-paste the output into a local file named `medical_sentiment_dataset.csv`.

### 9.2 Train on AutoTrain

1. Go to [https://huggingface.co/autotrain](https://huggingface.co/autotrain)
2. Click **"New Project"** → Name: `medical-sentiment` → Task: **Text Classification**
3. Click **"Create Project"**
4. Upload `medical_sentiment_dataset.csv`
5. Text column: `text` | Label column: `label`
6. Base model: `distilbert-base-uncased` | Epochs: `3`
7. Click **"Train"** — takes ~10 minutes in HF cloud

### 9.3 Use Your Fine-Tuned Model

After training completes:
1. Go to your HF profile → find `yourname/autotrain-medical-sentiment-12345`
2. Make it **Public** in repo settings
3. Update the model ID on your VM:

```bash
nano ~/mediassist-ai/app/utils/hf_api.py
```

Find:
```python
"sentiment": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
```

Replace with your model:
```python
"sentiment": "yourname/autotrain-medical-sentiment-12345",
```

Save: `Ctrl+O` → `Enter` → `Ctrl+X`

---

## 10. Step 7 — Part III: Add a PDF for RAG

> ⚠️ Many public PDF URLs silently redirect to HTML pages.
> Always verify with the `file` command. **Option C is the most reliable.**

### Option A — Download (Primary)

```bash
wget -O ~/mediassist-ai/rag_docs/medical_guide.pdf \
  "https://iris.who.int/bitstream/handle/10665/371090/WHO-MHP-HPS-EML-2023.02-eng.pdf"
file ~/mediassist-ai/rag_docs/medical_guide.pdf
# Must say: PDF document
```

### Option B — Download (Fallback)

```bash
wget -O ~/mediassist-ai/rag_docs/medical_guide.pdf \
  "https://www.nlm.nih.gov/nichsr/esmallbook.pdf"
file ~/mediassist-ai/rag_docs/medical_guide.pdf
```

### Option C — Generate Locally ✅ (Most Reliable — Recommended)

```bash
source ~/mediassist-ai/venv/bin/activate
cd ~/mediassist-ai

python3 - << 'EOF'
from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Helvetica", size=12)

content = [
    "Medical Reference Guide",
    "",
    "Ibuprofen: NSAID used for pain, fever, inflammation.",
    "Dose: 200-400mg every 4-6 hours. Max 1200mg/day OTC.",
    "Take with food. Avoid in peptic ulcer disease.",
    "",
    "Paracetamol (Acetaminophen): Analgesic and antipyretic.",
    "Dose: 500mg-1g every 4-6 hours. Max 4g/day.",
    "Avoid in liver disease. Safe in pregnancy.",
    "",
    "Amoxicillin: Broad-spectrum penicillin antibiotic.",
    "Dose: 250-500mg three times daily for 5-7 days.",
    "Common side effects: nausea, diarrhea, rash.",
    "",
    "Metformin: First-line treatment for Type 2 Diabetes.",
    "Dose: 500mg twice daily with meals, up to 2g/day.",
    "Contraindicated in renal impairment (eGFR < 30).",
    "",
    "Atorvastatin: Statin for high cholesterol.",
    "Dose: 10-80mg once daily. Take at same time each day.",
    "Monitor liver enzymes. Avoid grapefruit juice.",
    "",
    "Omeprazole: Proton pump inhibitor for acid reflux.",
    "Dose: 20mg once daily before meals.",
    "Used for GERD, peptic ulcers, H. pylori treatment.",
    "",
    "Salbutamol (Albuterol): Bronchodilator for asthma.",
    "Inhaler: 100-200mcg (1-2 puffs) as needed.",
    "Side effects: tremor, palpitations, headache.",
    "",
    "Lisinopril: ACE inhibitor for hypertension and heart failure.",
    "Dose: 5-40mg once daily. Monitor potassium and renal function.",
    "Contraindicated in pregnancy and bilateral renal artery stenosis.",
]

for line in content:
    pdf.cell(0, 10, txt=line, new_x="LMARGIN", new_y="NEXT")

pdf.output("rag_docs/medical_guide.pdf")
print("Done! PDF created at rag_docs/medical_guide.pdf")
EOF
```

> ⚠️ Always use `python3 -` (with the dash) for heredoc scripts.
> Never use `sudo python3` — it bypasses the venv.

Verify:

```bash
file ~/mediassist-ai/rag_docs/medical_guide.pdf
ls -lh ~/mediassist-ai/rag_docs/
# Must show: PDF document, non-zero size
```

### Option D — Upload via the App

Once the app is running, use the RAG Chatbot sidebar to upload any PDF, then click **Build / Rebuild Index**.

---

## 11. Step 8 — Run the App

```bash
tmux new -s mediassist
source ~/mediassist-ai/venv/bin/activate
cd ~/mediassist-ai/app
streamlit run main.py --server.port 8501 --server.address 0.0.0.0
```

Expected output:
```
You can now view your Streamlit app in your browser.
URL: http://0.0.0.0:8501
```

Press `Ctrl + B` then `D` to detach — app keeps running after disconnect.

---

## 12. Step 9 — Open the App in Your Browser

**On the VM desktop browser:**
```
http://localhost:8501
```

**From your laptop:**
```bash
curl -s http://checkip.amazonaws.com   # get your public IP
```
Then open: `http://<public-ip>:8501`

> If unreachable from your laptop, open TCP port 8501 in the AWS Security Group (source: `0.0.0.0/0`).

### Expected API Response Times

| Page | First Call | Subsequent Calls |
|---|---|---|
| Summarization | 10–30s | 3–8s |
| Sentiment Analysis | 5–15s | 2–5s |
| Question Answering | 5–15s | 2–5s |
| RAG Chatbot (Groq) | 3–8s | 2–5s |

> Groq is significantly faster than HuggingFace for generation — typically under 5 seconds.

---

## 13. Step 10 — Keep the App Running

| Action | Command |
|---|---|
| New session | `tmux new -s mediassist` |
| Detach (app keeps running) | `Ctrl + B` then `D` |
| Reattach | `tmux attach -t mediassist` |
| List sessions | `tmux ls` |
| Kill session | `tmux kill-session -t mediassist` |

---

## 14. Step 11 — Demo Guide & What to Show in Streamlit UI

Follow these steps in order for a clean, complete demonstration.

---

### 🏠 Home Page
1. Open `http://localhost:8501`
2. Show the landing page — title, tool table with API column, info card
3. **Take screenshot**

---

### 📋 Page 1 — Text Summarization

**What to do:**
1. Click **Summarization** in the sidebar
2. Paste this into the input box:

```
Ibuprofen is a nonsteroidal anti-inflammatory drug used widely in clinical practice
for the management of pain, fever, and inflammation. It works by inhibiting
cyclooxygenase enzymes, thereby reducing the synthesis of prostaglandins.
The standard adult dosage ranges from 200 to 400 mg taken every four to six hours,
with a maximum recommended dose of 1200 mg per day for over-the-counter use and up
to 3200 mg per day under medical supervision. Ibuprofen should be administered with
food or milk to minimize gastrointestinal side effects. It is contraindicated in
patients with peptic ulcer disease, severe renal impairment, and known hypersensitivity
to NSAIDs. Caution is advised in elderly patients and those with cardiovascular
or hepatic conditions. Prolonged use without medical supervision is discouraged.
```

3. Set **Max Summary Length** to `100`, **Min** to `40`
4. Click **Summarize** — wait for result
5. **Take screenshot** showing input + summary + word reduction %

---

### 💬 Page 2 — Sentiment Analysis

**What to do:**
1. Click **Sentiment Analysis** in the sidebar
2. Expand **"Try an example"** → click the positive example button
3. Click **Analyze Sentiment** → show ✅ POSITIVE + confidence %
4. **Take screenshot #1**
5. Clear the box, paste:
   `I waited 3 hours and the staff was rude. The diagnosis was wrong and I had to visit again.`
6. Click **Analyze Sentiment** → show ❌ NEGATIVE + confidence %
7. **Take screenshot #2**

---

### ❓ Page 3 — Question Answering

**What to do:**
1. Click **Question Answering** in the sidebar
2. Expand **"Try this example"** → click **Load example**
3. Click **Get Answer** — show extracted answer + confidence % + highlighted span
4. **Take screenshot #1**
5. Change the question to: `When should ibuprofen be taken with food?`
6. Click **Get Answer** again
7. **Take screenshot #2**

---

### 🤖 Page 4 — RAG Chatbot

**Prerequisites:** `medical_guide.pdf` must exist in `rag_docs/` and index must be built.

**What to do:**
1. Click **RAG Chatbot** in the sidebar
2. Confirm `medical_guide.pdf` is listed under **In rag_docs/:**
3. Click **Build / Rebuild Index** → wait for "✅ Index built!" in terminal and sidebar
4. **Take screenshot** of the sidebar showing PDF listed + index built message

**Ask these questions one by one:**

| Question to type | What it demonstrates |
|---|---|
| `What is ibuprofen used for?` | Factual Q&A — Groq generates grounded answer |
| `What is the dose of metformin?` | Dosage extraction via LLM |
| `Which medicine should be avoided in liver disease?` | Reasoning over context |
| `What are the side effects of amoxicillin?` | Multi-fact synthesis |
| `summarize the document` | RAG + BART summarization fallback |
| `What is lisinopril contraindicated in?` | Contraindication Q&A |
| `What drug is used for acid reflux?` | Multi-turn context awareness |

5. After at least one answer, expand the **Sources** panel (click the triangle)
6. **Take screenshot** showing multi-turn chat + Sources open
7. **Take screenshot** showing the generated Llama answer (not extracted span)

---

### 🎓 AutoTrain — Fine-Tuning

1. Open [https://huggingface.co/autotrain](https://huggingface.co/autotrain)
2. Open the `medical-sentiment` project → show training metrics / accuracy
3. Open the model page (`yourname/autotrain-medical-sentiment-12345`) on HF Hub
4. **Take screenshot** of training results
5. **Take screenshot** of published model page

---

## 15. Step 12 — Screenshots Checklist

Take all 14 screenshots in this order for your submission document.

| # | Page | What Must Be Visible |
|---|---|---|
| 1 | **Home** | Full landing page — title, tool table with API column, info card |
| 2 | **Summarization** | Input text (50+ words) + generated summary + word count reduction % |
| 3 | **Sentiment — Positive** | Positive feedback + ✅ POSITIVE badge + confidence % progress bar |
| 4 | **Sentiment — Negative** | Negative feedback + ❌ NEGATIVE badge + confidence % |
| 5 | **Q&A — Example 1** | Ibuprofen context + "max daily dose" question + answer highlighted in context |
| 6 | **Q&A — Example 2** | Different question + extracted answer + confidence score |
| 7 | **RAG — Index Built** | Sidebar showing PDF filename + "✅ Index built!" message |
| 8 | **RAG — Groq Answer** | Chat showing question + Groq-generated synthesized answer + Sources open |
| 9 | **RAG — Summary** | Chat showing "summarize the document" + summary response |
| 10 | **RAG — Multi-turn** | At least 3 question/answer pairs in chat history |
| 11 | **AutoTrain — Training** | HuggingFace AutoTrain project showing training metrics / accuracy |
| 12 | **AutoTrain — Model Page** | HuggingFace Hub page of your published fine-tuned model |
| 13 | **Terminal — App Running** | Terminal showing `You can now view your Streamlit app` startup message |
| 14 | **Terminal — Index Build** | Terminal showing `FAISS index saved — N vectors, dim=384` |

---

## 16. Troubleshooting

### HF Token invalid
```bash
cat ~/mediassist-ai/.env
# Must show: HF_TOKEN=hf_xxxxxxxx
nano ~/mediassist-ai/.env
```

### Groq API key invalid / AuthenticationError
```bash
cat ~/mediassist-ai/.env
# Must show: GROQ_API_KEY=gsk_xxxxxxxx
# Get key from https://console.groq.com → API Keys
```

### "Stream has ended unexpectedly" during index build
```bash
file ~/mediassist-ai/rag_docs/medical_guide.pdf
# If "HTML document" → delete and regenerate:
rm ~/mediassist-ai/rag_docs/medical_guide.pdf
# Then use Option C from Step 7
```

### "No module named fpdf" when generating PDF
```bash
# Never use sudo python3 — always activate venv first
source ~/mediassist-ai/venv/bin/activate
python3 - << 'EOF'
from fpdf import FPDF
print("fpdf OK")
EOF
```

### "No module named groq"
```bash
source ~/mediassist-ai/venv/bin/activate
pip install groq
```

### App spinner runs forever (HuggingFace)
Model is cold-starting. Wait up to 2 minutes — `wait_for_model: True` is set in all HF requests.

### Groq rate limit (429)
Groq free tier allows 30 requests/minute. Wait a few seconds and retry.

### "No module named X"
```bash
source ~/mediassist-ai/venv/bin/activate
pip install --no-cache-dir -r ~/mediassist-ai/requirements.txt
```

### RAG gives wrong answer
Try rephrasing with words that appear in the PDF. Groq's Llama will generate a grounded answer from whatever context was retrieved — if retrieval finds the wrong chunks, rephrase the query.

### "No space left on device"
```bash
pip cache purge
df -h /
```

### PDF downloaded but shows as HTML
```bash
file ~/mediassist-ai/rag_docs/medical_guide.pdf
# If HTML document → delete and use Option C (generate locally)
rm ~/mediassist-ai/rag_docs/medical_guide.pdf
```

---

## 17. Group Contributions

| Sl. No | BITS ID | Name | Contribution | % |
|---|---|---|---|---|
| 1 | | | Part I — Summarization page + HF API integration | |
| 2 | | | Part I — Sentiment Analysis page + HF API integration | |
| 3 | | | Part I — Question Answering page + HF API integration | |
| 4 | | | Part II — Dataset prep + AutoTrain fine-tuning | |
| 5 | | | Part III — RAG pipeline + Groq integration + Chatbot page + VM deployment | |

---

## 18. References

- [HuggingFace Inference Providers](https://huggingface.co/docs/inference-providers/en/index)
- [HF Inference Provider — hf-inference](https://huggingface.co/docs/inference-providers/en/providers/hf-inference)
- [huggingface_hub Python SDK](https://huggingface.co/docs/huggingface_hub/index)
- [HuggingFace AutoTrain](https://huggingface.co/autotrain)
- [Groq API Documentation](https://console.groq.com/docs/openai)
- [Groq Free Tier Models](https://console.groq.com/docs/models)
- [FAISS by Facebook Research](https://github.com/facebookresearch/faiss)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn)
- [distilbert/distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)
- [deepset/roberta-base-squad2](https://huggingface.co/deepset/roberta-base-squad2)
- [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [fpdf2 Documentation](https://py-pdf.github.io/fpdf2/)

---

> **✅ Submission Checklist**
> - [ ] HuggingFace token set in `.env` and HF API test passes (`HuggingFace Token OK!`)
> - [ ] Groq API key set in `.env` and Groq test passes (`Groq API OK!`)
> - [ ] All 11 project files created and verified with `find`
> - [ ] `pip install` completed — all 6 packages verified with `pip show`
> - [ ] All four Streamlit pages loading without errors
> - [ ] Summarization page: summary + word reduction stat shown
> - [ ] Sentiment page: both POSITIVE and NEGATIVE demonstrated
> - [ ] Q&A page: answer highlighted in context shown
> - [ ] Valid PDF confirmed (`file` shows `PDF document`)
> - [ ] RAG index built (`FAISS index saved — N vectors, dim=384` in terminal)
> - [ ] RAG chatbot generating answers via Groq (Llama-3.1-8B)
> - [ ] RAG chatbot showing Sources panel open
> - [ ] Fine-tuned model trained on AutoTrain — model ID updated in `hf_api.py`
> - [ ] All 14 screenshots taken and labelled
> - [ ] Video demo recorded and uploaded to Google Drive
> - [ ] Each team member uploads report individually to portal
> - [ ] Submitted before **27-April-2026, 11:55 PM IST**
