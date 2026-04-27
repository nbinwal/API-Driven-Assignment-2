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
    batch_size = 2
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
      Retrieve  → MiniLM embeddings + FAISS (HuggingFace API)
      Augment   → Top-k chunks as context
      Generate  → Llama-3.1-8B via Groq free API (true generation)
      Fallback  → BART (summaries) or RoBERTa (Q&A) if Groq unavailable
    """
    relevant = retrieve_chunks(query, index, chunks)
    sources  = list(set([c["source"] for c in relevant]))
    context  = "\n\n".join([c["text"] for c in relevant])

    # Primary: true LLM generation via Groq
    try:
        answer = generate_answer(prompt=query, context=context, question=query)
        return {"answer": answer, "sources": sources}
    except Exception as groq_err:
        print(f"Groq generation failed ({groq_err}), falling back...")

    # Fallback: BART for summaries, RoBERTa for Q&A
    q_lower = query.lower()
    is_summary = any(w in q_lower for w in [
        "summarize", "summary", "summarise", "overview",
        "what is this", "what does this", "describe the document"
    ])

    if is_summary:
        words = context.split()
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
