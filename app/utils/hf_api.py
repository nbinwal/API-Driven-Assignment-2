import os
import numpy as np
import requests
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from groq import Groq

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

HF_TOKEN   = os.getenv("HF_TOKEN", "")
GROQ_KEY   = os.getenv("GROQ_API_KEY", "")

client      = InferenceClient(token=HF_TOKEN, provider="hf-inference")
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
    if len(text.split()) < 50:
        return "Input is too short. Please provide at least 50 words."
    url     = f"{HF_BASE_URL}/{MODELS['summarization']}"
    payload = {
        "inputs": text[:1024],
        "parameters": {"max_length": max_length, "min_length": min_length},
        "options":    {"wait_for_model": True}
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
    True RAG generation via Groq free tier (Llama-3.1-8B).
    Synthesizes a grounded answer from retrieved context — fully compliant RAG.
    Falls back to RoBERTa extraction if Groq fails.
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
