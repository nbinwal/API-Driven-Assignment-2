import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from utils.hf_api import answer_question

st.set_page_config(page_title="Medical Q&A", page_icon="❓")
st.title("❓ Medical Question Answering")
st.markdown("Extract answers from medical passages via **RoBERTa** on HuggingFace API.")
st.caption("Model: `deepset/roberta-base-squad2`")

CONFIDENCE_THRESHOLD = 10.0  # Below this % → answer not found in context

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
                score  = result["score"]
                answer = result["answer"]

                if score < CONFIDENCE_THRESHOLD or not answer.strip():
                    st.warning(
                        f"⚠️ The answer to this question was **not found** in the provided passage.\n\n"
                        f"Confidence: {score:.1f}% (below {CONFIDENCE_THRESHOLD}% threshold).\n\n"
                        f"Make sure your question is answerable from the text above."
                    )
                else:
                    st.success("Answer found!")
                    st.markdown(f"### 💡 Answer: **{answer}**")
                    st.metric("Confidence", f"{score:.1f}%")
                    st.progress(int(score) / 100)
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
