import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from utils.hf_api import analyze_sentiment

st.set_page_config(page_title="Patient Feedback Analyzer", page_icon="💬")
st.title("💬 Patient Feedback Sentiment Analyzer")
st.markdown("Classify patient feedback as **Positive** or **Negative** via **DistilBERT** on HuggingFace API.")
st.caption("Model: `distilbert-base-uncased-finetuned-sst-2-english`")

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
