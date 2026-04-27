import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from utils.hf_api import summarize

st.set_page_config(page_title="Medical Text Summarizer", page_icon="📋")
st.title("📋 Medical Text Summarizer")
st.markdown("Summarize clinical notes or research abstracts via **BART** on the HuggingFace API.")
st.caption("Model: `facebook/bart-large-cnn`")

user_input = st.text_area("📝 Input Text", height=300,
    placeholder="Paste your medical text here (minimum 50 words)...")

col1, col2 = st.columns(2)
max_len = col1.slider("Max Summary Length (words)", 50, 300, 150)
min_len = col2.slider("Min Summary Length (words)", 20, 100, 50)

if st.button("Summarize", type="primary"):
    if user_input.strip():
        with st.spinner("Calling HuggingFace API... (first call may take ~20s to wake model)"):
            try:
                summary = summarize(user_input, max_len, min_len)
                st.success("Done!")
                st.markdown("### 📄 Summary")
                st.info(summary)
                orig  = len(user_input.split())
                summ  = len(summary.split())
                st.caption(f"Original: {orig} words → Summary: {summ} words "
                           f"({round((1 - summ/orig)*100)}% reduction)")
            except Exception as e:
                st.error(f"❌ {e}")
    else:
        st.warning("Please enter some text first.")
