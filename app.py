import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="Fake News Detector ID",
    page_icon="📰",
    layout="centered"
)

@st.cache_resource
def load_pipeline():
    with open("model/model_pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)
    return pipeline

pipeline = load_pipeline()

st.title("📰 Fake News Detector Bahasa Indonesia")
st.markdown("Model: TF-IDF + Logistic Regression")

st.divider()

input_text = st.text_area("Masukkan teks berita:", height=200)

if st.button("🔍 Analisis Berita"):

    if input_text.strip() == "":
        st.warning("Silakan masukkan teks terlebih dahulu.")
    else:
        prediction = pipeline.predict([input_text])[0]
        probabilities = pipeline.predict_proba([input_text])[0]
        confidence = np.max(probabilities) * 100

        st.divider()

        if prediction == "FAKE":
            st.error("⚠️ Berita terindikasi FAKE / HOAX")
        else:
            st.success("✅ Berita terindikasi REAL / FAKTA")

        st.metric("Confidence Score", f"{confidence:.2f}%")
        st.progress(int(confidence))

st.divider()
st.caption("Built with Scikit-learn Pipeline")