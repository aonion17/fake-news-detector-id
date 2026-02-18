import streamlit as st
import pickle
import numpy as np

# =========================
# Load Model & Vectorizer
# =========================

@st.cache_resource
def load_model():
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

# =========================
# UI CONFIG
# =========================

st.set_page_config(
    page_title="Fake News Detector ID",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detector Bahasa Indonesia")
st.markdown("Deteksi berita **FAKE** atau **REAL** menggunakan Machine Learning (TF-IDF + Logistic Regression)")

st.divider()

# =========================
# INPUT AREA
# =========================

input_text = st.text_area("Masukkan teks berita:", height=200)

# =========================
# PREDICTION
# =========================

if st.button("🔍 Analisis Berita"):

    if input_text.strip() == "":
        st.warning("Silakan masukkan teks berita terlebih dahulu.")
    else:
        vector = vectorizer.transform([input_text])
        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)

        confidence = np.max(probability) * 100

        st.divider()

        if prediction == "FAKE":
            st.error("⚠️ Berita terindikasi **FAKE / HOAX**")
        else:
            st.success("✅ Berita terindikasi **REAL / FAKTA**")

        st.metric("Confidence Score", f"{confidence:.2f}%")

        st.progress(int(confidence))

st.divider()

st.caption("Model: Logistic Regression | Feature Extraction: TF-IDF")
