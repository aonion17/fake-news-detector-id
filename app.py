import streamlit as st
import pickle
import numpy as np
import pandas as pd
import time

# --- Setup Page Configuration ---
st.set_page_config(
    page_title="Fake News Detector ID",
    page_icon="�️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Load Model ---
@st.cache_resource
def load_pipeline():
    try:
        with open("model/model_pipeline.pkl", "rb") as f:
            pipeline = pickle.load(f)
        return pipeline
    except FileNotFoundError:
        return None

pipeline = load_pipeline()

# --- Custom CSS (Safe Implementation) ---
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #FFFFFF;
    }

    /* Modern Dark Theme Background */
    .stApp {
        background: #0f172a;
        background-image: 
            radial-gradient(at 0% 0%, hsla(253,16%,7%,1) 0, transparent 50%), 
            radial-gradient(at 50% 0%, hsla(225,39%,30%,1) 0, transparent 50%), 
            radial-gradient(at 100% 0%, hsla(339,49%,30%,1) 0, transparent 50%);
    }

    /* Styling for Inputs */
    .stTextArea textarea {
        background-color: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border-radius: 12px !important;
    }
    .stTextArea textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.5) !important;
    }

    /* Buttons */
    div[data-testid="stHorizontalBlock"] button {
        border-radius: 10px;
        height: 3rem;
        font-weight: bold;
        transition: transform 0.2s;
        width: 100%;
    }
    
    div[data-testid="stHorizontalBlock"] button:hover {
        transform: scale(1.02);
    }

    /* Hide Default Header/Footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Result Animation */
    @keyframes slide-up {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .result-container {
        animation: slide-up 0.6s ease-out;
        padding: 2rem;
        border-radius: 16px;
        margin-top: 2rem;
        text-align: center;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.markdown("""
<div style='text-align: center; padding: 2rem 0;'>
    <h1 style='font-size: 3rem; margin-bottom: 0; background: linear-gradient(to right, #60a5fa, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Fake News Detector</h1>
    <p style='color: #94a3b8; font-size: 1.1rem; margin-top: 0.5rem;'>Verifikasi kebenaran berita dengan AI</p>
</div>
""", unsafe_allow_html=True)

# --- Main Interaction Area ---
with st.container():
    # Helper to clear text
    def clear_text():
        st.session_state["input_text"] = ""

    # Input Area
    input_text = st.text_area(
        "📝 Teks Berita",
        height=200,
        placeholder="Tempelkan isi berita yang ingin Anda periksa di sini...",
        label_visibility="collapsed",
        key="input_text"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Action Buttons
    col1, col2 = st.columns([2, 1])
    with col1:
        analyze = st.button("🚀 Analisis Berita", type="primary", use_container_width=True)
    with col2:
        st.button("🔄 Reset", type="secondary", use_container_width=True, on_click=clear_text)

    # --- Processing & Results ---
    if analyze:
        if not input_text.strip():
            st.warning("⚠️ Silakan masukkan teks berita terlebih dahulu.")
        elif pipeline is None:
            st.error("❌ Model tidak ditemukan. Pastikan file model tersedia.")
        else:
            with st.spinner("🔍 Menganalisis pola bahasa..."):
                time.sleep(1)  # Simulate processing for UX
                
                try:
                    # Prediction
                    prediction = pipeline.predict([input_text])[0]
                    probabilities = pipeline.predict_proba([input_text])[0]
                    confidence = np.max(probabilities) * 100
                    
                    # Logic for Display
                    if prediction == "FAKE":
                        status = "HOAX / PALSU"
                        icon = "🚫"
                        color = "#ef4444" # Red
                        bg_gradient = "linear-gradient(135deg, rgba(239,68,68,0.2), rgba(185,28,28,0.2))"
                        desc = "Sistem mendeteksi indikasi kuat disinformasi."
                    else:
                        status = "FAKTA / ASLI"
                        icon = "✅"
                        color = "#10b981" # Green
                        bg_gradient = "linear-gradient(135deg, rgba(16,185,129,0.2), rgba(6,95,70,0.2))"
                        desc = "Berita ini memiliki kredibilitas bahasa yang baik."

                    # Render Result with HTML
                    st.markdown(f"""
                    <div class="result-container" style="background: {bg_gradient}; border-color: {color};">
                        <div style="font-size: 50px; margin-bottom: 10px;">{icon}</div>
                        <h2 style="color: {color}; margin: 0; font-weight: 800;">{status}</h2>
                        <p style="color: #cbd5e1; margin-top: 10px;">{desc}</p>
                        <div style="margin-top: 20px; font-size: 0.9rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px;">Tingkat Keyakinan AI</div>
                        <div style="font-size: 2.5rem; font-weight: 700; color: white;">{confidence:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Detailed Metrics in Expander (Native Streamlit)
                    st.markdown("<br>", unsafe_allow_html=True)
                    with st.expander("📊 Lihat Detail Statistik"):
                        classes = pipeline.classes_ if hasattr(pipeline, "classes_") else ["FAKE", "REAL"]
                        chart_data = pd.DataFrame({
                            "Kategori": classes,
                            "Probabilitas": probabilities
                        })
                        st.bar_chart(
                            chart_data.set_index("Kategori"),
                            color=(color if prediction == "FAKE" else "#10b981")
                        )

                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")

# --- Footer ---
st.markdown("""
<div style='text-align: center; color: #475569; margin-top: 4rem; padding-bottom: 2rem; font-size: 0.85rem;'>
    &copy; 2026 Fake News Detector ID • Powered by Streamlit & Scikit-learn
</div>
""", unsafe_allow_html=True)

