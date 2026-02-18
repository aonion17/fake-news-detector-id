# 📰 Fake News Detector Bahasa Indonesia

Machine Learning project untuk mendeteksi berita palsu (Fake News) berbahasa Indonesia menggunakan Natural Language Processing (NLP) dan Logistic Regression.

---

## 📌 Project Overview

Dataset awal memiliki 5 kelas:

- FAKTA
- SALAH
- PENIPUAN
- BELUM TERBUKTI
- SATIR

Untuk tujuan Fake News Detection, label dikonversi menjadi binary classification:

- REAL  → FAKTA
- FAKE  → selain FAKTA

---

## 🛠 Tech Stack

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Streamlit (Web App)

---

## ⚙️ Machine Learning Pipeline

1. Load Dataset
2. Drop Duplicate Data
3. Label Conversion (Multiclass → Binary)
4. Stratified Train-Test Split
5. TF-IDF Vectorization (fit only on training data)
6. Logistic Regression Training
7. Model Evaluation
8. Cross Validation

---

## 📊 Model Evaluation

Confusion Matrix:

![Confusion Matrix](assets/confusion_matrix.png)

Results:

- Accuracy: ~99%
- Error Rate: ~1.2%
- False Positive: 39
- False Negative: 25

Cross Validation Mean Score: ~0.98–0.99

---

## 🔍 Key Insights

- Dataset sangat linearly separable menggunakan TF-IDF.
- Tidak ditemukan data leakage (vectorizer hanya di-fit pada training set).
- Model menunjukkan generalization yang baik melalui cross-validation.

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🎯 Skills Demonstrated

- Natural Language Processing
- Text Classification
- Feature Engineering
- Model Validation
- Handling Class Imbalance
- Binary Problem Framing
- ML Evaluation Techniques

---

## 👤 Author

aonion