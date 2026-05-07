# 🍽️ Analisis Sentimen Program Makan Bergizi Gratis (MBG)

> Perbandingan **Naïve Bayes** vs **Logistic Regression** untuk klasifikasi sentimen tweet masyarakat terhadap Program MBG — dilengkapi web app interaktif berbasis Streamlit.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn)
![PySastrawi](https://img.shields.io/badge/PySastrawi-Stemming-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 📌 Tentang Proyek

Proyek ini merupakan implementasi analisis sentimen terhadap **6.419 tweet** berbahasa Indonesia terkait Program Makan Bergizi Gratis (MBG) yang dikumpulkan dari platform X (Twitter) pada periode **Januari – Oktober 2025**.

Berbeda dari penelitian sejenis yang hanya menggunakan satu algoritma dan dua kelas sentimen, proyek ini:
- Membandingkan **dua algoritma** (Naïve Bayes vs Logistic Regression)
- Menggunakan **tiga kelas sentimen** (Negatif / Netral / Positif)
- Menambahkan **Confidence Score** sebagai indikator kualitas label otomatis
- Menyertakan **web app interaktif** untuk prediksi sentimen secara real-time

---

## 📊 Hasil Model

| Model | Akurasi |
|---|---|
| Naïve Bayes | 71.81% |
| **Logistic Regression** | **81.15%** ⭐ |

**Distribusi Sentimen Dataset:**

| Kelas | Jumlah | Persentase |
|---|---|---|
| Netral | ~3.355 | 52.3% |
| Negatif | ~1.541 | 24.0% |
| Positif | ~1.521 | 23.7% |

---

## 🗂️ Struktur Proyek

```
mbg-sentiment-analysis/
│
├── 📓 analisis_sentimen_mbg_fixed2.ipynb   # Notebook utama (preprocessing, training, evaluasi)
├── 🌐 app.py                               # Aplikasi Streamlit
│
├── model/
│   ├── nb_model.pkl                        # Model Naïve Bayes (generated dari notebook)
│   ├── lr_model.pkl                        # Model Logistic Regression (generated dari notebook)
│   └── tfidf_vectorizer.pkl               # TF-IDF Vectorizer (generated dari notebook)
│
├── Data_Sentimen_MBG.csv                   # Dataset tweet MBG
├── requirements.txt                        # Daftar dependensi
└── README.md
```

> ⚠️ File `model/*.pkl` **tidak disertakan** di repo. Generate terlebih dahulu dengan menjalankan notebook.

---

## ⚙️ Pipeline Penelitian

```
Raw Tweet (6.419 data)
        │
        ▼
┌─────────────────────┐
│  1. Text Cleaning   │  HTML decode, hapus mention/URL/emoji/angka, lowercase
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  2. Stopword Removal│  150+ kata (slang, sapaan, nama tokoh, nama tempat)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  3. Stemming        │  PySastrawi — bentuk kata dasar bahasa Indonesia
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  4. Auto Labeling   │  Lexicon-based + Confidence Score
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  5. TF-IDF          │  5.000 fitur, bigram (1,2)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  6. Split 80/20     │  Stratified split
└─────────────────────┘
        │
        ▼
   ┌────┴────┐
   NB       LR
   └────┬────┘
        │
        ▼
┌─────────────────────┐
│  7. Evaluasi        │  Accuracy, Precision, Recall, F1, Confusion Matrix
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  8. Streamlit App   │  Prediksi real-time, visualisasi statistik
└─────────────────────┘
```

---

## 🚀 Cara Menjalankan

### 1. Clone repository

```bash
git clone https://github.com/username/mbg-sentiment-analysis.git
cd mbg-sentiment-analysis
```

### 2. Install dependensi

```bash
pip install -r requirements.txt
```

### 3. Jalankan notebook untuk generate model

Buka dan jalankan seluruh cell di `analisis_sentimen_mbg_fixed2.ipynb`. Pastikan cell penyimpanan model sudah ada:

```python
import joblib
import os

os.makedirs('model', exist_ok=True)
joblib.dump(nb_model,   'model/nb_model.pkl')
joblib.dump(lr_model,   'model/lr_model.pkl')
joblib.dump(tfidf_vec,  'model/tfidf_vectorizer.pkl')
```

### 4. Jalankan aplikasi Streamlit

```bash
streamlit run app.py
```

Buka browser di `http://localhost:8501`

---

## 📦 Requirements

```
streamlit
scikit-learn
pandas
numpy
matplotlib
seaborn
joblib
PySastrawi
```

Atau install sekaligus:

```bash
pip install streamlit scikit-learn pandas numpy matplotlib seaborn joblib PySastrawi
```

---

## 🔬 Metodologi

### Text Preprocessing
| Tahap | Keterangan |
|---|---|
| HTML Decode | Konversi `&amp;`, `&quot;`, dll → karakter asli |
| Cleaning | Hapus mention, URL, hashtag symbol, karakter khusus, angka |
| Lowercase | Normalisasi kapitalisasi |
| Stopword Removal | 150+ kata kustom (umum, slang, sapaan, konteks MBG) |
| Stemming | PySastrawi (Enhanced Confix Stripping) |

### Labeling Otomatis
- **Metode:** Lexicon-based classification
- **Kamus positif:** 55 kata (bagus, bermanfaat, berhasil, apresiasi, ...)
- **Kamus negatif:** 60 kata (korupsi, gagal, kecewa, pencitraan, ...)
- **Confidence Score:** `|pos - neg| / (total + 1)` — indikator keandalan label

### Fitur
- **TF-IDF** dengan 5.000 fitur top, n-gram (1,2)

### Model
| Algoritma | Karakteristik |
|---|---|
| Multinomial Naïve Bayes | Probabilistik, sensitif kata tunggal, cepat |
| Logistic Regression | Mempelajari kombinasi fitur, lebih akurat untuk konteks |

---

## 📱 Fitur Aplikasi Streamlit

| Tab | Fitur |
|---|---|
| 🔍 Prediksi Sentimen | Input teks bebas, pilih model (NB / LR / Keduanya), hasil prediksi + confidence score + highlight kata |
| 📊 Statistik Dataset | Distribusi sentimen, perbandingan akurasi, confusion matrix, F1-score |
| 📄 Tentang Penelitian | Metodologi, perbandingan dengan jurnal, novelty, limitasi |

---

## 🆚 Perbandingan dengan Jurnal Referensi

| Aspek | Samuel et al. (2026) | Penelitian Ini |
|---|---|---|
| Dataset | 10.000 tweet (5.000 dipakai) | **6.419 tweet** |
| Periode | Nov 2024 – Mar 2025 | **Jan – Okt 2025** |
| Algoritma | Naïve Bayes saja | **NB + Logistic Regression** |
| Jumlah Kelas | 2 (Positif/Negatif) | **3 (Negatif/Netral/Positif)** |
| Akurasi | 97.4% *(data imbalanced 97.5% negatif)* | **81.15%** *(data balanced)* |
| Distribusi | 97.5% negatif — sangat bias | **~24% / 52% / 24%** |
| Confidence Score | ✗ | **✓** |
| Web App | ✗ | **✓ Streamlit** |

---

## ⚠️ Limitasi

- Labeling otomatis tanpa validasi manual — beberapa label mungkin tidak akurat
- Kamus lexicon belum menangani variasi slang/typo (contoh: `bosok` ≠ `busuk`)
- Beberapa kata kontekstual penting (`anggaran`, `dana`) masuk stopword sehingga kalimat seperti "anggaran bengkak" kehilangan sinyal negatifnya
- Data hanya dari platform X (Twitter), belum multi-platform

---

## 🔭 Rekomendasi Pengembangan

- [ ] Validasi manual minimal 10-20% data (prioritaskan confidence score rendah)
- [ ] Perluas kamus lexicon atau gunakan Word2Vec/FastText
- [ ] Hapus kata kontekstual dari stopword (`anggaran`, `dana`, `uang`)
- [ ] Implementasi SMOTE untuk antisipasi class imbalance
- [ ] Ekspansi data ke Instagram, YouTube, TikTok
- [ ] Coba IndoBERT untuk pemahaman konteks yang lebih dalam

---

## 👥 Tim

Tugas UTS Pembelajaran Mesin — Pertemuan 7  
Program Studi: *[isi nama prodi]*  
Universitas: *[isi nama universitas]*

| Nama | NPM |
|---|---|
| *[Nama 1]* | *[NPM]* |
| *[Nama 2]* | *[NPM]* |
| *[Nama 3]* | *[NPM]* |
| *[Nama 4]* | *[NPM]* |

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan akademis. Dataset tweet merupakan data publik dari platform X.

---

<div align="center">
  <sub>Made with ❤️ for UTS Pembelajaran Mesin</sub>
</div>
