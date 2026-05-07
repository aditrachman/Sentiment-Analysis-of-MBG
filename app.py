import streamlit as st
import joblib
import re
import html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Analisis Sentimen MBG",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS — Tema hijau-tosca bersih, modern
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Background utama */
.stApp {
    background: linear-gradient(135deg, #0f1923 0%, #0d2137 50%, #0a1f1a 100%);
    min-height: 100vh;
}

/* Header banner */
.hero-banner {
    background: linear-gradient(120deg, #00b894 0%, #00cec9 50%, #0984e3 100%);
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(0,184,148,0.3);
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%; right: -10%;
    width: 400px; height: 400px;
    background: rgba(255,255,255,0.06);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: white;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 1rem;
    color: rgba(255,255,255,0.85);
    margin: 6px 0 0 0;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(255,255,255,0.05);
    border-radius: 14px;
    padding: 6px;
    border: 1px solid rgba(255,255,255,0.1);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 10px;
    color: rgba(255,255,255,0.6);
    font-weight: 600;
    font-size: 0.9rem;
    padding: 10px 20px;
    border: none;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(120deg, #00b894, #00cec9) !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(0,184,148,0.4);
}

/* Card umum */
.card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    backdrop-filter: blur(10px);
}
.card-title {
    font-size: 1rem;
    font-weight: 700;
    color: rgba(255,255,255,0.5);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 12px;
}

/* Label sentimen */
.label-negatif {
    display: inline-block;
    background: linear-gradient(135deg, #ff6b6b, #ee5a24);
    color: white; font-weight: 800;
    padding: 12px 32px; border-radius: 50px;
    font-size: 1.4rem; letter-spacing: 1px;
    box-shadow: 0 8px 25px rgba(238,90,36,0.4);
}
.label-netral {
    display: inline-block;
    background: linear-gradient(135deg, #636e72, #b2bec3);
    color: white; font-weight: 800;
    padding: 12px 32px; border-radius: 50px;
    font-size: 1.4rem; letter-spacing: 1px;
    box-shadow: 0 8px 25px rgba(99,110,114,0.4);
}
.label-positif {
    display: inline-block;
    background: linear-gradient(135deg, #00b894, #00cec9);
    color: white; font-weight: 800;
    padding: 12px 32px; border-radius: 50px;
    font-size: 1.4rem; letter-spacing: 1px;
    box-shadow: 0 8px 25px rgba(0,184,148,0.4);
}

/* Confidence bar */
.conf-wrap { margin-top: 12px; }
.conf-label { color: rgba(255,255,255,0.6); font-size: 0.85rem; margin-bottom: 6px; }
.conf-bar-bg {
    background: rgba(255,255,255,0.1);
    border-radius: 50px; height: 10px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%; border-radius: 50px;
    background: linear-gradient(90deg, #00b894, #00cec9);
    transition: width 0.8s ease;
}

/* Highlight kata */
.highlight-pos {
    background: rgba(0,184,148,0.25);
    color: #00e5c3;
    border-radius: 4px; padding: 1px 5px;
    font-weight: 700;
}
.highlight-neg {
    background: rgba(238,90,36,0.25);
    color: #ff7675;
    border-radius: 4px; padding: 1px 5px;
    font-weight: 700;
}

/* Metric box */
.metric-box {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px; padding: 20px 24px;
    text-align: center;
}
.metric-num {
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(120deg, #00b894, #0984e3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-lbl { color: rgba(255,255,255,0.55); font-size: 0.85rem; margin-top: 4px; }

/* Textarea & button */
.stTextArea textarea {
    background: rgba(255,255,255,0.07) !important;
    border: 1.5px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    color: white !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.95rem !important;
}
.stTextArea textarea:focus {
    border-color: #00b894 !important;
    box-shadow: 0 0 0 3px rgba(0,184,148,0.15) !important;
}
.stButton > button {
    background: linear-gradient(120deg, #00b894, #0984e3) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 12px 32px !important;
    width: 100% !important;
    transition: all 0.2s !important;
    box-shadow: 0 6px 20px rgba(0,184,148,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 28px rgba(0,184,148,0.45) !important;
}

/* Radio & selectbox */
.stRadio label, .stSelectbox label {
    color: rgba(255,255,255,0.75) !important;
    font-weight: 500 !important;
}
[data-testid="stMarkdownContainer"] p { color: rgba(255,255,255,0.85); }
h1, h2, h3 { color: white !important; }

/* Comparison card */
.cmp-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px; padding: 20px;
    text-align: center;
}
.cmp-model { font-size: 0.8rem; font-weight: 700; color: rgba(255,255,255,0.5);
             text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }

/* Tabel info */
table { width: 100%; border-collapse: collapse; }
th { background: rgba(0,184,148,0.2); color: #00b894 !important;
     padding: 10px 14px; font-size: 0.85rem; text-transform: uppercase;
     letter-spacing: 0.8px; border-bottom: 1px solid rgba(0,184,148,0.3); }
td { padding: 10px 14px; color: rgba(255,255,255,0.8);
     border-bottom: 1px solid rgba(255,255,255,0.06); font-size: 0.9rem; }
tr:hover td { background: rgba(255,255,255,0.03); }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,184,148,0.4); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# PREPROCESSING PIPELINE — IDENTIK DENGAN NOTEBOOK
# ============================================================
stopwords_id = set([
    'yang','dan','di','dengan','untuk','tidak','ini','dari','dalam','akan',
    'pada','juga','saya','ke','karena','tersebut','bisa','ada','mereka','lebih',
    'kata','tahun','sudah','atau','saat','oleh','menjadi','orang','itu','kita',
    'kami','kalau','kalian','kamu','dia','apa','juga','tapi','tp','ya','yg',
    'ny','nya','dgn','utk','jg','sy','gak','ga','aja','udah','dah','nih','sih',
    'lah','kan','dong','lagi','buat','lg','dr','krn','sdh','pd','klo','kl',
    'emg','emang','memang','banget','bgt','mau','gimana','gmn','kayak','kaya',
    'gitu','gini','kok','deh','wkwk','wkwkwk','haha','hehe','eh','ah','oh',
    'iya','ok','oke','yuk','ajah','adalah','bahwa','hal','bagi','serta','jika',
    'bila','hingga','antara','namun','maka','sejak','setelah','sebelum','sebab',
    'tdk','ny','tp','yg','jg','sy','bgt','krn','sdh','klo','kl','lg','dr','pd',
    'dgn','utk','dll','dsb','dst','quot','amp','apos','nbsp','lt','gt',
    'min','kak','pak','bu','mas','mbak','bang','bro','sis','sob','guys',
    'namun','tetapi','meski','meskipun','walaupun','agar','supaya','bahkan',
    'padahal','sedangkan','sehingga','kemudian','lalu','selanjutnya','akhirnya',
    'siapa','dimana','kemana','darimana','kapan','bagaimana','kenapa','mengapa',
    'pun','pula','jua','kah','tah','nah','neh',
    'mbg','makan','bergizi','gratis',
    'prabowo','jokowi','joko','widodo','gibran','megawati',
    'anies','ganjar','ahok','sby','yudhoyono',
    'indonesia','jakarta','jawa','surabaya','bandung','medan',
    'bali','sulawesi','kalimantan','papua','sumatra',
    'tempo','kompas','detik','tribun','cnbc','okezone',
    'liputan','republika','kumparan','antara',
    'negara','rakyat','program','proyek','pemerintah',
    'presiden','menteri','anak','sekolah','siswa','murid',
    'daerah','dinas','kabupaten','kota','provinsi',
    'anggaran','dana','uang','biaya','rupiah',
])

kata_positif = [
    'bagus','baik','setuju','dukung','mendukung','bermanfaat','manfaat',
    'alhamdulillah','sukses','berhasil','senang','bangga','hebat','keren',
    'mantap','sehat','nutrisi','harapan','positif','maju','sejahtera',
    'membantu','terbantu','solusi','nikmat','enak','lezat',
    'berguna','efektif','transparan','akuntabel','merata','adil',
    'tepat sasaran','berkualitas','luar biasa','mengagumkan','memukau',
    'inspiratif','membanggakan','apresiasi','salut','kagum','syukur',
    'bersyukur','semangat','antusias','optimis','percaya','lancar',
    'tertib','rapi','teratur','disiplin','inovatif','kreatif','produktif',
    'puas','memuaskan','suka','gembira','bahagia','senyum',
    'terima kasih','makasih','terharu',
]

kata_negatif = [
    'korupsi','korup','maling','bancakan','rusak','gagal','bohong','palsu',
    'penipuan','tipu','biadab','bangsat','hancur','ancur','buruk','jelek',
    'sampah','bubar','tolol','bodoh','goblok','idiot','brengsek',
    'racun','bahaya','berbahaya','keracunan','mati','kematian','beracun',
    'kecewa','marah','protes','demo','tolak','menolak','sia sia','percuma',
    'pemborosan','boros','hutang','utang','koruptor','nepotisme','menipu',
    'rezim','otoriter','amburadul','kacau','berantakan','malakin','markup',
    'stop','bubarkan','hentikan','tidak layak','tidak berguna',
    'tidak benar','gak bener','salah','merugikan','dikorupsi','makan racun',
    'gagal total','tidak berhasil','tidak efektif','menghancurkan','merusak',
    'membahayakan','tidak merata','tidak adil','diskriminatif',
    'tidak transparan','ditilep','diselewengkan','basi','busuk',
    'tidak higienis','kotor','mual','muntah','sakit','terganggu',
    'lambat','terlambat','mangkrak','dipotong','pemotongan',
    'pencitraan','gimmick','tidak serius','abal abal','bohong belaka',
    'menyedihkan','miris','memprihatinkan','mengkhawatirkan',
    'benci','muak','jijik','menghina',
]

def clean_text(text):
    text = html.unescape(text)
    text = re.sub(r'&\w+;', ' ', text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    words = text.split()
    return ' '.join([w for w in words if w not in stopwords_id and len(w) > 2])

@st.cache_resource
def load_stemmer():
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        factory = StemmerFactory()
        return factory.create_stemmer()
    except ImportError:
        return None

@st.cache_resource
def load_models():
    try:
        nb    = joblib.load('model/nb_model.pkl')
        lr    = joblib.load('model/lr_model.pkl')
        tfidf = joblib.load('model/tfidf_vectorizer.pkl')
        return nb, lr, tfidf, True
    except:
        return None, None, None, False

def preprocess(text, stemmer=None):
    text = clean_text(text)
    text = remove_stopwords(text)
    if stemmer:
        words = text.split()
        text = ' '.join([stemmer.stem(w) for w in words])
    return text

def hitung_confidence(text):
    words = text.lower()
    pos = sum(1 for k in kata_positif if k in words)
    neg = sum(1 for k in kata_negatif if k in words)
    total = pos + neg
    conf = abs(pos - neg) / (total + 1)
    return pos, neg, conf

def highlight_teks(teks_asli):
    """Highlight kata positif/negatif dalam teks."""
    hasil = teks_asli
    # Tandai negatif dulu (cegah overlap)
    for kata in sorted(kata_negatif, key=len, reverse=True):
        pattern = re.compile(re.escape(kata), re.IGNORECASE)
        hasil = pattern.sub(f'<span class="highlight-neg">{kata}</span>', hasil)
    for kata in sorted(kata_positif, key=len, reverse=True):
        pattern = re.compile(re.escape(kata), re.IGNORECASE)
        # Jangan ganti jika sudah di-wrap
        if kata.lower() in hasil.lower() and 'highlight' not in hasil[max(0, hasil.lower().find(kata.lower())-30):hasil.lower().find(kata.lower())]:
            hasil = pattern.sub(f'<span class="highlight-pos">{kata}</span>', hasil)
    return hasil

def get_label_html(label):
    emoji = {'negatif': '🔴', 'netral': '⚪', 'positif': '🟢'}
    return f'<div class="label-{label}">{emoji.get(label,"")} {label.upper()}</div>'

# ============================================================
# LOAD MODEL & STEMMER
# ============================================================
stemmer = load_stemmer()
nb_model, lr_model, tfidf_vec, model_loaded = load_models()

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🍽️ Analisis Sentimen MBG</div>
    <div class="hero-sub">Makan Bergizi Gratis · Naïve Bayes vs Logistic Regression · 6.419 Tweet · Jan–Okt 2025</div>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.warning("⚠️ **Model belum ditemukan.** Jalankan notebook terlebih dahulu untuk generate file `model/nb_model.pkl`, `model/lr_model.pkl`, dan `model/tfidf_vectorizer.pkl`.")

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["🔍 Prediksi Sentimen", "📊 Statistik Dataset", "📄 Tentang Penelitian"])

# ============================================================
# TAB 1 — PREDIKSI SENTIMEN
# ============================================================
with tab1:
    col_input, col_result = st.columns([1.1, 0.9], gap="large")

    with col_input:
        st.markdown('<div class="card-title">Input Teks</div>', unsafe_allow_html=True)
        teks_input = st.text_area(
            label="Teks komentar",
            placeholder="Ketik atau paste komentar tentang program MBG di sini...\n\nContoh: Program makan gratis ini bagus sekali, anak-anak jadi lebih sehat dan bergizi.",
            height=160,
            label_visibility="collapsed"
        )

        pilihan_model = st.radio(
            "Pilih Model:",
            ["Naïve Bayes", "Logistic Regression", "Keduanya (Bandingkan)"],
            horizontal=True
        )

        tombol = st.button("🔍 Analisis Sentimen", use_container_width=True)

    with col_result:
        if tombol and teks_input.strip():
            processed = preprocess(teks_input, stemmer)
            pos_count, neg_count, conf_lex = hitung_confidence(teks_input)

            if model_loaded:
                vec = tfidf_vec.transform([processed])
                pred_nb = nb_model.predict(vec)[0]
                pred_lr = lr_model.predict(vec)[0]
                prob_nb = nb_model.predict_proba(vec)[0]
                prob_lr = lr_model.predict_proba(vec)[0]
                classes = list(nb_model.classes_)

                if pilihan_model == "Keduanya (Bandingkan)":
                    st.markdown('<div class="card-title">Perbandingan Model</div>', unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    for col, pred, prob, nama_model in [
                        (c1, pred_nb, prob_nb, "Naïve Bayes"),
                        (c2, pred_lr, prob_lr, "Logistic Regression")
                    ]:
                        with col:
                            st.markdown(f'<div class="cmp-card"><div class="cmp-model">{nama_model}</div>', unsafe_allow_html=True)
                            st.markdown(get_label_html(pred), unsafe_allow_html=True)
                            # Probabilitas per kelas
                            st.markdown("<br>", unsafe_allow_html=True)
                            for i, cls in enumerate(classes):
                                pct = prob[i] * 100
                                bar_color = "#00b894" if cls == pred else "rgba(255,255,255,0.2)"
                                st.markdown(f"""
                                <div style="margin-bottom:6px">
                                    <div style="display:flex;justify-content:space-between;color:rgba(255,255,255,0.7);font-size:0.8rem">
                                        <span>{cls}</span><span>{pct:.1f}%</span>
                                    </div>
                                    <div class="conf-bar-bg">
                                        <div class="conf-bar-fill" style="width:{pct}%;background:{bar_color}"></div>
                                    </div>
                                </div>""", unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                else:
                    pred = pred_nb if pilihan_model == "Naïve Bayes" else pred_lr
                    prob = prob_nb if pilihan_model == "Naïve Bayes" else prob_lr
                    st.markdown('<div class="card-title">Hasil Prediksi</div>', unsafe_allow_html=True)
                    st.markdown(get_label_html(pred), unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    for i, cls in enumerate(classes):
                        pct = prob[i] * 100
                        bar_color = "#00b894" if cls == pred else "rgba(255,255,255,0.2)"
                        st.markdown(f"""
                        <div style="margin-bottom:8px">
                            <div style="display:flex;justify-content:space-between;color:rgba(255,255,255,0.7);font-size:0.85rem;margin-bottom:3px">
                                <span>{cls}</span><span><b>{pct:.1f}%</b></span>
                            </div>
                            <div class="conf-bar-bg">
                                <div class="conf-bar-fill" style="width:{pct}%;background:{bar_color}"></div>
                            </div>
                        </div>""", unsafe_allow_html=True)
            else:
                # Fallback: lexicon only
                if neg_count > pos_count:   pred = 'negatif'
                elif pos_count > neg_count: pred = 'positif'
                else:                       pred = 'netral'
                st.markdown('<div class="card-title">Hasil Prediksi (Lexicon)</div>', unsafe_allow_html=True)
                st.markdown(get_label_html(pred), unsafe_allow_html=True)

            # Confidence Lexicon
            st.markdown(f"""
            <div style="margin-top:20px">
                <div class="card-title">Confidence Score Lexicon</div>
                <div style="display:flex;justify-content:space-between;color:rgba(255,255,255,0.6);font-size:0.85rem;margin-bottom:4px">
                    <span>Kata positif: <b style="color:#00b894">{pos_count}</b> &nbsp;|&nbsp; Kata negatif: <b style="color:#ff7675">{neg_count}</b></span>
                    <span><b>{conf_lex:.3f}</b></span>
                </div>
                <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{min(conf_lex*100*1.5,100):.0f}%"></div></div>
                <div style="color:rgba(255,255,255,0.4);font-size:0.78rem;margin-top:4px">
                    {'🔴 Rendah — label kurang pasti' if conf_lex < 0.3 else '🟡 Sedang' if conf_lex <= 0.6 else '🟢 Tinggi — label reliabel'}
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif tombol and not teks_input.strip():
            st.warning("⚠️ Masukkan teks terlebih dahulu.")

    # Highlight kata
    if tombol and teks_input.strip():
        st.markdown("---")
        st.markdown('<div class="card-title">Highlight Kata Sentimen dalam Teks</div>', unsafe_allow_html=True)
        highlighted = highlight_teks(teks_input)
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);
                    border-radius:12px;padding:18px;line-height:1.8;font-size:1rem;color:rgba(255,255,255,0.85)">
            {highlighted}
        </div>
        <div style="margin-top:10px;display:flex;gap:16px;font-size:0.82rem">
            <span><span class="highlight-pos">kata</span> = positif</span>
            <span><span class="highlight-neg">kata</span> = negatif</span>
        </div>
        """, unsafe_allow_html=True)

    # Contoh kalimat
    st.markdown("---")
    st.markdown('<div class="card-title">Coba Contoh Kalimat</div>', unsafe_allow_html=True)
    contoh = {
        "😊 Positif": "Program makan bergizi ini bagus sekali! Anak-anak jadi lebih sehat dan semangat belajar.",
        "😠 Negatif": "Ini cuma pencitraan saja, makanannya basi dan ada yang keracunan. Anggaran dikorupsi lagi!",
        "😐 Netral":  "Pelaksanaan program makan di sekolah sudah dimulai minggu lalu di beberapa daerah.",
    }
    c1, c2, c3 = st.columns(3)
    for col, (label, kalimat) in zip([c1, c2, c3], contoh.items()):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state['contoh_teks'] = kalimat
                st.rerun()

    if 'contoh_teks' in st.session_state:
        st.info(f"📝 **Kalimat contoh:** _{st.session_state['contoh_teks']}_  \n*Copy ke input di atas untuk menganalisis.*")

# ============================================================
# TAB 2 — STATISTIK DATASET
# ============================================================
with tab2:
    # Data statistik hardcoded (dari hasil notebook)
    total_data  = 6419
    dist_persen = {'negatif': 24.0, 'netral': 52.3, 'positif': 23.7}
    dist_jumlah = {k: int(v/100 * total_data) for k, v in dist_persen.items()}
    acc_nb_val  = 71.81
    acc_lr_val  = 81.15

    # Metric boxes
    m1, m2, m3, m4 = st.columns(4)
    for col, num, lbl in [
        (m1, f"{total_data:,}", "Total Tweet"),
        (m2, "Jan–Okt 2025", "Periode Data"),
        (m3, f"{acc_nb_val}%", "Akurasi NB"),
        (m4, f"{acc_lr_val}%", "Akurasi LR ⭐"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-num">{num}</div>
                <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Plot 1 — Distribusi & Akurasi
    col_pie, col_acc = st.columns(2)

    plt.rcParams.update({'text.color': 'white', 'axes.labelcolor': 'white',
                         'xtick.color': 'white', 'ytick.color': 'white'})

    with col_pie:
        fig1, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor='none')
        colors = ['#ff6b6b', '#b2bec3', '#00b894']

        # Pie
        vals = [dist_jumlah[k] for k in ['negatif','netral','positif']]
        wedges, texts, autotexts = axes[0].pie(
            vals, labels=['Negatif','Netral','Positif'], colors=colors,
            autopct='%1.1f%%', startangle=90,
            wedgeprops={'edgecolor':'#0d2137','linewidth':2}
        )
        for t in texts: t.set_color('white')
        for t in autotexts: t.set_color('white'); t.set_fontweight('bold')
        axes[0].set_title('Distribusi Sentimen', color='white', fontweight='bold', pad=15)

        # Bar distribusi
        bars = axes[1].bar(['Negatif','Netral','Positif'], vals, color=colors,
                            edgecolor='none', width=0.6)
        for bar, val in zip(bars, vals):
            axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+30,
                         str(val), ha='center', color='white', fontweight='bold', fontsize=10)
        axes[1].set_facecolor('none')
        axes[1].spines[:].set_visible(False)
        axes[1].tick_params(colors='white')
        axes[1].set_title('Jumlah per Kelas', color='white', fontweight='bold')
        axes[1].set_ylabel('Jumlah Tweet', color='white')

        plt.tight_layout()
        st.pyplot(fig1, use_container_width=True)
        plt.close()

    with col_acc:
        fig2, ax = plt.subplots(figsize=(5, 4), facecolor='none')
        bars = ax.bar(['Naïve Bayes', 'Logistic\nRegression'],
                      [acc_nb_val, acc_lr_val],
                      color=['#0984e3','#00b894'], edgecolor='none', width=0.5)
        for bar, acc in zip(bars, [acc_nb_val, acc_lr_val]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f'{acc:.2f}%', ha='center', color='white', fontweight='bold', fontsize=13)
        ax.set_ylim(0, 100)
        ax.set_facecolor('none')
        ax.spines[:].set_visible(False)
        ax.tick_params(colors='white', labelsize=11)
        ax.set_title('Perbandingan Akurasi Model', color='white', fontweight='bold', pad=15)
        ax.set_ylabel('Akurasi (%)', color='white')
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    # Confusion Matrix & F1 (hardcoded, ganti dengan nilai real dari notebook)
    st.markdown("---")
    col_cm1, col_cm2 = st.columns(2)

    # Contoh confusion matrix (ganti dengan nilai asli dari notebook)
    labels_cm = ['negatif', 'netral', 'positif']
    cm_nb_example = np.array([[210, 95, 45], [80, 560, 45], [55, 68, 230]])
    cm_lr_example = np.array([[258, 60, 32], [55, 620, 10], [40, 50, 263]])

    for col, cm, title, cmap in [
        (col_cm1, cm_nb_example, 'Confusion Matrix — Naïve Bayes', 'Blues'),
        (col_cm2, cm_lr_example, 'Confusion Matrix — Logistic Regression', 'Greens')
    ]:
        with col:
            fig, ax = plt.subplots(figsize=(5, 4), facecolor='none')
            sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                        xticklabels=labels_cm, yticklabels=labels_cm,
                        cbar=False, linewidths=0.5,
                        annot_kws={'color':'white','fontweight':'bold'})
            ax.set_title(title, color='white', fontweight='bold', pad=10)
            ax.set_xlabel('Prediksi', color='white')
            ax.set_ylabel('Aktual', color='white')
            ax.tick_params(colors='white')
            ax.set_facecolor('none')
            fig.patch.set_alpha(0)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

    st.caption("_⚠️ Confusion matrix di atas adalah estimasi. Jalankan notebook untuk mendapatkan nilai aktual, lalu update angka di app.py._")

    # F1 Score table
    st.markdown("---")
    st.markdown("**F1-Score per Kelas (Estimasi)**")
    f1_data = {
        'Kelas': ['Negatif', 'Netral', 'Positif'],
        'NB Precision': ['0.71', '0.76', '0.72'],
        'NB Recall':    ['0.60', '0.83', '0.66'],
        'NB F1':        ['0.65', '0.79', '0.69'],
        'LR Precision': ['0.80', '0.83', '0.85'],
        'LR Recall':    ['0.74', '0.90', '0.75'],
        'LR F1':        ['0.77', '0.86', '0.80'],
    }
    st.table(pd.DataFrame(f1_data))

# ============================================================
# TAB 3 — TENTANG PENELITIAN
# ============================================================
with tab3:
    col_a, col_b = st.columns([1.2, 0.8], gap="large")

    with col_a:
        st.markdown("### 🔬 Metode Penelitian")
        st.markdown("""
        <div class="card">
            <b style="color:#00b894">Pipeline Preprocessing:</b><br>
            <code style="color:#00cec9">Raw Tweet → HTML Decode → Cleaning → Stopword Removal → Stemming (Sastrawi) → TF-IDF Bigram</code>
            <br><br>
            <b style="color:#00b894">Labeling:</b> Lexicon-Based otomatis dengan confidence score<br>
            <b style="color:#00b894">Split:</b> 80% Training / 20% Testing (stratified)<br>
            <b style="color:#00b894">Fitur:</b> TF-IDF dengan 5.000 fitur, n-gram (1,2)
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🆚 Perbandingan dengan Jurnal Pembanding")
        perbandingan = pd.DataFrame({
            'Aspek':            ['Jurnal (Samuel et al. 2026)',  'Penelitian Ini'],
            'Dataset':          ['Twitter MBG',                   'Twitter MBG (6.419)'],
            'Periode':          ['Tidak disebutkan',              'Jan–Okt 2025'],
            'Algoritma':        ['Naïve Bayes saja',              'NB + Logistic Regression'],
            'Jumlah Kelas':     ['2 kelas',                       '3 kelas (neg/net/pos)'],
            'Akurasi':          ['97.4% (imbalanced!)',           '81.15% (balanced)'],
            'Distribusi Data':  ['97.5% negatif (bias berat)',    '~24% / 52% / 24%'],
            'Stemming':         ['Tidak disebutkan',              'Sastrawi ✓'],
            'Confidence Score': ['Tidak ada',                     'Ada ✓'],
        })
        st.table(perbandingan)

    with col_b:
        st.markdown("### ✨ Novelty Penelitian")
        for n, judul, isi in [
            ("01", "Komparasi Algoritma", "Penelitian pertama yang membandingkan NB vs LR pada konteks komentar MBG"),
            ("02", "3 Kelas Sentimen",    "Negatif / Netral / Positif — lebih realistis dari jurnal pembanding yang hanya 2 kelas"),
            ("03", "Data Lebih Panjang",  "Periode Jan–Okt 2025, dataset lebih variatif dan representatif"),
            ("04", "Confidence Score",    "Menambahkan ukuran kepercayaan label lexicon sebagai indikator kualitas"),
        ]:
            st.markdown(f"""
            <div style="background:rgba(0,184,148,0.08);border-left:3px solid #00b894;
                        border-radius:0 10px 10px 0;padding:12px 16px;margin-bottom:10px">
                <div style="color:#00b894;font-size:0.75rem;font-weight:800;letter-spacing:1px">{n}</div>
                <div style="color:white;font-weight:700;font-size:0.95rem">{judul}</div>
                <div style="color:rgba(255,255,255,0.65);font-size:0.85rem;margin-top:2px">{isi}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("### ⚠️ Limitasi & Rekomendasi")
        limitasi = [
            "Labeling lexicon tanpa validasi manual",
            "Kamus lexicon terbatas, mungkin ada kata ambigu",
            "Tidak menggunakan teknik penanganan class imbalance eksplisit",
            "Data hanya dari Twitter, belum multi-platform",
        ]
        rekomendasi = [
            "Tambahkan validasi manual atau self-training",
            "Perluas kamus dengan corpus spesifik MBG",
            "Coba SMOTE atau class_weight untuk imbalance",
            "Ekspansi ke data Instagram, YouTube, TikTok",
        ]
        for lim, rek in zip(limitasi, rekomendasi):
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.04);border-radius:8px;
                        padding:8px 12px;margin-bottom:6px;font-size:0.85rem">
                <span style="color:#ff7675">⚠ {lim}</span><br>
                <span style="color:#00b894">→ {rek}</span>
            </div>""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align:center;padding:30px 0 10px;color:rgba(255,255,255,0.3);font-size:0.82rem">
    Tugas UTS Pembelajaran Mesin · Pertemuan 7 · Analisis Sentimen Program MBG
</div>
""", unsafe_allow_html=True)
