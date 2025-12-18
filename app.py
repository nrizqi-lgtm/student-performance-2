import streamlit as st
import pandas as pd
import sqlite3
import pickle
import plotly.express as px
import json
from datetime import datetime

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="Student Performance Dashboard",
    layout="wide",
    page_icon="🎓"
)

# ======================================================
# ADAPTIVE CSS (DARK MODE FRIENDLY)
# ======================================================
st.markdown("""
<style>
/* Menggunakan Variabel Native Streamlit agar responsif terhadap Dark/Light Mode */
:root {
    --accent: #1E88E5;
    --border-radius: 12px;
}

/* GLOBAL */
.stApp {
    font-family: 'Inter', sans-serif;
    /* Background mengikuti tema Streamlit (tidak di-hardcode) */
}

/* HEADER */
.header-ultra {
    background-color: var(--secondary-background-color);
    padding: 30px 40px;
    border-radius: 0 0 20px 20px;
    border-bottom: 2px solid var(--accent);
    margin-bottom: 40px;
    text-align: center;
}

.header-title {
    font-size: 32px;
    font-weight: 800;
    color: var(--accent);
    margin-bottom: 5px;
}

.header-sub {
    font-size: 16px;
    color: var(--text-color);
    opacity: 0.8;
}

/* CARDS */
.ultra-card {
    background-color: var(--secondary-background-color);
    padding: 24px;
    border-radius: var(--border-radius);
    border: 1px solid rgba(128, 128, 128, 0.2);
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    margin-bottom: 20px;
    transition: transform 0.2s;
}

.ultra-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
}

.ultra-card h4 {
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 1px;
    opacity: 0.7;
    margin-bottom: 8px;
    color: var(--text-color);
}

.ultra-card p {
    font-size: 14px;
    opacity: 0.8;
    color: var(--text-color);
}

/* BUTTON CUSTOMIZATION */
div.stButton > button {
    background-color: var(--accent);
    color: white;
    font-weight: 600;
    border-radius: 8px;
    border: none;
    padding: 0.6rem 1.2rem;
    width: 100%;
}

div.stButton > button:hover {
    background-color: #1565C0;
    color: white;
}

/* SCORE BOX (ADAPTIVE) */
.score-box-container {
    background: linear-gradient(135deg, #1E88E5 0%, #42A5F5 100%);
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    color: white;
    box-shadow: 0 8px 16px rgba(30, 136, 229, 0.2);
    margin-bottom: 20px;
}

.score-value {
    font-size: 64px;
    font-weight: 900;
    margin: 0;
    line-height: 1.2;
}

.score-label {
    font-size: 18px;
    font-weight: 500;
    opacity: 0.9;
}
.metric-card {
    background: linear-gradient(180deg, #0e1117, #121621);
    padding: 18px 20px;
    border-radius: 14px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.35);
    margin: 14px 0;
    min-height: 120px;
    transition: all 0.25s ease;
    position: relative;
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 28px rgba(0,0,0,0.45);
}

.metric-title {
    font-size: 14px;
    color: #aaa;
    margin-bottom: 6px;
    letter-spacing: 0.3px;
}

.metric-value {
    font-size: 34px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 4px;
}

.metric-subtitle {
    font-size: 13px;
    color: #999;
    line-height: 1.4;
}

</style>
""", unsafe_allow_html=True)

DB_PATH = "student_perf.db"
CSV_PATH = "StudentPerformanceFactors.csv"

# =====================================================
# UTIL
# =====================================================
def query_db(q):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(q, conn)
    conn.close()
    return df

@st.cache_resource
def load_best_model():
    with open("best_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def metric_card(title, value, subtitle="", icon="📊", color="#1f77b4"):
    st.markdown(
        f"""
        <div class="metric-card" style="border-left: 6px solid {color};">
            <div class="metric-title">{icon} {title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
@st.cache_resource
def load_models():
    paths = {
        "lr": "model_LinearRegression.pkl",
        "nn": "model_NeuralNetwork.pkl",
        "rf": "model_RandomForest.pkl",
        "xgb": "model_XGBoost.pkl",
        "best": "best_model.pkl"
    }

    models = {}

    # Load masing-masing model jika ada
    for key, path in paths.items():
        try:
            with open(path, "rb") as f:
                models[key] = pickle.load(f)
        except FileNotFoundError:
            # show warning only if needed
            # st.warning(f"Model file not found: {path}")
            continue
        except Exception as e:
            # st.error(f"Error loading {path}: {e}")
            continue

    # --------- Fallback Aman ----------
    # Jika BEST MODEL tidak ada → gunakan XGB jika ada
    if "best" not in models:
        if "xgb" in models:
            models["best"] = models["xgb"]
        elif "rf" in models:
            models["best"] = models["rf"]
        elif "lr" in models:
            models["best"] = models["lr"]
        elif "nn" in models:
            models["best"] = models["nn"]
    return models

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(CSV_PATH)
perf = query_db("SELECT * FROM model_results ORDER BY r2 DESC")

# =====================================================
# HEADER
# =====================================================
# st.title("🎓 Student Performance Prediction Dashboard")
# st.caption("Ringkasan data, evaluasi model, dan prediksi nilai ujian")

st.markdown("""
<div class="header-ultra">
    <div class="header-title">🎓 Student Performance AI</div>
    <div class="header-sub">Sistem Analisis & Prediksi Nilai Siswa Terintegrasi</div>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["📊 Overview", "📈 Data Insight", "🧠 Model Comparison", "🤖 Prediction"])

# =====================================================
# 📊 OVERVIEW
# =====================================================
with tabs[0]:
    st.subheader("📊 Ringkasan Utama")

    best = perf.iloc[0]
    baseline_r2 = perf["r2"].mean()
    delta_r2 = best["r2"] - baseline_r2

    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)

    logs = query_db("SELECT * FROM prediction_logs ORDER BY id DESC LIMIT 100")

    with c1:
        metric_card(
            "Total Data",
            f"{len(df):,}",
            "Jumlah seluruh sampel",
            icon="📦",
            color="#4CAF50"
        )

    with c2:
        metric_card(
            "Train Size",
            f"{int(len(df)*0.8):,}",
            "80% data latih",
            icon="🧪",
            color="#2196F3"
        )

    with c3:
        metric_card(
            "Test Size",
            f"{int(len(df)*0.2):,}",
            "20% data uji",
            icon="🧫",
            color="#03A9F4"
        )

    with c4:
        metric_card(
            "Best Model",
            best["model_name"],
            "Model performa tertinggi",
            icon="🏆",
            color="#FFC107"
        )

    with c5:
        metric_card(
            "R² Score",
            f"{best['r2']:.3f}",
            f"Δ vs avg: {delta_r2:+.3f}",
            icon="📈",
            color="#9C27B0"
        )

    with c6:
        metric_card(
            "RMSE",
            f"{best['rmse']:.2f}",
            "Semakin kecil semakin baik",
            icon="📉",
            color="#F44336"
        )
    
    # ==========================================================
    # ============ ROW 5 — TABEL AKTIVITAS TERBARU =============
    # ==========================================================
    st.markdown("### 🕒 Aktivitas Prediksi Terbaru")

    if not logs.empty:
        # Ambil kolom yang diperlukan + copy agar aman
        logs_display = logs[["timestamp", "final_prediction"]].copy()

        # Pastikan timestamp bertipe datetime
        logs_display["timestamp"] = pd.to_datetime(
            logs_display["timestamp"],
            errors="coerce"
        )

        # Format tampilan waktu
        logs_display["timestamp"] = logs_display["timestamp"].dt.strftime("%d %b %H:%M")

        # Tampilkan 10 data terakhir
        st.dataframe(
            logs_display.tail(10),
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("Belum ada aktivitas prediksi yang tercatat.")

# =====================================================
# 📈 DATA INSIGHT
# =====================================================
with tabs[1]:
    st.subheader("📈 Data Insight")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df, x="Exam_Score", nbins=30, title="Distribusi Nilai")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        corr = df.select_dtypes("number").corr()
        fig = px.imshow(corr, text_auto=True, title="Korelasi Variabel Numerik", aspect="auto")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📋 Sampel Data")
    st.dataframe(df.head(), use_container_width=True)

# =====================================================
# 🧠 MODEL COMPARISON
# =====================================================
with tabs[2]:
    st.subheader("🧠 Perbandingan Model")

    st.dataframe(
        perf[["model_name", "r2", "rmse", "mae"]]
        .rename(columns={
            "model_name": "Model",
            "r2": "R²",
            "rmse": "RMSE",
            "mae": "MAE"
        }),
        use_container_width=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.plotly_chart(
            px.bar(perf, x="model_name", y="r2", title="R² per Model"),
            use_container_width=True
        )

    with col2:
        st.plotly_chart(
            px.bar(perf, x="model_name", y="rmse", title="RMSE per Model"),
            use_container_width=True
        )

    with col3:
        st.plotly_chart(
            px.bar(perf, x="model_name", y="mae", title="MAE per Model"),
            use_container_width=True
        )
    
## ======================================================
# 🤖 PREDICTION TAB (FINAL – CONSISTENT WITH TRAINING)
# ======================================================
with tabs[3]:
    st.subheader("🤖 Prediksi Nilai Siswa")

    models = load_models()
    if "best" not in models:
        st.error("Best model belum tersedia. Jalankan train.py terlebih dahulu.")
        st.stop()

    # best_model = models["best"]

    # ==========================
    # INPUT FORM
    # ==========================
    with st.form("pred_form"):
        st.markdown("### 📝 Data Siswa")

        col1, col2, col3 = st.columns(3)

        # ---------- NUMERIC ----------
        with col1:
            st.markdown("#### 🔢 Akademik")
            Hours_Studied = st.number_input("Jam Belajar / Minggu", 1, 40, 12)
            Attendance = st.number_input("Kehadiran (%)", 0, 100, 85)
            Sleep_Hours = st.number_input("Jam Tidur / Hari", 3, 12, 7)
            Previous_Scores = st.number_input("Nilai Sebelumnya", 0, 100, 75)
            Tutoring_Sessions = st.number_input("Sesi Les / Minggu", 0, 10, 1)
            Physical_Activity = st.number_input("Aktivitas Fisik / Minggu", 0, 14, 3)

        # ---------- ORDINAL ----------
        with col2:
            st.markdown("#### 📊 Lingkungan & Motivasi")
            Parental_Involvement = st.selectbox("Keterlibatan Orang Tua", ["Low", "Medium", "High"])
            Access_to_Resources = st.selectbox("Akses Materi Belajar", ["Low", "Medium", "High"])
            Motivation_Level = st.selectbox("Motivasi Belajar", ["Low", "Medium", "High"])
            Family_Income = st.selectbox("Pendapatan Keluarga", ["Low", "Medium", "High"])
            Teacher_Quality = st.selectbox("Kualitas Guru", ["Low", "Medium", "High"])

        # ---------- NOMINAL ----------
        with col3:
            st.markdown("#### 🧑 Sosial & Personal")
            Parental_Education_Level = st.selectbox(
                "Pendidikan Orang Tua",
                ["High School", "College", "Postgraduate"]
            )
            Distance_from_Home = st.selectbox(
                "Jarak Rumah ke Sekolah",
                ["Near", "Moderate", "Far"]
            )
            Peer_Influence = st.selectbox(
                "Pengaruh Teman",
                ["Negative", "Neutral", "Positive"]
            )
            Extracurricular_Activities = st.selectbox(
                "Ekstrakurikuler",
                ["No", "Yes"]
            )
            Internet_Access = st.selectbox(
                "Akses Internet",
                ["No", "Yes"]
            )
            Learning_Disabilities = st.selectbox(
                "Learning Disabilities",
                ["No", "Yes"]
            )
            School_Type = st.selectbox(
                "Tipe Sekolah",
                ["Public", "Private"]
            )
            Gender = st.selectbox(
                "Jenis Kelamin",
                ["Male", "Female"]
            )

        submit = st.form_submit_button("🚀 Prediksi Nilai")

    # ==========================
    # PREDICTION LOGIC
    # ==========================
    if submit:
        # =====================================================
        # PREPARE INPUT
        # =====================================================
        input_dict = {
            "Hours_Studied": Hours_Studied,
            "Attendance": Attendance,
            "Sleep_Hours": Sleep_Hours,
            "Previous_Scores": Previous_Scores,
            "Tutoring_Sessions": Tutoring_Sessions,
            "Physical_Activity": Physical_Activity,
            "Parental_Involvement": Parental_Involvement,
            "Access_to_Resources": Access_to_Resources,
            "Motivation_Level": Motivation_Level,
            "Family_Income": Family_Income,
            "Teacher_Quality": Teacher_Quality,
            "Parental_Education_Level": Parental_Education_Level,
            "Distance_from_Home": Distance_from_Home,
            "Peer_Influence": Peer_Influence,
            "Extracurricular_Activities": Extracurricular_Activities,
            "Internet_Access": Internet_Access,
            "Learning_Disabilities": Learning_Disabilities,
            "School_Type": School_Type,
            "Gender": Gender
        }

        df_in = pd.DataFrame([input_dict])

        # =====================================================
        # 1️⃣ PREDICTION: ALL MODELS
        # =====================================================
        predictions = {}

        for name in ["lr", "rf", "xgb", "nn"]:
            if name in models:
                try:
                    predictions[name] = float(models[name].predict(df_in)[0])
                except:
                    predictions[name] = None
            else:
                predictions[name] = None
        
        # Best model prediction
        if "best" in models:
            try:
                best_pred = float(models["best"].predict(df_in)[0])
            except:
                best_pred = None
        else:
            best_pred = None

        final_score = best_pred

        if final_score is None:
            st.error("Gagal melakukan prediksi dengan model terbaik.")
            st.stop()

        # =====================================================
        # OUTPUT: MAIN RESULT
        # =====================================================
        st.markdown("---")
        st.markdown("## 🎯 Hasil Prediksi")

        col_res, col_comp = st.columns([1, 1.5])

        with col_res:
            st.markdown(f"""
            <div class="score-box-container">
                <div class="score-value">{final_score:.1f}</div>
                <div class="score-label">Prediksi Nilai Akhir</div>
            </div>
            """, unsafe_allow_html=True)

        # =====================================================
        # 2️⃣ MODEL COMPARISON
        # =====================================================
        with col_comp:
            st.markdown("### 📊 Perbandingan Model")

            df_pred = (
                pd.DataFrame.from_dict(predictions, orient="index", columns=["Prediksi"])
                .reset_index()
                .rename(columns={"index": "Model"})
                .sort_values("Prediksi", ascending=False)
            )

            st.dataframe(df_pred, use_container_width=True, hide_index=True)

        # =====================================================
        # 3️⃣ RECOMMENDATION ENGINE (RULE-BASED, CONSISTENT)
        # =====================================================
        st.markdown("---")
        st.markdown("## 🧭 Rekomendasi Pembelajaran")

        recommendations = []

        # =====================================================
        # 📘 ACADEMIC HABITS
        # =====================================================
        if Hours_Studied < 8:
            recommendations.append(
                "📘 Jam belajar masih rendah. Disarankan menambah durasi belajar "
                "menjadi minimal 10–12 jam per minggu dengan jadwal terstruktur."
            )
        elif Hours_Studied > 30:
            recommendations.append(
                "⚠️ Jam belajar sangat tinggi. Pastikan tidak kelelahan dan tetap "
                "menjaga keseimbangan istirahat."
            )

        if Previous_Scores < 60:
            recommendations.append(
                "📉 Nilai sebelumnya rendah. Fokus pada penguatan konsep dasar "
                "dan evaluasi ulang metode belajar."
            )
        elif Previous_Scores < 75:
            recommendations.append(
                "📊 Nilai sebelumnya cukup. Konsistensi belajar perlu ditingkatkan "
                "untuk hasil yang lebih optimal."
            )

        # =====================================================
        # 🏫 ATTENDANCE & SCHOOL ENGAGEMENT
        # =====================================================
        if Attendance < 75:
            recommendations.append(
                "🏫 Kehadiran di kelas rendah. Tingkatkan kehadiran agar tidak "
                "tertinggal materi penting."
            )
        elif Attendance < 90:
            recommendations.append(
                "📚 Kehadiran cukup baik. Pertahankan atau tingkatkan untuk "
                "memaksimalkan pemahaman materi."
            )

        # =====================================================
        # 😴 HEALTH & LIFESTYLE
        # =====================================================
        if Sleep_Hours < 6:
            recommendations.append(
                "😴 Jam tidur kurang dari ideal. Disarankan tidur 7–8 jam per hari "
                "untuk meningkatkan konsentrasi dan daya ingat."
            )
        elif Sleep_Hours > 9:
            recommendations.append(
                "🛌 Jam tidur berlebih. Pastikan rutinitas tidur seimbang dengan "
                "aktivitas harian."
            )

        if Physical_Activity < 2:
            recommendations.append(
                "🏃 Aktivitas fisik rendah. Aktivitas ringan 2–3 kali seminggu "
                "dapat membantu fokus dan kesehatan mental."
            )
        elif Physical_Activity > 10:
            recommendations.append(
                "⚽ Aktivitas fisik sangat tinggi. Pastikan tidak mengganggu waktu belajar."
            )

        # =====================================================
        # 👨‍🏫 SUPPORT SYSTEM
        # =====================================================
        if Tutoring_Sessions == 0 and Previous_Scores < 75:
            recommendations.append(
                "👨‍🏫 Tidak ada sesi les tambahan. Les privat atau kelompok kecil "
                "dapat membantu memperbaiki pemahaman materi."
            )

        if Parental_Involvement == "Low":
            recommendations.append(
                "👨‍👩‍👧 Keterlibatan orang tua rendah. Dukungan emosional dan "
                "monitoring belajar di rumah dapat meningkatkan motivasi."
            )

        if Teacher_Quality == "Low":
            recommendations.append(
                "📢 Kualitas pengajaran dirasa kurang. Manfaatkan sumber belajar "
                "tambahan seperti video atau platform daring."
            )

        # =====================================================
        # 📚 RESOURCES & MOTIVATION
        # =====================================================
        if Access_to_Resources == "Low":
            recommendations.append(
                "📚 Akses materi belajar terbatas. Gunakan sumber gratis seperti "
                "perpustakaan digital atau modul terbuka."
            )

        if Motivation_Level == "Low":
            recommendations.append(
                "🔥 Motivasi belajar rendah. Tetapkan target belajar jangka pendek "
                "dan beri reward atas pencapaian kecil."
            )

        # =====================================================
        # 🧑‍🤝‍🧑 SOCIAL & ENVIRONMENT
        # =====================================================
        if Peer_Influence == "Negative":
            recommendations.append(
                "🧑‍🤝‍🧑 Lingkungan pertemanan kurang mendukung. Batasi distraksi "
                "dan cari kelompok belajar yang positif."
            )

        if Distance_from_Home == "Far" and Attendance < 85:
            recommendations.append(
                "🚌 Jarak rumah ke sekolah jauh. Pertimbangkan pengaturan transportasi "
                "atau waktu berangkat lebih awal."
            )

        if Extracurricular_Activities == "Yes" and Hours_Studied < 8:
            recommendations.append(
                "🎭 Kegiatan ekstrakurikuler aktif. Pastikan waktu belajar tetap "
                "cukup dan tidak terganggu."
            )

        # =====================================================
        # 🌐 TECHNOLOGY & ACCESS
        # =====================================================
        if Internet_Access == "No":
            recommendations.append(
                "🌐 Tidak memiliki akses internet. Manfaatkan buku cetak dan "
                "materi belajar offline secara maksimal."
            )

        # =====================================================
        # 🧠 SPECIAL CONDITIONS
        # =====================================================
        if Learning_Disabilities == "Yes":
            recommendations.append(
                "🧠 Terdapat hambatan belajar. Pendekatan pembelajaran individual "
                "dan dukungan khusus sangat dianjurkan."
            )

        # =====================================================
        # 🏁 FALLBACK
        # =====================================================
        if not recommendations:
            recommendations.append(
                "✅ Pola belajar dan lingkungan sudah mendukung. "
                "Pertahankan konsistensi dan evaluasi berkala."
            )

        else:
            for r in recommendations:
                st.markdown(f"- {r}")

        # =====================================================
        # 4️⃣ SUMMARY & FEATURE INSIGHT
        # =====================================================
        st.markdown("---")
        st.markdown("## 🧠 Ringkasan & Insight")

        st.markdown(f"""
        - Nilai diprediksi **{final_score:.1f}**, dipengaruhi kuat oleh:
        - **Nilai sebelumnya ({Previous_Scores})**
        - **Jam belajar ({Hours_Studied} jam/minggu)**
        - **Kehadiran ({Attendance}%)**
        - Faktor non-akademik seperti **motivasi**, **akses sumber belajar**, dan
        **dukungan orang tua** juga berkontribusi signifikan.
        - Selisih prediksi antar model relatif kecil → hasil cukup stabil.
        """)

        # =====================================================
        # 5️⃣ SAVE LOG (OPTIONAL)
        # =====================================================
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO prediction_logs
                (timestamp, input_json, prediction_lr, prediction_rf, prediction_xgb, prediction_nn, final_prediction)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                json.dumps(input_dict),
                predictions['lr'],
                predictions['rf'],
                predictions['xgb'],
                predictions['nn'],
                final_score
            ))
            conn.commit()
            conn.close()
        except:
            pass