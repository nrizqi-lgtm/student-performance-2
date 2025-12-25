# ============================================================
# STREAMLIT APP – STUDENT PERFORMANCE PREDICTION
# Clean | Stable | No Sidebar
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import pickle
import json
import os
from datetime import datetime
import plotly.express as px

# ============================================================
# CONFIG
# ============================================================

DB_PATH = "student_perf.db"
MODEL_DIR = "models"
DATA_PATH = "StudentPerformanceFactors.csv"
TARGET_COL = "Exam_Score"

st.set_page_config(
    page_title="Student Performance Prediction",
    layout="wide"
)

# ============================================================
# DB UTIL
# ============================================================

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def query_db(query, params=None):
    conn = get_conn()
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def insert_prediction_log(input_dict, preds, ensemble):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO prediction_logs
        (timestamp, input_json,
         prediction_lr, prediction_rf,
         prediction_xgb, prediction_nn,
         ensemble_prediction)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        json.dumps(input_dict),
        preds.get("lr"),
        preds.get("rf"),
        preds.get("xgb"),
        preds.get("nn"),
        ensemble
    ))
    conn.commit()
    conn.close()

# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():
    files = {
        "lr": "model_LinearRegression.pkl",
        "rf": "model_RandomForest.pkl",
        "xgb": "model_XGBoost.pkl",
        "nn": "model_NeuralNetwork.pkl"
    }
    models = {}
    for k, f in files.items():
        path = os.path.join(MODEL_DIR, f)
        if os.path.exists(path):
            with open(path, "rb") as fh:
                models[k] = pickle.load(fh)
    return models

models = load_models()

@st.cache_data
def load_training_summary():
    return query_db("""
        SELECT model_name,
               AVG(r2) AS mean_r2,
               AVG(rmse) AS mean_rmse,
               AVG(mae)  AS mean_mae,
               COUNT(*) AS n_runs
        FROM training_log
        GROUP BY model_name
    """)

summary_df = load_training_summary()

@st.cache_data
def load_dataset():
    if not os.path.exists(DATA_PATH):
        return None, None

    df = pd.read_csv(DATA_PATH)

    if TARGET_COL in df.columns:
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]
    else:
        X = df
        y = None

    return X, y

# ============================================================
# LOAD DATASET (for overview only)
# ============================================================

# if os.path.exists(DATA_PATH):
#     df_data = pd.read_csv(DATA_PATH)
#     total_samples = len(df_data)
# else:
#     df_data = None
#     total_samples = 0

# ============================================================
# HELPER UI
# ============================================================

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

def info_card(title, value, subtitle="", color="#1E88E5"):
    st.markdown(
        f"""
        <div class="metric-card" style="border-left: 6px solid {color};">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# HEADER
# ============================================================

st.title("🎓 Student Performance Prediction System")
st.markdown("---")

# ============================================================
# TABS
# ============================================================

tabs = st.tabs([
    "🏠 Dashboard",
    "🧠 Model Comparison",
    "🤖 Prediction",
    "📜 Logs"
])

# ============================================================
# 🏠 DASHBOARD
# ============================================================

with tabs[0]:
    st.subheader("🏠 System Overview")

    # ==========================
    # SAFETY CHECK
    # ==========================
    if summary_df.empty:
        st.warning("⚠️ Belum ada hasil training. Jalankan `train.py` terlebih dahulu.")
        st.stop()

    # ==========================
    # METADATA
    # ==========================
    best_row = summary_df.sort_values("mean_r2", ascending=False).iloc[0]

    total_models = len(summary_df)
    total_runs = int(summary_df["n_runs"].sum())

    X_data, y_data = load_dataset()

    # dataset info (aman)
    if X_data is not None:
        total_samples = len(X_data)
        n_features = X_data.shape[1]
    else:
        total_samples = "N/A"
        n_features = "N/A"

    # last training time
    last_train_df = query_db(
        "SELECT MAX(timestamp) AS last_ts FROM training_log"
    )
    last_train = (
        last_train_df["last_ts"].iloc[0]
        if not last_train_df.empty else "N/A"
    )

    # ==========================
    # INFO CARDS
    # ==========================
    col1, col2, col3 = st.columns(3)
    col4, col5 = st.columns(2)

    with col1:
        info_card(
            "📦 Total Data",
            f"{total_samples:,}" if total_samples != "N/A" else "N/A",
            f"{n_features} Features | Target: Exam_Score",
            "#4CAF50"
        )

    with col2:
        info_card(
            "🧠 Models",
            total_models,
            "Linear | Tree | Boosting | Neural Net",
            "#2196F3"
        )

    with col3:
        info_card(
            "🏆 Best Model",
            best_row["model_name"],
            f"Avg R² = {best_row['mean_r2']:.3f} | {int(best_row['n_runs'])} runs",
            "#FFC107"
        )

    with col4:
        info_card(
            "🔁 Training Runs",
            total_runs,
            "Randomized retraining",
            "#9C27B0"
        )

    with col5:
        info_card(
            "⏱ Last Training",
            last_train.split("T")[0] if last_train != "N/A" else "N/A",
            "Model freshness",
            "#FF7043"
        )

    st.markdown("---")

    # ==========================
    # PERFORMANCE SUMMARY
    # ==========================
    st.markdown("### 📈 Average Model Performance (R²)")

    st.plotly_chart(
        px.bar(
            summary_df.sort_values("mean_r2", ascending=False),
            x="model_name",
            y="mean_r2",
            text_auto=".3f",
            labels={
                "model_name": "Model",
                "mean_r2": "Average R²"
            },
            title="Average Cross-Run R² per Model"
        ),
        use_container_width=True
    )

    # ==========================
    # SYSTEM INSIGHT
    # ==========================
    st.markdown("### 📌 System Insight")

    st.markdown(f"""
    - Sistem menggunakan **{total_models} model Machine Learning** dengan skema
    **repeated randomized training**.
    - Model terbaik saat ini adalah **{best_row['model_name']}**
    dengan **rata-rata R² = {best_row['mean_r2']:.3f}** dari
    **{int(best_row['n_runs'])} kali training**.
    - Total **{total_runs} run training** tersimpan dan dapat diaudit ulang.
    - Sistem siap digunakan untuk **ensemble prediction & analisis stabilitas**.
    """)

# ============================================================
# 🧠 MODEL COMPARISON
# ============================================================

with tabs[1]:
    st.subheader("🧠 Model Performance Comparison")

    # =====================================================
    # MODEL PERFORMANCE TABLE + METRICS
    # =====================================================
    if summary_df.empty:
        st.info("Belum ada data training.")
    else:
        st.dataframe(summary_df, use_container_width=True)

        # ============================
        # Row 1: R², RMSE, MAE
        # ============================
        col1, col2, col3 = st.columns(3)

        with col1:
            st.plotly_chart(
                px.bar(
                    summary_df,
                    x="model_name",
                    y="mean_r2",
                    text_auto=".3f",
                    title="R² Comparison"
                ),
                use_container_width=True
            )

        with col2:
            st.plotly_chart(
                px.bar(
                    summary_df,
                    x="model_name",
                    y="mean_rmse",
                    text_auto=".2f",
                    title="RMSE Comparison"
                ),
                use_container_width=True
            )

        with col3:
            st.plotly_chart(
                px.bar(
                    summary_df,
                    x="model_name",
                    y="mean_mae",
                    text_auto=".2f",
                    title="MAE Comparison"
                ),
                use_container_width=True
            )

    st.markdown("---")
    st.subheader("📌 Korelasi Variabel & Distribusi Nilai")

    # =====================================================
    # LOAD DATASET
    # =====================================================
    X, y = load_dataset()

    if X is None:
        st.warning("Dataset tidak ditemukan.")
        st.stop()

    df_full = X.copy()
    if y is not None:
        df_full[TARGET_COL] = y

    numeric_cols = [
        "Hours_Studied", "Attendance", "Sleep_Hours",
        "Previous_Scores", "Tutoring_Sessions",
        "Physical_Activity", TARGET_COL
    ]

    corr_df = df_full[numeric_cols].corr()

    # =====================================================
    # Row 2: Correlation & Distribution (SIDE BY SIDE)
    # =====================================================
    col_corr, col_dist = st.columns(2)

    with col_corr:
        st.plotly_chart(
            px.imshow(
                corr_df,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Correlation Heatmap (Numerical Variables)"
            ),
            use_container_width=True
        )

    with col_dist:
        st.plotly_chart(
            px.histogram(
                df_full,
                x=TARGET_COL,
                nbins=30,
                marginal="box",
                title="Distribusi Nilai Siswa"
            ),
            use_container_width=True
        )

    # =====================================================
    # AUTOMATIC INSIGHT
    # =====================================================
    st.markdown("---")
    st.subheader("🧠 Insight Otomatis")

    # Correlation to target
    target_corr = (
        corr_df[TARGET_COL]
        .drop(TARGET_COL)
        .sort_values(key=abs, ascending=False)
    )

    st.markdown("**🔗 Variabel dengan korelasi tertinggi terhadap nilai:**")

    for feat, val in target_corr.items():
        emoji = "📈" if val > 0 else "📉"
        st.markdown(
            f"- {emoji} **{feat}** → korelasi `{val:.2f}`"
        )

    # Multicollinearity detection
    high_corr_pairs = (
        corr_df.abs()
        .where(lambda x: (x > 0.7) & (x < 1))
        .stack()
        .reset_index()
    )

    if high_corr_pairs.empty:
        st.success("✅ Tidak ditemukan multikolinearitas tinggi antar variabel.")
    else:
        st.warning("⚠️ Terdapat korelasi tinggi antar beberapa variabel:")
        for _, row in high_corr_pairs.iterrows():
            st.markdown(
                f"- **{row['level_0']}** ↔ **{row['level_1']}** "
                f"(corr = {row[0]:.2f})"
            )
# ============================================================
# 🤖 PREDICTION
# ============================================================

with tabs[2]:
    st.subheader("🤖 Prediksi Nilai Siswa")

    models = load_models()
    summary_df = load_training_summary()

    if not models or summary_df.empty:
        st.error("❌ Model belum tersedia. Jalankan train.py terlebih dahulu.")
        st.stop()

    best_model_name = (
        summary_df.sort_values("mean_r2", ascending=False)
        .iloc[0]["model_name"]
    )

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
        
        predictions = {}
        weights = {}

        for key, model in models.items():
            try:
                pred = float(model.predict(df_in)[0])
                predictions[key] = pred

                # mapping key → model_name DB
                map_name = {
                    "lr": "LinearRegression",
                    "rf": "RandomForest",
                    "xgb": "XGBoost",
                    "nn": "NeuralNetwork"
                }

                if key in map_name:
                    r2_row = summary_df[
                        summary_df["model_name"] == map_name[key]
                    ]
                    if not r2_row.empty:
                        weights[key] = float(r2_row["mean_r2"].iloc[0])

            except:
                predictions[key] = None
            
            valid_preds = {k: v for k, v in predictions.items() if v is not None}

            if not valid_preds:
                st.error("Gagal melakukan prediksi.")
                st.stop()

            if weights:
                final_score = sum(
                    valid_preds[k] * weights.get(k, 0)
                    for k in valid_preds
                ) / sum(weights.values())
            else:
                final_score = np.mean(list(valid_preds.values()))
            
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
        
        std_pred = np.std(list(valid_preds.values()))
        ci_low = final_score - 1.96 * std_pred
        ci_high = final_score + 1.96 * std_pred

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
                (timestamp, input_json,
                prediction_lr, prediction_rf,
                prediction_xgb, prediction_nn,
                final_prediction)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                json.dumps(input_dict),
                predictions.get("lr"),
                predictions.get("rf"),
                predictions.get("xgb"),
                predictions.get("nn"),
                final_score
            ))
            conn.commit()
            conn.close()
        except:
            pass

# ============================================================
# 📜 LOGS
# ============================================================

with tabs[3]:
    st.subheader("📜 System Logs")

    st.markdown("### 🧠 Training Logs")
    st.dataframe(query_db("SELECT * FROM training_log ORDER BY timestamp DESC LIMIT 100"))

    st.markdown("### 🤖 Prediction Logs")
    st.dataframe(query_db("SELECT * FROM prediction_logs ORDER BY timestamp DESC LIMIT 100"))
