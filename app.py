import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import json
import pickle
import shap
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(
    page_title="Student Performance Prediction",
    layout="wide",
    page_icon="🎓"
)

# ======================================================
# ULTRA BRIGHT BLUE CSS
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

</style>
""", unsafe_allow_html=True)

# ======================================================
# LOAD MODELS
# ======================================================
DB_PATH = "student_perf.db"
CSV_PATH = "StudentPerformanceFactors.csv"
# =====================================
# LOAD CSV SEKALI SAJA DI AWAL APP
# =====================================
MODEL_FEATURES = [
    "Hours_Studied", "Attendance", "Sleep_Hours",
    "Tutoring_Sessions", "Previous_Scores",
    "Access_to_Resources", "Family_Income",
    "Exam_Score"  # target
]

try:
    df_full = pd.read_csv(CSV_PATH)
    df_model = df_full[[c for c in MODEL_FEATURES if c in df_full.columns]]
except:
    st.error("CSV tidak ditemukan.")
    df_full = pd.DataFrame()
    df_model = pd.DataFrame()


@st.cache_resource

def load_models():
    paths = {
        "lr": "model_LinearRegression.pkl",
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

    return models

def query_db(q):
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(q, conn)
        conn.close()
        return df
    except:
        return pd.DataFrame()

# ======================================================
# HEADER
# ======================================================
st.markdown("""
<div class="header-ultra">
    <div class="header-title">🎓 Student Performance Prediction</div>
    <div class="header-sub">Sistem Analisis dan Prediksi Nilai Ujian</div>
</div>
""", unsafe_allow_html=True)

# ======================================================
# TABS
# ======================================================
tab_dash, tab_data, tab_model, tab_pred = st.tabs([
    "📊 Dashboard ",
    "📁 Analisis Data",
    "🧠 Analisis Model",
    "🤖 Prediksi Individu"
])

# ======================================================
# ======================= DASHBOARD ======================
# ======================================================
with tab_dash:

    st.subheader("📊 Dashboard")

    # ==========================================================
    # LOAD DATA FROM DB
    # ==========================================================
    meta = query_db("SELECT * FROM meta_info")
    shap_df = query_db("SELECT * FROM shap_summary ORDER BY shap_value DESC LIMIT 10")
    logs = query_db("SELECT * FROM prediction_logs ORDER BY id DESC LIMIT 100")

    # Load CSV
    df_raw = df_model.copy()

    # Convert meta to dictionary
    meta_dict = {m["key"]: float(m["value"]) if str(m["value"]).replace('.','',1).isdigit() else m["value"]
                 for _, m in meta.iterrows()} if not meta.empty else {}

    # ==========================================================
    # ============ ROW 1 — NATIONAL INSIGHT CARDS ==============
    # ==========================================================
    colA, colB, colC, colD = st.columns(4)

    with colA:
        st.markdown(f"""
        <div class="ultra-card">
            <h4>Total Data</h4>
            <h2 style='color:#1E88E5'>{meta_dict.get("total_data","0")}</h2>
            <p>Jumlah total data akademik .</p>
        </div>
        """, unsafe_allow_html=True)

    with colB:
        st.markdown(f"""
        <div class="ultra-card">
            <h4>Data Latih</h4>
            <h2 style='color:#1E88E5'>{meta_dict.get("train_data","0")}</h2>
            <p>Data digunakan untuk melatih model AI.</p>
        </div>
        """, unsafe_allow_html=True)

    with colC:
        st.markdown(f"""
        <div class="ultra-card">
            <h4>Rasio Latih (%)</h4>
            <h2 style='color:#1E88E5'>{float(meta_dict.get("train_ratio",0))*100:.1f}%</h2>
            <p>Proporsi data training.</p>
        </div>
        """, unsafe_allow_html=True)

    with colD:
        pred_count = len(logs)
        st.markdown(f"""
        <div class="ultra-card">
            <h4>Total Prediksi</h4>
            <h2 style='color:#1E88E5'>{pred_count}</h2>
            <p>Aktivitas prediksi yang tercatat.</p>
        </div>
        """, unsafe_allow_html=True)


    st.markdown("---")

    # ==========================================================
    # ============ ROW 2 — DISTRIBUSI NILAI  ===========
    # ==========================================================
    st.markdown("### 🎯 Distribusi Nilai")

    col1, col2 = st.columns(2)

    if "Exam_Score" in df_raw:

        with col1:
            fig1 = px.histogram(df_raw, x="Exam_Score", nbins=30,
                                color_discrete_sequence=["#1E88E5"])
            fig1.update_layout(title="Distribusi Nilai")
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.box(df_raw, y="Exam_Score",
                          color_discrete_sequence=["#1E88E5"])
            fig2.update_layout(title="Boxplot Nilai")
            st.plotly_chart(fig2, use_container_width=True)

    else:
        st.info("Kolom 'Exam_Score' tidak ditemukan di CSV.")

    st.markdown("---")

    # ==========================================================
    # ============ ROW 3 — SHAP NATIONAL IMPACT ===============
    # ==========================================================
    st.markdown("### 🧠 Faktor Paling Berpengaruh (SHAP)")

    col3, col4 = st.columns(2)

    with col3:
        if not shap_df.empty:
            fig_shap = px.bar(
                shap_df,
                x="shap_value", y="feature",
                orientation="h",
                color="shap_value",
                color_continuous_scale="Blues"
            )
            fig_shap.update_layout(title="Top 10 Fitur Paling Berpengaruh")
            st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.warning("SHAP belum tersedia. Jalankan training.py terlebih dahulu.")
        
        if not shap_df.empty:
            top_f = shap_df.iloc[0]
            st.markdown(f"""
            <div class="ultra-card">
                <h4>🔥 Faktor Dominan</h4>
                <h2 style='color:#1E88E5'>{top_f["feature"]}</h2>
                <p>Fitur ini memiliki pengaruh terbesar terhadap nilai siswa.</p>
            </div>
            """, unsafe_allow_html=True)

    with col4:
        if "Hours_Studied" in df_raw.columns:
            fig_corr = px.scatter(
                df_raw, x="Hours_Studied", y="Exam_Score",
                trendline="ols",
                color_discrete_sequence=["#1E88E5"]
            )
            fig_corr.update_layout(title="Hubungan Jam Belajar vs Nilai")
            st.plotly_chart(fig_corr, use_container_width=True)
        
        if "Sleep_Hours" in df_raw.columns:
            fig_corr = px.scatter(
                df_raw, x="Sleep_Hours", y="Exam_Score",
                trendline="ols",
                color_discrete_sequence=["#1E88E5"]
            )
            fig_corr.update_layout(title="Hubungan Jam Tidur vs Nilai")
            st.plotly_chart(fig_corr, use_container_width=True)


    st.markdown("---")

    # ==========================================================
    # ============ ROW 4 — TREND PREDIKSI  =============
    # ==========================================================
    st.markdown("### 📈 Tren Prediksi")

    if not logs.empty:
        logs["timestamp"] = pd.to_datetime(logs["timestamp"])

        fig_log = px.line(logs, x="timestamp", y="final_prediction",
                          markers=True, color_discrete_sequence=["#1E88E5"])
        fig_log.update_layout(title="Tren Nilai Prediksi dari Waktu ke Waktu")

        st.plotly_chart(fig_log, use_container_width=True)

    else:
        st.info("Belum ada aktivitas prediksi.")

    st.markdown("---")

    # ==========================================================
    # ============ ROW 5 — TABEL AKTIVITAS TERBARU =============
    # ==========================================================
    st.markdown("### 🕒 Aktivitas Prediksi Terbaru")

    if not logs.empty:
        logs_display = logs[["timestamp", "final_prediction"]]
        logs_display["timestamp"] = logs_display["timestamp"].dt.strftime("%d %b %H:%M")
        st.dataframe(logs_display.tail(10),
                     hide_index=True, use_container_width=True)
    else:
        st.info("Belum ada aktivitas prediksi yang tercatat.")

# ======================================================
# ======================= DATA TAB ======================
# ======================================================
with tab_data:

    st.subheader("📁 Analisis Dataset")

    # ------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------
    df = df_model.copy()

    # ------------------------------------------------------
    # ROW 1 — CARD SUMMARY
    # ------------------------------------------------------
    colA, colB, colC, colD = st.columns(4)

    if not df.empty:

        with colA:
            st.markdown(f"""
            <div class="ultra-card">
                <h4>Total Baris</h4>
                <h2 style='color:#1E88E5'>{len(df)}</h2>
                <p>Jumlah total sampel siswa.</p>
            </div>
            """, unsafe_allow_html=True)

        with colB:
            st.markdown(f"""
            <div class="ultra-card">
                <h4>Fitur</h4>
                <h2 style='color:#1E88E5'>{len(df.columns)-1}</h2>
                <p>Total fitur input model.</p>
            </div>
            """, unsafe_allow_html=True)

        with colC:
            st.markdown(f"""
            <div class="ultra-card">
                <h4>Nilai Rata-rata</h4>
                <h2 style='color:#1E88E5'>{df['Exam_Score'].mean():.1f}</h2>
                <p>Rata-rata nilai siswa .</p>
            </div>
            """, unsafe_allow_html=True)

        with colD:
            missing = df.isnull().sum().sum()
            st.markdown(f"""
            <div class="ultra-card">
                <h4>Missing Value</h4>
                <h2 style='color:#E53935'>{missing}</h2>
                <p>Kekosongan data terdeteksi.</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ------------------------------------------------------
    # ROW 2 — STATISTIK DESKRIPTIF & ANALISIS CATEGORY
    # ------------------------------------------------------
    st.markdown("##### 📌 Sampel Data")
    st.dataframe(df_model.head(), use_container_width=True)

    st.markdown("### 📌 Statistik Deskriptif")
    st.dataframe(df.describe(), use_container_width=True)

    st.markdown("---")

    st.markdown("### 📊 Distribusi Variabel Kategorikal")

    cat_cols = df.select_dtypes(include="object").columns.tolist()

    if len(cat_cols) == 0:
        st.info("Tidak ada kolom kategorikal dalam dataset.")
    else:
        # Hitung jumlah baris grid
        rows = (len(cat_cols) + 2) // 3
        index = 0

        for r in range(rows):
            colA, colB, colC = st.columns(3)
            cols = [colA, colB, colC]

            for c in cols:
                if index < len(cat_cols):
                    colname = cat_cols[index]
                    fig_cat = px.pie(df, names=colname, title=f"Distribusi {colname}")
                    c.plotly_chart(fig_cat, use_container_width=True)
                    index += 1

    st.markdown("---")

    # ------------------------------------------------------
    # ROW 3 — KORELASI (2 grafik berdampingan)
    # ------------------------------------------------------
    st.markdown("### 🔗 Korelasi Antar Variabel")

    numeric_cols = df.select_dtypes(include=['int64','float64']).columns

    col3, col4 = st.columns(2)

    with col3:
        fig_corr = px.imshow(df[numeric_cols].corr(),
                             text_auto=True, aspect="auto",
                             color_continuous_scale="Blues")
        fig_corr.update_layout(title="Heatmap Korelasi")
        st.plotly_chart(fig_corr, use_container_width=True)

    with col4:
        fig_pair = px.scatter_matrix(df, dimensions=numeric_cols,
                                     color_discrete_sequence=["#1E88E5"])
        fig_pair.update_layout(title="Scatter Matrix")
        st.plotly_chart(fig_pair, use_container_width=True)

    st.markdown("---")

    # ------------------------------------------------------
    # ROW 4 — DISTRIBUSI TIAP FITUR (GRID 3 KOLOM)
    # ------------------------------------------------------
    st.markdown("### 📉 Distribusi Tiap Variabel Numerik")

    num_cols = df.select_dtypes(include='number').columns
    rows = (len(num_cols) + 2) // 3
    cols_per_row = 3

    index = 0
    for r in range(rows):
        cA, cB, cC = st.columns(3)
        col_list = [cA, cB, cC]

        for c in col_list:
            if index < len(num_cols):
                colname = num_cols[index]
                fig = px.histogram(df, x=colname, nbins=30,
                                   color_discrete_sequence=["#1E88E5"])
                fig.update_layout(title=f"Distribusi {colname}")
                c.plotly_chart(fig, use_container_width=True)
                index += 1

    st.markdown("---")

    # ------------------------------------------------------
    # ROW 5 — AUTO INSIGHTS (READABLE VERSION)
    # ------------------------------------------------------
    st.markdown("### 🤖 Insight (Statistical Intelligence)")

    insights = []

    # Pola jam belajar
    if "Hours_Studied" in df.columns:
        corr = df["Hours_Studied"].corr(df["Exam_Score"])
        if corr > 0.4:
            insights.append({
                "title": "Jam Belajar Mempengaruhi Nilai",
                "detail": f"Ada korelasi kuat antara jam belajar dan nilai ujian (r = {corr:.2f}). "
                        f"Semakin tinggi jam belajar, semakin besar kecenderungan nilai meningkat.",
                "icon": "📘"
            })
        else:
            insights.append({
                "title": "Jam Belajar Kurang Signifikan",
                "detail": f"Korelasi jam belajar terhadap nilai relatif lemah (r = {corr:.2f}). "
                        f"Faktor lain tampaknya lebih berpengaruh.",
                "icon": "📘"
            })

    # Kehadiran
    if "Attendance" in df.columns:
        corr_att = df["Attendance"].corr(df["Exam_Score"])
        insights.append({
            "title": "Kehadiran dan Konsistensi Belajar",
            "detail": f"Kehadiran memiliki korelasi {corr_att:.2f} dengan nilai. "
                    f"Ini menunjukkan kehadiran yang baik cenderung membantu peningkatan nilai.",
            "icon": "🕒"
        })

    # Tidur
    if "Sleep_Hours" in df.columns:
        corr_sleep = df["Sleep_Hours"].corr(df["Exam_Score"])
        insights.append({
            "title": "Durasi Tidur",
            "detail": f"Durasi tidur memiliki korelasi {corr_sleep:.2f} terhadap nilai. "
                    f"Ini mengindikasikan bahwa pola istirahat berperan bagi performa akademik.",
            "icon": "😴"
        })

    # ------------------------------------------------------
    # DISPLAY INSIGHTS — 2 CARDS PER ROW
    # ------------------------------------------------------
    if insights:
        rows = (len(insights) + 1) // 2
        index = 0

        for _ in range(rows):
            colA, colB = st.columns(2)
            cols = [colA, colB]

            for c in cols:
                if index < len(insights):
                    item = insights[index]
                    c.markdown(f"""
                    <div class="ultra-card" style="min-height:150px;">
                        <h4>{item['icon']} {item['title']}</h4>
                        <p style="opacity:0.8; margin-top:8px;">{item['detail']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    index += 1

    else:
        st.info("Tidak ada insight otomatis yang tersedia untuk dataset ini.")


# ======================================================
# ================ MODEL ANALYSIS =======================
# ======================================================
with tab_model:

    st.subheader("🧠 Model Performance")

    # =====================================
    # LOAD DATA FROM DB
    # =====================================
    perf = query_db("SELECT * FROM model_results ORDER BY r2 DESC")
    params = query_db("SELECT * FROM model_params")
    cv = query_db("SELECT * FROM cv_scores")
    residuals = query_db("SELECT * FROM residuals")

    if perf.empty:
        st.warning("Model belum ditraining. Jalankan training.py terlebih dahulu.")
        st.stop()

    # ============================================================
    # 📌 CARD SECTION — SUMMARY OF BEST MODEL
    # ============================================================
    best_row = perf.iloc[0]
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class='ultra-card'>
            <h3>🏆 Best Model</h3>
            <h2 style='color:#1E88E5'>{}</h2>
            <p>Dipilih berdasarkan nilai R² tertinggi.</p>
        </div>
        """.format(best_row["model_name"]), unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class='ultra-card'>
            <h3>📈 R² Score</h3>
            <h2 style='color:#1E88E5'>{:.3f}</h2>
            <p>Semakin mendekati 1 semakin baik.</p>
        </div>
        """.format(best_row["r2"]), unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class='ultra-card'>
            <h3>📉 RMSE</h3>
            <h2 style='color:#E53935'>{:.2f}</h2>
            <p>Error absolut rata-rata model.</p>
        </div>
        """.format(best_row["rmse"]), unsafe_allow_html=True)

    st.markdown("---")

    # ============================================================
    # 📌 ROW 1 — METRIC COMPARISON CHARTS
    # ============================================================
    st.markdown("### 📊 Perbandingan Metrik Utama Per Model")

    colA, colB, colC = st.columns(3)

    # --- R2 ---
    with colA:
        fig_r2 = px.bar(perf, x="model_name", y="r2",
                        color="r2", text="r2",
                        color_continuous_scale="Blues")
        st.plotly_chart(fig_r2, use_container_width=True)

    # --- RMSE ---
    with colB:
        fig_rmse = px.bar(perf, x="model_name", y="rmse",
                          color="rmse", text="rmse",
                          color_continuous_scale="Reds")
        st.plotly_chart(fig_rmse, use_container_width=True)

    # --- MAE ---
    with colC:
        fig_mae = px.bar(perf, x="model_name", y="mae",
                         color="mae", text="mae",
                         color_continuous_scale="Greens")
        st.plotly_chart(fig_mae, use_container_width=True)

    st.markdown("---")

    # ============================================================
    # 📌 ROW 2 — TRAIN vs TEST R² & OVERFITTING CHECK
    # ============================================================
    st.markdown("### 🔍 Analisis Train/Test R² dan Overfitting")

    col1, col2 = st.columns(2)

    # --- Train vs Test R² ---
    with col1:
        fig_tt = go.Figure()
        fig_tt.add_trace(go.Bar(
            x=perf["model_name"],
            y=perf["train_r2"],
            name="Train R²"
        ))
        fig_tt.add_trace(go.Bar(
            x=perf["model_name"],
            y=perf["test_r2"],
            name="Test R²"
        ))
        fig_tt.update_layout(barmode="group", title="Perbandingan Train vs Test R²")
        st.plotly_chart(fig_tt, use_container_width=True)

    # --- Overfitting Detector ---
    with col2:
        perf["gap"] = perf["train_r2"] - perf["test_r2"]
        fig_gap = px.bar(perf, x="model_name", y="gap",
                         color="gap", text="gap",
                         color_continuous_scale="Oranges")
        fig_gap.update_layout(title="Gap Overfitting (Train R² - Test R²)")
        st.plotly_chart(fig_gap, use_container_width=True)

    st.markdown("---")

    # ============================================================
    # 📌 ROW 3 — CV SCORES & RESIDUAL DISTRIBUTION
    # ============================================================
    st.markdown("### 🧪 Cross-Validation & Residual Analysis")

    colx, coly = st.columns(2)

    # --- CV SCORE CHART ---
    with colx:
        fig_cv = px.scatter(cv, x="fold", y="score", color="model_name",
                            title="Cross-Validation R² per Fold",
                            symbol="model_name")
        st.plotly_chart(fig_cv, use_container_width=True)

    # --- RESIDUALS ---
    with coly:
        fig_res = px.box(residuals, x="model_name", y="residual",
                         color="model_name",
                         title="Distribusi Residual Per Model")
        st.plotly_chart(fig_res, use_container_width=True)

    st.markdown("---")

    # ============================================================
    # 📌 ROW 4 — MODEL PARAMETERS (CARD STYLE)
    # ============================================================
    st.markdown("### ⚙️ Parameter Setiap Model")

    for m in params["model_name"].unique():
        st.markdown(f"#### 🔧 {m}")
        p = params[params["model_name"] == m]
        st.dataframe(p[["param_key", "param_value"]],
                     hide_index=True,
                     use_container_width=True)

    st.markdown("---")

    # ============================================================
    # 📌 ROW 5 — SHAP SUMMARY
    # ============================================================
    st.markdown("### 🧠 SHAP Feature Importance (Top Features)")

    shap_df = query_db("SELECT * FROM shap_summary ORDER BY shap_value DESC LIMIT 20")
    if not shap_df.empty:
        fig_shap = px.bar(
            shap_df,
            x="shap_value", y="feature",
            orientation="h",
            color="shap_value",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_shap, use_container_width=True)
    else:
        st.info("SHAP belum dihitung.")

# ======================================================
# ================ PREDIKSI ======================
# ======================================================
with tab_pred:
    # =================================================================
    # TAB: PREDICTION
    # =================================================================
    st.markdown("## 🔮 Prediksi Nilai Akhir Siswa")

    # -------------------------------
    # LOAD MODELS
    # -------------------------------
    models = load_models()


    # =================================================================
    # INPUT FORM
    # =================================================================
    with st.form("pred_form"):

        st.markdown("### 📝 Input Data Siswa")
        st.markdown("Masukkan data berikut untuk melakukan prediksi nilai akhir.")
        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🎓 Akademik")
            Hours_Studied = st.number_input("Hours Studied per Day", 1, 12, 6)
            Previous_Scores = st.number_input("Previous Score", 0, 100, 75)
            Tutoring_Sessions = st.number_input("Tutoring Sessions per Week", 0, 10, 1)

        with col2:
            st.markdown("#### 📚 Karakter & Kondisi")
            Attendance = st.number_input("Attendance (%)", 50, 100, 85)
            Sleep_Hours = st.number_input("Sleep Hours", 3, 12, 7)
            # Family_Income = st.number_input("Family Income (1–10)", 1, 10, 5)
            Access_to_Resources = st.selectbox(
                "Access to Learning Resources",
                ["Low", "Medium", "High"]
            )           
            Family_Income = st.selectbox(
                "Family Income",
                ["Low", "Medium", "High"]
            ) 
            

        submit = st.form_submit_button("🔮 Predict", use_container_width=True)



    # =================================================================
    # WHEN SUBMIT CLICKED — PROCESS
    # =================================================================
    if submit:

        # Prepare data
        input_dict = {
            "Hours_Studied": Hours_Studied,
            "Attendance": Attendance,
            "Sleep_Hours": Sleep_Hours,
            "Tutoring_Sessions": Tutoring_Sessions,
            "Previous_Scores": Previous_Scores,
            "Access_to_Resources": Access_to_Resources,
            "Family_Income": Family_Income,
        }

        df_in = pd.DataFrame([input_dict])

        # -------------------------------
        # PREDICT USING EACH MODEL
        # -------------------------------
        preds = {}
        for name in ["lr", "rf", "xgb"]:
            if name in models:
                try:
                    preds[name] = float(models[name].predict(df_in)[0])
                except:
                    preds[name] = None
            else:
                preds[name] = None

        # Best model prediction
        if "best" in models:
            try:
                best_pred = float(models["best"].predict(df_in)[0])
            except:
                best_pred = None
        else:
            best_pred = None


        # =================================================================
        # SECTION: SCORE + ANALISIS INPUT
        # =================================================================
        st.markdown("### 🎯 Prediksi Nilai & Analisis Input")

        col_left, col_right = st.columns([1.5, 1])

        # LEFT = FINAL SCORE
        with col_left:
            st.markdown("#### 🏆 Prediksi Nilai Akhir")

            if best_pred is None or pd.isna(best_pred):
                score_html = "N/A"
            else:
                score_html = f"{best_pred:.1f}"

            st.markdown(
                f"""
                <div style="
                    background-color:#1f77b4;
                    padding:25px;
                    border-radius:12px;
                    color:white;
                    text-align:center;
                    font-size:32px;
                    font-weight:700;">
                    {score_html}
                </div>
                """,
                unsafe_allow_html=True
            )

        # RIGHT = INPUT ANALYSIS
        with col_right:
            st.markdown("#### 🔍 Analisis Faktor Input")

            def card(text, color):
                return f"""
                <div style="
                    background-color:{color};
                    padding:15px;
                    border-radius:10px;
                    margin-bottom:10px;
                    color:white;
                    font-size:15px;
                    font-weight:500;">
                    {text}
                </div>
                """

            # Hours Studied
            if Hours_Studied < 4:
                st.markdown(card("📘 Jam Belajar Rendah", "#d9534f"), unsafe_allow_html=True)
            elif Hours_Studied < 7:
                st.markdown(card("📙 Jam Belajar Cukup", "#f0ad4e"), unsafe_allow_html=True)
            else:
                st.markdown(card("📗 Jam Belajar Tinggi", "#5cb85c"), unsafe_allow_html=True)

            # Attendance
            if Attendance < 75:
                st.markdown(card("📝 Kehadiran Rendah", "#d9534f"), unsafe_allow_html=True)
            elif Attendance < 90:
                st.markdown(card("📝 Kehadiran Cukup", "#f0ad4e"), unsafe_allow_html=True)
            else:
                st.markdown(card("📝 Kehadiran Baik", "#5cb85c"), unsafe_allow_html=True)

            # Previous Score
            if Previous_Scores < 60:
                st.markdown(card("📊 Nilai Sebelumnya Rendah", "#d9534f"), unsafe_allow_html=True)
            elif Previous_Scores < 80:
                st.markdown(card("📊 Nilai Sebelumnya Cukup", "#f0ad4e"), unsafe_allow_html=True)
            else:
                st.markdown(card("📊 Nilai Sebelumnya Tinggi", "#5cb85c"), unsafe_allow_html=True)



        # =================================================================
        # SECTION: RECOMMENDATIONS
        # =================================================================
        st.markdown("### 💡 Rekomendasi Personal")

        recomm = []

        # Hours Studied
        if Hours_Studied < 6:
            recomm.append("Tingkatkan jam belajar minimal +1 jam per hari.")

        # Attendance
        if Attendance < 80:
            recomm.append("Perbaiki tingkat kehadiran untuk stabilitas nilai.")

        # Sleep
        if Sleep_Hours < 7:
            recomm.append("Tidur 7–8 jam/hari akan membantu fokus belajar.")

        # Tutoring
        if Tutoring_Sessions < 2:
            recomm.append("Ikuti sesi les tambahan untuk memperdalam pemahaman materi.")

        # Previous Scores
        if Previous_Scores < 75:
            recomm.append("Evaluasi kelemahan materi dari nilai sebelumnya untuk perbaikan hasil.")

        # Access to Resources
        if Access_to_Resources == "Low":
            recomm.append("Akses sumber belajar rendah — gunakan platform gratis seperti YouTube, e-book, dan aplikasi belajar.")

        # Family Income
        if Family_Income < "Low":
            recomm.append("Gunakan sumber belajar gratis untuk efisiensi biaya.")

        if recomm:
            for r in recomm:
                st.info("• " + r)
        else:
            st.success("Semua faktor sudah optimal! Pertahankan.")



        # =================================================================
        # SECTION: MODEL COMPARISON
        # =================================================================
        st.markdown("### 📊 Perbandingan Prediksi Antar Model")

        comp_df = pd.DataFrame({
            "Model": ["Linear Regression", "Random Forest", "XGBoost"],
            "Prediksi": [
                preds["lr"],
                preds["rf"],
                preds["xgb"]
            ]
        })

        fig = px.bar(comp_df, x="Model", y="Prediksi", text="Prediksi", color="Prediksi")
        st.plotly_chart(fig, use_container_width=True)



        # =================================================================
        # SAVE LOG INTO DATABASE
        # =================================================================
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO prediction_logs
            (timestamp, input_json, prediction_lr, prediction_rf, prediction_xgb, final_prediction)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            json.dumps(input_dict),
            preds["lr"],
            preds["rf"],
            preds["xgb"],
            best_pred
        ))
        conn.commit()
        conn.close()
