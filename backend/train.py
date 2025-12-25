# ============================================================
# TRAINING PIPELINE
# Student Academic Performance Prediction
# Academic-grade | Reproducible | Logged | Ensemble-ready
# ============================================================

import os
import time
import json
import pickle
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# =========================
# SKLEARN / ML
# =========================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, OrdinalEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error
)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

# ============================================================
# CONFIG
# ============================================================
DB_PATH = "student_perf.db"
DATA_PATH = "StudentPerformanceFactors.csv"
MODEL_DIR = "models"

TARGET = "Exam_Score"
N_RETRAIN = 10          # jumlah retrain acak (statistik kuat)
TEST_SIZE = 0.2

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# FEATURE DEFINITIONS
# ============================================================
NUMERIC_FEATURES = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity"
]

ORDINAL_FEATURES = {
    "Parental_Involvement": ["Low", "Medium", "High"],
    "Access_to_Resources": ["Low", "Medium", "High"],
    "Motivation_Level": ["Low", "Medium", "High"],
    "Family_Income": ["Low", "Medium", "High"],
    "Teacher_Quality": ["Low", "Medium", "High"],
    "Parental_Education_Level": ["High School", "College", "Postgraduate"],
    "Distance_from_Home": ["Near", "Moderate", "Far"],
    "Peer_Influence": ["Negative", "Neutral", "Positive"]
}

NOMINAL_FEATURES = [
    "Extracurricular_Activities",
    "Internet_Access",
    "Learning_Disabilities",
    "Gender",
    "School_Type"
]

ALL_FEATURES = (
    NUMERIC_FEATURES
    + list(ORDINAL_FEATURES.keys())
    + NOMINAL_FEATURES
)

# ============================================================
# DATABASE INITIALIZATION
# ============================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # =========================
    # TRAINING LOG
    # =========================
    cur.execute("""
        CREATE TABLE IF NOT EXISTS training_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            model_name TEXT,
            r2 REAL,
            rmse REAL,
            mae REAL,
            train_time REAL,
            random_seed INTEGER,
            timestamp TEXT
        )
    """)

    # =========================
    # MODEL REGISTRY
    # =========================
    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_registry (
            model_name TEXT PRIMARY KEY,
            model_path TEXT,
            best_r2 REAL,
            updated_at TEXT
        )
    """)

    # =========================
    # PREDICTION LOGS
    # (USED BY app.py)
    # =========================
    cur.execute("""
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            input_json TEXT,
            prediction_lr REAL,
            prediction_rf REAL,
            prediction_xgb REAL,
            prediction_nn REAL,
            ensemble_prediction REAL
        )
    """)

    conn.commit()
    conn.close()

# ============================================================
# DATA LOADING
# ============================================================
def load_data():
    df = pd.read_csv(DATA_PATH)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    return df[ALL_FEATURES], df[TARGET]

# ============================================================
# PREPROCESSOR
# ============================================================
def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("ord", OrdinalEncoder(
            categories=[ORDINAL_FEATURES[c] for c in ORDINAL_FEATURES]
        ), list(ORDINAL_FEATURES.keys())),
        ("nom", OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        ), NOMINAL_FEATURES)
    ])

# ============================================================
# MODELS
# ============================================================
def get_models(seed):
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, random_state=seed
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            objective="reg:squarederror",
            random_state=seed
        ),
        "NeuralNetwork": MLPRegressor(
            hidden_layer_sizes=(128, 64),
            max_iter=600,
            random_state=seed
        )
    }

# ============================================================
# SINGLE TRAIN RUN
# ============================================================
def train_once(X, y, run_id, seed):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=seed
    )

    preprocessor = build_preprocessor()
    models = get_models(seed)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    trained = []

    for name, model in models.items():
        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", model)
        ])

        start = time.perf_counter()
        pipe.fit(Xtr, ytr)
        train_time = time.perf_counter() - start

        preds = pipe.predict(Xte)

        r2 = r2_score(yte, preds)
        rmse = np.sqrt(mean_squared_error(yte, preds))
        mae = mean_absolute_error(yte, preds)

        cur.execute("""
            INSERT INTO training_log
            (run_id, model_name, r2, rmse, mae, train_time, random_seed, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id, name, r2, rmse, mae,
            train_time, seed, datetime.now().isoformat()
        ))

        trained.append((name, pipe, r2))

    conn.commit()
    conn.close()
    return trained

# ============================================================
# FULL RETRAIN PIPELINE
# ============================================================
def retrain_pipeline(X, y, n_runs=N_RETRAIN):
    all_models = {}

    for run_id in tqdm(range(1, n_runs + 1), desc="🔁 Retraining"):
        seed = np.random.randint(0, 100_000)
        results = train_once(X, y, run_id, seed)

        for name, pipe, r2 in results:
            all_models.setdefault(name, []).append((pipe, r2))

    # =========================
    # SAVE BEST MODEL PER TYPE
    # =========================
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    for name, entries in all_models.items():
        best_pipe, best_r2 = sorted(
            entries, key=lambda x: x[1], reverse=True
        )[0]

        model_path = f"{MODEL_DIR}/model_{name}.pkl"

        with open(model_path, "wb") as f:
            pickle.dump(best_pipe, f)

        cur.execute("""
            INSERT OR REPLACE INTO model_registry
            (model_name, model_path, best_r2, updated_at)
            VALUES (?, ?, ?, ?)
        """, (
            name, model_path, best_r2,
            datetime.now().isoformat()
        ))

    conn.commit()
    conn.close()

    # =========================
    # SAVE ENSEMBLE METADATA
    # =========================
    with open("ensemble_models.pkl", "wb") as f:
        pickle.dump(all_models, f)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("🔧 Initializing database...")
    init_db()

    print("📥 Loading data...")
    X, y = load_data()

    print("🚀 Starting training & retraining...")
    retrain_pipeline(X, y)

    print("✅ TRAINING PIPELINE SELESAI")
