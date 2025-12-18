import pandas as pd
import numpy as np
import sqlite3
import pickle
import os
from datetime import datetime

# =========================
# ML
# =========================
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

# =========================
# CONFIG
# =========================
DB_PATH = "student_perf.db"
DATA_PATH = "StudentPerformanceFactors.csv"
MODEL_OUTPUT_BEST = "best_model.pkl"
TARGET = "Exam_Score"

NUMERIC_FEATURES = [
    "Hours_Studied", "Attendance", "Sleep_Hours",
    "Previous_Scores", "Tutoring_Sessions", "Physical_Activity"
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

ALL_FEATURES = NUMERIC_FEATURES + list(ORDINAL_FEATURES.keys()) + NOMINAL_FEATURES

# =========================
# INIT DB
# =========================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT,
            rmse REAL,
            mae REAL,
            r2 REAL,
            train_r2 REAL,
            test_r2 REAL,
            is_best INTEGER,
            created_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_params (
            model_name TEXT,
            param_key TEXT,
            param_value TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS cv_scores (
            model_name TEXT,
            fold INTEGER,
            r2 REAL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS nn_feature_weights (
            feature TEXT,
            importance REAL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            input_json TEXT,
            prediction_lr REAL,
            prediction_rf REAL,
            prediction_xgb REAL,
            prediction_nn REAL,
            final_prediction REAL
        )
    """)

    conn.commit()
    conn.close()
    print("DB initialized.")

# =========================
# LOAD & PREPROCESS
# =========================
def load_and_preprocess():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Dataset not found.")

    df = pd.read_csv(DATA_PATH)

    for col in ALL_FEATURES + [TARGET]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    X = df[ALL_FEATURES]
    y = df[TARGET]
    return X, y

# =========================
# TRAIN & EVALUATE
# =========================
def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    ordinal_encoder = OrdinalEncoder(
        categories=[ORDINAL_FEATURES[c] for c in ORDINAL_FEATURES]
    )

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("ord", ordinal_encoder, list(ORDINAL_FEATURES.keys())),
        ("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False), NOMINAL_FEATURES)
    ])

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        "XGBoost": XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        ),
        "NeuralNetwork": MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=500,
            random_state=42
        )
    }

    results = []
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # clean old
    cur.execute("DELETE FROM model_results")
    cur.execute("DELETE FROM model_params")
    cur.execute("DELETE FROM cv_scores")
    cur.execute("DELETE FROM nn_feature_weights")
    conn.commit()

    for name, model in models.items():
        print(f"Training {name}...")

        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        train_r2 = pipe.score(X_train, y_train)
        test_r2 = pipe.score(X_test, y_test)

        # params
        for k, v in model.get_params().items():
            cur.execute(
                "INSERT INTO model_params VALUES (?, ?, ?)",
                (name, k, str(v))
            )

        # cv
        try:
            scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
            for i, s in enumerate(scores):
                cur.execute(
                    "INSERT INTO cv_scores VALUES (?, ?, ?)",
                    (name, i + 1, float(s))
                )
        except:
            pass

        results.append({
            "model_name": name,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "pipeline": pipe
        })

        with open(f"model_{name}.pkl", "wb") as f:
            pickle.dump(pipe, f)

        print(f"{name} | R2={r2:.4f}")

    conn.commit()
    conn.close()
    return results, X_train

# =========================
# SAVE NN WEIGHTS
# =========================
def save_nn_weights(pipeline, feature_names):
    model = pipeline.named_steps["model"]
    if not hasattr(model, "coefs_"):
        return

    weights = model.coefs_[0]
    importance = np.mean(np.abs(weights), axis=1)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    for f, v in zip(feature_names, importance):
        cur.execute(
            "INSERT INTO nn_feature_weights VALUES (?, ?)",
            (f, float(v))
        )

    conn.commit()
    conn.close()

# =========================
# FINALIZE
# =========================
def finalize(results, X_train):
    best = max(results, key=lambda x: x["r2"])

    with open(MODEL_OUTPUT_BEST, "wb") as f:
        pickle.dump(best["pipeline"], f)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    for r in results:
        cur.execute("""
            INSERT INTO model_results
            (model_name, rmse, mae, r2, train_r2, test_r2, is_best, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            r["model_name"],
            r["rmse"],
            r["mae"],
            r["r2"],
            r["train_r2"],
            r["test_r2"],
            1 if r["model_name"] == best["model_name"] else 0,
            datetime.now().isoformat()
        ))

    conn.commit()
    conn.close()

    if best["model_name"] == "NeuralNetwork":
        prep = best["pipeline"].named_steps["prep"]
        feature_names = prep.get_feature_names_out()
        save_nn_weights(best["pipeline"], feature_names)

    print(f"Training finished. Best model: {best['model_name']}")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    init_db()
    X, y = load_and_preprocess()
    results, X_train = train_and_evaluate(X, y)
    finalize(results, X_train)
