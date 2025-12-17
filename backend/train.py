import pandas as pd
import numpy as np
import sqlite3
import shap
import pickle
import os
from datetime import datetime

# ==== ML ====
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ======================================
# CONFIG
# ======================================
DB_PATH = "student_perf.db"
DATA_PATH = "StudentPerformanceFactors.csv"
MODEL_OUTPUT_BEST = "best_model.pkl"

# The 6 features we will train on (as requested)
FEATURES = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Tutoring_Sessions",
    "Previous_Scores",
    "Access_to_Resources",
    "Family_Income"
]

TARGET = "Exam_Score"

# ======================================
# 1. DATABASE INIT
# ======================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS meta_info (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT,
            rmse REAL,
            mae REAL,
            r2 REAL,
            train_r2 REAL,
            test_r2 REAL,
            train_date TEXT,
            best_model INTEGER
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
        CREATE TABLE IF NOT EXISTS test_predictions (
            actual REAL,
            pred REAL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS residuals (
            model_name TEXT,
            pred REAL,
            actual REAL,
            residual REAL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS cv_scores (
            model_name TEXT,
            fold INTEGER,
            score REAL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS shap_summary (
            feature TEXT,
            shap_value REAL
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
            final_prediction REAL
        )
    """)

    conn.commit()
    conn.close()
    print("DB initialized.")


# ======================================
# 2. LOAD & CLEAN DATA (select only 6 features)
# ======================================
def load_and_preprocess():
    # -----------------------------
    # 1. LOAD DATA
    # -----------------------------
    if not os.path.exists(DATA_PATH):
        print("CSV not found — generating dummy dataset...")

        df = pd.DataFrame({
            "Hours_Studied": np.random.randint(1, 10, 200),
            "Attendance": np.random.randint(60, 100, 200),
            "Sleep_Hours": np.random.randint(4, 10, 200),
            "Tutoring_Sessions": np.random.randint(0, 5, 200),

            # NEW FEATURES
            "Previous_Scores": np.random.randint(50, 100, 200),
            "Access_to_Resources": np.random.choice(["Low", "Medium", "High"], 200),
            "Family_Income": np.random.randint(1, 10, 200),

            "Exam_Score": np.random.randint(50, 100, 200)
        })

    else:
        df = pd.read_csv(DATA_PATH)


    # -----------------------------
    # 2. ENSURE ALL FEATURES EXIST
    # -----------------------------
    for feat in FEATURES:
        if feat not in df.columns:
            print(f"Feature '{feat}' missing → adding default values")

            # Numeric defaults
            if feat in ["Hours_Studied"]:
                df[feat] = 5
            elif feat == "Attendance":
                df[feat] = 80
            elif feat == "Sleep_Hours":
                df[feat] = 7
            elif feat == "Tutoring_Sessions":
                df[feat] = 1
            elif feat == "Previous_Scores":
                df[feat] = 70

            # Categorical defaults
            elif feat == "Family_Income":
                df[feat] = "Medium"
            elif feat == "Access_to_Resources":
                df[feat] = "Medium"

    # Ensure target exists
    if TARGET not in df.columns:
        print(f"Target '{TARGET}' missing → generating dummy values")
        df[TARGET] = np.random.randint(50, 100, len(df))


    # -----------------------------
    # 3. KEEP ONLY SELECTED FEATURES
    # -----------------------------
    df = df[FEATURES + [TARGET]].copy()


    # -----------------------------
    # 4. FILL MISSING VALUES CLEANLY
    # -----------------------------
    for col in df.columns:
        if df[col].dtype == "object":
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode().iloc[0])
        else:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())


    # -----------------------------
    # 5. SPLIT OUTPUT
    # -----------------------------
    X = df[FEATURES]
    y = df[TARGET]

    return X, y, len(df)



# ======================================
# 3. TRAIN & EVALUATE
# ======================================
def train_and_evaluate(X, y):
    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # categorical columns (explicit)
    categorical_cols = [c for c in FEATURES if c in ["Family_Income", "Access_to_Resources"] and c in X.columns]
    numeric_cols = [c for c in FEATURES if c not in categorical_cols]

    encoder = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
    ], remainder="passthrough")

    # define base learners
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=150, random_state=42),
        "XGBoost": XGBRegressor(
            objective='reg:squarederror',
            n_estimators=200,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1,
            random_state=42,
            verbosity=0
        )
    }

    results = []
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # clear previous run measurements (keep schema)
    cur.execute("DELETE FROM cv_scores")
    cur.execute("DELETE FROM model_params")
    cur.execute("DELETE FROM residuals")
    cur.execute("DELETE FROM test_predictions")
    conn.commit()

    for name, model in models.items():
        print(f"\nTraining {name} ...")
        pipe = Pipeline([
            ("encoder", encoder),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        train_r2 = pipe.score(X_train, y_train)
        test_r2 = pipe.score(X_test, y_test)

        # save pipeline with consistent filename used by app.py
        fname = f"model_{name}.pkl"
        with open(fname, "wb") as f:
            pickle.dump(pipe, f)

        # store model params (estimator params)
        try:
            est = pipe.named_steps["model"]
            for k, v in est.get_params().items():
                cur.execute("""
                    INSERT INTO model_params (model_name, param_key, param_value)
                    VALUES (?, ?, ?)
                """, (name, k, str(v)))
        except Exception:
            pass

        # test predictions & residuals
        for a, p in zip(y_test, preds):
            cur.execute("INSERT INTO test_predictions (actual, pred) VALUES (?, ?)", (float(a), float(p)))
            cur.execute("""
                INSERT INTO residuals (model_name, pred, actual, residual)
                VALUES (?, ?, ?, ?)
            """, (name, float(p), float(a), float(a - p)))

        # cross-val
        try:
            cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
            for i, score in enumerate(cv_scores):
                cur.execute("""
                    INSERT INTO cv_scores (model_name, fold, score)
                    VALUES (?, ?, ?)
                """, (name, i+1, float(score)))
        except Exception as e:
            print(f"CV error for {name}: {e}")

        results.append({
            "model_name": name,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "pipeline": pipe
        })

        print(f"   🔹 {name}: R2={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}")

    conn.commit()
    conn.close()
    return results, X_train, len(X_train), len(X_test)


# ======================================
# 4. SAVE RESULTS + SHAP
# ======================================
def save_results_to_db(results, total_data, train_size, test_size, X_train):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # clear previous
    cur.execute("DELETE FROM model_results")
    cur.execute("DELETE FROM shap_summary")
    cur.execute("DELETE FROM meta_info")

    # determine best
    best_result = sorted(results, key=lambda x: x["r2"], reverse=True)[0]
    best = best_result["pipeline"]
    best_name = best_result["model_name"]
    print(f"\nBest Model: {best_name}")

    for r in results:
        cur.execute("""
            INSERT INTO model_results
            (model_name, rmse, mae, r2, train_r2, test_r2, train_date, best_model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            r["model_name"], r["rmse"], r["mae"], r["r2"],
            r["train_r2"], r["test_r2"],
            datetime.now().isoformat(),
            1 if r["model_name"] == best_name else 0
        ))

    # save best pipeline to standardized filename for app
    try:
        with open(MODEL_OUTPUT_BEST, "wb") as f:
            pickle.dump(best, f)
    except Exception as e:
        print(f"Error saving best model: {e}")

    print("Calculating SHAP...")

    try:
        enc = best.named_steps["encoder"]
        model = best.named_steps["model"]

        # Build correct feature names
        feature_names = []

        # Extract categorical OHE names
        for name, trans, cols in enc.transformers_:
            if name == "cat":
                ohe = trans
                cats = ohe.categories_
                for col, cat_list in zip(cols, cats):
                    for cat in cat_list:
                        feature_names.append(f"{col}_{cat}")

        # Numeric passthrough
        for col in [c for c in FEATURES if c not in ["Family_Income", "Access_to_Resources"]]:
            feature_names.append(col)


        # Prepare sample
        sample = X_train.sample(min(200, len(X_train)))
        XX = enc.transform(sample)
        XX = np.array(XX)

        # SHAP calculation
        if best_name in ["RandomForest", "XGBoost"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(XX)
            shap_mean = np.abs(shap_values.values).mean(axis=0)
        else:
            # Fallback for LinearRegression or unsupported models
            explainer = shap.KernelExplainer(model.predict, XX[:30])
            shap_values = explainer.shap_values(XX[:50])
            shap_mean = np.abs(shap_values).mean(axis=0)

        print(f"SHAP computed → {len(shap_mean)} values, {len(feature_names)} features")

        # Insert into DB if matching
        if len(shap_mean) == len(feature_names):
            for feat, val in zip(feature_names, shap_mean):
                cur.execute(
                    "INSERT INTO shap_summary (feature, shap_value) VALUES (?, ?)",
                    (feat, float(val))
                )
        else:
            print("SHAP mismatch → skipping database insertion.")

    except Exception as e:
        print(f"SHAP error: {e}")

    # save meta
    meta = {
        "total_data": total_data,
        "train_data": train_size,
        "test_data": test_size,
        "train_ratio": round(train_size/total_data, 3)
    }
    for k, v in meta.items():
        cur.execute("INSERT INTO meta_info (key, value) VALUES (?, ?)", (k, str(v)))

    conn.commit()
    conn.close()
    print("Backend processing complete.")


# ======================================
# MAIN RUN
# ======================================
if __name__ == "__main__":
    init_db()
    X, y, total_len = load_and_preprocess()
    results, X_train, train_len, test_len = train_and_evaluate(X, y)
    save_results_to_db(results, total_len, train_len, test_len, X_train)
    print("\nTraining Completed — RUN STREAMLIT DASHBOARD")
