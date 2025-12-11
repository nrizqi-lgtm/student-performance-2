import pandas as pd
import numpy as np
import sqlite3
import shap
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
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
MODEL_OUTPUT = "best_model.pkl"

# ======================================
# MAIN TRAINING SCRIPT
# ======================================
def load_data():
    df = pd.read_csv(DATA_PATH)

    # Drop columns as requested
    drop_cols = [
        "Parental_Education_Level",
        "Distance_from_Home",
        "Teacher_Quality"
    ]
    df = df.drop(columns=drop_cols, errors="ignore")
    return df


def preprocess(df):
    X = df.drop("Exam_Score", axis=1)
    y = df["Exam_Score"]

    categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
    numeric_cols = [c for c in X.columns if X[c].dtype != "object"]

    encoder = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ], remainder="passthrough")

    return X, y, encoder


def train_models(X_train, y_train, encoder):
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        "XGBoost": XGBRegressor(objective='reg:squarederror', n_estimators=300, max_depth=6)
    }

    trained = {}

    for name, model in models.items():
        pipe = Pipeline([
            ("encoder", encoder),
            ("model", model)
        ])
        pipe.fit(X_train, y_train)
        trained[name] = pipe

    return trained


def evaluate_models(models, X_test, y_test):
    results = []

    for name, model in models.items():
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        pickle.dump(model, open(f"model_{name}.pkl", "wb"))
        results.append((name, rmse, mae, r2))

    return results


def save_to_db(results, total, train_size, test_size, shap_df):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Clear previous data
    cur.execute("DELETE FROM model_results")
    cur.execute("DELETE FROM shap_summary")
    cur.execute("DELETE FROM meta_info")

    # Insert model performance
    best_model = sorted(results, key=lambda x: x[1])[0][0]

    for model_name, rmse, mae, r2 in results:
        cur.execute("INSERT INTO model_results (model_name, rmse, mae, r2, train_date, best_model) VALUES (?, ?, ?, ?, ?, ?)",
                    (model_name, rmse, mae, r2, datetime.now().isoformat(), 1 if model_name == best_model else 0))

    # Insert SHAP summary
    for _, row in shap_df.iterrows():
        cur.execute("INSERT INTO shap_summary (feature, shap_value) VALUES (?, ?)", (row["feature"], row["shap_value"]))

    # Meta info
    meta = {
        "total_data": total,
        "train_data": train_size,
        "test_data": test_size,
        "train_ratio": train_size / total
    }

    for k, v in meta.items():
        cur.execute("INSERT INTO meta_info (key, value) VALUES (?, ?)", (k, str(v)))

    conn.commit()
    conn.close()

    return best_model


def compute_shap(best_pipe, X_sample):
    print("\n=== Computing SHAP values for best model ===")
    print(f"Model: {best_pipe['model'].__class__.__name__}")
    print(f"Sample size for SHAP: {len(X_sample)} rows")

    # Transform data with encoder
    print("Encoding data...")
    X_encoded = best_pipe["encoder"].transform(X_sample)

    # Use model.predict as the callable function
    print("Initializing SHAP explainer...")
    explainer = shap.Explainer(best_pipe["model"].predict, X_encoded)

    # Compute SHAP values
    print("Computing SHAP values (this may take a moment)...")
    shap_values = explainer(X_encoded)

    # Mean absolute SHAP value per feature
    shap_mean = np.abs(shap_values.values).mean(axis=0)

    # Get feature names after encoding
    feature_names = best_pipe["encoder"].get_feature_names_out()

    df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_mean
    })

    print("\n=== Top 10 Most Important Features Based on SHAP ===")
    print(df.head(10).to_string(index=False))

    return df.sort_values(by="shap_value", ascending=False)


def main():
    print("\n=== Loading Dataset ===")
    df = load_data()
    print(f"Total data loaded: {len(df)} rows")

    print("\n=== Preprocessing ===")
    X, y, encoder = preprocess(df)
    print(f"Features: {list(X.columns)}")
    print(f"Target: Exam_Score")

    print("\n=== Splitting Train/Test ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    print("\n=== Training Models ===")
    models = train_models(X_train, y_train, encoder)
    print("Models trained:", ", ".join(models.keys()))

    print("\n=== Evaluating Models ===")
    results = evaluate_models(models, X_test, y_test)

    # Pick best model
    best_model_name = sorted(results, key=lambda x: x[1])[0][0]
    best_pipe = models[best_model_name]
    print(f"\n=== Best Model Selected: {best_model_name} ===")

    print("\n=== SHAP Analysis ===")
    shap_df = compute_shap(best_pipe, X_train.sample(200))

    print("\n=== Saving Results to Database ===")
    best_model = save_to_db(
        results,
        total=len(df),
        train_size=len(X_train),
        test_size=len(X_test),
        shap_df=shap_df
    )
    print("Database updated successfully.")

    # Save best model
    print("\n=== Saving Best Model ===")
    with open(MODEL_OUTPUT, "wb") as f:
        pickle.dump(best_pipe, f)

    print(f"Training completed! Best model: {best_model}")
    print("Process finished successfully.\n")

if __name__ == "__main__":
    main()