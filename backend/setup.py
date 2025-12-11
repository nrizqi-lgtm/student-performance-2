import sqlite3

DB_PATH = "student_perf.db"

def create_tables():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # ===========================
    # TABLE: model_results
    # ===========================
    cur.execute("""
    CREATE TABLE IF NOT EXISTS model_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT,
        rmse REAL,
        mae REAL,
        r2 REAL,
        train_date TEXT,
        best_model INTEGER DEFAULT 0
    )
    """)

    # ===========================
    # TABLE: shap_summary
    # ===========================
    cur.execute("""
    CREATE TABLE IF NOT EXISTS shap_summary (
        feature TEXT,
        shap_value REAL
    )
    """)

    # ===========================
    # TABLE: prediction_logs
    # ===========================
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

    # ===========================
    # TABLE: meta_info
    # ===========================
    cur.execute("""
    CREATE TABLE IF NOT EXISTS meta_info (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    """)

    conn.commit()
    conn.close()
    print("Database & tables created successfully!")

if __name__ == "__main__":
    create_tables()
