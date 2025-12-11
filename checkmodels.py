import pickle
import streamlit as st
import pandas as pd
import numpy as np

def load_models():
    paths = {
        "lr": "model_LinearRegression.pkl",
        "rf": "model_RandomForest.pkl",
        "xgb": "model_XGBoost.pkl",
        "best": "best_model.pkl"
    }

    models = {}

    # ============================
    # Load models with logging
    # ============================
    for key, path in paths.items():
        try:
            with open(path, "rb") as f:
                models[key] = pickle.load(f)
                st.success(f"✅ Loaded model '{key}' from {path}")
        except FileNotFoundError:
            st.warning(f"⚠️ Model file not found: {path}")
        except Exception as e:
            st.error(f"❌ Error loading {key} from {path}: {e}")

    # ============================
    # Fallback for best model
    # ============================
    if "best" not in models or models.get("best") is None:
        if "xgb" in models:
            models["best"] = models["xgb"]
            st.info("ℹ️ Best model not found → fallback to XGBoost")
        elif "rf" in models:
            models["best"] = models["rf"]
            st.info("ℹ️ Best model not found → fallback to Random Forest")
        elif "lr" in models:
            models["best"] = models["lr"]
            st.info("ℹ️ Best model not found → fallback to Linear Regression")
        else:
            st.warning("⚠️ No model available at all!")

    st.write("Current models loaded:", models.keys())
    return models

# ============================
# Example usage for prediction
# ============================
def predict_best(models, input_dict):
    df_in = pd.DataFrame([input_dict])

    # Encode categorical column
    st.write("Input DataFrame dtypes after encoding:", df_in.dtypes)

    best_pred = None
    if "best" in models and models["best"] is not None:
        try:
            pred_raw = models["best"].predict(df_in)
            if isinstance(pred_raw, (list, np.ndarray, pd.Series)):
                best_pred = float(pred_raw[0])
            else:
                best_pred = float(pred_raw)
            st.success(f"✅ Prediction successful: {best_pred:.2f}")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            best_pred = None
    else:
        st.warning("⚠️ No best model available for prediction.")

    return best_pred

# ============================
# Test
# ============================
if __name__ == "__main__":
    models = load_models()

    test_input = {
        "Hours_Studied": 6,
        "Attendance": 85,
        "Sleep_Hours": 7,
        "Tutoring_Sessions": 1,
        "Previous_Scores": 75,
        "Access_to_Resources": "Medium",
        "Family_Income": "Medium"
    }

    best_pred = predict_best(models, test_input)
