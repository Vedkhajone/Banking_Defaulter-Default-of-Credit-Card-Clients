from src.preprocessing import full_preprocessing
from src.classification import run_classification

import os
import joblib

# -------------------------------
# 🔹 Step 1: Load + Preprocess Data
# -------------------------------
X, y, df = full_preprocessing("data/raw/data.csv")

print("✅ Data loaded and preprocessed")
print("Shape of X:", X.shape)

# -------------------------------
# 🔹 Step 2: Train Models
# -------------------------------
log_model, rf_model, scaler = run_classification(X, y)

print("✅ Models trained successfully")

# -------------------------------
# 🔹 Step 3: Create models folder
# -------------------------------
os.makedirs("models", exist_ok=True)

# -------------------------------
# 🔹 Step 4: Save Models
# -------------------------------
joblib.dump(rf_model, "models/rf_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Models saved")

# -------------------------------
# 🔹 Step 5: Save Feature Columns
# -------------------------------
joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")

print("✅ Feature columns saved")

print("\n🎉 All files saved successfully!")