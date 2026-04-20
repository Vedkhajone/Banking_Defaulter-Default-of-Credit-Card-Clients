import os
import joblib

# Create folder if not exists
os.makedirs("models", exist_ok=True)

joblib.dump(rf_model, "models/rf_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(kmeans, "models/kmeans.pkl")