from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "student_model.joblib")
model = joblib.load(MODEL_PATH)

FEATURES = ["study_hours","attendance","previous_score","assignments","sleep_hours","engagement"]

@app.get("/")
def home():
    return jsonify({
        "status": "online",
        "service": "NeuraLab Student Performance ML API",
        "model": "RandomForestRegressor"
    })

@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    try:
        values = [
            float(data["study_hours"]),
            float(data["attendance"]),
            float(data["previous_score"]),
            float(data["assignments"]),
            float(data["sleep_hours"]),
            float(data["engagement"])
        ]
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Send all six numeric fields."}), 400

    if not (0 <= values[0] <= 12 and 0 <= values[1] <= 100 and
            0 <= values[2] <= 100 and 0 <= values[3] <= 100 and
            0 <= values[4] <= 12 and 1 <= values[5] <= 3):
        return jsonify({"error": "One or more values are outside the allowed range."}), 400

    prediction = float(model.predict(np.array([values]))[0])
    prediction = round(max(0, min(100, prediction)), 1)

    if prediction >= 85:
        status, risk, grade = "Excellent outlook", "Low", "A"
    elif prediction >= 70:
        status, risk, grade = "Strong outlook", "Low", "B+"
    elif prediction >= 55:
        status, risk, grade = "Needs improvement", "Medium", "B"
    elif prediction >= 40:
        status, risk, grade = "At risk", "High", "C"
    else:
        status, risk, grade = "High risk", "High", "D"

    # Approximate confidence for UI; this is not a calibrated probability.
    confidence = round(min(94, max(70, 70 + abs(prediction - 50) * .45)), 1)

    return jsonify({
        "prediction": prediction,
        "status": status,
        "risk": risk,
        "grade": grade,
        "confidence": confidence,
        "model": "RandomForestRegressor"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
