from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd


app = Flask(__name__)
CORS(app)

model = joblib.load("melanoma_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    """ Accept JSON from frontend and return prediction """
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON received"}), 400

    try:
        # Use strings for categorical columns
        sex = data.get("sex", "unknown")
        location = data.get("anatom_site_general_challenge", "unknown")

        # Convert numeric fields
        age_approx = float(data.get("age_approx", 0))
        width = float(data.get("width", 0))
        height = float(data.get("height", 0))

        # Build DataFrame for pipeline
        X = pd.DataFrame(
            [[sex, age_approx, location, width, height]],
            columns=['sex', 'age_approx', 'anatom_site_general_challenge', 'width', 'height']
        )

        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0].max()

        print(f"Prediction: {prediction}, Probability: {probability}")

        return jsonify({
            "prediction": int(prediction),    # 0 or 1
            "confidence": float(probability)  # 0.0 - 1.0
        })

    except Exception as e:
        print('Error during prediction:', e)
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)