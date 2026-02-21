from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # allow requests from frontend


model = joblib.load("melanoma_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    """ Get data from the request, preprocess it, and return the prediction and confidence. """
    data = request.get_json()

    try:
        features = [
            data["sex"],
            float(data["age_approx"]),
            data["anatom_site_general_challenge"],
            float(data["width"]),
            float(data["height"])
        ]

        # reshape for single prediction
        X = np.array([features])
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X).max()

        return jsonify({
            "prediction": int(prediction),  # 0 = benign, 1 = malignant
            "confidence": float(probability)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)