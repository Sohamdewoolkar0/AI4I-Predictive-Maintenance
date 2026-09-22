
from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd

app = Flask(__name__)

package = joblib.load("predictive_maintenance_model.pkl")

model = package["model"]
preprocessor = package["preprocessor"]
threshold = package["threshold"]

@app.route("/")
def home():
    return send_from_directory(".", "index.html")
    
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    machine_data = pd.DataFrame([data])
    processed_data = preprocessor.transform(machine_data)

    probability = float(
        model.predict_proba(processed_data)[:, 1][0]
    )

    prediction = int(probability >= threshold)

    return jsonify({
        "failure_probability": round(probability, 4),
        "prediction": "Failure" if prediction == 1 else "Normal"
    })


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
