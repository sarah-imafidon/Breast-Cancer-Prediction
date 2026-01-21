from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        try:
            # Get input values from form
            radius_mean = float(request.form["radius_mean"])
            texture_mean = float(request.form["texture_mean"])
            perimeter_mean = float(request.form["perimeter_mean"])
            area_mean = float(request.form["area_mean"])
            smoothness_mean = float(request.form["smoothness_mean"])

            # Create feature array
            features = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean]])

            # Scale features
            features_scaled = scaler.transform(features)

            # Predict
            pred = model.predict(features_scaled)[0]

            # Convert 0/1 to labels
            prediction = "Malignant" if pred == 0 else "Benign"
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
