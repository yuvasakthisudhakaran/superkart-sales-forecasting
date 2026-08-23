
from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("superkart_sales_forecast_model.pkl")

@app.route("/")
def home():
    return {"message": "SuperKart Sales Forecast API Running"}

@app.route("/v1/predict", methods=["POST"])
def predict():

    data = request.get_json()

    input_df = pd.DataFrame([data])

    prediction = model.predict(input_df)

    return jsonify(
        {
            "Predicted_Sales": float(prediction[0])
        }
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
