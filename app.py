from flask import Flask, render_template, request
import joblib
import pandas as pd
from pathlib import Path

app = Flask(__name__)
MODEL_PATH = Path(__file__).parent / "models" / "resale_model.joblib"
model = joblib.load(MODEL_PATH)

BRANDS = ["Apple", "Samsung", "OnePlus", "Google", "Xiaomi", "Motorola", "Oppo", "Vivo"]
MODELS = {
    "Apple": ["iPhone SE", "iPhone 12", "iPhone 13", "iPhone 14", "iPhone 15"],
    "Samsung": ["Galaxy A", "Galaxy S21", "Galaxy S22", "Galaxy S23", "Galaxy S24"],
    "OnePlus": ["Nord", "OnePlus 9", "OnePlus 10", "OnePlus 11", "OnePlus 12"],
    "Google": ["Pixel 6a", "Pixel 7", "Pixel 7a", "Pixel 8", "Pixel 8a"],
    "Xiaomi": ["Redmi Note", "Redmi Note Pro", "Xiaomi 11", "Xiaomi 12", "Xiaomi 13"],
    "Motorola": ["Moto G", "Edge 30", "Edge 40", "Edge 50"],
    "Oppo": ["A Series", "Reno 8", "Reno 10", "Reno 11"],
    "Vivo": ["Y Series", "V Series", "V27", "V30"],
}
CONDITIONS = ["Like New", "Good", "Fair", "Poor"]

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    selected = {"brand":"Apple", "model":"iPhone 13", "age_years":"2", "condition":"Good",
                "storage_gb":"128", "ram_gb":"6", "battery_health_pct":"85"}
    if request.method == "POST":
        selected = {k: request.form.get(k, "") for k in selected}
        try:
            row = pd.DataFrame([{
                "brand": selected["brand"],
                "model": selected["model"],
                "age_years": int(selected["age_years"]),
                "condition": selected["condition"],
                "storage_gb": int(selected["storage_gb"]),
                "ram_gb": int(selected["ram_gb"]),
                "battery_health_pct": int(selected["battery_health_pct"])
            }])
            estimate = float(model.predict(row)[0])
            result = f"₹{estimate:,.0f}"
        except Exception as exc:
            error = "Could not calculate an estimate. Check the entered values and try again."
    return render_template("index.html", result=result, error=error, selected=selected,
                           brands=BRANDS, models=MODELS, conditions=CONDITIONS)

@app.route("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
