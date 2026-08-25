"""
CarDekho Kolkata Used Car Price Predictor
Flask app with per-prediction LIME explanations, deployable free on Render.

Expects a `models/` folder alongside this file containing:
  best_model.pkl, scaler.pkl, feature_names.pkl, X_train_background.csv
(all produced by the training notebook)
"""

import base64
import io

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
import lime
import lime.lime_tabular
import matplotlib

matplotlib.use("Agg")  # no GUI backend needed on a server
import matplotlib.pyplot as plt

app = Flask(__name__)

# --------------------------------------------------------------------------
# Load artifacts once at startup
# --------------------------------------------------------------------------
MODEL = joblib.load("models/best_model.pkl")
SCALER = joblib.load("models/scaler.pkl")
FEATURE_NAMES = joblib.load("models/feature_names.pkl")
X_TRAIN_BG = pd.read_csv("models/X_train_background.csv")[FEATURE_NAMES]

NEEDS_SCALING = isinstance(MODEL, (Ridge, SVR))


FUEL_OPTIONS = ["CNG", "Diesel", "Electric", "Petrol"]
TRANSMISSION_OPTIONS = ["Automatic", "Manual"]
BRAND_OPTIONS = [
    "Hyundai", "Kia", "MG", "Mahindra", "Maruti Suzuki", "Mercedes-Benz",
    "Renault", "Skoda", "Tata", "Toyota", "Volkswagen", "Honda", "Other",
]

# --------------------------------------------------------------------------
# Feature vector construction - mirrors the training notebook's pd.get_dummies
# exactly: only the non-baseline category gets a 1, everything else is 0.
# --------------------------------------------------------------------------
def build_feature_row(form):
    row = {name: 0 for name in FEATURE_NAMES}

    row["KM_Driven"] = float(form["km_driven"])
    row["Car_Age"] = float(form["car_age"])
    row["Owner_Number"] = float(form["owner_number"])
    row["Latitude"] = float(form["latitude"])
    row["Longitude"] = float(form["longitude"])

    fuel = form["fuel"]
    fuel_col = f"Fuel_{fuel}"
    if fuel_col in row:
        row[fuel_col] = 1
    # else: fuel == the dropped baseline category -> leave all Fuel_* at 0

    if form["transmission"] == "Manual":
        row["Trans_Manual"] = 1
    # else Automatic -> leave at 0 (the baseline)

    brand = form["brand"]
    brand_col = f"Brand_{brand}"
    if brand_col in row:
        row[brand_col] = 1
    # else: brand is the unconfirmed dropped baseline -> falls through as all-zero,
    # which is only correct if that brand really is the true baseline. See TODO above.

    return pd.DataFrame([row], columns=FEATURE_NAMES)


def predict_price(feature_row):
    """Returns predicted price in rupees (inverse of the log1p target transform)."""
    X = feature_row.copy()
    if NEEDS_SCALING:
        X = pd.DataFrame(SCALER.transform(X), columns=FEATURE_NAMES)
    pred_log = MODEL.predict(X)[0]
    return float(np.expm1(pred_log))


def predict_price_batch(X_raw):
    """Same as predict_price but for LIME's batched perturbed samples (numpy array in)."""
    X = pd.DataFrame(X_raw, columns=FEATURE_NAMES)
    if NEEDS_SCALING:
        X = pd.DataFrame(SCALER.transform(X), columns=FEATURE_NAMES)
    pred_log = MODEL.predict(X)
    return np.expm1(pred_log)


# --------------------------------------------------------------------------
# LIME explainer - built once at startup against the training background
# --------------------------------------------------------------------------
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_TRAIN_BG.values,
    feature_names=FEATURE_NAMES,
    mode="regression",
    verbose=False,
)


def explain_prediction(feature_row):
    explanation = explainer.explain_instance(
        feature_row.values[0],
        predict_price_batch,
        num_features=8,
    )
    fig = explanation.as_pyplot_figure()
    fig.set_size_inches(7, 4.5)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
PAGE_TEMPLATE = """
<!doctype html>
<html>
<head>
    <title>AutoXplain - Explainable Used Car Valuation System (Kolkata)</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 16px; }
        h1 { font-size: 1.5rem; }
        label { display: block; margin-top: 12px; font-weight: bold; }
        input, select { width: 100%; padding: 8px; margin-top: 4px; box-sizing: border-box; }
        button { margin-top: 20px; padding: 10px 20px; font-size: 1rem; cursor: pointer; }
        .result { margin-top: 30px; padding: 16px; background: #f0f4f8; border-radius: 8px; }
        .price { font-size: 1.6rem; font-weight: bold; color: #1a5d1a; }
        img { max-width: 100%; margin-top: 16px; }
    </style>
</head>
<body>
    <h1>AutoXplain - Explainable Used Car Valuation System (Kolkata)</h1>
    <p>Fill in the car's details to get a predicted price and see which features drove that prediction.</p>
    <form method="POST">
        <label>Brand</label>
        <select name="brand">
            {% for b in brand_options %}<option value="{{ b }}">{{ b }}</option>{% endfor %}
        </select>

        <label>Fuel Type</label>
        <select name="fuel">
            {% for f in fuel_options %}<option value="{{ f }}">{{ f }}</option>{% endfor %}
        </select>

        <label>Transmission</label>
        <select name="transmission">
            {% for t in transmission_options %}<option value="{{ t }}">{{ t }}</option>{% endfor %}
        </select>

        <label>Car Age (years)</label>
        <input type="number" name="car_age" step="1" min="0" value="5" required>

        <label>KM Driven</label>
        <input type="number" name="km_driven" step="1" min="0" value="40000" required>

        <label>Number of Previous Owners</label>
        <input type="number" name="owner_number" step="1" min="0" value="1" required>

        <label>Latitude</label>
        <input type="number" name="latitude" step="0.0001" value="22.5726" required>

        <label>Longitude</label>
        <input type="number" name="longitude" step="0.0001" value="88.3639" required>

        <button type="submit">Predict Price</button>
    </form>

    {% if price is not none %}
    <div class="result">
        <div>Predicted Price</div>
        <div class="price">\u20b9{{ "{:,.0f}".format(price) }}</div>
        <p>Feature contributions for this specific prediction (LIME):</p>
        <img src="data:image/png;base64,{{ lime_img }}" alt="LIME explanation">
    </div>
    {% endif %}
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    price = None
    lime_img = None

    if request.method == "POST":
        feature_row = build_feature_row(request.form)
        price = predict_price(feature_row)
        lime_img = explain_prediction(feature_row)

    return render_template_string(
        PAGE_TEMPLATE,
        brand_options=BRAND_OPTIONS,
        fuel_options=FUEL_OPTIONS,
        transmission_options=TRANSMISSION_OPTIONS,
        price=price,
        lime_img=lime_img,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
