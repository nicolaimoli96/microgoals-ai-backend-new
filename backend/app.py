from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS
import os

# --- App setup ---
app = Flask(__name__)
CORS(app, origins=[
    "http://localhost:3000",
    "https://waiter-frontend.netlify.app",
    "https://waiter-app-v2.netlify.app"
])

# --- File paths ---
BASE_DIR = os.path.dirname(__file__)

# Load the trained model and categories
# (Your actual files in the repo root)
model_path = os.path.join(BASE_DIR, '..', 'category_sales_predictor.joblib')
categories_path = os.path.join(BASE_DIR, '..', 'category_list.pkl')

# Load model + categories
try:
    model = joblib.load(model_path)
    categories = joblib.load(categories_path)
except Exception as e:
    print(f"⚠️ Error loading model files: {e}")
    model, categories = None, []

# Encoder and CSV are currently missing, so comment them out for now
enc = None
waiters = ["John", "Sarah", "Mike"]  # Placeholder values


# --- API endpoints ---

@app.route("/api/waiters", methods=["GET"])
def get_waiters():
    """Return list of available waiters (placeholder if CSV missing)."""
    return jsonify({"waiters": waiters})


@app.route("/api/simulate-daily", methods=["POST"])
def simulate_daily():
    """
    Placeholder endpoint for future logic.
    Validates input and can be expanded to simulate daily performance.
    """
    try:
        data = request.get_json()
        day_of_week = data.get("day_of_week")
        weather = data.get("weather")
        daily_target = data.get("daily_target", 0)
        sales_done_today = data.get("sales_done_today", 0)

        if not all([
            day_of_week is not None,
            weather in ["rain", "cloud", "wind", "sunny"],
            daily_target > 0
        ]):
            return jsonify({"error": "Invalid input"}), 400

        # This is where you'd use your model to simulate daily predictions.
        return jsonify({
            "message": "Simulation endpoint placeholder",
            "day_of_week": day_of_week,
            "weather": weather,
            "daily_target": daily_target,
            "sales_done_today": sales_done_today
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/recommend-categories", methods=["POST"])
def recommend_categories():
    """
    Recommend top 3 categories based on input day, session, weather, waiter.
    """
    try:
        if model is None or not categories:
            return jsonify({"error": "Model not loaded on server"}), 500

        data = request.get_json()
        day = data.get("day")          # e.g. 'Mon'
        session = data.get("session")  # e.g. 'Lunch' or 'Dinner'
        weather = data.get("weather")  # e.g. 'Rain'
        waiter = data.get("waiter")    # e.g. 'ornella (Mgt)'

        # Map frontend session to internal representation
        if session == "Lunch":
            session = "Before5pm"
        elif session == "Dinner":
            session = "After5pm"

        # Validate inputs
        if not all([day, session, weather, waiter]):
            return jsonify({"error": "Missing or invalid input"}), 400

        # Create input DataFrame
        input_df = pd.DataFrame({
            "Weekday": [day],
            "Session": [session],
            "Weather": [weather],
            "Clerk Name": [waiter]
        })

        # NOTE: Encoder is currently missing, so we skip encoding
        # and directly simulate predictions for now.
        # Replace this block when 'encoder.joblib' is available.
        preds = model.predict(pd.DataFrame([[1] * len(model.feature_names_in_)],
                                           columns=model.feature_names_in_))[0]

        # Map predictions to categories
        cat_preds = {cat: preds[i] for i, cat in enumerate(categories)}

        # Sort by predicted quantity descending and take top 3
        sorted_cats = sorted(cat_preds.items(),
                             key=lambda x: x[1],
                             reverse=True)[:3]

        # Prepare recommendations with +20% target
        recommendations = []
        for cat, pred_qty in sorted_cats:
            target_qty = int(round(pred_qty * 1.2))
            recommendations.append({
                "category": cat,
                "predicted_quantity": round(pred_qty, 2),
                "target_quantity": target_qty
            })

        return jsonify({"recommendations": recommendations})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Run locally ---
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
