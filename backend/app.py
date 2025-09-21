from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from flask_cors import CORS  # To handle CORS for React

app = Flask(__name__)
CORS(app)  # Enable CORS to allow React frontend to call API

# Load the trained model and category list
model = joblib.load('../category_sales_predictor.joblib')
categories = joblib.load('../category_list.pkl')  # Load the saved category list

@app.route('/api/suggest-category', methods=['POST'])
def suggest_category():
    try:
        # Get request data (day_of_week, hour, weather)
        data = request.get_json()
        day_of_week = data.get('day_of_week')
        hour = data.get('hour')
        weather = data.get('weather')  # Expected: 'rain', 'cloud', 'wind', 'sunny'

        # Validate inputs
        if not all([day_of_week is not None, hour is not None, weather in ['rain', 'cloud', 'wind', 'sunny']]):
            return jsonify({'error': 'Invalid input'}), 400

        # Create feature DataFrame
        current_features = pd.DataFrame({
            'day_of_week': [day_of_week],
            'hour': [hour],
            'weather_rain': [1 if weather == 'rain' else 0],
            'weather_cloud': [1 if weather == 'cloud' else 0],
            'weather_wind': [1 if weather == 'wind' else 0],
            'weather_sunny': [1 if weather == 'sunny' else 0]
        })

        # Make prediction
        predictions = model.predict(current_features)[0]
        # Get top 3 indices sorted by prediction descending
        top_indices = np.argsort(predictions)[-3:][::-1]  # Top 3 highest
        top_suggestions = []
        for idx in top_indices:
            category = categories[idx]
            qty = round(predictions[idx] * 1.4)
            top_suggestions.append({'category': category, 'quantity': qty})

        return jsonify({'suggestions': top_suggestions})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)