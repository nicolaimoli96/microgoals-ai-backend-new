import joblib
import pandas as pd
import numpy as np

# Load the trained model
model = joblib.load('category_sales_predictor.joblib')

# Load the saved list of categories
categories = joblib.load('category_list.pkl')

# Test with specific conditions: Monday (day_of_week=1), 19:00 (hour=19), sunny
current_features = pd.DataFrame({
    'day_of_week': [5],  # Monday
    'hour': [19],
    'weather_rain': [0],
    'weather_cloud': [0],
    'weather_wind': [0],
    'weather_sunny': [1]
})

# Make prediction
predictions = model.predict(current_features)[0]

# Get top 5 indices sorted by prediction descending
top_indices = np.argsort(predictions)[-5:][::-1]  # Top 5 highest

print("Top 5 Suggestions for Monday 19:00 sunny:")
for rank, idx in enumerate(top_indices, 1):
    category = categories[idx]
    qty = round(predictions[idx] * 1.1)
    print(f"{rank}. Sell {qty} in {category}")