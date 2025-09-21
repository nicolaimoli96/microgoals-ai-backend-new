import joblib
import pandas as pd
import numpy as np

# Load the trained model
model = joblib.load('item_sales_predictor.joblib')

# Load the saved list of items
items = joblib.load('item_list.pkl')

# Test with specific conditions: Monday (day_of_week=1), 19:00 (hour=19), sunny
current_features = pd.DataFrame({
    'day_of_week': [1],  # tue
    'hour': [19],
    'weather_rain': [0],
    'weather_cloud': [0],
    'weather_wind': [0],
    'weather_sunny': [1]
})

# Make prediction
predictions = model.predict(current_features)[0]

best_idx = np.argmax(predictions)
best_item = items[best_idx]
best_qty = round(predictions[best_idx] * 1.1)

print(f"Suggested for Monday 19:00 sunny: Sell {best_qty} of {best_item}")