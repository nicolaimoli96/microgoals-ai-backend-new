import pandas as pd
import sqlite3
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib
from datetime import datetime
import numpy as np  # For argmax in test

# Load the CSV data into a Pandas DataFrame
df = pd.read_csv('sales_data.csv')

# (Optional) Load data into SQLite for persistence
conn = sqlite3.connect('sales.db')
df.to_sql('sales_data', conn, if_exists='replace', index=False)
conn.close()
# To load from DB instead: df = pd.read_sql('SELECT * FROM sales_data', sqlite3.connect('sales.db'))

# Preprocess the data
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)  # Fix for DD/MM/YYYY format

# Map Week day string to numeric day_of_week (0=Sun, 1=Mon, ..., 6=Sat) for consistency
weekday_map = {'Sun': 0, 'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu': 4, 'Fri': 5, 'Sat': 6}
df['day_of_week'] = df['Week day'].map(weekday_map)

# Extract exact hour as numerical feature
df['hour'] = pd.to_datetime(df['Hour'], format='%H:%M').dt.hour

# Map weather code to strings for one-hot
weather_map = {1: 'rain', 2: 'cloud', 3: 'wind', 4: 'sunny'}
df['weather'] = df['Weather code'].map(weather_map)

# Use the existing 'Category' column and aggregate (sum quantity/sales per date, hour, category, conditions)
df['category'] = df['Category'].fillna('Other')  # Handle unmapped as 'Other' or drop
aggregated_df = df.groupby(['Date', 'day_of_week', 'hour', 'weather', 'category'], observed=True).agg({
    'Quantity': 'sum',
    'Sales': 'sum'
}).reset_index()

# Filter out 'Other' if needed
aggregated_df = aggregated_df[aggregated_df['category'] != 'Other']

# Exclude specific categories from recommendations
exclude_categories = ['Souvlaki', 'Childrens Menu', 'Meal Deals']
aggregated_df = aggregated_df[~aggregated_df['category'].isin(exclude_categories)]

# One-hot encode categoricals (only weather now, since hour is numerical)
aggregated_df = pd.get_dummies(aggregated_df, columns=['weather'], dtype=int)

# Pivot to wide format: one row per date/conditions, columns for each category's quantity
pivot_df = aggregated_df.pivot_table(index=['Date', 'day_of_week', 'hour', 
                                            'weather_rain', 'weather_cloud', 'weather_wind', 'weather_sunny'],
                                     columns='category', values='Quantity', fill_value=0).reset_index()

# Features: all except Date and category quantities
category_columns = [col for col in pivot_df.columns if col not in ['Date', 'day_of_week', 'hour', 
                                                                   'weather_rain', 'weather_cloud', 'weather_wind', 'weather_sunny']]
X = pivot_df.drop(columns=['Date'] + category_columns)
y = pivot_df[category_columns]

# Train the multi-output model
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X, y)

# Save the trained model
joblib.dump(model, 'category_sales_predictor.joblib')

# Save the list of categories
joblib.dump(list(y.columns), 'category_list.pkl')
print("Model trained and saved as 'category_sales_predictor.joblib'")
print("Category list saved as 'category_list.pkl'")

# Test the model with specific conditions: Monday (day_of_week=1), 19:00 (hour=19), sunny
current_features = pd.DataFrame({
    'day_of_week': [1],  # Monday
    'hour': [19],
    'weather_rain': [0],
    'weather_cloud': [0],
    'weather_wind': [0],
    'weather_sunny': [1]
})
predictions = model.predict(current_features)[0]
categories = y.columns  # e.g., ['Dips', 'Dessert', ...]
best_idx = np.argmax(predictions)
best_category = categories[best_idx]
best_qty = round(predictions[best_idx] * 1.1)
print(f"Suggested for Monday 19:00 sunny: Sell {best_qty} in {best_category}")