import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib
from datetime import datetime
import numpy as np  # For argmax in test

# Load the CSV data into a Pandas DataFrame
df = pd.read_csv('sales_data.csv')

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

# Exclude items from specific categories
exclude_categories = ['Souvlaki', 'Childrens Menu', 'Meal Deals']
df = df[~df['Category'].isin(exclude_categories)]

# Aggregate (sum quantity/sales per date, hour, item, conditions)
aggregated_df = df.groupby(['Date', 'day_of_week', 'hour', 'weather', 'Item'], observed=True).agg({
    'Quantity': 'sum',
    'Sales': 'sum'
}).reset_index()

# One-hot encode categoricals (only weather now, since hour is numerical)
aggregated_df = pd.get_dummies(aggregated_df, columns=['weather'], dtype=int)

# Pivot to wide format: one row per date/conditions, columns for each item's quantity
pivot_df = aggregated_df.pivot_table(index=['Date', 'day_of_week', 'hour', 
                                            'weather_rain', 'weather_cloud', 'weather_wind', 'weather_sunny'],
                                     columns='Item', values='Quantity', fill_value=0).reset_index()

# Features: all except Date and item quantities
item_columns = [col for col in pivot_df.columns if col not in ['Date', 'day_of_week', 'hour', 
                                                               'weather_rain', 'weather_cloud', 'weather_wind', 'weather_sunny']]
X = pivot_df.drop(columns=['Date'] + item_columns)
y = pivot_df[item_columns]

# Train the multi-output model
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X, y)

# Save the trained model
joblib.dump(model, 'item_sales_predictor.joblib')

# Save the list of items for use in testing
joblib.dump(item_columns, 'item_list.pkl')

print("Model trained and saved as 'item_sales_predictor.joblib'")
print("Item list saved as 'item_list.pkl'")