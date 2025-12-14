import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np

# 1. Create a date range (3 years of weekly data)
dates = pd.date_range(start='2010-02-05', end='2012-11-01', freq='W-FRI')

# 2. Generate synthetic sales data with a pattern
# Base trend + Seasonality (Sine wave) + Random Noise
n = len(dates)
trend = np.linspace(20000, 30000, n)  # Sales growing from 20k to 30k
seasonality = 10000 * np.sin(np.linspace(0, 3 * np.pi, n)) # Seasonal waves
noise = np.random.normal(0, 2000, n) # Random bumps

weekly_sales = trend + seasonality + noise

# 3. Create the DataFrame
df_mock = pd.DataFrame({
    'Store': 1,
    'Dept': 1,
    'Date': dates,
    'Weekly_Sales': weekly_sales,
    'IsHoliday': False # Simplified for this test
})

# 4. Save to CSV so your main script can find it
df_mock.to_csv('train.csv', index=False)
print("Mock 'train.csv' created successfully. You can now run the forecasting script.")

# 1. Load the data
# Make sure 'train.csv' is in the same folder as your script
df = pd.read_csv('train.csv')

# 2. Preprocess
# Convert the 'Date' column to a proper datetime format
df['Date'] = pd.to_datetime(df['Date'])

# 3. Filter for simplicity
# Let's focus only on Store 1, and aggregate sales for the whole store (all departments)
store_1 = df[df['Store'] == 1].groupby('Date')['Weekly_Sales'].sum().reset_index()

# 4. Prepare for Prophet
# Prophet requires two specific column names: 'ds' (Date) and 'y' (Value to predict)
store_1.columns = ['ds', 'y']

print("Data Prepared. First 5 rows:")
print(store_1.head())

# 5. Build the Model
# We turn on daily_seasonality=False because this is weekly data
# We add 'yearly_seasonality' to capture the Christmas spikes
model = Prophet(yearly_seasonality=True, daily_seasonality=False)
model.fit(store_1)

# 6. Make a Forecast
# Create a dataframe to hold future dates (let's predict 52 weeks into the future)
future = model.make_future_dataframe(periods=52, freq='W')
forecast = model.predict(future)

# 7. Visualize
# This plots the actual data (black dots) and the prediction (blue line)
print("Plotting forecast...")
fig1 = model.plot(forecast)
plt.title("Walmart Store 1 Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.show()

#See the components (Trend vs Seasonality)
ig2 = model.plot_components(forecast)
plt.show()


