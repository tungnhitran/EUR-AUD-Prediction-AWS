import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Fetch historical EUR/AUD data from Yahoo Finance
end_date = datetime.now()
start_date = end_date - timedelta(days=365 * 1)  # Last 1 years
data = yf.download("EURAUD=X", start=start_date, end=end_date)

# Save to CSV
df = data[['Close']].reset_index()
df.columns = ['Date', 'EUR_AUD']
df.to_csv('data/euraud_historical.csv', index=False)
print("Data fetched and saved to data/euraud_historical.csv")
print(df.tail(10))  # Print last 10 days for verification