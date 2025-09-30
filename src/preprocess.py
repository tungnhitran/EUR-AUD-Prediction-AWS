import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# Load data
df = pd.read_csv('data/euraud_historical.csv', parse_dates=['Date'])
df.set_index('Date', inplace=True)
print("Data loaded:", df.shape)

# Feature engineering
df['MA10'] = df['EUR_AUD'].rolling(window=10).mean()
df['Lag1'] = df['EUR_AUD'].shift(1)
df['Lag10'] = df['EUR_AUD'].shift(10)
df['Volume'] = df['Volume']  # Retain Volume for potential future use
df.dropna(inplace=True)

# Normalization
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['EUR_AUD', 'MA10', 'Lag1', 'Lag10', 'Volume']])
joblib.dump(scaler, 'data/scaler.pkl')

# Create sequence for LSTM (using past 90 days to predict next day)
def create_sequences(data, seq_length=90):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])  # Predicting EUR_AUD
    return np.array(X), np.array(y)
X, y = create_sequences(scaled_data)
print(f"Sequences created: {X.shape}, {y.shape}")

# Save preprocessed data
os.makedirs('data', exist_ok=True)
np.save('data/X.npy', X)
np.save('data/y.npy', y)