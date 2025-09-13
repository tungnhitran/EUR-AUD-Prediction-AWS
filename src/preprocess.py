import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_csv('data/euraud_historical.csv', parse_dates=['Date'])
df.set_index('Date', inplace=True)
print("Data loaded:")

# Feature engineering
df['MA10'] = df['EUR_AUD'].rolling(window=10).mean()
df['Lag1'] = df['EUR_AUD'].shift(1)
df['Lag10'] = df['EUR_AUD'].shift(10)
df.dropna(inplace=True)

# Normalization
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['EUR_AUD', 'MA10', 'Lag1', 'Lag10']])

# Create sequence for LSTM (using past 90 days to predict next day)
def create_sequences(data, seq_length=90):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])  # Predicting EUR_AUD
    return np.array(X), np.array(y)
X, y = create_sequences(scaled_data)
print(f"Sequences created: {X.shape}, {y.shape}")
np.save('data/X.npy', X)
np.save('data/y.npy', y)