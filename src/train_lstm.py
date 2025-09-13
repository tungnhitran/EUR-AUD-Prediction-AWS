import tensorflow as tf
import numpy as np
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', 'model/'))
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', 'data/'))
    return parser.parse_args()
args = parse_args()

# Load preprocessed data
X = np.load(os.path.join(os.environ.get('SM_CHANNEL_TRAIN', 'data/'), 'X.npy'))
y = np.load(os.path.join(os.environ.get('SM_CHANNEL_TRAIN', 'data/'), 'y.npy'))

# Define LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate), loss='mse')
model.summary()

# Train the model
model.fit(X, y, epochs=args.epochs, batch_size=args.batch_size, validation_split=0.2)
model.save(os.environ['SM_MODEL_DIR'] + '/1')
print("Model training complete and saved.")