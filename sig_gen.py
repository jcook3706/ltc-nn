import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Signal Generator
def generate_signal(length, dims, num_components=3, trend_range=(-0.1, 0.1)):
    t = np.linspace(0, 10, length)
    signals = []
    for d in range(dims):
        signal = np.zeros_like(t)
        freqs = np.random.uniform(1, 20, size=num_components)
        weights = np.random.uniform(0.1, 2.0, size=num_components)
        for freq, weight in zip(freqs, weights):
            signal += weight * np.sin(2 * np.pi * freq * t)
        trend_slope = np.random.uniform(trend_range[0], trend_range[1])
        signal += trend_slope * t  # linear trend
        signals.append(signal)
    return np.stack(signals, axis=-1)

# Prepare sequences
def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i:i+seq_len]
        y = data[i+1:i+seq_len+1]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

if __name__ == '__main__':
    # Parameters
    dims = 8
    signal_length = 1000
    sequence_length = 50
    batch_size = 32
    epochs = 20

    # Generate signal and save
    signal = generate_signal(signal_length, dims)
    np.save('generated_signal.npy', signal)

    # Prepare data for LSTM
    x, y = create_sequences(signal, sequence_length)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Model setup
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(sequence_length, dims)),
        LSTM(64, return_sequences=True),
        Dense(dims)
    ])

    model.compile(optimizer='adam', loss='mse')

    # Train
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size)

    # Evaluate
    train_loss = model.evaluate(x_train, y_train, verbose=0)
    test_loss = model.evaluate(x_test, y_test, verbose=0)
    print(f"Final Train Loss: {train_loss:.4f}")
    print(f"Final Test Loss: {test_loss:.4f}")

    # Predict
    y_pred = model.predict(x_test)

    # Plot predictions
    for i in range(0, dims, 4):
        plt.figure(figsize=(12, 10))
        for j in range(4):
            if i + j >= dims:
                break
            plt.subplot(2, 2, j+1)
            plt.plot(y_test[0,:,i+j], label='True')
            plt.plot(y_pred[0,:,i+j], label='Predicted', linestyle='--')
            plt.title(f'Dimension {i+j+1}')
            plt.legend()
        plt.tight_layout()
        plt.show()
