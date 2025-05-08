import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import RNN, Dense, Flatten, Dropout, LSTM, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class LTCCell(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(LTCCell, self).__init__(**kwargs)
        self.units = units
    
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.units), initializer='glorot_uniform', trainable=True)
        self.U = self.add_weight(shape=(self.units, self.units), initializer='orthogonal', trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True)
        self.tau = self.add_weight(shape=(self.units,), initializer='ones', trainable=True)
    
    def call(self, inputs, states):
        prev_output = states[0]
        new_output = prev_output + (1.0 / self.tau) * (tf.matmul(inputs, self.W) + tf.matmul(prev_output, self.U) + self.b - prev_output)
        return new_output, [new_output]
    
    @property
    def state_size(self):
        return self.units

# Prepare sequences
def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i:i+seq_len]
        y = data[i+seq_len]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Parameters
dims = 256
sequence_length = 10
batch_size = 32
epochs = 100
signal = np.load('generated_signal.npy')

x, y = create_sequences(signal, sequence_length)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=69)

# Build the model
ltc_rnn = Sequential([
    SimpleRNN(32, input_shape=(sequence_length, dims)),
    # Dense(dims*2),
    Dense(dims)
])

ltc_rnn.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])
ltc_rnn.summary()

# Train the model
history = ltc_rnn.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))

# Evaluate the model
ltc_rnn.evaluate(x_test, y_test)

# Plot accuracy over epochs
plt.plot(history.history['mean_squared_error'], label='Train MSE')
plt.plot(history.history['val_mean_squared_error'], label='Validation MSE')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.title('Model Mean Squared Error Over Epochs')
plt.legend()
plt.show()

# Evaluate
train_loss = ltc_rnn.evaluate(x_train, y_train, verbose=0)
test_loss = ltc_rnn.evaluate(x_test, y_test, verbose=0)
print(f"Final Train Loss: {train_loss}")
print(f"Final Test Loss: {test_loss}")

# Predict
y_pred = ltc_rnn.predict(x)
print(y_pred.shape)

# Plot predictions
for i in range(0, dims, 4):
    plt.figure(figsize=(12, 10))
    for j in range(4):
        if i + j >= dims:
            break
        plt.subplot(2, 2, j+1)
        plt.plot(y[:200,i+j], label='True')
        plt.plot(y_pred[:200,i+j], label='Predicted', linestyle='--')
        plt.title(f'Dimension {i+j+1}')
        plt.legend()
    plt.tight_layout()
    plt.show()
