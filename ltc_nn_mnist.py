import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import RNN, Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
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

# Load and preprocess Sequential MNIST
def preprocess_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = preprocess_mnist()

# Build the model
ltc_rnn = Sequential([
    RNN(LTCCell(128)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

ltc_rnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
ltc_rnn.summary()

# Train the model
history = ltc_rnn.fit(x_train, y_train, epochs=50, batch_size=64, validation_data=(x_test, y_test))

# Evaluate the model
ltc_rnn.evaluate(x_test, y_test)

# Plot accuracy over epochs
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.legend()
plt.show()