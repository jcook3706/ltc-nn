import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import RNN, Dense, Flatten, Input, Dropout
from tensorflow.keras.models import Model

# LTC Cell Definition
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

# Load and preprocess dataset
def load_dataset(image_dir, csv_path, img_size=(28, 28)):
    df = pd.read_csv(csv_path)
    X, y = [], []
    class_names = sorted(df.iloc[:, 1].unique())
    class_to_index = {cls: i for i, cls in enumerate(class_names)}

    for idx, row in df.iterrows():
        img_path = os.path.join(image_dir, row[0])
        img = load_img(img_path, target_size=img_size, color_mode='grayscale')
        img_array = img_to_array(img) / 255.0
        X.append(img_array)
        y.append(class_to_index[row[1]])

    return np.array(X), to_categorical(np.array(y), num_classes=len(class_names)), class_names

# Paths
image_dir = 'Human_Action_Recognition/images'
csv_path = 'Human_Action_Recognition/images.csv'

# Load data
X, y, class_names = load_dataset(image_dir, csv_path)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2])  # Make sequential (timesteps, features)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
input_layer = Input(shape=(X.shape[1], X.shape[2]))
x = RNN(LTCCell(128))(input_layer)
# x = Dropout(0.2),
x = Dense(64, activation='relu')(x)
out = Dense(len(class_names), activation='softmax')(x)
model = Model(inputs=input_layer, outputs=out)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))

# Evaluate
model.evaluate(X_test, y_test)

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('LTC-NN Accuracy on Human Action Recognition')
plt.legend()
plt.show()
