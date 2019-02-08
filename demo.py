# Developed by Cyril

import tensorflow as tf
from tensorflow import keras
# import numpy as np 

model = keras.Sequential()

input_layer = keras.layers.Dense(3, input_shape=[3], activation='tanh')
model.add(input_layer)

output_layer = keras.layers.Dense(1, activation='sigmoid')
model.add(output_layer)

gd = tf.train.GradientDescentOptimizer(0.01)

model.compile(optimizer=gd, loss='mse')

# Use a numpy array if you run into some trouble
# training_x = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 0], [-1, 1, 0], [-1, 0, 0], [-1, 0, 1],[0, 0, 1], [1, 1, 0], [1, 0, 0], [-1, 0, 0], [1, 0, 1], [0, 1, 1], [0, 0, 0], [-1, 1, 1]])
# training_y = np.array([[0], [0], [1], [1], [1], [0], [1],[0], [1], [1], [1], [1], [1], [0]])

training_x = tf.Variable([[1, 1, 0], [1, 1, 1], [0, 1, 0], [-1, 1, 0], [-1, 0, 0], [-1, 0, 1],[0, 0, 1], [1, 1, 0], [1, 0, 0], [-1, 0, 0], [1, 0, 1], [0, 1, 1], [0, 0, 0], [-1, 1, 1]])
training_y = tf.Variable([[0], [0], [1], [1], [1], [0], [1],[0], [1], [1], [1], [1], [1], [0]])

model.fit(training_x, training_y, epochs=1000, steps_per_epoch=10)
# model.save_weights('demo_model.h5')
# model.load_weights('demo_model.h5')

# Use a numpy array if you run into some trouble
# text_x = np.array([[1, 0, 0]])
text_x = tf.Variable([[1, 0, 0]])
prediction = model.predict(text_x, verbose=0, steps=1)


print(prediction)
