

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = keras.Sequential()
sess = tf.Session()

input_layer = keras.layers.Dense(3, input_shape=[3], activation='tanh')
model.add(input_layer)

output_layer = keras.layers.Dense(1, activation='sigmoid')
model.add(output_layer)

gd = tf.train.GradientDescentOptimizer(0.01)

model.compile(optimizer=gd, loss='mse')

training_x = tf.Variable([[1, 1, 0], [1, 1, 1], [0, 1, 0], [-1, 1, 0], [-1, 0, 0], [-1, 0, 1],[0, 0, 1], [1, 1, 0], [1, 0, 0], [-1, 0, 0], [1, 0, 1], [0, 1, 1], [0, 0, 0], [-1, 1, 1]])
training_y = tf.Variable([[0], [0], [1], [1], [1], [0], [1],[0], [1], [1], [1], [1], [1], [0]])

model.fit(training_x, training_y, epochs=1000, steps_per_epoch=10)
# model.save_weights('demo_model.h5')
# model.load_weights('demo_model.h5')

text_x = tf.Variable([[1, 0, 0]])
test_y = model.predict(text_x, verbose=0, steps=1)


print(test_y)
