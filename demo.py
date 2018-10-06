
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential()

input_layer = keras.layers.Dense(3, input_shape=[3], activation='tanh')

model.add(input_layer)

output_layer = keras.layers.Dense(1, activation='sigmoid')

model.add(output_layer)

gd = tf.train.GradientDescentOptimizer(0.01)

model.compile(optimizer=gd, loss='mse')

x = tf.Variable([[1,1,0],[1,1,1],[0,1,0],[-1,1,0],[-1,0,0],[-1,0,1],[0,0,1],[1,1,0],[1,0,0],[-1,0,0],[1,0,1],[0,1,1],[0,0,0],[-1,1,1]])
y = tf.Variable([[0],[0],[1],[1],[1],[0],[1],[0],[1],[1],[1],[1],[1],[0]])

model.fit(x, y, epochs=5000, steps_per_epoch=10)
model.save_weights('demo_model.h5')

results = model.predict(x, verbose=0, steps=1)

test = tf.Variable([[1,0,0]])
test = model.predict(test, verbose=0, steps=1)

print(results)
print(test)