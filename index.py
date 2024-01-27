import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt

# X = np.array([-10, -5, 0, 5, 10, 15, 20, 25])
X = tf.range(-100,100,5)
# print(len(X))
X_train = X[:40]
X_test = X[40:]
y = X + 12
y_train = y[:40]
y_test = y[40:]

print(X, y)
# y = np.array([2, 7, 12, 17, 22, 27, 32, 37])

print(y == X + 12)

X= tf.constant(X)
y=tf.constant(y)

tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    # tf.keras.layers.Dense(100, activation=None),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.mae,
    # optimizer=tf.keras.optimizers.SGD(),
    # optimizer=tf.keras.optimizers.Adam(),
    # optimizer=tf.keras.optimizers.Adam(lr=.01),
    optimizer=tf.keras.optimizers.Adam(lr=.0001),
    # optimizer=tf.keras.optimizers.Adam(learning_rate=.00001),              
    metrics=["mae"]
)

model.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs = 100, verbose=0)

# print(model.predict([17]))

# print("Answer should be 45")
predict = model.predict([33])
print(predict)
print(f'error difference: {45 - predict}')
# answer is 45

