import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt

X = tf.range(-100, 100, 4)
X_train = X[:40]
X_test = X[40:]
y = X + 10
y_train = y[:40]
y_test = y[40:]
# print(X)
print(X_train)
print(X_test)
# print(y)

# plt.plot(X, y)
# plt.scatter(X, y)

# plt.figure(figsize=(10,7))
# plt.scatter(X_train, y_train, c="b", label="Training data")
# plt.scatter(X_test, y_test, c="g", label="Testing data")
# plt.legend()
# plt.show()

# 3 sets

# training set, test set, 

tf.random.set_seed(11)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1])
])

model.compile(loss=tf.keras.losses.mae,
optimizer=tf.keras.optimizers.Adam(lr=.00001),
metrics=["mae"])

# model.build()

# model.summary()

model.fit(X_train, y_train, epochs=100, verbose=0)

y_pred = model.predict(X_test)

plt.figure(figsize=(10,7))
plt.scatter(X_train, y_train, c="b", label="Training data")
plt.scatter(X_test, y_test, c="g", label="testing data")
plt.scatter(X_test, y_pred, c="y", label="Prediction data")
plt.legend()
plt.show()