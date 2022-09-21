import numpy as np
import tensorflow as tf
import pandas as pd
import keras
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
x_data = digits.data
y_data = digits.target
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)
x_valid, x_train = x_train[:100] , x_train[100:]
y_valid, y_train = y_train[:100], y_train[100:]
class_names = ["0","1","2","3","4","5","6","7","8","9"]

model = keras.models.Sequential()
model.add(keras.layers.Dense(300,input_shape=[64,],activation="relu"))
model.add(keras.layers.Dense(300,activation="relu"))
model.add(keras.layers.Dense(10,activation="softmax"))
model.compile(loss="sparse_categorical_crossentropy",metrics=["accuracy"], optimizer="sgd")
history = model.fit(x_train, y_train ,epochs=30,verbose=1,validation_data=(x_test, y_test))

model_drop = keras.models.Sequential()
model_drop.add(keras.layers.Dense(300,kernel_initializer='glorot_uniform',activation="relu",input_shape=[64,]))
model_drop.add(keras.layers.Dropout(0.3))
model_drop.add(keras.layers.Dense(300,kernel_initializer='glorot_uniform',activation="relu"))
model_drop.add(keras.layers.Dropout(0.3))
model_drop.add(keras.layers.Dense(10,kernel_initializer='glorot_uniform',activation="softmax"))
model_drop.compile(loss="sparse_categorical_crossentropy",metrics=["accuracy"], optimizer="sgd")
history_drop = model_drop.fit(x_train, y_train ,epochs=30,verbose=1,validation_data=(x_test, y_test))

plt.figure(figsize=(16, 10))
key='accuracy'
histories=[('Normal', history), ('Dropout', history_drop)]
for name, history in histories:
    val = plt.plot(history.epoch, history.history['val_' + key],
                   '--', label=name.title() + ' Val')
    plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title() + ' Train')

plt.xlabel('Epochs')
plt.ylabel(key.replace('_', ' ').title())
plt.legend()

plt.xlim([0, max(history.epoch)])
plt.show()