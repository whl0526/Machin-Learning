import numpy as np
import tensorflow as tf
import pandas as pd
import keras
import time
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
class_names = ["0","1","2","3","4","5","6","7","8","9"]
def makemodel(X_train, y_train, X_valid, y_valid):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(300, input_shape=[64,], activation="relu"))
    model.add(tf.keras.layers.Dense(100, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    model.summary()
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="sgd",
                  metrics=["accuracy"])
    #tb_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
    start = time.time()
    history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid), verbose=1)
    print("time :", time.time() - start)
    return model, history

def data_normalization(X_train_full, y_train_full):
    X_valid, X_train = X_train_full[:200] , X_train_full[200:]
    y_valid, y_train = y_train_full[:200], y_train_full[200:]
    return X_valid, X_train, y_valid, y_train

def printmod(model, x_test,y_test) :
    model.evaluate(x_test, y_test)
    x_new = x_test[:10]
    y_proba = model.predict(x_new)
    plt.figure(figsize=(10 * 1.2, 10 * 1.2))
    for i in range(10):
        pic = x_test[i].reshape(8,8)
        plt.subplot(1, 10, i+1)
        plt.imshow(pic, cmap="binary", interpolation="nearest")
        plt.axis('off')
        yindex = list(y_proba[i]).index(y_proba[i].max())
        print(yindex)
        plt.title(class_names[y_test[i]], fontsize=12)
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    plt.show()
def main():
    digits = load_digits()
    x_data = digits.data
    y_data = digits.target
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)
    x_valid, x_train, y_valid, y_train = data_normalization(x_train, y_train)
    model, history= makemodel(x_train, y_train, x_valid, y_valid)
    printmod(model,x_test, y_test)
main()