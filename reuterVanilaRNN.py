
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from keras.datasets import reuters
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Activation
from keras.utils import to_categorical

def load_data():
    num_words = 30000
    maxlen = 50
    test_split = 0.3

    (X_train, y_train), (X_test, y_test) = reuters.load_data(num_words = num_words, maxlen = maxlen, test_split = test_split)

    # pad the sequences with zeros
    # padding parameter is set to 'post' => 0's are appended to end of sequences
    X_train = pad_sequences(X_train, padding = 'post')
    X_test = pad_sequences(X_test, padding = 'post')


    X_train = np.array(X_train).reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = np.array(X_test).reshape((X_test.shape[0], X_test.shape[1], 1))

    y_data = np.concatenate((y_train, y_test))
    y_data = to_categorical(y_data)


    y_train = y_data[:1395]
    y_test = y_data[1395:]

    return X_train,X_test, y_train,y_test

def make_simplemodel(X_train, y_train):
    model = Sequential()
    model.add(SimpleRNN(50, input_shape=(49, 1), return_sequences=False))
    model.add(Dense(46))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history= model.fit(X_train, y_train,epochs=100)

    return model,history

def make_deepmodel(X_train, y_train):
    model = Sequential()
    model.add(SimpleRNN(50, input_shape=(49, 1), return_sequences=True))
    model.add(SimpleRNN(50, return_sequences=True))
    model.add(SimpleRNN(50, return_sequences=False))

    model.add(Dense(46))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history= model.fit(X_train, y_train,epochs=100)

    return model,history

def model_predict(model,X_test,y_test ):

    y_pred = model.predict(X_test)
    y_pred_ = np.argmax(y_pred, axis = 1)
    y_test_ = np.argmax(y_test, axis = 1)
    print(accuracy_score(y_pred_, y_test_))

def plot_history(histories, key='loss'):
  plt.figure(figsize=(16,10))

  for name, history in histories:
    #val = plt.plot(history.epoch, history.history[key],
    #               '--', label=name.title())
    plt.plot(history.epoch, history.history[key],
             label=name.title())

  plt.xlabel('Epochs')
  plt.ylabel(key.replace('_',' ').title())
  plt.legend()

  plt.xlim([0,max(history.epoch)])
  plt.show()

def main():
    X_train,X_test, y_train,y_test= load_data()
    #레이어 한개
    simple_model,history = make_simplemodel(X_train, y_train)
    #레이어 세개
    deep_model, deephistory = make_deepmodel(X_train, y_train)

    model_predict(simple_model, X_test, y_test)

    model_predict(deep_model, X_test, y_test)

    plot_history([('SimpleRNN', history),('DeepRNN', deephistory)])

main()