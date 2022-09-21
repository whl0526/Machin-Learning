import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Activation


X = []
Y = []
for i in range(4):
    lst = list(range(i,i+4))
    X.append(list(map(lambda c: [c/10], lst)))
    Y.append((i+4)/10)
X = np.array(X)
Y = np.array(Y)
print(X)
print(Y)



model = Sequential()
model.add(SimpleRNN(50,  return_sequences=False, input_shape=(4,1)))
model.add(Dense(1))
model.summary()
model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X,Y,epochs=200,  verbose=2)
print(model.predict(X))

X_test = np.array([[[0.8],[0.9],[1.0],[1.1]]])
print(model.predict(X_test))

