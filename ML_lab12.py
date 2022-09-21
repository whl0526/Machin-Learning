
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import SimpleRNN,TimeDistributed, Dense

X = []
Y = []
for i in range(4):
    lst = list(range(i+1,i+4))
    X.append(list(map(lambda c: [c/10], lst)))
    Y.append(list(map(lambda c: [c/10+0.1], lst)))
X = np.array(X)
Y = np.array(Y)
print(X)
print(Y)

X_test1 = np.array([[[0.5], [0.6], [0.7]]])
X_test2 = np.array([[[0.6], [0.7], [0.8]]])

model1 = Sequential()
model1.add(SimpleRNN(100,
                     return_sequences=True,
                     input_shape=(3,1)))

model1.add(TimeDistributed(Dense(1)))
model1.summary()
model1.compile(loss='mse',optimizer='adam')
model1.fit(X,Y,epochs=200)

print(model1.predict(X_test1))
print(model1.predict(X_test2))

model2 = Sequential()
model2.add(SimpleRNN(100,
                     return_sequences=True,
                     input_shape=(3,1)))

model2.add(SimpleRNN(100,
                     return_sequences=True,
                     input_shape=(3,1)))

model2.add(TimeDistributed(Dense(1)))
model2.summary()
model2.compile(loss='mse', optimizer='adam')
model2.fit(X,Y,epochs=200)

print(model2.predict(X_test1))
print(model2.predict(X_test2))

