import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN

def convertToMatrix(data, step):
    X, Y = [], []
    for i in range(len(data)- step):
        d =i + step
        X.append(data[i:d, ])
        Y.append(data[d,])
    return np.array(X), np.array(Y)

step = 4
N = 1000
Tp = 800

t = np.arange(0, N)
x = np.sin(0.02 * t) + 2 * np.random.rand(N)
df = pd.DataFrame(x)
df.head()

plt.plot(df)

values = df.values
train, test = values[0:Tp, :], values[Tp:N, :]
print(train.shape)

test = np.append(test, np.repeat(test[-1,], step))
train = np.append(train, np.repeat(train[-1,], step))

trainX, trainY = convertToMatrix(train, step)
testX, testY = convertToMatrix(test, step)

trainX = trainX.reshape(800,4,1)
testX = testX.reshape(200,4,1)
plus = np.concatenate([trainX,testX],axis=0)

model = Sequential()
model.add(SimpleRNN(50, return_sequences=False,input_shape =(4,1)))
model.add(Dense(1))
model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(trainX,trainY,epochs=200,verbose=2)
pred = model.predict(plus)
df_1 = pd.DataFrame(pred)
df_1.head()
plt.plot(df_1)
plt.show()