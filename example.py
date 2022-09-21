from tensorflow.keras import Model
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    # 먼저 MNIST 데이터셋을 로드하겠습니다. 케라스는 `keras.datasets`에 널리 사용하는 데이터셋을 로드하기 위한 함수를 제공합니다. 이 데이터셋은 이미 훈련 세트와 테스트 세트로 나누어져 있습니다. 훈련 세트를 더 나누어 검증 세트를 만드는 것이 좋습니다:

    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data() # mnist 데이터를 불러들여서 (X_train_full, y_train_full), (X_test, y_test)에 각각 값을 넣습니다.
    X_train_full = X_train_full.astype(np.float32) # X_train_full을 실수형으로 변환합니다.
    X_test = X_test.astype(np.float32) # X_test를 실수형으로 변환합니다.
    #print(X_train_full.shape, y_train_full.shape)
    #print(X_test.shape, y_test.shape)
    return X_train_full, y_train_full, X_test, y_test # X_train_full, y_train_full, X_test, y_test를 리턴합니다.

def data_normalization(X_train_full, X_test):
    # 전체 훈련 세트를 검증 세트와 (조금 더 작은) 훈련 세트로 나누어 보죠. 또한 픽셀 강도를 255로 나누어 0~1 범위의 실수로 바꾸겠습니다.

    X_train_full = X_train_full / 255. # X_train_full의 값을 255로 나누어서 0~1 사이의 실수로 바꿔준다.

    X_test = X_test / 255. # X_test의 값을 255로 나누어서 0~1 사이의 실수로 바꿔준다.
    train_feature = np.expand_dims(X_train_full, axis=3) # train_feature
    test_feature = np.expand_dims(X_test, axis=3) # test_feature

    print(train_feature.shape, train_feature.shape) # 행렬 차원을 print합니다.
    print(test_feature.shape, test_feature.shape) # 행렬 차원을 print합니다.

    return train_feature,  test_feature # train_feature,  test_feature 값들을 반환합니다.


def draw_digit(num):
    for i in num:
        for j in i:
            if j == 0:
                print('0', end='') # j가 0이면 0을 print 한다.
            else :
                print('1', end='') # j가 0이 아닌 다른 수이면 1을 print 한다.
        print()





def makemodel(X_train, y_train, X_valid, y_valid, weight_init):
    model = Sequential() # Sequential 한 모델을 생성한다.
    model.add(Conv2D(32, kernel_size=(3, 3),  activation='relu')) # 커널 사이즈가 (3,3)
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']) # 모델을 loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']로 컴파일 해준다.

    return model

def plot_history(histories, key='accuracy'):
    plt.figure(figsize=(16,10)) # 플롯의 크기 설정

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],
                       '--', label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                 label=name.title()+' Train')

    plt.xlabel('Epochs') # xlabel의 값 설정
    plt.ylabel(key.replace('_',' ').title()) # ylabel의 값 설정
    plt.legend()

    plt.xlim([0,max(history.epoch)]) # plot의 x축 길이의 한계 설정
    plt.show()
import random # 랜덤 모듈 import


def draw_prediction(pred, k,X_test,y_test,yhat):
    samples = random.choices(population=pred, k=16)

    count = 0
    nrows = ncols = 4
    plt.figure(figsize=(12,8))

    for n in samples:
        count += 1
        plt.subplot(nrows, ncols, count)
        plt.imshow(X_test[n].reshape(28, 28), cmap='Greys', interpolation='nearest')
        tmp = "Label:" + str(y_test[n]) + ", Prediction:" + str(yhat[n])
        plt.title(tmp)

    plt.tight_layout()
    plt.show()

def evalmodel(X_test,y_test,model):
    yhat = model.predict(X_test)
    yhat = yhat.argmax(axis=1)

    print(yhat.shape)
    answer_list = []

    for n in range(0, len(y_test)):
        if yhat[n] == y_test[n]:
            answer_list.append(n)

    draw_prediction(answer_list, 16,X_test,y_test,yhat)

    answer_list = []

    for n in range(0, len(y_test)):
        if yhat[n] != y_test[n]:
            answer_list.append(n)

    draw_prediction(answer_list, 16,X_test,y_test,yhat)

def main():
    X_train, y_train, X_test, y_test = load_data() # load_data() 함수를 불러와 X_train, y_train, X_test, y_test에 대입한다.
    X_train, X_test = data_normalization(X_train,  X_test) # data_normalization 함수를 불러와 X_train과 X_test에 대입한다.

    #show_oneimg(X_train)
    #show_40images(X_train, y_train)

    model= makemodel(X_train, y_train, X_test, y_test,'glorot_uniform') # 다층 퍼셉트론 glorot_uniform을 이용하여 모델을 만든다.



    baseline_history = model.fit(X_train,
                                 y_train,
                                 epochs=2,
                                 batch_size=512,
                                 validation_data=(X_test, y_test),
                                 verbose=2) # 모델을 X_train, Y_train을 2번 반복하여 batch size가 512로 하고 교차검증은 X_test와 Y_test를 하게 fit을 해준다.

    evalmodel(X_test, y_test, model) # evalmodel 함수를 불러온다.
    plot_history([('baseline', baseline_history)]) # plot_history 함수를 불러와 그림을 그린다.

main()