from tensorflow.keras import Model #Model 불러옴
from tensorflow import keras #keras 불러옴
from tensorflow.keras.models import Sequential #Sequential 불러옴
from tensorflow.keras.layers import Dense, Dropout, Flatten # Dense, Dropout, Flatten 불러옴
from tensorflow.keras.layers import Conv2D, MaxPooling2D #Conv2D, MaxPooling2D 불러옴
from tensorflow.keras.datasets import mnist #mnist 불러옴
import matplotlib.pyplot as plt #matplotlib.pyplot불러와서 'plt'로 줄여서 사용
import numpy as np #numpy불러와서 'np'로 줄여서 사용

def load_data(): #data 불러오는 함수
    # 먼저 MNIST 데이터셋을 로드하겠습니다. 케라스는 `keras.datasets`에 널리 사용하는 데이터셋을 로드하기 위한 함수를 제공합니다. 이 데이터셋은 이미 훈련 세트와 테스트 세트로 나누어져 있습니다. 훈련 세트를 더 나누어 검증 세트를 만드는 것이 좋습니다:

    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data() #mnist데이터를 X에는 훈련데이터와 테스트데이터를 Y에는 그에 맞는 라벨값을 저장
    X_train_full = X_train_full.astype(np.float32) #훈련용 데이터 실수형 변환
    X_test = X_test.astype(np.float32)#테스트 데이터 실수형 변환
    return X_train_full, y_train_full, X_test, y_test #훈련용,테스트 데이터 반환

def data_normalization(X_train_full, X_test):
    # 전체 훈련 세트를 검증 세트와 (조금 더 작은) 훈련 세트로 나누어 보죠. 또한 픽셀 강도를 255로 나누어 0~1 범위의 실수로 바꾸겠습니다.

    X_train_full = X_train_full / 255.  # X훈련용 데이터를 0~1의 수로 하기위해 255를 나누어줌

    X_test = X_test / 255. # X 테스트값을 0~1의 수로 하기위해 255를 나누어줌
    train_feature = np.expand_dims(X_train_full, axis=3) #tensorflow 사용을 위해 3차원으로 차원수를 1개 늘려줌
    test_feature = np.expand_dims(X_test, axis=3) #tensorflow 사용을 위해 3차원으로 차원수를 1개 늘려줌

    print(train_feature.shape, train_feature.shape) #값 출력
    print(test_feature.shape, test_feature.shape) #값 출력

    return train_feature,  test_feature


def draw_digit(num):
    for i in num:
        for j in i:
            if j == 0:
                print('0', end='')
            else :
                print('1', end='')
        print()


def makemodel(X_train, y_train, X_valid, y_valid, weight_init): #모델 생성 함수
    model = Sequential() #Sequential() 모델 생성
    model.add(Conv2D(32, kernel_size=(3, 3),  activation='relu')) # 32개의 3*3 크기의 커널을 만들고 activation은 relu로 사용
    model.add(MaxPooling2D(pool_size=2))  #이미지의 사이즈를 줄이기 위해서 MAXPOOLING 함
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu')) # 64개의 3*3 크기의 커널을 만들고 activation은 relu로 사용
    model.add(MaxPooling2D(pool_size=2))  #또 이미지의 사이즈를 줄이기 위해서 MAXPOOLING 함
    model.add(Dropout(0.25)) #전체의 0.25% 만큼 드롭아웃 시킨다.
    model.add(Flatten()) #이미지를 차례대로 Flatten 시켜준다.
    model.add(Dense(128, activation='relu')) # 입력값과 가중치를 계산하여 128개의 출력하는층  activation은 relu로 사용
    model.add(Dense(10, activation='softmax')) # 마지막에 0~9까지의 10개 값을 출력하는 층 activation은 softmax로 사용

    model.compile(loss='sparse_categorical_crossentropy', #모델을 컴파일, loss함수는 'sparse_categorical_crossentropy'사용
                  optimizer='adam', #optimizer은 'adam'사용
                  metrics=['accuracy'])#정확도를 보여줌
    return model


def plot_history(histories, key='accuracy'):
    plt.figure(figsize=(16,10)) #그림판을 16*10 사이즈로 생성

    for name, history in histories: #name은 'baseline' history는 피팅시킨 모델을 가져온다
        val = plt.plot(history.epoch, history.history['val_'+key],
                       '--', label=name.title()+' Val') #x축은 모델의 epoch값 y축은 val(test)의 정확도값으로 하고 '--'형태로 그림을 그리고 baseline Val이라고 라벨링함
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                 label=name.title()+' Train') #x축은 모델의 epoch값 y축은 train의 정확도값으로 하고  baseline Train이라고 라벨링함

    plt.xlabel('Epochs') #x축 라벨 'Epochs' 설정
    plt.ylabel(key.replace('_',' ').title()) #y축 라벨 accuracy 설정
    plt.legend() #이름표 생성

    plt.xlim([0,max(history.epoch)]) #x값 0 에서 history.epoch의 최대값까지 설정
    plt.show() #그림 보여줌
import random # 랜덤함수 사용위해 불러옴


def draw_prediction(pred, k,X_test,y_test,yhat):
    samples = random.choices(population=pred, k=16) #만들었던 pred함수에서 16개를 무작위로 뽑아서 samples 변수에 저장
    count = 0
    nrows = ncols = 4
    plt.figure(figsize=(12,8)) #12*10 사이즈 그림판 생성

    for n in samples:
        count += 1 # for n번째 마다 count++1
        plt.subplot(nrows, ncols, count) #4*4 서브플롯 그림판 생성 count에 따라 위치 표시
        plt.imshow(X_test[n].reshape(28, 28), cmap='Greys', interpolation='nearest')# 위에서 얻은 16개의 데이터를 28 by 28로 바꾸어 나타냄
        tmp = "Label:" + str(y_test[n]) + ", Prediction:" + str(yhat[n]) # Label은 y테스트 값을 문자열로 변환한 것으로, Predictiondms  X_test값을 넣어 예측된 값을 문자열로 변환해서 표시
        plt.title(tmp)# 그림의 타이틀은 tmp 값으로 표시

    plt.tight_layout()
    plt.show()

def evalmodel(X_test,y_test,model):
    yhat = model.predict(X_test)# X_test값을 넣은 y의 예측값을 yhat에 저장
    yhat = yhat.argmax(axis=1) #y예측값중 가장 큰값을 yhat에 저장

    print(yhat.shape) #yhat 차원 표시
    answer_list = []#answer_list 초기 배열 생성

    for n in range(0, len(y_test)):
        if yhat[n] == y_test[n]:
            answer_list.append(n) #for 구문으로 예측값과 테스트 값이 같다면 배열에 추가 저장

    draw_prediction(answer_list, 16,X_test,y_test,yhat)# 함수 사용

    answer_list = [] #예측 라벨값과 실제 라벨값이 다를때를 보여주기 위해 다시 배열 초기화

    for n in range(0, len(y_test)):
        if yhat[n] != y_test[n]:
            answer_list.append(n) #for 구문으로 예측값과 테스트 값이 다르다면 배열에 추가 저장

    draw_prediction(answer_list, 16,X_test,y_test,yhat)# 함수 사용

def main():
    X_train, y_train, X_test, y_test = load_data()
    X_train, X_test = data_normalization(X_train,  X_test)  #data_normalization()에서 리턴한 값들을 저장
                                                            #X값들만 0~1사이의 값으로 만들어줌

    #show_oneimg(X_train)
    #show_40images(X_train, y_train)

    model= makemodel(X_train, y_train, X_test, y_test,'glorot_uniform') #만든 모델들을 model 변수에 저장



    baseline_history = model.fit(X_train,
                                 y_train,
                                 epochs=2,
                                 batch_size=512,
                                 validation_data=(X_test, y_test),
                                 verbose=2) # x에는 X_train y에는 y_train값을 넣고 epochs는 2번만 돌도록 batch size는 512로 모델 피팅
                                            #  validation data는 각 테스트값과 라벨값을 넣고 verbose =2 로 설정

    evalmodel(X_test, y_test, model) #evalmodel함수 사용
    plot_history([('baseline', baseline_history)]) #plot_history함수 사용

main()
