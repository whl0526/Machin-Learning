import matplotlib.pylab as plt #그림그리기 위해 matplotlib.pylab을 불러오고 이를  plt 약자로 줄임.
import numpy as np #python 대규모 다차원 배열처리를 지원하는 파이썬 라이브러리 불러오고 간편히 사용하기 위해 np약자로 줄임

from sklearn.linear_model import LinearRegression #선형회귀분석,사이킷런 라이브러리의 LincearRegression 클래스 함수는 통계학의 선형회귀분석 처리.
from sklearn import datasets #데이터 준비
from sklearn.model_selection import train_test_split
diabetes = datasets.load_diabetes() #당뇨병:나이,성별,혈압 등 load_data
# print(diabetes.feature_names)

#데이터프레임 X,y를 각각 트레이닝 데이터, 테스팅 데이터로 나누어준다.
#인스턴스 수 :442개의 행, 속성 수: 10개 숫자 예측 값의 열
diabetes_X_train = diabetes.data[:-20,:]#처음 부터 -19 (0~422행 *10열 까지 훈련용X 데이터 저장)
diabetes_X_test = diabetes.data[-20:,:]#-20부터 끝까지 (422 ~ 442행 *10열 까지 테스트X 데이터 저장)

diabetes_y_train = diabetes.target[:-20] #처음 부터 -19 (0~422행 까지 훈련용y 데이터 저장)
diabetes_y_test = diabetes.target[-20:]#-20부터 끝까지 (422 ~ 442행 까지 테스트y 데이터 저장)

model = LinearRegression() #LinearRegression 클래스 선형회귀 객체를 생성하고, 변수 model에 저장

model.fit(diabetes_X_train,diabetes_y_train) #머신러닝 모델 객체에 학습데이터 diabetes_X_train와 diabetes_y_train을 주입후 학습시킴.
#실행하면 예측값의 오차가 가장 작은 함수 관계식을 찾게됨.

y_pred = model.predict(diabetes_X_test)# 모델을 가지고 예측하는데 predict 메소드 사용, x변수값을 입력받아서 y값을 예측하기 됨 예측값 변수 y_pred에 저장.
plt.plot(diabetes_y_test,y_pred,'.')#테스트값 , 예측값 점 찍기

x = np.linspace(0,330,100)#(시작,끝,개수)의 파라미터로 같은 간격을 가진 숫자들을 손쉽게 어레이에 담을 수있다.(0에서 229까지 330개를 100개로 쪼갠다)
y = x #y에 x값을 대입
plt.plot(x,y)# x,y연산값을 점을 찍어 시각화를 위한 처리
plt.show()#플로팅된 점들을 시각화

