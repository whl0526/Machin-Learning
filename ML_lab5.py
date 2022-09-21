import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X1=np.random.randint(101,size=(100,3))
X2=np.random.randint(50,101,size=(100,3))
# print(X1,X2)
Y1=np.zeros(100,dtype=np.int)
Y2=np.ones(100,dtype=np.int)
#print(Y1,Y2)
X=np.vstack((X1,X2))
Y=np.vstack((Y1,Y2)).ravel()
# print('X,Y',X,Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3)
model= LogisticRegression()
model.fit(X_train,Y_train)
pred=model.predict(X_test)
score= accuracy_score(Y_test,pred)
print('테스트값 :',Y_test)
print('\n')
print('예측값 :',pred)
print('\n')
print('스코어 값 :',score)

