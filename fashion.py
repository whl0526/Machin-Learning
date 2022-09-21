import sys
assert sys.version_info >= (3, 5)
from tensorflow import keras
# 사이킷런 ≥0.20 필수
import sklearn
assert sklearn.__version__ >= "0.20"
# 텐서플로 ≥2.0 필수
import tensorflow as tf
assert tf.__version__ >= "2.0"
# 공통 모듈 임포트
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
# 먼저 MNIST 데이터셋을 로드하겠습니다.
# 케라스는 `keras.datasets`에 널리 사용하는 데이터셋을 로드하기 위한 함수를
#니다.
# 이 데이터셋은 이미 훈련 세트와 테스트 세트로 나누어져 있습니다.
# 훈련 세트를 더 나누어 검증 세트를 만드는 것이 좋습니다:
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()


X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.
plt.imshow(X_train[0], cmap="binary")
plt.axis('off')
plt.show()