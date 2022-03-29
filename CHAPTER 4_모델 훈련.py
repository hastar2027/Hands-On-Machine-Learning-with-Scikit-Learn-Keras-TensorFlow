# 선형 회귀 #
# 정규방정식 #
# 정규방정식을 테스트하기 위해 선형처럼 보이는 데이터를 생성
import numpy as np

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 정규방정식을 사용해 계산
# 넘파이 선형대수 모듈에 있는 inv() 함수를 사용해 역행렬을 계산하고 dot() 메서드를 사용해 행렬 곱셈
X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# 모델의 예측을 그래프에 표현
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()

# 경사 하강법 #
# 배치 경사 하강법 #
# 경사 하강법을 간단히 구현
eta = 0.1  # learning rate
n_iterations = 1000
m = 100

theta = np.random.randn(2,1)  # random initialization

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients
    
# 확률적 경사 하강법 #
# 간단한 학습 스케줄을 사용한 확률적 경사 하강법의 구현
n_epochs = 50
t0, t1 = 5, 50  # learning schedule hyperparameters

def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2,1)  # random initialization

for epoch in range(n_epochs):
    for i in range(m):
      random_index = np.random.randint(m)
      xi = X_b[random_index:random_index+1]
      yi = y[random_index:random_index+1]
      gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
      eta = learning_schedule(epoch * m + i)
      theta = theta - eta * gradients
      
# 일반적으로 한 반복에서 m번 되풀이 되고, 이때 각 반복을 에포크(epoch)라고 한다

# 최대 1,000번 에포크 동안 실행
# 또는 한 에포크에서 0.001보다 적게 손실이 줄어들 때까지 실행
# 학습률 0.1로 기본 학습 스케줄 사용
# 규제는 전혀 사용하지 x
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1, random_state=42)
sgd_reg.fit(X, y.ravel())

# 다항 회귀 #
# 간단한 2차방정식(quadratic equation)으로 비선형 데이터 생성
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

