# 그레이디언트 소실과 폭주 문제 #
# 글로럿과 He 초기화 #
"""
케라스는 기본적으로 균등분포의 글로럿 초기화 사용
층을 만들 때 kernel_initialiser = "he_uniform"이나 kernel_initialiser = "he_normal"로 바꾸어 He 초기화를 사용할 수 있다
"""
keras.layers.Dense(10, activation="relu", kernel_initializer="he_normal")

# fan_in 대신 fan_avg 기반의 균등분포 He 초기화를 사용하고 싶다면 Variance Scaling을 사용할 수 있다
he_avg_init = keras.initializers.VarianceScaling(scale=2., mode='fan_avg',
                                                 distribution='uniform')
keras.layers.Dense(10, activation="sigmoid", kernel_initializer=he_avg_init)

# 수렴하지 않는 활성화 함수 #
# LeakyReLU 활성화 함수를 사용하려면 LeakyReLU 층을 만들고 모델에서 적용하려는 층 뒤에 추가
model = keras.models.Sequential([
                                 keras.layers.Dense(10, kernel_initializer="he_normal"),
                                 keras.layers.LeakyReLU(alpha=0.2),
])

# SELU 활성화 함수를 사용하려면 층을 만들 때 activation="selu"와 kernel_initializer="lecun_normal"로 지정
layer = keras.layers.Dense(10, activation="selu",
                           kernel_initializer="lecun_normal")

# 배치 정규화 #
# 케라스로 배치 정규화 구현하기 #
"""
은닉층의 활성화 함수 전이나 후에 BatchNormalization 층을 추가하면 된다
모델의 첫 번째 층으로 배치 정규화 층을 추가할 수도 있다
각 은닉층 다음과 모델의 첫 번째 층으로 배치 정규화 층 적용
"""
model = keras.models.Sequential([
                                 keras.layers.Flatten(input_shape=[28, 28]),
                                 keras.layers.BatchNormalization(),
                                 keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
                                 keras.layers.BatchNormalization(),
                                 keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
                                 keras.layers.BatchNormalization(),
                                 keras.layers.Dense(10, activation="softmax")
])

"""
배치 정규화 논문의 저자들은 활성화 함수 이후보다 활성화 함수 이전에 배치 정규화 층을 추가하는 것이 좋다고 조언
하지만 작업에 따라 선호되는 방식이 달라서 이 조언에 대해서는 논란이 조금 있다
두 가지 방법 모두 실험해보고 어떤 것이 주어진 데이터셋에 가장 잘 맞는지 확인하는 것이 좋다
활성화 함수 전에 배치 정규화 층을 추가하려면 은닉층에서 활성화 함수를 지정하지 말고 배치 정규화 층 뒤에 별도의 층으로 추가해야 한다
배치 정규화 층은 입력마다 이동 파라미터를 포함하기 때문에 이전 층에서 편향을 뺄 수 있다
"""
model = keras.models.Sequential([
                                 keras.layers.Flatten(input_shape=[28, 28]),
                                 keras.layers.BatchNormalization(),
                                 keras.layers.Dense(300, kernel_initializer="he_normal", use_bias=False),
                                 keras.layers.BatchNormalization(),
                                 keras.layers.Activation("elu"),
                                 keras.layers.Dense(100, kernel_initializer="he_normal", use_bias=False),
                                 keras.layers.BatchNormalization(),
                                 keras.layers.Activation("elu"),
                                 keras.layers.Dense(10, activation="softmax")
])

# 그레이디언트 클리핑 #
# 케라스에서 그레이디언트 클리핑을 구현하려면 옵티마이저를 만들 때 clipvalue와 clipnorm 매개변수를 지정하면 된다
optimizer = keras.optimizers.SGD(clipvalue=1.0)
model.compile(loss="mse", optimizer=optimizer)

# 사전훈련된 층 재사용하기 #
# 케라스를 사용한 전이 학습 #
"""
먼저 모델 A를 로드하고 이 모델의 층을 기반으로 새로운 모델을 만든다
출력층만 제외하고 모든 층을 재사용하겠다
"""
model_A = keras.models.load_model("my_model_A.h5")
model_B_on_A = keras.models.Sequential(model_A.layers[:-1])
model_B_on_A.add(keras.layers.Dense(1, activation="sigmoid"))

"""
model_A와 model_B_on_A는 일부 층 공유
model_B_on_A를 훈련할 때 model_A도 영향을 받는다
이를 원치 않는다면 층을 재사용하기 전에 model_A를 클론(clone)하자
clone_model() 메서드로 모델 A의 구조를 복제한 후 가중치 복사
"""
model_A_clone = keras.models.clone_model(model_A)
model_A_clone.set_weights(model_A.get_weights())

"""
이제 작업 B를 위해 model_B_on_A을 훈련할 수 있다
하지만 새로운 출력층이 랜덤하게 초기화되어 있으므로 큰 오차를 만들 것이다
따라서 큰 오차 그레이디언트가 재사용된 가중치를 망칠 수 있다
이를 피하는 한 가지 방법은 처음 몇 번의 에포크 동안 재사용된 층을 동결하고 새로운 층에게 적절한 가중치를 학습할 시간을 주는 것이다
이를 위해 모든 층의 trainable 속성을 False로 지정하고 모델 컴파일
"""
for layer in model_B_on_A.layers[:-1]:
  layer.trainable=False

model_B_on_A.compile(loss="binary_crossentropy", optimizer="sgd",
                     metrics=["accuracy"])

"""
이제 몇 번의 에포크 동안 모델을 훈련할 수 있다
그다음 재사용된 층의 동결을 해제하고 작업 B에 맞게 재사용된 층을 세밀하게 튜닝하기 위해 훈련을 계속
일반적으로 재사용된 층의 동결을 해제한 후에 학습률을 낮추는 것이 좋다
이렇게 하면 재사용된 가중치가 망가지는 것을 막아준다
"""
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=4,
                           validation_data=(X_valid_B, y_valid_B))

for layer in model_B_on_A.layers[:-1]:
  layer.trainable = True

optimizer = keras.optimizers.SGD(lr=1e-4) # 기본 학습률은 1e-2
model_B_on_A.compile(loss="binary_crossentropy", optimizer=optimizer,
                     metrics=["accuracy"])
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=16,
                           validation_data=(X_valid_B, y_valid_B))

# 고속 옵티마이저 #
# 모멘텀 최적화 #
"""
케라스에서 모멘텀 최적화를 구현하는 것은 아주 쉽다
SGD 옵티마이저를 사용하고 momentum 매개변수를 지정하고 기다리면 된다
"""
optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)

# 네스테로프 가속 경사 #
"""
NAG는 일반적으로 기본 모멘텀 최적화보다 훈련 속도가 빠르다
이를 사용하려면 SGD 옵티마이저를 만들 때 use_nesterov=True라고 설정하면 된다
"""
optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)

# RMSProp #
# 케라스에는 RMSprop 옵티마이저가 있다
optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9)

# Adam과 Nadam 최적화 #
"""
모멘텀 감쇠 하이퍼파라미터 beta_1은 보통 0.9로 초기화하고 스케일 감쇠 하이퍼파라미터 beta_2는 0.999로 초기화하는 경우가 많다
안정된 계산을 위해 epsilon은 보통 10^(-7) 같은 아주 작은 수로 초기화
이것이 Adam 클래스의 기본값
케라스에서 Adam 옵티마이저를 만드는 방법
"""
optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

# 학습률 스케줄링 #
"""
케라스에서 거듭제곱 기반 스케줄링이 가장 구현하기 쉽다
옵티마이저를 만들 때 decay 매개변수만 지정하면 된다
"""
optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-4)

"""
지수 기반 스케줄링과 구간별 스케줄링도 꽤 간단하다
먼저 현재 에포크를 받아 학습률을 반환하는 함수를 정의해야 한다
예를 들면 지수 기반 스케줄링을 구현해보겠다
"""
def exponential_decay_fn(epoch):
  return 0.01 * 0.1 ** (epoch / 20)

# eta_0와 s를 하드코딩하고 싶지 않다면 이 변수를 설정한 클로저(closure)를 반환하는 함수를 만들 수 있다
def exponential_decay(lr0, s):
  def exponential_decay_fn(epoch):
    return lr0 * 0.1 ** (epoch / s)
  return exponential_decay_fn

exponential_decay_fn = exponential_decay(lr0=0.01, s=20)

"""
그다음 이 스케줄링 함수를 전달하여 LearningRateScheduler 콜백을 만든다
그리고 이 콜백을 fit() 메서드에 전달
"""
lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)
history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                    validation_data=(X_valid_scaled, y_valid),
                    callbacks=[lr_scheduler])

"""
스케줄 함수는 두 번째 매개변수로 현재 학습률을 받을 수 있다
예를 들어 이전 학습률에 0.1^(1/20)을 곱하여 동일한 지수 감쇠 효과를 낸다
"""
def exponential_decay_fn(epoch, lr):
  return lr * 0.1 ** (1 / 20)

# 구간별 고정 스케줄링을 위해서 지수 기반 스케줄링에서 했던 것처럼 스케줄 함수로 LearningRateScheduler 콜백을 만들어 fit() 메서드에 전달
def piecewise_constant_fn(epoch):
  if epoch < 5:
    return 0.01
  elif epoch < 15:
    return 0.005
  else:
    return 0.001
  
"""
성능 기반 스케줄링을 위해서는 ReduceLROnPlateau 콜백 사용
예를 들어 fit() 메서드에 전달하면 최상의 검증 손실이 다섯 번의 연속적인 에포크 동안 향상되지 않을 때마다 학습률에 0.5를 곱한다
"""
lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)

"""
tf.keras는 학습률 스케줄링을 위한 또 다른 방법 제공
keras.optimizers.schedules에 있는 스케줄 중에 하나를 사용해 학습률을 정의하고 이 학습률을 옵티마이저에 전달
이렇게 하면 에포크가 아니라 매 스텝마다 학습률 업데이트
예를 들어 exponential_decay_fn()와 동일한 지수 기반 스케줄링을 구현하는 방법
"""
s = 20 * len(X_train) // 32 # 20번 에포크에 담긴 전체 스텝 수
learning_rate = keras.optimizers.schedules.ExponentialDecay(0.01, s, 0.1)
optimizer = keras.optimizers.SGD(learning_rate)

# 규제를 사용해 과대적합 피하기 #
# l_1과 l_2 규제 #
"""
신경망의 연결 가중치를 제한하기 위해 l_2 규제를 사용하거나 희소 모델을 만들기 위해 l_1 규제를 사용할 수 있다
케라스 층의 연결 가중치에 규제 강도 0.01을 사용하여 l_2 규제를 적용하는 방법을 보여준다
"""
layer = keras.layers.Dense(100, activation="elu",
                           kernel_initializer="he_normal",
                           kernel_regularizer=keras.regularizers.l2(0.01))

"""
일반적으로 네트워크의 모든 은닉층에 동일한 활성화 함수, 동일한 초기화 전략을 사용하거나 모든 층에 동일한 규제를 적용하기 때문에 동일한 매개변수 값을 반복하는 경우가 많다
이는 코드를 읽기 어렵게 만들고 버그를 만들기 쉽다
이를 피하려면 반복문을 사용하도록 코드를 리팩터링(refactoring)할 수 있다
또 다른 방법은 파이썬의 functools.partial() 함수를 사용하여 기본 매개변수 값을 사용하여 함수 호출을 감싸는 것이다
"""
from functools import partial 

RegularizedDense = partial(keras.layers.Dense,
                           activation="elu",
                           kernel_initializer="he_normal",
                           kernel_regularizer=keras.regularizers.l2(0.01))

model = keras.models.Sequential([
                                 keras.layers.Flatten(input_shape=[28, 28]),
                                 RegularizedDense(300),
                                 RegularizedDense(100),
                                 RegularizedDense(10, activation="softmax",
                                                  kernel_initializer="glorot_uniform")
])

# 드롭아웃 #
"""
케라스에서는 keras.layers.Dropout 층을 사용하여 드롭아웃 구현
이 층은 훈련하는 동안 일부 입력을 랜덤하게 버린다
그다음 남은 입력을 보존 확률로 나눈다
훈련이 끝난 후에는 어떤 작업도 하지 않는다
입력을 다음 층으로 그냥 전달
드롭아웃 비율 0.2를 사용한 드롭아웃 규제를 모든 Dense 층 이전에 적용하는 코드
"""
model = keras.models.Seuqential([
                                 keras.layers.Flatten(input_shape=[28, 28]),
                                 keras.layers.Dropout(rate=0.2),
                                 keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
                                 keras.layers.Dropout(rate=0.2),
                                 keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
                                 keras.layers.Dropout(rate=0.2),
                                 keras.layers.Dense(10, activation="softmax")
])

# 몬테 카를로 드롭아웃 #
# 드롭아웃 모델을 재훈련하지 않고 성능을 향상시키는 완전한 MC 드롭아웃 구현
y_probas = np.stack([model(X_test_scaled, training=True)
for sample in range(100)])
y_proba = y_probas.mean(axis=0)

"""
모델이 훈련하는 동안 다르게 작동하는 층을 가지고 있다면 훈련 모드를 강제로 설정해서는 안 된다
대신 Dropout 층을 MCDropout 클래스로 바꿔주자
"""
class MCDropout(keras.layers.Dropout):
  def call(self, inputs): 
    return super().call(inputs, training=True)
  
# 맥스-노름 규제 #
# 케라스에서 맥스-노름 규제를 구현하려면 적절한 최댓값으로 지정한 max_norm()이 반환한 객체로 은닉층의 kernel_constraint 매개변수 지정
keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal",
                   kernel_constraint=keras.constraints.max_norm(1.))
