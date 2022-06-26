# 시계열 예측하기 #
'''
간단하게 generate_time_series() 함수로 생성한 시계열을 사용하겠다.
'''

def generate_time_series(batch_size, n_steps):
  freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
  time = np.linspace(0, 1, n_steps)
  series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10)) # 사인 곡선 1
  series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # + 사인 곡선 2
  series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5) # + 잡음
  return series[..., np.newaxis].astype(np.float32)
  
'''
이 함수는 요청한 만큼 n_steps 길이의 여러 시계열을 만든다. 각 시계열에는 타임 스텝마다 하나의 값만 있다. 이 함수는 [배치 크기, 타임 스텝 수, 1] 크기의 넘파이 배열을 반환한다. 각 시계열은 진폭이 같고 진동 수와 위상이 랜덤한 두 개의 사인 곡선을 더하고 약간의 잡음을 추가한다.
이제 이 함수를 사용해 훈련 세트, 검증 세트, 테스트 세트를 만들어보자.
'''

n_steps = 50
series = generate_time_series(10000, n_steps + 1)
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps], series[9000:, -1]

'''
X_train은 7,000개의 시계열을 담고, X_valid는 2,000개, X_test는 1,000개를 담고 있다. 각 시계열마다 하나의 값을 예측해야 하기 때문에 타깃은 열 벡터이다.
'''

## 기준 성능 ##
'''
간단한 방법은 완전 연결 네트워크를 사용하는 것이다. 이 네트워크는 입력마다 1차원 특성 배열을 기대하기 때문에 Flatten 층을 추가해야 한다. 시계열 값의 선형 조합으로 예측하기 위해 간단한 선형 회귀 모델을 사용하겠다.
'''

model = keras.models.Sequential([
                                 keras.layers.Flatten(input_shape=[50, 1]),
                                 keras.layers.Dense(1)
])

'''
MSE 손실과 Adam 옵티마이저를 사용해 이 모델을 컴파일하고 20 에포크 동안 훈련 세트에서 훈련하여 검증 세트에서 평가하면 약 0.004의 MSE 값을 얻는다. 순진한 예측보다 훨씬 낫다!
'''

## 간단한 RNN 구현하기 ##
'''
간단한 RNN을 사용해 이 성능을 앞지를 수 있는지 확인해보자.
'''

model = keras.models.Sequential([
                                 keras.layers.SimpleRNN(1, input_shape=[None, 1])
])

'''
가장 간단하게 만들 수 있는 RNN이다. 하나의 뉴런으로 이루어진 하나의 층을 가진다. 순환 신경망은 어떤 길이의 타임 스텝도 처리할 수 있기 때문에 입력 시퀀스의 길이를 지정할 필요가 없다. 기본적으로 SimpleRNN 층은 하이퍼볼릭 탄젠트 활성화 함수를 사용한다. 초기 상태를 0으로 설정하고 첫 번째 타임 스텝과 함께 하나의 순환 뉴런으로 전달한다. 뉴런은 이 값의 가중치 합을 계산하고 하이퍼볼릭 탄젠트 활성화 함수를 적용하여 결과를 만들어 첫 번째를 출력한다. 기본 RNN에서는 이 출력이 새로운 상태가 된다. 이 새로운 상태는 다음 입력 값과 함께 동일한 순환 뉴런으로 전달된다. 이 과정이 마지막 타임 스텝까지 반복된다. 그다음 이 층은 마지막 값을 출력한다. 모든 시계열에 대해 이 과정이 모두 동시에 수행된다.
'''

## 심층 RNN ##
'''
tf.keras로 심층 RNN을 구현하는 것은 매우 쉽다. 그냥 순환 층을 쌓으면 된다. 이 예에서는 세 개의 SimpleRNN 층을 사용한다.
'''

model = keras.models.Sequential([
                                 keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
                                 keras.layers.SimpleRNN(20, return_sequences=True),
                                 keras.layers.SimpleRNN(1)
])

'''
이 모델을 컴파일, 훈련, 평가하면 0.003의 MSE에 도달할 수 있다. 드디어 선형 모델을 앞질렀다!
마지막 층은 이상적이지 않다. 단변량 시계열을 예측하기 때문에 하나의 유닛이 필요하고 이는 타임 스텝마다 하나의 출력을 만들어야 한다는 뜻이다. 하나의 유닛을 가진다는 것은 은닉 상태가 하나의 숫자라는 뜻이다. 많지는 않지만 쓸모가 있진 않다. 아마 이 RNN은 한 타임 스텝에서 다음 타임 스텝으로 필요한 모든 정보를 나르기 위해 다른 순환 층의 은닉 상태를 주로 사용할 것이다. 마지막 층의 은닉 상태는 크게 필요하지 않다. 또한 SimpleRNN 층은 기본적으로 tanh 활성화 함수를 사용하기 때문에 예측된 값이 -1과 1 사이 범위에 놓인다. 다른 활성화 함수를 사용하려면 어떻게 할까? 이런 이유로 출력층을 Dense 층으로 바꾸는 경우가 많다. 더 빠르면서 정확도는 거의 비슷하다. 또한 원하는 활성화 함수를 선택할 수 있다. 이렇게 바꾸려면 두 번째 순환 층에서 return_sequences=True를 제거한다.
'''

model = keras.models.Sequential([
                                 keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
                                 keras.layers.SimpleRNN(20),
                                 keras.layers.Dense(1)
])

'''
이 모델을 훈련하면 빠르게 수렴하고 성능도 좋다. 또한 출력층의 활성화 함수를 원하는 함수로 바꿀 수 있다.
'''

## 여러 타임 스텝 앞을 예측하기 ##
'''
첫 번째 방법은 이미 훈련된 모델을 사용하여 다음 값을 예측한 다음 이 값을 입력으로 추가하는 것이다. 이 모델을 사용해 다시 다음 값을 예측하는 식이다. 다음 코드와 같다.
'''

series = generate_time_series(1, n_steps + 10)
X_new, Y_new = series[:, :n_steps], series[:, n_steps:]
X = X_new
for step_ahead in range(10):
  y_pred_one = model.predict(X[:, step_ahead:])[:, np.newaxis, :]
  X = np.concatenate([X, y_pred_one], axis=1)

Y_pred = X[:, n_steps:]  

'''
예상할 수 있듯이 다음 스텝에 대한 예측은 보통 더 미래의 타임 스텝에 대한 예측보다 정확하다. 미래의 타임 스텝은 오차가 누적될 수 있기 때문이다. 이 방식을 검증 세트에 적용하면 약 0.029의 MSE를 얻는다. 이전 모델보다 크게 높지만 훨씬 어려운 작업이므로 단순히 비교하기 어렵다. 이 성능을 단순한 예측이나 간단한 선형 모델과 비교하는 것이 더 의미가 있다. 단순한 방식은 성능이 아주 나쁘다. 선형 모델은 0.0188의 MSE를 낸다. 이 모델은 한 번에 하나의 미래 스텝을 예측하기 위해 RNN을 사용하는 것보다 낫다. 또한 훈련과 실행 속도도 더 빠르다. 조금 더 복잡한 문제에서 몇 개의 타임 스텝 앞을 예측할 때도 이 방식이 잘 적용될 수 있다.
두 번째 방법은 RNN을 훈련하여 다음 값 10개를 한 번에 예측하는 것이다. 시퀀스-투-벡터 모델을 사용하지만 1개가 아니라 값 10개를 출력한다. 먼저 타깃을 다음 10개의 값이 담긴 벡터로 바꾸어야 한다.
'''

series = generate_time_series(10000, n_steps + 10)
X_train, Y_train = series[:7000, :n_steps], series[:7000, -10:, 0]
X_valid, Y_valid = series[7000:9000, :n_steps], series[7000:9000, -10:, 0]
X_test, Y_test = series[9000:, :n_steps], series[9000:, -10:, 0]

'''
이제 1개의 유닛이 아니라 10개 유닛을 가진 출력층이 필요하다.
'''

model = keras.models.Sequential([
                                 keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
                                 keras.layers.SimpleRNN(20),
                                 keras.layers.Dense(10)
])

'''
이 모델을 훈련한 후에 한 번에 다음 값 10개를 매우 쉽게 예측할 수 있다.
'''

Y_pred = model.predict(X_new)

'''
이 모델은 잘 작동한다. 다음 10개 타임 스텝에 대한 MSE는 약 0.008이다. 선형 모델보다 훨씬 좋다. 하지만 더 개선할 여지가 있다. 마지막 타임 스텝에서만 다음 값 10개를 예측하도록 모델을 훈련하는 대신 모든 타임 스텝에서 다음 값 10개를 예측하도록 모델을 훈련할 수 있다. 다르게 말하면 이 시퀀스-투-벡터 RNN을 시퀀스-투-시퀀스 RNN으로 바꿀 수 있다. 이 방식의 장점은 마지막 타임 스텝에서의 출력뿐만 아니라 모든 타임 스텝에서 RNN 출력에 대한 항이 손실에 포함된다는 것이다. 이 말은 더 많은 오차 그레이디언트가 모델로 흐른다는 뜻이고 시간에 따라서만 흐를 필요가 없다. 각 타임 스텝의 출력에서 그레이디언트가 흐를 수 있다. 이는 훈련을 안정적으로 만들고 훈련 속도를 높인다.
구체적으로 설명하면 타임 스텝 0에서 모델이 타임 스텝 1에서 10까지 예측을 담은 벡터를 출력할 것이다. 그다음 타임 스텝 1에서 이 모델은 타임 스텝 2에서 11까지 예측할 것이다. 이런 식으로 계속한다. 각 타깃은 입력 시퀀스와 동일한 길이의 시퀀스이다. 이 시퀀스는 타임 스텝마다 10차원 벡터를 담고 있다. 타깃 시퀀스를 준비해보자.
'''

Y = np.empty((10000, n_steps, 10)) # 각 타깃은 10D 벡터의 시퀀스이다.
for step_ahead in range(1, 10 + 1):
  Y[:, :, step_ahead - 1] = series[:, step_ahead:step_ahead + n_steps, 0]
Y_train = Y[:7000]
Y_valid = Y[7000:9000]
Y_test = Y[9000:]  

'''
이 모델을 시퀀스-투-시퀀스 모델로 바꾸려면 모든 순환 층에 return_sequences=True를 지정해야 한다. 그다음 모든 타임 스템에서 출력을 Dense 층에 적용해야 한다. 케라스는 바로 이런 목적을 위해 TimeDistributed 층을 제공한다. 이 층은 다른 층을 감싸서 입력 시퀀스의 모든 타임 스텝에 이를 적용한다. 각 타임 스텝을 별개의 샘플처럼 다루도록 입력의 크기를 바꾸어 이를 효과적으로 수행한다. 그다음 Dense 층에 적용한다. 마지막으로 출력 크기를 시퀀스로 되돌린다. 다음이 개선된 모델이다.
'''

model = keras.models.Sequential([
                                 keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
                                 keras.layers.SimpleRNN(20, return_sequences=True),
                                 keras.layers.TimeDistributed(keras.layers.Dense(10))
])

'''
Dense 층이 실제로 시퀀스를 입력으로 받을 수 있다. 마치 TimeDistributed(Dense(...))처럼 입력을 다룬다. 즉 마지막 입력 차원에만 적용된다. 따라서 마지막 층을 그냥 Dense(10)으로 바꿀 수 있다. 하지만 명확하게 하기 위해 TimeDistributed(Dense(10))을 그대로 사용하겠다. Dense 층을 타임 스텝마다 독립적으로 적용하고 모델이 하나의 벡터가 아니라 시퀀스를 출력한다는 것을 잘 드러내기 때문이다.
훈련하는 동안 모든 출력이 필요하지만 예측과 평가에는 마지막 타임 스텝의 출력만 사용된다. 훈련을 위해 모든 출력에 걸쳐 MSE를 계산했다. 평가를 위해서는 마지막 타임 스텝의 출력에 대한 MSE만을 계산하는 사용자 정의 지표를 사용하겠다.
'''

def last_time_step_mse(Y_true, Y_pred):
  return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])

optimizer = keras.optimizers.Adam(lr=0.01)
model.compile(loss="mse", optimizer=optimizer, metrics=[last_time_step_mse])  

'''
검증 MSE로 0.006을 얻었다. 이전 모델보다 25%나 향상된 것이다. 이런 방식을 처음 모델과 결합할 수 있다. 이 RNN을 사용해 다음 값 10개를 예측하고 이 값을 입력 시계열에 연결한다. 그다음 모델을 다시 사용해 다음 값 10개를 예측한다. 이 과정을 필요한 만큼 반복한다. 이런 식으로 어떤 길이의 시퀀스로 생성할 수 있다. 장기간 예측을 하면 정확도가 떨어지겠지만 새로운 음악이나 텍스트를 생성하는 것이 목적이라면 문제가 되지 않는다.
'''

# 긴 시퀀스 다루기 #
## 불안정한 그레이디언트 문제와 싸우기 ##
'''
tf.keras를 사용해 간단한 메모리 셀 안에 층 정규화를 구현해보겠다. 이렇게 하려면 사용자 정의 메모리 셀을 정의해야 한다. 이 층은 call() 메서드가 다음 두 개의 매개변수를 받는 것을 제외하고는 일반적인 층이다. 현재 타임 스텝의 inputs과 이전 타임 스텝의 은닉 states이다. states 매개변수는 하나 이상의 텐서를 담은 리스트이다. 간단한 RNN 셀의 경우 이전 타임 스텝의 출력과 동일한 하나의 텐서를 담고 있다. 다른 셀의 경우에는 여러 상태 텐서를 가질 수 있다. 셀은 state_size 속성과 output_size 속성을 가져야 한다. 간단한 RNN에서는 둘 다 모두 유닛 개수와 동일하다. SimpleRNNCell처럼 작동하는 사용자 정의 메모리 셀을 구현한다. 다른 점은 각 타임 스텝마다 층 정규화를 적용한다.
'''

class LNSimpleRNNCell(keras.layers.Layer):
  def __init__(self, units, activation="tanh", **kwargs):
    super().__init__(**kwargs)
    self.state_size = units
    self.output_size = units
    self.simple_rnn_cell = keras.layers.SimpleRNNCell(units,
                                                      activation=None)
    self.layer_norm = keras.layers.LayerNormalization()
    self.activation = keras.activations.get(activation)
  def call(self, inputs, states):
    outputs, new_states = self.simple_rnn_cell(inputs, states)
    norm_outputs = self.activation(self.layer_norm(outputs))
    return norm_outputs, [norm_outputs]
  
'''
매우 직관적이다. LNSimpleRNNCell 클래스는 다른 사용자 정의 층과 마찬가지로 keras.layers.Layer 클래스를 상속한다. 생성자는 유닛 개수와 활성화 함수를 매개변수로 받고 state_size와 output_size 속성을 설정한 다음 활성화 함수 없이 SimpleRNNCell을 만든다. 그다음 생성자는 LayerNormalization 층을 만들고 마지막으로 원하는 활성화 함수를 선택한다. call() 메서드는 먼저 간단한 RNN 셀을 적용하여 현재 입력과 이전 은닉 상태의 선형 조합을 계산한다. 이 셀은 두 개의 결과를 반환한다. 그다음에 call() 메서드는 층 정규화와 활성화 함수를 차례대로 적용한다. 마지막으로 출력을 두 번 반환한다. 이 사용자 정의 셀을 사용하려면 keras.layers.RNN 층을 만들어 이 셀의 객체를 전달하면 된다.
'''

model = keras.models.Sequential([
                                 keras.layers.RNN(LNSimpleRNNCell(20), return_sequences=True,
                                                  input_shape=[None, 1]),
                                 keras.layers.RNN(LNSimpleRNNCell(20), return_sequences=True),
                                 keras.layers.TimeDistributed(keras.layers.Dense(10))
])

## 단기 기억 문제 해결하기 ##
model = keras.models.Sequential([
                                 keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1]),
                                 keras.layers.LSTM(20, return_sequences=True),
                                 keras.layers.TimeDistributed(keras.layers.Dense(10))
])

'''
또는 범용 목적의 keras.layers.RNN 층에 LSTMCell을 매개변수로 지정할 수도 있다.
'''

model = keras.models.Sequential([
                                 keras.layers.RNN(keras.layers.LSTMCell(20), return_sequences=True,
                                                  input_shape=[None, 1]),
                                 keras.layers.RNN(keras.layers.LSTMCell(20), return_sequences=True),
                                 keras.layers.TimeDistributed(keras.layers.Dense(10))
])

'''
하지만 LSTM 층이 GPU에서 실행할 때 최적화된 구현을 사용하므로 일반적으로 선호된다.
2D 합성곱 층이 이미지에 대해 몇 개의 매우 작은 커널이 슬라이딩하여 2D 특성 맵을 만든다는 것을 보았다. 비슷하게 1D 합성곱 층이 몇 개의 커널을 시퀀스 위를 슬라이딩하여 커널마다 1D 특성 맵을 출력한다. 각 커널은 매우 짧은 하나의 순차 패턴을 감지하도록 학습된다. 10개의 커널을 사용하면 이 층의 출력은 10개의 1차원 시퀀스로 구성된다. 또는 이 출력을 10차원 시퀀스 하나로 볼 수 있다. 이는 순환 층과 1D 합성곱 층을 섞어서 신경망을 구성할 수 있다는 뜻이다. 스트라이드 1과 "same" 패딩으로 1D 합성곱 층을 사용하면 출력 시퀀스의 길이는 입력 시퀀스와 같다. 하지만 "valid" 패딩과 1보다 큰 스트라이드를 사용하면 출력 시퀀스는 입력 시퀀스보다 짧아진다. 따라서 적절한 타깃이 만들어지는지 확인하자. 예를 들어, 다음 모델은 앞의 모델과 같다. 다만 스트라이드 2를 사용해 입력 시퀀스를 두 배로 다운샘플링하는 1D 합성곱 층으로 시작하는 것이 다르다. 커널 크기가 스트라이드보다 크므로 모든 입력을 사용하여 이 층의 출력을 계산한다. 따라서 모델이 중요하지 않은 세부 사항은 버리고 유용한 정보를 보존하도록 학습할 수 있다. 합성곱 층으로 시퀀스 길이를 줄이면 GRU 층이 더 긴 패턴을 감지하는 데 도움이 된다. 타깃에서 처음 세 개의 타임 스텝을 버리고 두 배로 다운샘플해야 한다.
'''

model = keras.models.Sequential([
                                 keras.layers.Conv1D(filters=20, kernel_size=4, strides=2, padding="valid",
                                                     input_shape=[None, 1]),
                                 keras.layers.GRU(20, return_sequences=True),
                                 keras.layers.GRU(20, return_sequences=True),
                                 keras.layers.TimeDistributed(keras.layers.Dense(10))
])

model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
history = model.fit(X_train, Y_train[:, 3::2], epochs=20,
                    validation_data=(X_valid, Y_valid[:, 3::2]))

'''
이 모델을 훈련하고 평가하면 지금까지 중에서 가장 좋은 모델임을 알 수 있다. 합성곱 층이 정말로 도움이 된다. 사실 순환 층을 완전히 제거하고 1D 합성곱 층만 사용할 수도 있다!
WaveNet 논문에서 저자들은 실제로 팽창 비율이 각각 1, 2, 4, 8, ..., 256, 512인 합성곱 층 10개를 쌓았다. 그다음 동일한 층 10개를 따로 그룹지어 쌓았다. 이런 팽창 비율을 가진 합성곱 층 10개가 1,024 크기의 커널 한 개로 이루어진 매우 효율적인 합성곱 층처럼 작동한다는 것을 보였다. 이런 이유 때문에 이 블럭을 3개 쌓았다. 각 층 이전의 팽창 비율과 동일한 개수의 0을 입력 시퀀스 왼쪽에 패딩으로 추가하여 네트워크를 통과하는 시퀀스 길이를 동일하게 만들었다. 다음은 앞과 동일한 시퀀스를 처리하는 간단한 WaveNet 구현이다.
'''

model = keras.models.Sequential() 
model.add(keras.layers.InputLayer(input_shape=[None, 1]))
for rate in (1, 2, 4, 8) * 2:
  model.add(keras.layers.Conv1D(filters=20, kernel_size=2, padding="causal",
                                activation="relu", dilation_rate=rate))
model.add(keras.layers.Conv1D(filters=10, kernel_size=1))
model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
history = model.fit(X_train, Y_train, epochs=20,
                    validation_data=(X_valid, Y_valid))

'''
이 Sequential 모델은 명시적인 입력층으로 시작한다. 그다음 이어서 "causal" 패딩을 사용한 1D 합성곱 층을 추가한다. 이렇게 하면 합성곱 층이 예측을 만들 때 미래의 시퀀스를 훔쳐보게 되지 않는다. 이때 동일하게 팽창 비율이 늘어나는 일련의 층을 반복한다. 즉 팽창 비율 1, 2, 4, 8의 층을 추가하고 다시 팽창 비율 1, 2, 4, 8의 층을 추가한다. 마지막으로 출력층을 추가한다. 이 층은 크기가 1인 필터 10개를 사용하고 활성화 함수가 없는 합성곱 층이다. 층에 추가한 패딩 덕분에 모든 합성곱 층은 입력 시퀀스의 길이와 동일한 시퀀스를 출력한다. 따라서 훈련하는 동안 전체 시퀀스를 타깃으로 사용할 수 있다. 잘라내거나 다운샘플링할 필요가 없다.
'''

