# 넘파이처럼 텐서플로 사용하기 #
# 변수 #
"""
tf.Variable은 tf.Tensor와 비슷하게 동작한다. 동일한 연산을 수행할 수 있고 넘파이와도 잘 호환된다.
까다로운 데이터 타입도 마찬가지이다. 하지만 assign() 메서드를 사용하여 변숫값을 바꿀 수 있다.
또한 원소의 assign() 메서드나 scatter_update(), scatter_nd_update() 메서드를 사용하여 개별 원소를 수정할 수도 있다.
"""
v.assign(2 * v) # => [[2., 4., 6.], [8., 10., 12.]]
v[0, 1].assign(42) # => [[2., 42., 6.], [8., 10., 12.]]
v[:, 2].assign([0., 1.]) # => [[2., 42., 0.], [8., 10., 1.]]
v.scatter_nd_update(indices = [[0, 0], [1, 2]], updates=[100., 200.]) # => [[100., 42., 0.], [8., 10., 200.]]

# 사용자 정의 모델과 훈련 알고리즘 #
# 사용자 정의 손실 함수 #
"""
회귀 모델을 훈련하는 데 훈련 세트에 잡음 데이터가 조금 있는 경우, 평균 제곱 오차 대신 후버(Huber) 손실을 사용하면 좋다.
후버 손실은 아직 공식 케라스 API에서 지원하지 않는다. tf.keras에서는 지원하기는 하지만, 마치 없는 것처럼 생각하고 구현해보겠다.
레이블과 예측을 매개변수로 받는 함수를 만들고 텐서플로 연산을 사용해 샘플의 손실을 계산하면 된다.
"""
def huber_fn(y_true, y_pred):
  error = y_true - y_pred
  is_small_error = tf.abs(error) < 1
  squared_loss = tf.square(error) / 2
  linear_loss = tf.abs(error) - 0.5
  return tf.where(is_small_error, squared_loss, linear_loss)

"""
이 손실을 사용해 케라스 모델의 컴파일 메서드를 호출하고 모델을 훈련할 수 있다.
"""
model.compile(loss=huber_fn, optimizer="nadam")
model.fit(X_train, y_train, epochs=2,
          validation_data=(X_valid_scaled, y_valid))

# 사용자 정의 요소를 가진 모델을 저장하고 로드하기 #
"""
케라스가 함수 이름을 저장하므로 사용자 정의 손실 함수를 사용하는 모델은 아무 이상 없이 저장된다.
모델을 로드(load)할 때는 함수 이름과 실제 함수를 매핑한 딕셔너리를 전달해야 한다.
좀 더 일반적으로 사용자 정의 객체를 포함한 모델을 로드할 때는 그 이름과 객체를 매핑해야 한다.
"""
model = keras.models.load_model("my_model_with_a_custom_loss.h5",
                                custom_objects={"huber_fn": huber_fn})

"""
앞서 구현한 함수는 -1과 1 사이의 오차는 작은 것으로 간주한다.
다른 기준이 필요할 때는 어떻게 해야 할까? 한 가지 방법은 매개변수를 받을 수 있는 함수를 만드는 것이다.
"""
def create_huber(threshold=1.0):
  def huber_fn(y_true, y_pred):
    error = y_true - y_pred
    is_small_error = tf.abs(error) < threshold
    squared_loss = tf.square(error) / 2
    linear_loss = threshold * tf.abs(error) - threshold**2 / 2
    return tf.where(is_small_error, squared_loss, linear_loss)
  return huber_fn 
model.compile(loss=create_huber(2.0), optimizer="nadam")

"""
안타깝지만 모델을 저장할 때 이 threshold 값은 저장되지 않는다.
따라서 모델을 로드할 때 threshold 값을 지정해야 한다.
"""
model = keras.models.load_model("my_model_with_a_custom_loss_threshold_2.h5",
                                custom_objects={"huber_fn": create_huber(2.0)})

"""
이 문제는 keras.losses.Loss 클래스를 상속하고 get_config() 메서드를 구현하여 해결할 수 있다.
"""
class HuberLoss(keras.losses.Loss):
  def __init__(self, threshold=1.0, **kwargs):
    self.threshold = threshold
    super().__init__(**kwargs)
  def call(self, y_true, y_pred):
    error = y_true - y_pred
    is_small_error = tf.abs(error) < self.threshold
    squared_loss = tf.square(error) / 2
    linear_loss = self.threshold * tf.abs(error) - self.threshold**2 / 2
    return tf.where(is_small_error, squared_loss, linear_loss)
  def get_config(self):
    base_config = super().get_config()
    return {**base_config, "threshold": self.threshold}

"""
모델을 컴파일할 때 이 클래스의 인스턴스를 사용할 수 있다.
"""
model.compile(loss=HuberLoss(2.), optimizer="nadam")

"""
이 모델을 저장할 때 임곗값도 함께 저장된다.
모델을 로드할 때 클래스 이름과 클래스 자체를 매핑해주어야 한다.
"""
model = keras.models.load_model("my_model_with_a_custom_loss_class.h5",
                                custom_objects={"HuberLoss": HuberLoss})

# 활성화 함수, 초기화, 규제, 제한을 커스터마이징하기 #
"""
손실, 규제, 제한, 초기화, 지표, 활성화 함수, 층, 모델과 같은 대부분의 케라스 기능은 유사한 방법으로 커스터마이징할 수 있다.
대부분의 경우 적절한 입력과 출력을 가진 간단한 함수를 작성하면 된다.
"""
# 사용자 정의 활성화 함수, 사용자 정의 글로럿 초기화, 사용자 정의 l_1 규제, 양수인 가중치만 남기는 사용자 정의 제한
def my_softplus(z): # tf.nn.softplus(z)가 큰 입력을 더 잘 다룬다
  return tf.math.log(tf.exp(z) + 1.0)

def my_glorot_initializer(shape, dtype=tf.float32):
  stddev = tf.sqrt(2. / (shape[0] + shape[1]))
  return tf.random.normal(shape, stddev=stddev, dtype=dtype)

def my_l1_regularizer(weights): 
  return tf.reduce_sum(tf.abs(0.01 * weights))

def my_positive_weights(weights): # tf.nn.relu(weights)와 반환값이 같다
  return tf.where(weights < 0., tf.zeros_like(weights), weights)

"""
매개변수는 사용자 정의하려는 함수의 종류에 따라 다르다.
만들어진 사용자 정의 함수는 보통의 함수와 동일하게 사용할 수 있다.
"""
layer = keras.layers.Dense(30, activation=my_softplus,
                           kernel_initializer=my_glorot_initializer,
                           kernel_regularizer=my_l1_regularizer,
                           kernel_constraint=my_positive_weights)

"""
함수가 모델과 함꼐 저장해야 할 하이퍼파라미터를 가지고 있다면 keras.regularizers.Regularizer, keras.constraints.Constraint, keras.initializers.Initializer,
keras.layers.Layer와 같이 적절한 클래스를 상속한다.
"""
# factor 하이퍼파라미터를 저장하는 l_1 규제를 위한 간단한 클래스
class MyL1Regularizer(keras.regularizers.Regularizer):
  def __init__(self, factor):
    self.factor = factor
  def __call__(self, weights):
    return tf.reduce_sum(tf.abs(self.factor * weights))
  def get_config(self):
    return {"factor": self.factor}
  
# 사용자 정의 지표 #
"""
대부분의 경우 사용자 지표 함수를 만드는 것은 사용자 손실 함수를 만드는 것과 동일하다.
후버 손실 함수는 지표로도 사용해도 잘 동작한다.
"""
model.compile(loss="mse", optimizer="nadam", metrics=[create_huber(2.0)])

"""
스트리밍 지표를 만들고 싶다면 keras.metrics.Metric 클래스를 상속한다.
"""
# 전체 후버 손실과 지금까지 처리한 샘플 수를 기록하는 클래스
# 결괏값을 요청하면 평균 후버 손실이 반환
class HuberMetric(keras.metrics.Metric):
  def __init__(self, threshold=1.0, **kwargs):
    super().__init__(**kwargs) # 기본 매개변수 처리(예, dtype)
    self.threshold = threshold
    self.huber_fn = create_huber(threshold)
    self.total = self.add_weight("total", initializer="zeros")
    self.count = self.add_weight("count", initializer="zeros")
  def update_state(self, y_true, y_pred, sample_weight=None):
    metric = self.huber_fn(y_true, y_pred)
    self.total.assign_add(tf.reduce_sum(metric))
    self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
  def result(self):
    return self.total / self.count
  def get_config(self):
    base_config = super().get_config()
    return {**base_config, "threshold": self.threshold}
  
# 사용자 정의 층 #
"""
keras.layers.Flatten나 keras.layers.ReLU와 같은 층은 가중치가 없다.
가중치가 필요 없는 사용자 정의 층을 만들기 위한 가장 간단한 방법은 파이썬 함수를 만든 후 keras.layers.Lambda 층으로 감싸는 것이다.
"""
# 입력에 지수 함수를 적용하는 층
exponential_layer = keras.layers.Lambda(lambda x: tf.exp(x))

"""
상태가 있는 층을 만들려면 keras.layers.Layer를 상속해야 한다.
"""
# Dense 층의 간소화 버전
class MyDense(keras.layers.Layer):
  def __init__(self, units, activation=None, **kwargs):
    super().__init__(**kwargs)
    self.units = units
    self.activation = keras.activations.get(activation)

  def build(self, batch_input_shape):
    self.kernel = self.add_weight(
        name="kernel", shape=[batch_input_shape[-1], self.units],
        initializer="glorot_normal")
    self.bias = self.add_weight(
        name="bias", shape=[self.units], initializer="zeros")
    super().build(batch_input_shape) # 마지막에 호출해야 한다.

  def call(self, X):
    return self.activation(X @ self.kernel + self.bias)

  def compute_ouput_shape(self, batch_input_shape):
    return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.units])

  def get_config(self):
    base_config = super().get_config()
    return {**base_config, "units": self.units,
            "activation": keras.activations.serialize(self.activation)}
  
"""
여러 가지 입력을 받는 층을 만들려면 call() 메서드에 모든 입력이 포함된 튜플을 매개변수 값으로 전달해야 한다.
비슷하게 compute_output_shape() 메서드의 매개변수도 각 입력의 배치 크기를 담은 튜플이어야 한다.
여러 출력을 가진 층을 만들려면 call() 메서드가 출력의 리스트를 반환해야 한다.
compute_output_shape() 메서드는 배치 출력 크기의 리스트를 반환해야 한다.
"""
# 두 개의 입력과 세 개의 출력을 만드는 층
class MyMultiLayer(keras.layers.Layer):
  def call(self, X):
    X1, X2 = X
    return [X1 + X2, X1 * X2, X1 / X2]

  def compute_output_shape(self, batch_input_shape):
    b1, b2 = batch_input_shape
    return [b1, b1, b1] # 올바르게 브로드캐스팅되어야 한다.
  
"""
훈련과 테스트에서 다르게 동작하는 층이 필요하다면 call() 메서드에 training 매개변수를 추가하여 훈련인지 테스트인지를 결정해야 한다.
"""
# 훈련하는 동안 가우스 잡음을 추가하고 테스트 시에는 아무것도 하지 않는 층
class MyGaussianNoise(keras.layers.Layer):
  def __init__(self, stddev, **kwargs):
    super().__init__(**kwargs)
    self.stddev = stddev

  def call(self, X, training=None):
    if training:
      noise = tf.random.normal(tf.shape(X), stddev=self.stddev)
      return X + noise
    else:
      return X

  def compute_output_shape(self, batch_input_shape):
    return batch_input_shape
  
# 사용자 정의 모델 #
"""
입력이 첫 번째 완전 연결 층을 통과하여 두 개의 완전 연결 층과 스킵 연결로 구성된 잔차 블록(residual block)으로 전달된다.
그다음 동일한 잔차 블록에 세 번 더 통과시킨다.
그다음 두 번째 잔차 블록을 지나 마지막 출력이 완전 연결된 출력 층에 전달된다.
이런 구조는 실제 사용되는 것이 아니다. 필요하다면 반복문이나 스킵 연결도 있는 어떤 종류의 모델도 쉽게 만들 수 있다는 것을 보이기 위한 예시일 뿐이다.
"""
# 이 모델을 구현하려면 동일한 블록을 여러 개 만들어야 하므로 먼저 ResidualBlock 층을 만들겠다.
class ResidualBlock(keras.layers.Layer):
  def __init__(self, n_layers, n_neurons, **kwargs):
    super().__init__(**kwargs)
    self.hidden = [keras.layers.Dense(n_neurons, activation="elu",
                                      kernel_initializer="he_normal")
    for _ in range(n_layers)]
    
  def call(self, inputs):
    Z = inputs
    for layer in self.hidden:
      Z = layer(Z)
    return inputs + Z
  
"""
이 층은 다른 층을 포함하고 있기 때문에 조금 특별하다.
케라스가 알아서 추적해야 할 객체가 담긴 hidden 속성을 감지하고 필요한 변수를 자동으로 이 층의 변수 리스트에 추가한다.
이 클래스의 나머지는 그 자체로 이해할 수 있다.
"""
# 서브클래싱 API를 사용해 이 모델 정의
class ResidualRegressor(keras.Model):
  def __init__(self, output_dim, **kwargs):
    super().__init__(**kwargs)
    self.hidden1 = keras.layers.Dense(30, activation="elu",
                                      kernel_initializer="he_normal")
    self.block1 = ResidualBlock(2, 30)
    self.block2 = ResidualBlock(2, 30)
    self.out = keras.layers.Dense(output_dim)

  def call(self, inputs):
    Z = self.hidden1(inputs)
    for _ in range(1 + 3):
      Z = self.block1(Z)
    Z = self.block2(Z)
    return self.out(Z)
  
# 모델 구성 요소에 기반한 손실과 지표 #
"""
모델 구성 요소에 기반한 손실을 정의하고 계산하여 add_loss() 메서드에 그 결과를 전달한다.
예를 들어 다섯 개의 은닉층과 출력층으로 구성된 회귀용 MLP 모델을 만들어보자.
이 모델은 맨 위의 은닉층에 보조 출력을 가진다. 이 보조 출력에 연결된 손실을 재구성 손실(reconstruction loss)이라고 부르겠다.
즉 재구성과 입력 사이의 평균 제곱 오차이다.
재구성 손실을 주 손실에 더하여 회귀 작업에 직접적으로 도움이 되지 않은 정보일지라도 모델이 은닉층을 통과하면서 가능한 많은 정보를 유지하도록 유도한다.
사실 이런 손실이 이따금 일반화 성능을 향상시킨다.
"""
# 사용자 정의 재구성 손실을 가지는 모델을 만드는 코드
class ReconstructingRegressor(keras.Model):
  def __init__(self, output_dim, **kwargs):
    super().__init__(**kwargs)
    self.hidden = [keras.layers.Dense(30, activation="selu",
                                      kernel_initializer="lecun_normal")
    for _ in range(5)]
    self.out = keras.layers.Dense(output_dim)

  def build(self, batch_input_shape):
    n_inputs = batch_input_shape[-1]
    self.reconstruct = keras.layers.Dense(n_inputs)
    super().build(batch_input_shape)

  def call(self, inputs):
    Z = inputs
    for layer in self.hidden:
      Z = layer(Z)
    reconstruction = self.reconstruct(Z)
    recon_loss = tf.reduce_mean(tf.square(reconstruction - inputs))
    self.add_loss(0.05 * recon_loss)
    return self.out(Z)
  
# 자동 미분을 사용하여 그레이디언트 계산하기 #
# 자동 미분을 사용하여 그레이디언트를 자동으로 계산하는 방법을 이해하기 위한 간단한 함수
def f(w1, w2):
  return 3 * w1 ** 2 + 2 * w1 * w2

"""
이 방법은 잘 동작하고 구현하기도 쉽다. 하지만 근삿값이고 무엇보다도 파라미터보다 적어도 한 번씩은 함수 f()를 호출해야 한다.
파라미터마다 적어도 한번씩 f()를 호출하므로 대규모 신경망에서는 적용하기 어려운 방법이다. 대신 자동 미분을 사용해보자.
텐서플로에서는 아주 쉽게 계산한다.
"""
w1, w2 = tf.Variable(5.), tf.Variable(3.)
with tf.GradientTape() as tape:
  z = f(w1, w2)

gradients = tape.gradient(z, [w1, w2])

"""
gradient() 메서드가 호출된 후에는 자동으로 테이프가 즉시 지워진다.
따라서 gradient() 메서드를 두 번 호출하면 예외가 발생한다.
"""
with tf.GradientTape() as tape:
  z = f(w1, w2)

dz_dw1 = tape.gradient(z, w1) # => 36.0 텐서
dz_dw2 = tape.gradient(z, w2) # 실행 에러!

"""
gradient() 메서드를 한 번 이상 호출해야 한다면 지속 가능한 테이프를 만들고 사용이 끝난 후 테이프를 삭제하여 리소스를 해제해야 한다.
"""
with tf.GradientTape(persistent=True) as tape:
  z = f(w1, w2)

dz_dw1 = tape.gradient(z, w1) # => 36.0 텐서
dz_dw2 = tape.gradient(z, w2) # => 10.0 텐서, 작동 이상 없음!
del tape

"""
기본적으로 테이프는 변수가 포함된 연산만을 기록한다.
만약 변수가 아닌 다른 객체에 대한 z의 그레이디언트를 계산하려면 None이 반환된다.
"""
c1, c2 = tf.constant(5.), tf.constant(3.)
with tf.GradientTape() as tape:
  z = f(c1, c2)

gradients = tape.gradient(z, [c1, c2]) # [None, None]이 반환

"""
하지만 필요한 어떤 텐서라도 감시하여 관련된 모든 연산을 기록하도록 강제할 수 있다.
그다음 변수처럼 이런 텐서에 대해 그레이디언트를 계산할 수 있다.
"""
with tf.GradientTape() as tape:
  tape.watch(c1)
  tape.watch(c2)
  z = f(c1, c2)

gradients = tape.gradient(z, [c1, c2]) # [36. 텐서, 10. 텐서] 반환

"""
어떤 경우에는 신경망의 일부분에 그레이디언트가 역전파되지 않도록 막을 필요가 있다.
이렇게 하려면 tf.stop_gradient() 함수를 사용해야 한다. 이 함수는 정방향 계산을 할 때 입력을 반환한다.
하지만 역전파 시에는 그레이디언트를 전파하지 않는다.
"""
def f(w1, w2):
  return 3 * w1 ** 2 + tf.stop_gradient(2 * w1 * w2)

with tf.GradientTape() as tape:
  z = f(w1, w2) # stop_gradient() 없을 때와 결과가 같다.

gradients = tape.gradient(z, [w1, w2]) # => [30. 텐서, None] 반환

"""
자동 미분을 사용하여 이 함수의 그레이디언트를 계산하는 것은 수치적으로 불안정하다. 즉 부동소수점 정밀도 오류로 인해 자동 미분이 무한 나누기 무한을 계산하게 된다.
다행히 수치적으로 안전한 소프트플러스(softplus)의 도함수 1 / (1 + 1 / exp(x))를 해석적으로 구할 수 있다. 
그다음 @tf.custom_gradient 데코레이터(decorator)를 사용하고 일반 출력과 도함수를 계산하는 함수를 반환하여 텐서플로가 my_softplus() 함수의 그레이디언트를 계산할 때
안전한 함수를 사용하도록 만들 수 있다.
"""
@tf.custom_gradient
def my_better_softplus(z):
  exp = tf.exp(z)
  def my_softplus_gradients(grad):
    return grad / (1 + 1 / exp)
  return tf.math.log(exp + 1), my_softplus_gradients

# 사용자 정의 훈련 반복 #
# 훈련 반복을 직접 다루기 때문에 컴파일할 필요가 없다.
l2_reg = keras.regularizers.l2(0.05)
model = keras.models.Sequential([
                                 keras.layers.Dense(30, activation="elu", kernel_initializer="he_normal",
                                                    kernel_regularizer=l2_reg),
                                 keras.layers.Dense(1, kernel_regularizer=l2_reg)
])

# 훈련 세트에서 샘플 배치를 랜덤하게 추출하는 작은 함수를 만든다.
def random_batch(X, y, batch_size=32):
  idx = np.random.randint(len(X), size=batch_size)
  return X[idx], y[idx]

# 현재 스텝 수, 전체 스텝 횟수, 에포크 시작부터 평균 손실, 그 외 다른 지표를 포함하여 훈련 상태를 출력하는 함수도 만든다.
def print_status_bar(iteration, total, loss, metrics=None):
  metrics = " - ".join(["{}: {:.4f}".format(m.name, m.result())
  for m in [loss] + (metrics or [])])
  end = "" if iteration < total else "\n"
  print("\r{}/{} - ".format(iteration, total) + metrics,
        end=end)
  
# 몇 개의 하이퍼파라미터를 정의하고 옵티마이저, 손실 함수, 지표를 선택해야 한다.
n_epochs = 5
batch_size = 32
n_steps = len(X_train) // batch_size
optimizer = keras.optimizers.Nadam(lr=0.01)
loss_fn = keras.losses.mean_squared_error
mean_loss = keras.metrics.Mean()
metrics = [keras.metrics.MeanAbsoluteError()]

for epoch in range(1, n_epochs + 1):
  print("에포크 {}/{}".format(epoch, n_epochs))
  for step in range(1, n_steps + 1):
    X_batch, y_batch = random_batch(X_train_scaled, y_train)
    with tf.GradientTape() as tape:
      y_pred = model(X_batch, training=True)
      main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
      loss = tf.add_n([main_loss] + model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    mean_loss(loss)
    for metric in metrics:
      metric(y_batch, y_pred)
    print_status_bar(step * batch_size, len(y_train), mean_loss, metrics)
  print_status_bar(len(y_train), len(y_train), mean_loss, metrics)
  for metric in [mean_loss] + metrics:
    metric.reset_states()
    
"""
모델에 가중치 제한을 추가하면 apply_gradients() 다음에 이 제한을 적용하도록 훈련 반복을 수정해야 한다.
"""
for variable in model.variables:
  if variable.constraint is not None:
    variable.assign(variable.constraint(variable))
    
# 텐서플로 함수와 그래프 #
"""
텐서플로 1에서 그래프는 텐서플로 API의 핵심이므로 피할 수가 없었다.
텐서플로 2에도 그래프가 있지만 이전만큼 핵심적이지는 않고 사용하기 매우 쉽다.
"""
# 입력의 세 제곱을 계산하는 함수
def cube(x):
  return x ** 3

"""
tf.function()을 사용하여 이 파이썬 함수를 텐서플로 함수(TensorFlow function)로 바꿀 수 있다.
내부적으로 tf.function()은 cube() 함수에서 수행되는 계산을 분석하고 동일한 작업을 수행하는 계산 그래프를 생성한다.
"""
# tf.function 데코레이터가 실제로는 더 널리 사용된다.
@tf.function
def tf_cube(x):
  return x ** 3
