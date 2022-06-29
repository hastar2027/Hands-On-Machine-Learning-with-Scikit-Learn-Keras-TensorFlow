# 과소완전 선형 오토인코더로 PCA 수행하기 #
from tensorflow import keras

encoder = keras.models.Sequential([keras.layers.Dense(2, input_shape=[3])])
decoder = keras.models.Sequential([keras.layers.Dense(3, input_shape=[2])])
autoencoder = keras.models.Sequential([encoder, decoder])

autoencoder.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate=1.5))

'''
- 오토인코더를 인코더와 디코더 두 개 컴포넌트로 구성한다. 둘 다 하나의 Dense 층을 가진 일반적인 Sequential 모델이다. 오토인코더는 인코더 다음에 디코더가 뒤따르는 Sequential 모델이다.
- 오토인코더의 출력 개수가 입력의 개수와 동일하다.
- 단순한 PCA를 수행하기 위해서는 활성화 함수를 사용하지 않으며, 비용 함수는 MSE이다.
'''

history = autoencoder.fit(X_train, X_train, epochs=20)
codings = encoder.predict(X_train)

'''
동일한 데이터셋 X_train이 입력과 타깃에도 사용된다는 것을 주목하자. 오토인코더는 데이터에 있는 분산이 가능한 많이 보존되도록 데이터를 투영할 최상의 2D 평면을 찾는다.
'''

# 적층 오토인코더 #
## 케라스를 사용하여 적층 오토인코더 구현하기 ##
stacked_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(30, activation="selu"),
])
stacked_decoder = keras.models.Sequential([
    keras.layers.Dense(100, activation="selu", input_shape=[30]),
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])
stacked_ae = keras.models.Sequential([stacked_encoder, stacked_decoder])
stacked_ae.compile(loss="binary_crossentropy",
                   optimizer=keras.optimizers.SGD(learning_rate=1.5), metrics=[rounded_accuracy])
history = stacked_ae.fit(X_train, X_train, epochs=20,
                         validation_data=(X_valid, X_valid))

'''
- 오토인코더 모델을 인코더와 디코더 두 개 서브 모델로 나눈다.
- 인코더는 28*28 픽셀의 흑백 이미지를 받는다. 그다음 각 이미지를 784 크기의 벡터로 표현하기 위해 펼친다. 이 벡터를 크기가 점점 줄어드는 Dense 층 두 개에 통과시킨다. 두 층은 모두 SELU 활성화 함수를 사용한다. 각 입력 이미지에 대해 인코더는 크기가 30인 벡터를 출력한다.
디코더는 크기가 30인 코딩을 받는다. 그다음 크기가 점점 커지는 Dense 층 두 개에 통과시킨다. 최종 벡터를 28*28 배열로 변경하여 디코더의 출력이 인코더의 입력과 동일한 크기가 되도록 만든다.
- 적층 오토인코더를 컴파일할 때 평균 제곱 오차 대신 이진 크로스 엔트로피 손실을 사용한다. 재구성 작업을 다중 레이블 이진 분류 문제로 다루는 것이다. 즉 각 픽셀의 강도는 픽셀이 검정일 확률을 나타낸다. 이런 식으로 문제를 정의하면 모델이 더 빠르게 수렴하는 경향이 있다.
- 마지막으로 X_train을 입력과 타깃으로 사용해 모델을 훈련한다.
'''

## 재구성 시각화 ##
def plot_image(image):
    plt.imshow(image, cmap="binary")
    plt.axis("off")
    
def show_reconstructions(model, images=X_valid, n_images=5):
    reconstructions = model.predict(images[:n_images])
    fig = plt.figure(figsize=(n_images * 1.5, 3))
    for image_index in range(n_images):
        plt.subplot(2, n_images, 1 + image_index)
        plot_image(images[image_index])
        plt.subplot(2, n_images, 1 + n_images + image_index)
        plot_image(reconstructions[image_index])
        
show_reconstructions(stacked_ae)

## 패션 MNIST 데이터셋 시각화 ##
from sklearn.manifold import TSNE

X_valid_compressed = stacked_encoder.predict(X_valid)
tsne = TSNE()
X_valid_2D = tsne.fit_transform(X_valid_compressed)

plt.scatter(X_valid_2D[:, 0], X_valid_2D[:, 1], c=y_valid, s=10, cmap="tab10")

## 가중치 묶기 ##
class DenseTranspose(keras.layers.Layer):
    def __init__(self, dense, activation=None, **kwargs):
        self.dense = dense
        self.activation = keras.activations.get(activation)
        super().__init__(**kwargs)
    def build(self, batch_input_shape):
        self.biases = self.add_weight(name="bias",
                                      shape=[self.dense.input_shape[-1]],
                                      initializer="zeros")
        super().build(batch_input_shape)
    def call(self, inputs):
        z = tf.matmul(inputs, self.dense.weights[0], transpose_b=True)
        return self.activation(z + self.biases)
      
'''
이 사용자 정의 층은 일반적인 Dense 층과 비슷하다. 하지만 다른 Dense 층의 전치된 가중치를 사용한다. 하지만 편향 벡터는 독자적으로 사용한다. 그다음 이전과 비슷하게 새로운 적층 오토인코더를 만든다. 이 디코더의 Dense 층은 인코더의 Dense 층과 묶여 있다.
'''

dense_1 = keras.layers.Dense(100, activation="selu")
dense_2 = keras.layers.Dense(30, activation="selu")

tied_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    dense_1,
    dense_2
])

tied_decoder = keras.models.Sequential([
    DenseTranspose(dense_2, activation="selu"),
    DenseTranspose(dense_1, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])

tied_ae = keras.models.Sequential([tied_encoder, tied_decoder])

# 합성곱 오토인코더 #
conv_encoder = keras.models.Sequential([
    keras.layers.Reshape([28, 28, 1], input_shape=[28, 28]),
    keras.layers.Conv2D(16, kernel_size=3, padding="SAME", activation="selu"),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Conv2D(32, kernel_size=3, padding="SAME", activation="selu"),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Conv2D(64, kernel_size=3, padding="SAME", activation="selu"),
    keras.layers.MaxPool2D(pool_size=2)
])
conv_decoder = keras.models.Sequential([
    keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="VALID", activation="selu",
                                 input_shape=[3, 3, 64]),
    keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, padding="SAME", activation="selu"),
    keras.layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding="SAME", activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])
conv_ae = keras.models.Sequential([conv_encoder, conv_decoder])

# 순환 오토인코더 #
recurrent_encoder = keras.models.Sequential([
    keras.layers.LSTM(100, return_sequences=True, input_shape=[28, 28]),
    keras.layers.LSTM(30)
])
recurrent_decoder = keras.models.Sequential([
    keras.layers.RepeatVector(28, input_shape=[30]),
    keras.layers.LSTM(100, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(28, activation="sigmoid"))
])
recurrent_ae = keras.models.Sequential([recurrent_encoder, recurrent_decoder])

'''
이 순환 오토인코더는 타임 스텝마다 28차원을 갖는 어떤 길이의 시퀀스로 처리할 수 있다. 편리하게도 이 말은 각 이미지를 행의 시퀀스로 간주하여 패션 MNIST 이미지를 처리할 수 있다는 뜻이다. 각 타임 스텝에서 이 RNN은 28픽셀의 행 하나를 처리한다. 당연히 어떤 종류의 시퀀스에도 순환 오토인코더를 사용할 수 있다. 타임 스텝마다 입력 벡터를 주입하기 위해 디코더의 첫 번째 층에 RepeatVector 층을 사용한 점을 주목하자.
'''

# 잡음 제거 오토인코더 #
'''
구현은 간단하다. 인코더의 입력에 적용한 Dropout 층이 있는 일반적인 적층 오토인코더이다. Dropout 층은 훈련하는 동안에만 활성화된다.
'''

dropout_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(30, activation="selu")
])
dropout_decoder = keras.models.Sequential([
    keras.layers.Dense(100, activation="selu", input_shape=[30]),
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])
dropout_ae = keras.models.Sequential([dropout_encoder, dropout_decoder])

# 희소 오토인코더 #
sparse_l1_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(300, activation="sigmoid"),
    keras.layers.ActivityRegularization(l1=1e-3)  # Alternatively, you could add
                                                  # activity_regularizer=keras.regularizers.l1(1e-3)
                                                  # to the previous layer.
])
sparse_l1_decoder = keras.models.Sequential([
    keras.layers.Dense(100, activation="selu", input_shape=[300]),
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])
sparse_l1_ae = keras.models.Sequential([sparse_l1_encoder, sparse_l1_decoder])

'''
ActivityRegularization 층은 입력을 그대로 반환하면서 훈련 손실에 입력의 절댓값의 합을 더한다. ActivityRegularization 층을 제거하고 이전 층에서 activity_regularizer=keras.regularizers.l1(1e-3)로 지정해도 동일하다. 이 규제는 신경망이 0에 가까운 코딩을 만들도록 유도하지만 입력을 올바르게 재구성하지 못하면 벌칙을 받기 때문에 적어도 0이 아닌 값이 조금은 출력되어야 한다. l2 노름 대신 l1 노름을 사용하면 신경망이 입력 이미지에서 불필요한 것을 제거하고 가장 중요한 코딩을 보전하도록 만든다.
'''

K = keras.backend
kl_divergence = keras.losses.kullback_leibler_divergence

class KLDivergenceRegularizer(keras.regularizers.Regularizer):
    def __init__(self, weight, target=0.1):
        self.weight = weight
        self.target = target
    def __call__(self, inputs):
        mean_activities = K.mean(inputs, axis=0)
        return self.weight * (
            kl_divergence(self.target, mean_activities) +
            kl_divergence(1. - self.target, 1. - mean_activities))
      
'''
코딩 층의 활성화에 KLDivergenceRegularizer를 적용해 희소 오토인코더를 만든다.
'''

kld_reg = KLDivergenceRegularizer(weight=0.05, target=0.1)
sparse_kl_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(300, activation="sigmoid", activity_regularizer=kld_reg)
])
sparse_kl_decoder = keras.models.Sequential([
    keras.layers.Dense(100, activation="selu", input_shape=[300]),
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])
sparse_kl_ae = keras.models.Sequential([sparse_kl_encoder, sparse_kl_decoder])

# 변이형 오토인코더 #
class Sampling(keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2) + mean 
      
'''
이 Sampling 층은 두 입력 mean와 log_var를 받는다. K.random_normal() 함수를 사용해 평균이 0이고 표준편차가 1인 정규분포에서 랜덤한 벡터를 샘플링한다. 그다음 exp(log_var / 2)를 곱하고 마지막으로 mean을 더한 결과를 반환한다. 이는 평균이 mean이고 표준편차가 sigma인 정규분포에서 코딩 벡터를 샘플링한다.
'''

codings_size = 10

inputs = keras.layers.Input(shape=[28, 28])
z = keras.layers.Flatten()(inputs)
z = keras.layers.Dense(150, activation="selu")(z)
z = keras.layers.Dense(100, activation="selu")(z)
codings_mean = keras.layers.Dense(codings_size)(z)
codings_log_var = keras.layers.Dense(codings_size)(z)
codings = Sampling()([codings_mean, codings_log_var])
variational_encoder = keras.models.Model(
    inputs=[inputs], outputs=[codings_mean, codings_log_var, codings])

'''
codings_mean와 codings_log_var를 출력하는 두 Dense 층이 동일한 입력을 사용한다. 그다음 codings_mean과 codings_log_var를 Sampling 층으로 전달한다. 마지막으로 variational_encoder 모델은 출력 세 개를 만든다. 여기에서는 조사 목적으로 codings_mean과 codings_log_var를 출력한다. 실제 사용하는 것은 마지막 출력이다.
'''

decoder_inputs = keras.layers.Input(shape=[codings_size])
x = keras.layers.Dense(100, activation="selu")(decoder_inputs)
x = keras.layers.Dense(150, activation="selu")(x)
x = keras.layers.Dense(28 * 28, activation="sigmoid")(x)
outputs = keras.layers.Reshape([28, 28])(x)
variational_decoder = keras.models.Model(inputs=[decoder_inputs], outputs=[outputs])

_, _, codings = variational_encoder(inputs)
reconstructions = variational_decoder(codings)
variational_ae = keras.models.Model(inputs=[inputs], outputs=[reconstructions])

latent_loss = -0.5 * K.sum(
    1 + codings_log_var - K.exp(codings_log_var) - K.square(codings_mean),
    axis=-1)
variational_ae.add_loss(K.mean(latent_loss) / 784.)
variational_ae.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=[rounded_accuracy])

'''
배치에 있는 각 샘플의 잠재 손실을 계산한다. 그다음 배치에 있는 모든 샘플의 평균 손실을 계산하고 재구성 손실에 비례해 적절한 크기가 되도록 784로 나눈다. 실제로 변이형 오토인코더의 재구성 손실은 픽셀마다 재구성 오차의 합이다. 하지만 케라스가 binary_crossentropy 손실을 계산할 때 합이 아니라 784개 전체 픽셀의 평균을 계산한다. 따라서 필요한 것보다 재구성 손실이 784배 작다. 평균이 아니라 합을 계산하는 사용자 정의 손실 함수를 정의할 수 있지만 잠재 손실을 784로 나누는 것이 더 간단하다.
이 문제에 잘 맞는 RMSprop 옵티마이저를 사용한다.
'''

history = variational_ae.fit(X_train, X_train, epochs=25, batch_size=128,
                             validation_data=(X_valid, X_valid))

## 패션 MNIST 이미지 생성하기 ##
codings = tf.random.normal(shape=[12, codings_size])
images = variational_decoder(codings).numpy()

'''
다음 코드 예제에서 12개의 코딩을 생성하여 이를 3*4 격자로 만든다. 텐서플로의 tf.image.resize() 함수를 사용해 이 격자를 5*7 크기로 바꾼다. 기본적으로 resize() 함수는 이중 선형 보간(bilinear interpolation)을 수행한다. 따라서 늘어난 모든 행과 열이 보간된 코딩을 가진다. 그다음 디코더로 이미지를 생성한다.
'''

codings_grid = tf.reshape(codings, [1, 3, 4, codings_size])
larger_grid = tf.image.resize(codings_grid, size=[5, 7])
interpolated_codings = tf.reshape(larger_grid, [-1, codings_size])
images = variational_decoder(interpolated_codings).numpy()

# 생성적 적대 신경망 #
'''
먼저 생성자와 판별자를 만들어야 한다. 생성자는 오토인코더의 디코더와 비슷하다. 판별자는 일반적인 이진 분류기이다. 각 훈련 반복의 두 번째 단계에서 생성자와 판별자가 연결된 전체 GAN 모델이 필요하다.
'''

codings_size = 30

generator = keras.models.Sequential([
    keras.layers.Dense(100, activation="selu", input_shape=[codings_size]),
    keras.layers.Dense(150, activation="selu"),
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])
discriminator = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(150, activation="selu"),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(1, activation="sigmoid")
])
gan = keras.models.Sequential([generator, discriminator])

'''
다음 이 모델들을 컴파일한다. 판별자는 이진 분류기이므로 자연스럽게 이진 크로스 엔트로피 손실을 사용한다. 생성자는 gan 모델을 통해서만 훈련되기 때문에 따로 컴파일할 필요가 없다. gan 모델도 이진 분류기이므로 이진 크로스 엔트로피 손실을 사용한다. 중요한 것은 두 번째 단계에서 판별자를 훈련하면 안 된다는 것이다. 따라서 gan 모델을 컴파일하기 전에 판별자가 훈련되지 않도록 설정해야 한다.
'''

discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")
discriminator.trainable = False
gan.compile(loss="binary_crossentropy", optimizer="rmsprop")

'''
훈련이 일반적인 반복이 아니기 때문에 fit() 메서드를 사용할 수 없다. 대신 사용자 정의 훈련 반복문을 만들겠다. 이를 위해 먼저 이미지를 순회하는 Dataset을 만들어야 한다.
'''

batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(1000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)

'''
이제 훈련 반복을 만들 준비가 되었다. 이를 train_gan() 함수로 감싼다.
'''

def train_gan(gan, dataset, batch_size, codings_size, n_epochs=50):
    generator, discriminator = gan.layers
    for epoch in range(n_epochs):
        for X_batch in dataset:
            # phase 1 - training the discriminator
            noise = tf.random.normal(shape=[batch_size, codings_size])
            generated_images = generator(noise)
            X_fake_and_real = tf.concat([generated_images, X_batch], axis=0)
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator.trainable = True
            discriminator.train_on_batch(X_fake_and_real, y1)
            # phase 2 - training the generator
            noise = tf.random.normal(shape=[batch_size, codings_size])
            y2 = tf.constant([[1.]] * batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y2)
            
train_gan(gan, dataset, batch_size, codings_size, n_epochs=1)

## 심층 합성곱 GAN ##
codings_size = 100

generator = keras.models.Sequential([
    keras.layers.Dense(7 * 7 * 128, input_shape=[codings_size]),
    keras.layers.Reshape([7, 7, 128]),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="SAME",
                                 activation="selu"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2DTranspose(1, kernel_size=5, strides=2, padding="SAME",
                                 activation="tanh"),
])
discriminator = keras.models.Sequential([
    keras.layers.Conv2D(64, kernel_size=5, strides=2, padding="SAME",
                        activation=keras.layers.LeakyReLU(0.2),
                        input_shape=[28, 28, 1]),
    keras.layers.Dropout(0.4),
    keras.layers.Conv2D(128, kernel_size=5, strides=2, padding="SAME",
                        activation=keras.layers.LeakyReLU(0.2)),
    keras.layers.Dropout(0.4),
    keras.layers.Flatten(),
    keras.layers.Dense(1, activation="sigmoid")
])
gan = keras.models.Sequential([generator, discriminator])

X_train_dcgan = X_train.reshape(-1, 28, 28, 1) * 2. - 1. # reshape and rescale

'''
판별자는 이진 분류를 위한 일반적인 CNN과 매우 비슷하다. 다만 이미지를 다운샘플링하기 위해 최대 풀링 층을 사용하지 않고 스트라이드 합성곱을 사용한다. 또한 LeakyReLU 활성화 함수를 사용한다는 것도 눈여겨보자.
전체적으로 DCGAN의 가이드라인을 따랐지만 판별자에 있는 BatchNormalization 층을 Dropout 층으로 바꾸고 생성자에서 ReLU를 SELU로 바꾸었다. 자유롭게 이 구조를 바꿔보자. 하이퍼파라미터에 얼마나 민감한지 볼 수 있다.
'''

