# 합성곱 층 #
## 텐서플로 구현 ##
'''
사이킷런의 load_sample_images()를 사용해 두 개의 샘플 이미지를 로드한다. 그다음 7*7 필터 두 개를 만들어 두 이미지에 모두 적용한다. 마지막으로, 만들어진 특성 맵 중 하나를 그린다. load_sample_images()를 사용하려면 pip로 Pillow 패키지를 설치해야 한다.
'''

from sklearn.datasets import load_sample_image

# 샘플 이미지 로드
china = load_sample_image("china.jpg") / 255
flower = load_sample_image("flower.jpg") / 255
images = np.array([china, flower])
batch_size, height, width, channels = images.shape

# 필터를 2개 만든다
filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters[:,3,:,0] = 1 # 수직선
filters[3,:,:,1] = 1 # 수평선

outputs = tf.nn.conv2d(images, filters, strides=1, padding="SAME")

plt.imshow(outputs[0,:,:,1], cmap="gray") # 첫 번째 이미지의 두 번째 특성 맵을 그린다
plt.show()

'''
- 각 컬러 채널의 픽셀 강도는 0에서 255 사이의 값을 가진 바이트 하나로 표현된다. 이 특성을 255로 나누어 0에서 1 사이의 실수로 바꾼다.
- 그다음 두 개의 7*7 필터를 만든다.
- 텐서플로 저수준 딥러닝 API 중 하나인 tf.nn.conv2d() 함수를 사용해 이 필터를 두 이미지에 적용한다. 이 예에서는 제로 패딩과 스트라이드 1을 사용한다.
- 마지막으로 만들어진 특성 맵 중 하나를 그래프로 그린다.
'''

'''
이 예에서는 필터를 직접 지정했지만 실제 CNN에서는 보통 훈련 가능한 변수로 필터를 정의하므로 신경망이 가장 잘 맞는 필터를 학습할 수 있다. 변수를 직접 만드는 것보다 keras.layers.Conv2D 층을 사용한다.
'''

conv = keras.layers.Conv2D(filters=32, kernel_size=3, strides=1,
                           padding="same", activation="relu")

'''
이 코드는 3*3 크기의 32개의 필터와 스트라이드 1, "same" 패딩을 사용하는 Conv2D 층을 만들고 출력을 위해 ReLU 활성화 함수를 적용한다. 여기에서 볼 수 있듯이 합성곱 층을 위해 꽤 많은 하이퍼파라미터가 필요하다. 필터의 수, 필터의 높이와 너비, 스트라이드, 패딩 종류이다. 항상 그렇듯이 정확한 하이퍼파라미터 값을 찾으려면 교차 검증을 사용해야 하지만 시간이 많이 걸린다.
'''

# 풀링 층 #
## 텐서플로 구현 ##
'''
텐서플로에서 최대 풀링 층을 구현하는 것은 아주 쉽다. 다음 코드는 2*2 커널을 사용해 최대 풀링 층을 만든다. 스트라이드의 기본값은 커널 크기이므로 이 층은 스트라이드 2를 사용한다. 기본적으로 "valid" 패딩을 사용한다.
'''

max_pool = keras.layers.MaxPool2D(pool_size=2)

'''
케라스는 깊이방향 풀링 층을 제공하지 않지만 텐서플로 저수준 딥러닝 API를 사용할 수 있다. tf.nn.max_pool() 함수를 사용하고 커널 크기와 스트라이드를 4개의 원소를 가진 튜플로 지정한다. 첫 번째 세 값은 1이어야 한다. 이는 배치, 높이, 너비 차원을 따라 커널 크기와 스트라이드가 1이란 뜻이다. 예를 들어 3과 같이 깊이 차원을 따라 원하는 커널 사이즈와 스트라이드를 마지막 값에 지정한다.
'''

output = tf.nn.max_pool(images,
                        ksize=(1,1,1,3),
                        strides=(1,1,1,3),
                        padding="valid")

'''
이를 케라스 모델의 층으로 사용하고 싶으면 Lambda 층으로 감싸면 된다.
'''

depth_pool = keras.layers.Lambda(
    lambda X: tf.nn.max_pool(X, ksize=(1,1,1,3), strides=(1,1,1,3),
                             padding="valid"))

global_avg_pool = keras.layers.GlobalAvgPool2D()

'''
이는 공간 방향을 따라 평균을 계산하는 Lambda 층과 동등하다.
'''

global_avg_pool = keras.layers.Lambda(lambda X: tf.reduce_mean(X, axis=[1, 2]))

# CNN 구조 #
'''
패션 MNIST 데이터셋 문제를 해결하기 위한 간단한 CNN이다.
'''

model = keras.models.Sequential([
                                 keras.layers.Conv2D(64, 7, activation="relu", padding="same",
                                                     input_shape=[28, 28, 1]),
                                 keras.layers.MaxPooling2D(2),
                                 keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
                                 keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
                                 keras.layers.MaxPooling2D(2),
                                 keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
                                 keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
                                 keras.layers.MaxPooling2D(2),
                                 keras.layers.Flatten(),
                                 keras.layers.Dense(128, activation="relu"),
                                 keras.layers.Dropout(0.5),
                                 keras.layers.Dense(64, activation="relu"),
                                 keras.layers.Dropout(0.5),
                                 keras.layers.Dense(10, activation="softmax")
])

'''
- 이미지가 아주 크지 않아서 첫 번째 층은 64개의 큰 필터와 스트라이드 1을 사용한다. 이미지가 28*28 픽셀 크기이고 하나의 컬러 채널이므로 input_shape=[28,28,1]로 지정한다.
- 그다음 풀링 크기가 2인 최대 풀링 층을 추가하여 공간 방향 차원을 절반으로 줄인다.
- 이와 동일한 구조를 두 번 반복한다. 최대 풀링 층이 뒤따르는 합성곱 층이 두 번 등장한다. 이미지가 클 때는 이 구조를 더 많이 반복할 수 있다.
- CNN이 출력층에 다다를수록 필터 개수가 늘어난다. 저수준 특성의 개수는 적지만 이를 연결하여 고수준 특성을 만들 수 있는 방법이 많기 때문에 이런 구조가 합리적이다. 풀링 층 다음에 필터 개수를 두 배로 늘리는 것이 일반적인 방법이다. 풀링 층이 공간 방향 차원을 절반으로 줄이므로 이어지는 층에서 파라미터 개수, 메모리 사용량, 계산 비용을 크게 늘리지 않고 특성 맵 개수를 두 배로 늘릴 수 있다.
- 그다음이 두 개의 은닉층과 하나의 출력층으로 구성된 완전 연결 네트워크이다. 밀집 네트워크는 샘플의 특성으로 1D 배열을 기대하므로 입력을 일렬로 펼쳐야 한다. 또 밀집 층 사이에 과대적합을 줄이기 위해 50%의 드롭아웃 비율을 가진 드롭아웃 층을 추가한다.
'''

# 케라스를 사용해 ResNet-34 CNN 구현하기 #
'''
케라스를 사용해 직접 ResNet-34 모델을 구현해보겠다. 먼저 ResidualUnit 층을 만든다.
'''

class ResidualUnit(keras.layers.Layer):
  def __init__(self, filters, strides=1, activation="relu", **kwargs):
    super().__init__(**kwargs)
    self.activation = keras.activations.get(activation)
    self.main_layers = [
                        keras.layers.Conv2D(filters, 3, strides=strides,
                                            padding="same", use_bias=False),
                        keras.layers.BatchNormalization(),
                        self.activation,
                        keras.layers.Conv2D(filters, 3, strides=1,
                                            padding="same", use_bias=False),
                        keras.layers.BatchNormalization()]
    self.skip_layers = []
    if strides > 1:
      self.skip_layers = [
                          keras.layers.Conv2D(filters, 1, strides=strides,
                                              padding="same", use_bias=False),
                          keras.layers.BatchNormalization()]

  def call(self, inputs):
    Z = inputs
    for layer in self.main_layers:
      Z = layer(Z) 
    skip_Z = inputs
    for layer in self.skip_layers:
      skip_Z = layer(skip_Z)
    return self.activation(Z + skip_Z)                          

'''
생성자에서 필요한 층을 모두 만든다. call() 메서드에서 입력을 main_layers와 skip_layers에 통과시킨 후 두 출력을 더하여 활성화 함수를 적용한다.
'''

'''
이 네트워크는 연속되어 길게 연결된 층이기 때문에 Sequential 클래스를 사용해 ResNet-34 모델을 만들 수 있다.
'''

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(64, 7, strides=2, input_shape=[224, 224, 3],
                              padding="same", use_bias=False))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"))
prev_filters = 64
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
  strides = 1 if filters == prev_filters else 2
  model.add(ResidualUnit(filters, strides=strides))
  prev_filters = filters
model.add(keras.layers.GlobalAvgPool2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10, activation="softmax"))

'''
조금 복잡한 부분은 모델에 ResidualUnit 층을 더하는 반복 루프이다. 처음 3개 RU는 64개의 필터를 가지고 그다음 4개 RU는 128개의 필터를 가지는 식이다. 필터 개수가 이전 RU와 동일할 경우는 스트라이드를 1로 설정한다. 아니면 스트라이드를 2로 설정한다. 그다음 ResidualUnit을 더하고 마지막에 prev_filters를 업데이트한다.
'''

# 케라스에서 제공하는 사전훈련된 모델 사용하기 #
'''
일반적으로 GoogLeNet이나 ResNet 같은 표준 모델을 직접 구현할 필요가 없다. keras.applications 패키지에 준비되어 있는 사전훈련된 모델을 코드 한 줄로 불러올 수 있다. 예를 들어 이미지넷 데이터셋에서 사전훈련된 ResNet-50 모델을 로드할 수 있다.
'''

model = keras.applications.resnet50.ResNet50(weights="imagenet")

'''
가중치를 다운로드한다. 이 모델을 사용하려면 이미지가 적절한 크기인지 확인해야 한다. ResNet-50 모델은 224*224 픽셀 크기의 이미지를 기대한다. 텐서플로의 tf.image.resize() 함수로 앞서 적재한 이미지의 크기를 바꿔보겠다.
'''

images_resized = tf.image.resize(images, [224, 224])

'''
사전훈련된 모델은 이미지가 적절한 방식으로 전처리되었다고 가정한다. 경우에 따라 0에서 1 사이 또는 -1에서 1 사이의 입력을 기대한다. 이를 위해 모델마다 이미지를 전처리해주는 preprocess_input() 함수를 제공한다. 이 함수는 픽셀값이 0에서 255 사이라고 가정한다. 따라서 images_resized에 255를 곱해야 한다.
'''

inputs = keras.applications.resnet50.preprocess_input(images_resized * 255)

'''
이제 사전훈련된 모델을 사용해 예측을 수행할 수 있다.
'''

Y_proba = model.predict(inputs)

'''
통상적인 구조대로 출력 Y_proba는 행이 하나의 이미지이고 열이 하나의 클래스인 행렬이다. 최상위 K개의 예측을 클래스 이름과 예측 클래스의 추정 확률을 출력하려면 decode_predictions() 함수를 사용한다. 각 이미지에 대해 최상위 K개의 예측을 담은 리스트를 반환한다. 각 예측은 클래스, 아이디, 이름, 확률을 포함한 튜플이다.
'''

top_K = keras.applications.resnet50.decode_predictions(Y_proba, top=3)
for image_index in range(len(images)):
	print("이미지 #{}".format(image_index))
	for class_id, name, y_proba in top_K[image_index]:
		print("	{} - {:12s} {:.2f}%".format(class_id, name, y_proba * 100))
	print()
	
# 사전훈련된 모델을 사용한 전이 학습 #
'''
충분하지 않은 훈련 데이터로 이미지 분류기를 훈련하려면 사전훈련된 모델의 하위층을 사용하는 것이 좋다. 예를 들어 사전훈련된 Xception 모델을 사용해 꽃 이미지를 분류하는 모델을 훈련해보겠다. 먼저 텐서플로 데이터셋을 사용해 데이터를 적재한다.
'''

import tensorflow_datasets as tfds

dataset, info = tfds.load("tf_flowers", as_supervised=True, with_info=True)
dataset_size = info.splits["train"].num_examples # 3670
class_names = info.features["label"].names # ["dandelion", "daisy", ...]
n_classes = info.features["label"].num_classes # 5

'''
with_info=True로 지정하면 데이터셋에 대한 정보를 얻을 수 있다. 여기에서는 데이터셋의 크기와 클래스의 이름을 얻는다. 이 데이터셋에는 "train" 세트만 있고 테스트 세트나 검증 세트는 없다. 따라서 훈련 세트를 나누어야 한다. TF 데이터셋에는 이를 위한 API가 제공된다. 예를 들어 데이터셋의 처음 10%를 테스트 세트로 사용하고 다음 15%를 검증 세트, 나머지 75%는 훈련 세트로 나눈다.
'''

test_set_raw, valid_set_raw, train_set_raw = tfds.load(
    "tf_flowers",
    split=["train[:10%]", "train[10%:25%]", "train[25%:]"],
    as_supervised=True)

'''
그다음 이미지를 전처리해야 한다. 이 CNN 모델은 224 * 224 크기 이미지를 기대하므로 크기를 조정해야 한다. 또한 xception에 패키지에 포함된 preprocess_input() 함수로 이미지를 전처리해야 한다.
'''
def central_crop(image):
    shape = tf.shape(image)
    min_dim = tf.reduce_min([shape[0], shape[1]])
    top_crop = (shape[0] - min_dim) // 4
    bottom_crop = shape[0] - top_crop
    left_crop = (shape[1] - min_dim) // 4
    right_crop = shape[1] - left_crop
    return image[top_crop:bottom_crop, left_crop:right_crop]

def random_crop(image):
    shape = tf.shape(image)
    min_dim = tf.reduce_min([shape[0], shape[1]]) * 90 // 100
    return tf.image.random_crop(image, [min_dim, min_dim, 3])

def preprocess(image, label, randomize=False):
    if randomize:
        cropped_image = random_crop(image)
        cropped_image = tf.image.random_flip_left_right(cropped_image)
    else:
        cropped_image = central_crop(image)
    resized_image = tf.image.resize(cropped_image, [224, 224])
    final_image = keras.applications.xception.preprocess_input(resized_image)
    return final_image, label

'''
훈련 세트를 섞은 다음 이 전처리 함수를 3개 데이터셋에 모두 적용한다. 그다음 배치 크기를 지정하고 프리페치를 적용한다.
'''

batch_size = 32
train_set = train_set_raw.shuffle(1000).repeat()
train_set = train_set.map(partial(preprocess, randomize=True)).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.map(preprocess).batch(batch_size).prefetch(1)
test_set = test_set_raw.map(preprocess).batch(batch_size).prefetch(1)

'''
그다음 이미지넷에서 사전훈련된 Xception 모델을 로드한다. include_top=False로 지정하여 네트워크의 최상층에 해당하는 전역 평균 풀링 층과 밀집 출력 층을 제외시킨다. 이 기반 모델의 출력을 바탕으로 새로운 전역 평균 풀링 층을 추가하고 그 뒤에 클래스마다 하나의 유닛과 소프트맥스 활성화 함수를 가진 밀집 출력 층을 놓는다. 마지막으로 케라스의 Model 클래스 객체를 만든다.
'''

base_model = keras.applications.xception.Xception(weights="imagenet",
                                                  include_top=False)
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation="softmax")(avg)
model = keras.models.Model(inputs=base_model.input, outputs=output)

'''
훈련 초기에는 사전훈련된 층의 가중치를 동결하는 것이 좋다.
'''

for layer in base_model.layers:
    layer.trainable = False
		
'''
마지막으로 모델을 컴파일하고 훈련을 시작한다.
'''

optimizer = keras.optimizers.SGD(learning_rate=0.2, momentum=0.9, decay=0.01)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])
history = model.fit(train_set,
                    steps_per_epoch=int(0.75 * dataset_size / batch_size),
                    validation_data=valid_set,
                    validation_steps=int(0.15 * dataset_size / batch_size),
                    epochs=5)

'''
모델을 몇 번의 에포크 동안 훈련하면 검증 정확도가 75~80%에 도달하고 더 나아지지 않을 것이다. 이는 새로 추가한 최상위 층이 잘 훈련되었다는 것을 의미한다. 따라서 이제 모든 층의 동결을 해제하고 훈련을 계속한다. 이때는 사전훈련된 가중치가 훼손되는 것을 피하기 위해 훨씬 작은 학습률을 사용한다.
'''

for layer in base_model.layers:
    layer.trainable = True

optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9,
                                 nesterov=True, decay=0.001)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])
history = model.fit(train_set,
                    steps_per_epoch=int(0.75 * dataset_size / batch_size),
                    validation_data=valid_set,
                    validation_steps=int(0.15 * dataset_size / batch_size),
                    epochs=40)

'''
잠시 후에 이 모델이 테스트 세트에서 95%의 정확도를 달성할 것이다. 이런 식으로 훌륭한 이미지 분류기를 훈련할 수 있다!
'''

# 분류와 위치 추정 #
'''
사진에서 물체의 위치를 추정하는 것은 회귀 작업으로 나타낼 수 있다. 물체 주위의 바운딩 박스(bounding box)를 예측하는 일반적인 방법은 물체 중심의 수평, 수직 좌표와 높이, 너비를 예측하는 것이다. 즉 네 개의 숫자를 예측해야 한다. 이 때문에 모델을 크게 바꿀 필요는 없다. 네 개의 유닛을 가진 두 번째 밀집 출력 층을 추가하고 MSE 손실을 사용해 훈련한다.
'''

base_model = keras.applications.xception.Xception(weights="imagenet",
                                                  include_top=False)
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
class_output = keras.layers.Dense(n_classes, activation="softmax")(avg)
loc_output = keras.layers.Dense(4)(avg)
model = keras.models.Model(inputs=base_model.input,
                           outputs=[class_output, loc_output])
model.compile(loss=["sparse_categorical_crossentropy", "mse"],
              loss_weights=[0.8, 0.2], # depends on what you care most about
              optimizer=optimizer, metrics=["accuracy"])

