# 데이터 API #
## 데이터 셔플링 ##
### 여러 파일에서 한 줄씩 번갈아 읽기
'''
train_filepaths가 훈련 파일 경로를 담은 리스트라고 가정한다.
이제 이런 파일 경로가 담긴 데이터셋을 만든다.
'''

filepath_dataset = tf.data.Dataset.list_files(train_filepaths, seed=42)

'''
기본적으로 list_files() 함수는 파일 경로를 섞은 데이터셋을 반환한다.
일반적으로 이는 바람직한 설정이지만 어떤 이유로 이를 원하지 않는다면 shuffle=False로 지정할 수 있다.
그다음 interleave() 메서드를 호출하여 한 번에 다섯 개의 파일을 한 줄씩 번갈아 읽는다.
'''

n_readers = 5
dataset = filepath_dataset.interleave(
    lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
    cycle_length=n_readers)

'''
interleave() 메서드는 filepath_dataset에 있는 다섯 개의 파일 경로에서 데이터를 읽는 데이터셋을 만든다.
이 메서드에 전달한 함수를 각 파일에 대해 호출하여 새로운 데이터셋을 만들 것이다.
명확하게 말해서 이 단계에서 총 7개의 데이터셋이 있다.
파일 경로 데이터셋, 인터리브 데이터셋, 인터리브 데이터셋에 의해 내부적으로 생성된 다섯 개의 TextLineDataset이 있다.
인터리브 데이터셋을 반복 구문에 사용하면 다섯 개의 TextLineDataset을 순회한다.
모든 데이터셋이 아이템이 소진될 때까지 한 번에 한 줄씩 읽는다.
그후에 filepath_dataset에서 다음 다섯 개의 파일 경로를 가져오고 동일한 방식으로 한 줄씩 읽는다.
모든 파일 경로가 소진될 때까지 계속된다.
'''

## 데이터 전처리 ##
'''
전처리를 수행하기 위한 간단한 함수를 만들어보자.
'''

n_inputs = 8

def preprocess(line): 
  defs = [0.] * n_inputs + [tf.constant([], dtype=tf.float32)]
  fields = tf.io.decode_csv(line, record_defaults=defs) 
  x = tf.stack(fields[:-1])
  y = tf.stack(fields[-1:])
  return(x - X_mean) / X_std, y

## 데이터 적재와 전처리를 합치기 ##
'''
재사용 가능한 코드를 만들기 위해 지금까지 언급한 모든 것을 하나의 헬퍼 함수로 만들겠다.
이 함수는 CSV 파일에서 캘리포니아 주택 데이터셋을 효율적으로 적재하고 전처리, 셔플링, 반복, 배치를 적용한 데이터셋을 만들어 반환한다.
'''

def csv_reader_dataset(filepaths, repeat=1, n_readers=5,
                       n_read_threads=None, shuffle_buffer_size=10000,
                       n_parse_threads=5, batch_size=32):
  dataset = tf.data.Dataset.list_files(filepaths).repeat(repeat) 
  dataset = dataset.interleave(
      lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
      cycle_length=n_readers, num_parallel_calls=n_read_threads)
  dataset = dataset.shuffle(shuffle_buffer_size)
  dataset = dataset.map(preprocess, num_parallel_calls=n_parse_threads)
  return dataset.batch(batch_size).prefetch(1)

## tf.keras와 데이터셋 사용하기 ##
'''
csv_reader_dataset() 함수로 훈련 세트로 사용할 데이터셋을 만들 수 있다.
tf.keras에서 반복을 처리하므로 반복을 지정할 필요가 없다.
검증 세트와 테스트 세트로 사용할 데이터셋도 만든다.
'''

train_set = csv_reader_dataset(train_filepaths)
valid_set = csv_reader_dataset(valid_filepaths)
test_set = csv_reader_dataset(test_filepaths)

'''
이제 케라스 모델을 만들고 이 데이터셋으로 훈련할 수 있다.
fit() 메서드에 X_train, y_train, X_valid, y_valid 대신 훈련 데이터셋과 검증 데이터셋을 전달하기만 하면 된다.
'''

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    keras.layers.Dense(1),
])
model.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate=1e-3))
model.fit(train_set, epochs=10, validation_data=valid_set)

'''
비슷하게 evaluate()와 predict() 메서드에 데이터셋을 전달할 수 있다.
'''

model.evaluate(test_set)
new_set = test_set.take(3).map(lambda X, y: X) # 새로운 샘플이 3개 있다고 가정한다.
model.predict(new_set) # 새로운 샘플이 들어 있는 데이터셋

'''
다른 세트와 달리 new_set은 레이블을 가지고 있지 않다.
이런 모든 경우에 데이터셋 대신에 넘파이 배열을 사용할 수 있다.
자신만의 훈련 반복을 만들고 싶다면 그냥 훈련 세트를 반복하면 된다.
'''

optimizer = keras.optimizers.Nadam(learning_rate=0.01)
loss_fn = keras.losses.mean_squared_error

n_epochs = 5
batch_size = 32
n_steps_per_epoch = len(X_train) // batch_size
total_steps = n_epochs * n_steps_per_epoch
global_step = 0
for X_batch, y_batch in train_set.take(total_steps):
    global_step += 1
    print("\rGlobal step {}/{}".format(global_step, total_steps), end="")
    with tf.GradientTape() as tape:
        y_pred = model(X_batch)
        main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
        loss = tf.add_n([main_loss] + model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))    
    
'''
사실 전체 훈련 반복을 수행하는 텐서플로 함수를 만들 수도 있다.
'''

@tf.function
def train(model, n_epochs, batch_size=32,
          n_readers=5, n_read_threads=5, shuffle_buffer_size=10000, n_parse_threads=5):
    train_set = csv_reader_dataset(train_filepaths, repeat=n_epochs, n_readers=n_readers,
                       n_read_threads=n_read_threads, shuffle_buffer_size=shuffle_buffer_size,
                       n_parse_threads=n_parse_threads, batch_size=batch_size)
    n_steps_per_epoch = len(X_train) // batch_size
    total_steps = n_epochs * n_steps_per_epoch
    global_step = 0
    for X_batch, y_batch in train_set.take(total_steps):
        global_step += 1
        if tf.equal(global_step % 100, 0):
            tf.print("\rGlobal step", global_step, "/", total_steps)
        with tf.GradientTape() as tape:
            y_pred = model(X_batch)
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
            loss = tf.add_n([main_loss] + model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
# TFRecord 포맷 #
'''
대용량 데이터를 저장하고 효율적으로 읽기 위해 텐서플로가 선호하는 포맷은 TFRecord이다.
TFRecord는 크기가 다른 연속된 이진 레코드를 저장하는 단순한 이진 포맷이다.
tf.io.TFRecordWriter 클래스를 사용해 TFRecord를 손쉽게 만들 수 있다.
'''

with tf.io.TFRecordWriter("my_data.tfrecord") as f:
  f.write(b"This is the first record")
  f.write(b"And this is the second record")
  
'''
그다음 tf.data.TFRecordDataset을 사용해 하나 이상의 TFRecord를 읽을 수 있다.
'''

filepaths = ["my_data.tfrecord"]
dataset = tf.data.TFRecordDataset(filepaths)
for item in dataset:
  print(item)

## 압축된 TFRecord 파일 ##
'''
이따금 TFRecord 파일을 압축할 필요가 있다.
특히 네트워크를 통해 읽어야 하는 경우이다.
options 매개변수를 사용하여 압축된 TFRecord 파일을 만들 수 있다.
'''

options = tf.io.TFRecordOptions(compression_type="GZIP")
with tf.io.TFRecordWriter("my_compressed.tfrecord", options) as f:
    f.write(b"This is the first record")
    f.write(b"And this is the second record")
    
'''
압축된 TFRecord 파일을 읽으려면 압축 형식을 지정해야 한다.
'''

dataset = tf.data.TFRecordDataset(["my_compressed.tfrecord"],
                                  compression_type="GZIP")

## 프로토콜 버퍼 개요 ##
'''
프로토콜 버퍼는 다음과 같은 간단한 언어를 사용하여 정의한다.
'''

%%writefile person.proto
syntax = "proto3";
message Person {
  string name = 1;
  int32 id = 2;
  repeated string email = 3;
}

## 텐서플로 프로토콜 버퍼 ##
'''
TFRecord 파일에서 사용하는 전형적인 주요 프로토콜 버퍼는 데이터셋에 있는 하나의 샘플을 표현하는 Example 프로토콜 버퍼이다.
이 프로토콜 버퍼는 이름을 가진 특성의 리스트를 가지고 있다.
각 특성은 바이트 스트링의 리스트나 실수의 리스트, 정수의 리스트 중 하나이다.
다음이 이 프로토콜 버퍼의 정의이다.
'''

syntax = "proto3";

message BytesList { repeated bytes value = 1; }
message FloatList { repeated float value = 1 [packed = true]; }
message Int64List { repeated int64 value = 1 [packed = true]; }
message Feature {
    oneof kind {
        BytesList bytes_list = 1;
        FloatList float_list = 2;
        Int64List int64_list = 3;
    }
};
message Features { map<string, Feature> feature = 1; };
message Example { Features features = 1; };

'''
BytesList, FloatList, Int64List의 정의는 이해하기 쉽다.
[packed=true]는 효율적인 인코딩을 위해 반복적인 수치 필드에 사용된다.
Feature는 BytesList, FloatList, Int64List 중 하나를 담고 있다.
Features는 특성 이름과 특성값을 매핑한 딕셔너리를 가진다.
마지막으로 Example은 하나의 Features 객체를 가진다.
다음은 앞 Person과 동일하게 표현한 tf.train.Example 객체를 만들고 TFRecord 파일에 저장하는 방법을 보여준다.
'''

from tensorflow.train import BytesList, FloatList, Int64List
from tensorflow.train import Feature, Features, Example

person_example = Example(
    features=Features(
        feature={
            "name": Feature(bytes_list=BytesList(value=[b"Alice"])),
            "id": Feature(int64_list=Int64List(value=[123])),
            "emails": Feature(bytes_list=BytesList(value=[b"a@b.com",
                                                          b"c@d.com"]))
        }
    )
)

'''
이 코드는 장황하고 중복이 많아 보이지만 오히려 이해하기는 쉽다.
Example 프로토콜 버퍼를 만들었으므로 SerializeToString() 메서드를 호출하여 직렬화하고 결과 데이터를 TFRecord 파일에 저장할 수 있다.
'''

with tf.io.TFRecordWriter("my_contacts.tfrecord") as f:
  f.write(person_example.SerializeToString())
  
## Example 프로토콜 버퍼를 읽고 파싱하기 ##
'''
다음 코드는 설명 딕셔너리를 정의하고 TFRecordDataset을 순회하면서 데이터셋에 포함된 직렬화된 Example 프로토콜 버퍼를 파싱한다.
'''

feature_description = {
    "name": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "id": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "emails": tf.io.VarLenFeature(tf.string),
}

for serialized_example in tf.data.TFRecordDataset(["my_contacts.tfrecord"]):
  parsed_example = tf.io.parse_single_example(serialized_example,
                                              feature_description)
  
'''
tf.io.parse_single_example()로 하나씩 파싱하는 대신 tf.io.parse_example()를 사용하여 배치 단위로 파싱할 수 있다.
'''
dataset = tf.data.TFRecordDataset(["my_contacts.tfrecord"]).batch(10)
for serialized_examples in dataset: 
  parsed_examples = tf.io.parse_example(serialized_examples,
                                        feature_description)
  
## SequenceExample 프로토콜 버퍼를 사용해 리스트의 리스트 다루기 ##
'''
다음이 SequenceExample 프로토콜 버퍼의 정의이다.
'''

message FeatureList { repeated Feature feature = 1; };
message FeatureLists { map<string, FeatureList> feature_list = 1; };
message SequenceExample {
  Features context = 1;
  FeatureLists feature_lists = 2;
};

'''
SequenceExample은 문맥 데이터를 위한 하나의 Features 객체와 이름이 있는 한 개 이상의 FeatureList를 가진 FeatureLists 객체를 포함한다.
각 FeatureList는 Feature 객체의 리스트를 포함하고 있다.
Feature 객체는 바이트 스트링의 리스트나 64비트 정수의 리스트, 실수의 리스트일 수 있다.
SequenceExample를 만들고 직렬화하고 파싱하는 것은 Example을 만들고 직렬화하고 파싱하는 것과 비슷하다.
하지만 하나의 SequenceExample를 파싱하려면 tf.io.parse_single_sequence_example()를 사용하고 배치를 파싱하려면 tf.io.parse_sequence_example()를 사용해야 한다.
두 함수는 모두 문맥 특성과 특성 리스트를 담은 튜플을 반환한다.
특성 리스트가 가변 길이의 시퀀스를 담고 있다면 tf.RaggedTensor.from_sparse()를 사용해 래그드 텐서로 바꿀 수 있다.
'''

context_feature_descriptions = {
    "author_id": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "title": tf.io.VarLenFeature(tf.string),
    "pub_date": tf.io.FixedLenFeature([3], tf.int64, default_value=[0, 0, 0]),
}
sequence_feature_descriptions = {
    "content": tf.io.VarLenFeature(tf.string),
    "comments": tf.io.VarLenFeature(tf.string),
}
parsed_context, parsed_feature_lists = tf.io.parse_single_sequence_example(
    serialized_sequence_example, context_feature_descriptions,
    sequence_feature_descriptions)

# 입력 특성 전처리 #
'''
다음 코드는 Lambda 층을 사용해 표준화를 수행하는 층을 구현하는 방법을 보여준다.
각 특성의 평균을 빼고 표준편차로 나눈다.
'''

means = np.mean(X_train, axis=0, keepdims=True)
stds = np.std(X_train, axis=0, keepdims=True) 
eps = keras.backend.epsilon()
model = keras.models.Sequential([
                                keras.layers.Lambda(lambda inputs: (inputs - means) / (stds + eps)),
                                [...] # 다른 층
])

'''
means, stds와 같은 전역 변수를 다루기보다 완전한 사용자 정의 층을 원할 수 있다.
'''

class Standardization(keras.layers.Layer):
  def adapt(self, data_sample): 
    self.means_ = np.mean(data_sample, axis=0, keepdims=True) 
    self.stds_ = np.std(data_sample, axis=0, keepdims=True) 
  def call(self, inputs):
    return (inputs - self.means_) / (self.stds_ + keras.backend.epsilon())
  
'''
이 Standardization 층을 모델에 추가하기 전에 데이터 샘플과 함께 adapt() 메서드를 호출해야 한다.
이렇게 해야 각 특성에 대해 적절한 평균과 표준편차를 사용할 수 있다.
'''

std_layer = Standardization() 
std_layer.adapt(data_sample)

'''
이 샘플은 전체 데이터셋을 대표할 만큼 충분히 커야 하지만 전체 훈련 세트일 필요는 없다.
일반적으로 랜덤하게 선택된 수백 개의 샘플이면 충분하다.
그다음 일반적인 층처럼 이 전처리 층을 사용할 수 있다.
'''

model = keras.Sequential()
model.add(std_layer) 
[...] # 모델을 구성한다
model.compile([...])
model.fit([...])

## 원-핫 벡터를 사용해 범주형 특성 인코딩하기 ##
'''
캘리포니아 주택 데이터셋에 있는 ocean_proximity 특성을 생각해보자.
이 특성은 "<1H OCEAN, "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"와 같은 다섯 개의 값이 가능한 범주형 특성이다. 이 특성을 신경망에 주입하기 전에 인코딩해야 한다.
범주 개수가 매우 작으므로 원-핫 인코딩을 사용할 수 있다. 이를 위해 먼저 룩업(lookup) 테이블을 사용해 각 범주를 인덱스로 매핑한다.
'''

vocab = ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
indices = tf.range(len(vocab), dtype=tf.int64)
table_init = tf.lookup.KeyValueTensorInitializer(vocab, indices)
num_oov_buckets = 2
table = tf.lookup.StaticVocabularyTable(table_init, num_oov_buckets)
         
## 임베딩을 사용해 범주형 특성 인코딩하기 ##
'''
임베딩의 동작 원리를 이해하기 위해 직접 구현하는 방법을 알아보겠다.
먼저 각 범주의 임베딩을 담은 임베딩 행렬(embedding matrix)을 만들어 랜덤하게 초기화해야 한다.
이 행렬은 범주와 oov 버킷마다 하나의 행이 있고 임베딩 차원마다 하나의 열을 가진다.
'''

embedding_dim = 2
embed_init = tf.random.uniform([len(vocab) + num_oov_buckets, embedding_dim])
embedding_matrix = tf.Variable(embed_init)         

'''
이 예에서는 2개의 차원을 사용하지만 일반적으로 임베딩은 작업과 어휘 사전 크기에 따라 10에서 300차원을 가진다.
'''

'''
케라스는 임베딩 행렬을 처리해주는 keras.layers.Embedding 층을 제공한다.
이 층이 생성될 때 임베딩 행렬을 랜덤하게 초기화하고 어떤 범주 인덱스로 호출될 때 임베딩 행렬에 있는 그 인덱스의 행을 반환한다.
'''

embedding = keras.layers.Embedding(input_dim=len(vocab) + num_oov_buckets,
                                   output_dim=embedding_dim)

'''
이를 모두 연결하면 범주형 특성을 처리하고 각 범주마다 임베딩을 학습하는 케라스 모델을 만들 수 있다.
'''

regular_inputs = keras.layers.Input(shape=[8])
categories = keras.layers.Input(shape=[], dtype=tf.string)
cat_indices = keras.layers.Lambda(lambda cats: table.lookup(cats))(categories)
cat_embed = keras.layers.Embedding(input_dim=6, output_dim=2)(cat_indices)
encoded_inputs = keras.layers.concatenate([regular_inputs, cat_embed])
outputs = keras.layers.Dense(1)(encoded_inputs)
model = keras.models.Model(inputs=[regular_inputs, categories],
                           outputs=[outputs])

'''
이 모델은 두 개의 입력을 받는다. 샘플마다 8개의 특성을 담은 입력과 하나의 범주형 입력이다.
Lambda 층을 사용해 범주의 인덱스를 찾은 다음 임베딩에서 이 인덱스를 찾는다. 그다음 이 임베딩과 일반 입력을 연결하여 신경망에 주입할 인코드된 입력을 만든다.
여기서부터는 어떤 신경망도 추가할 수 있지만 간단하게 완전 연결 층을 하나 추가하여 케라스 모델을 만든다.
'''

## 케라스 전처리 층 ##
'''
PreprocessingStage 클래스를 사용해 여러 전처리 층을 연결할 수 있다.
예를 들어 다음 코드는 먼저 입력을 정규화하고 그다음 이산화(discretization)하는 전처리 파이프라인을 만든다.
이 파이프라인을 샘플 데이터에 적응시킨 다음 일반적인 층처럼 모델에 사용할 수 있다.
'''

normalization = keras.layers.Normalization()
discretization = keras.layers.Discretization([...])
pipeline = keras.layers.PreprocessingStage([normalization, discretization])
pipeline.adapt(data_sample)

# TF 변환 #
'''
전처리 연산을 딱 한 번만 정의하려면 어떻게 해야 할까? 이것이 TF 변환(transform)이 만들어진 이유이다.
TF 변환은 텐서플로 모델 상품화를 위한 엔드-투-엔드(end-to-end) 플랫폼인 TFX(TensorFlow Extended)의 일부분이다.
TF 변환 같은 TFX 컴포넌트를 사용하려면 먼저 TFX를 설치해야 한다. TFX는 텐서플로와 함께 제공되지 않는다.
그다음 스케일링, 버킷 할당(bucketizing) 등과 같은 TF 변환 함수를 사용하여 전처리 함수를 한 번만 정의한다. 또 필요하면 어떤 텐서플로 연산도 사용할 수 있다.
다음은 두 개의 특성을 전처리하는 함수이다.
'''

import tensorflow_transform as tft

def preprocess(inputs): # inputs = 입력 특성의 배치
  median_age = inputs["housing_median_age"]
  ocean_proximity = inputs["ocean_proximity"]
  standardized_age = tft.scale_to_z_score(median_age)
  ocean_proximity_id = tft.compute_and_apply_vocabulary(ocean_proximity)
  return {
      "standardized_median_age": standardized_age,
      "ocean_proximity_id": ocean_proximity_id
  }

# 텐서플로 데이터셋(TFDS) 프로젝트 #
'''
TFDS는 텐서플로에 기본으로 포함되어 있지 않으므로 tensorflow-datasets 라이브러리를 설치해야 한다.
그다음 tfds.load() 함수를 호출하면 원하는 데이터를 다운로드하고 데이터셋의 딕셔너리로 데이터를 반환한다.
예를 들어 MNIST 데이터셋을 다운로드해보자.
'''

import tensorflow_datasets as tfds

dataset = tfds.load(name="mnist")
mnist_train, mnist_test = dataset["train"], dataset["test"]

'''
그다음 원하는 변환을 적용한 다음 모델을 훈련하기 위한 준비를 마친다.
다음은 예시 코드이다.
'''

mnist_train = mnist_train.repeat(5).batch(32).prefetch(1)
for item in mnist_train:
    images = item["image"]
    labels = item["label"]
    for index in range(5):
        plt.subplot(1, 5, index + 1)
        image = images[index, ..., 0]
        label = labels[index].numpy()
        plt.imshow(image, cmap="binary")
        plt.title(label)
        plt.axis("off")
    break # just showing part of the first batch
    
'''
데이터셋에 있는 각 아이템은 특성과 레이블을 담은 딕셔너리이다. 하지만 케라스는 두 원소를 담은 튜플 아이템을 기대한다.
map() 메서드를 사용해 데이터셋을 이런 식으로 변환할 수 있다.
'''

mnist_train = mnist_train.shuffle(10000).batch(32)
mnist_train = mnist_train.map(lambda items: (items["image"], items["label"]))
mnist_train = mnist_train.prefetch(1)

'''
하지만 as_supervised=True로 지정하여 load() 함수를 호출하는 것이 더 간단하다. 또한 원하는 배치 크기를 지정할 수도 있다.
그다음 tf.keras 모델에 바로 이 데이터셋을 전달할 수 있다.
'''

datasets = tfds.load(name="mnist", batch_size=32, as_supervised=True)
mnist_train = datasets["train"].repeat().prefetch(1)
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28, 1]),
    keras.layers.Lambda(lambda images: tf.cast(images, tf.float32)),
    keras.layers.Dense(10, activation="softmax")])
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(learning_rate=1e-3),
              metrics=["accuracy"])
model.fit(mnist_train, steps_per_epoch=60000 // 32, epochs=5)
