# Char-RNN을 사용해 셰익스피어 문체 생성하기 #
## 훈련 데이터셋 만들기 ##
'''
케라스의 편리한 get_file() 함수를 사용해 안드레이 카패시의 Char-RNN 프로젝트에서 셰익스피어 작품을 모두 다운로드한다.
'''

shakespeare_url = "https://homl.info/shakespeare" # 단축 URL
filepath = keras.utils.get_file("shakespeare.txt", shakespeare_url)
with open(filepath) as f:
  shakespeare_text = f.read()
  
'''
그다음 모든 글자를 정수로 인코딩해야 한다. 사용자 정의 전처리 층을 만드는 것이 한 방법이다. 여기에서는 더 간단하게 케라스의 Tokenizer 클래스를 사용한다. 먼저 이 클래스의 객체를 텍스트에 훈련해야 한다. 텍스트에서 사용되는 모든 글자를 찾아 각기 다른 글자 ID에 매핑한다. 이 ID는 1부터 시작해 고유한 글자 개수까지 만들어진다.
'''

tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts(shakespeare_text)

'''
char_level=True로 지정하여 단어 수준 인코딩 대신 글자 수준 인코딩을 만든다. 이 클래스는 기본적으로 텍스트를 소문자로 바꾼다. 이제 문장을 글자 ID로 인코딩하거나 반대로 디코딩할 수 있다. 이를 통해 텍스트에 있는 고유 글자 개수와 전체 글자 개수를 알 수 있다.
'''

[encoded] = np.array(tokenizer.texts_to_sequences([shakespeare_text])) - 1

## 순차 데이터셋을 나누는 방법 ##
'''
텍스트의 처음 90%를 훈련 세트로 사용한다. 이 세트에서 한 번에 한 글자씩 반환하는 tf.data.Dataset 객체를 만든다.
'''

train_size = dataset_size * 90 // 100
dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])

## 순차 데이터를 윈도 여러 개로 자르기 ##
'''
훈련 세트는 백만 개 이상의 글자로 이루어진 시퀀스 하나이다. 여기에 신경망을 직접 훈련시킬 수 없다. 이 RNN은 백만 개의 층이 있는 심층 신경망과 비슷하고 샘플 하나로 훈련하는 셈이 된다. 대신 데이터셋의 window() 메서드를 사용해 이 긴 시퀀스를 작은 많은 텍스트 윈도로 변환한다. 이 데이터셋의 각 샘플은 전체 텍스트에서 매우 짧은 부분 문자열이다. RNN은 이 부분 문자열 길이만큼만 역전파를 위해 펼쳐진다. 이를 TBPTT(truncated backpropagation through time)라고 부른다. window() 메서드를 호출하여 짧은 텍스트 윈도를 갖는 데이터셋을 만들어보자.
'''

n_steps = 100
window_length = n_steps + 1 # target = 1글자 앞의 input
dataset = dataset.window(window_length, shift=1, drop_remainder=True)

'''
기본적으로 window() 메서드는 윈도를 중복하지 않는다. shift=1로 지정하면 가장 큰 훈련 세트를 만들 수 있다. 첫 번째 윈도는 0에서 100번째 글자를 포함하고 두 번째 윈도는 1에서 101번째 글자를 포함하는 식이다. 모든 윈도가 동일하게 101개의 글자를 포함하도록 drop_remainder=True로 지정한다.
window() 메서드는 각각 하나의 데이터셋으로 표현되는 윈도를 포함하는 데이터셋을 만든다. 리스트의 리스트와 비슷한 중첩 데이터셋(nested dataset)이다. 이런 구조는 데이터셋 메서드를 호출하여 각 윈도를 변환할 때 유용하다. 하지만 모델은 데이터셋이 아니라 텐서를 기대하기 때문에 훈련에 중첩 데이터셋을 바로 사용할 수 없다. 따라서 중첩 데이터셋을 플랫 데이터셋(flat dataset)으로 변환하는 flat_map() 메서드를 호출해야 한다. flat_map() 메서드는 중첩 데이터셋을 평평하게 만들기 전에 각 데이터셋에 적용할 변환 함수를 매개변수로 받을 수 있다.
'''

dataset = dataset.flat_map(lambda window: window.batch(window_length))

'''
윈도마다 batch(window_length)를 호출한다. 이 길이는 윈도 길이와 같기 때문에 텐서 하나를 담은 데이터셋을 얻는다. 이 데이터셋은 연속된 101 글자 길이의 윈도를 담는다. 경사 하강법은 훈련 세트 샘플이 동일 독립 분포일 때 가장 잘 작동하기 때문에 이 윈도를 섞어야 한다. 그다음 윈도를 배치로 만들고 입력과 타깃을 분리하겠다.
'''

batch_size = 32
dataset = dataset.shuffle(10000).batch(batch_size)
dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))

dataset = dataset.map(
    lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))

dataset = dataset.prefetch(1)

## Char-RNN 모델 만들고 훈련하기 ##
'''
이전 글자 100개를 기반으로 다음 글자를 예측하기 위해 유닛 128개를 가진 GRU 층 2개와 입력과 은닉 상태에 20% 드롭아웃을 사용한다. 필요하면 나중에 이 하이퍼파라미터를 수정할 수 있다. 출력층은 TimeDistributed 클래스를 적용한 Dense 층이다. 텍스트에 있는 고유한 글자 수는 39개이므로 이 층은 39개의 유닛을 가져야 한다. 각 글자에 대한 확률을 출력할 수 있다. 타임 스텝에서 출력 확률의 합은 1이어야 하므로 Dense 층의 출력에 소프트맥스 함수를 적용한다. 그다음 "sparse_categorical_crossentropy" 손실과 Adam 옵티마이저를 사용해 모델의 compile() 메서드를 호출한다. 이제 여러 에포크 동안 모델을 훈련할 준비를 마쳤다.
'''

model = keras.models.Sequential([
                                 keras.layers.GRU(128, return_sequences=True, input_shape=[None, max_id],
                                                  dropout=0.2, recurrent_dropout=0.2),
                                 keras.layers.GRU(128, return_sequences=True,
                                                  dropout=0.2, recurrent_dropout=0.2),
                                 keras.layers.TimeDistributed(keras.layers.Dense(max_id,
                                                                                 activation="softmax"))
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
history = model.fit(dataset, epochs=20)

## Char-RNN 모델 사용하기 ##
def preprocess(texts): 
  X = np.array(tokenizer.texts_to_sequences(texts)) - 1
  return tf.one_hot(X, max_id)

## 가짜 셰익스피어 텍스트를 생성하기 ##
'''
Char-RNN 모델을 사용해 새로운 텍스트를 생성하려면 먼저 초기 텍스트를 주입하고 모델이 가장 가능성 있는 다음 글자를 예측한다. 이 글자를 텍스트 끝에 추가하고 늘어난 텍스트를 모델에 전달하여 다음 글자를 예측하는 식이다. 실제로는 이렇게 하면 같은 단어가 계속 반복되는 경우가 많다. 대신 텐서플로의 tf.random.categorical() 함수를 사용해 모델이 추정한 확률을 기반으로 다음 글자를 무작위로 선택할 수 있다. 이 방식은 더 다채롭고 흥미로운 텍스트를 생성한다. categorical() 함수는 클래스의 로그 확률을 전달하면 랜덤하게 클래스 인덱스를 샘플링한다. 생성된 텍스트의 다양성을 더 많이 제어하려면 온도(temperature)라고 불리는 숫자로 로짓을 나눈다. 온도는 원하는 값으로 설정할 수 있는데 0에 가까울수록 높은 확률을 가진 글자를 선택한다. 온도가 매우 높으면 모든 글자가 동일한 확률을 가진다. 다음 next_char() 함수는 이 방식을 사용해 다음 글자를 선택하고 입력 텍스트에 추가한다.
'''

def next_char(text, temperature=1):
  X_new = preprocess([text])
  y_proba = model(X_new)[0, -1:, :]
  rescaled_logits = tf.math.log(y_proba) / temperature 
  char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
  return tokenizer.sequences_to_texts(char_id.numpy())[0]

'''
그다음 next_char() 함수를 반복 호출하여 다음 글자를 얻고 텍스트에 추가하는 작은 함수를 만든다.
'''

def complete_text(text, n_chars=50, temperature=1):
  for _ in range(n_chars):
    text += next_char(text, temperature)
  return text

## 상태가 있는 RNN ##
'''
먼저 상태가 있는 RNN은 배치에 있는 각 입력 시퀀스가 이전 배치의 시퀀스가 끝난 지점에서 시작해야 한다. 따라서 상태가 있는 RNN을 만들기 위해 첫 번째로 할 일은 순차적이고 겹치지 않는 입력 시퀀스를 만드는 것이다. Dataset을 만들 때 window() 메서드에서 shift=n_steps를 사용한다. 또한 shuffle() 메서드를 호출해서는 안된다. 안타깝게도 상태가 있는 RNN을 위한 데이터셋은 상태가 없는 RNN의 경우보다 배치를 구성하기 더 힘들다. 실제 batch(32)라고 호출하면 32개의 연속적인 윈도가 같은 배치에 들어간다. 이 윈도가 끝난 지점부터 다음 배치가 계속되지 않는다. 첫 번째 배치는 윈도 1에서 32까지 포함하고 두 번째 배치는 윈도 33부터 64까지 포함한다. 따라서 각 배치의 첫 번째 윈도를 생각하면 연속적이지 않음을 알 수 있다. 이 문제에 대한 가장 간단한 해결책은 하나의 윈도를 갖는 배치를 만드는 것이다.
'''

dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])
dataset = dataset.window(window_length, shift=n_steps, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(window_length))
dataset = dataset.batch(1)
dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
dataset = dataset.map(
    lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))
dataset = dataset.prefetch(1)

'''
배치를 만드는 것이 어렵지만 불가능한 것은 아니다. 예를 들면 셰익스피어의 텍스트를 길이가 동일한 32개의 텍스트로 나누고 각 텍스트에 대해 연속적인 입력 시퀀스를 가진 데이터셋 하나를 만들 수 있다. 마지막으로 tf.train.Dataset.zip(datasets).map(lambda*windows: tf.stack(windows))를 사용해 연속적인 배치를 만든다. 여기에서 한 배치에서 n번째 입력 시퀀스의 시작은 정확히 이전 배치의 n번째 입력 시퀀스가 끝나는 지점이다.
이제 상태가 있는 RNN을 만들어보자. 첫째, 각 순환 층을 만들 때 stateful=True로 지정해야 한다. 둘째, 상태가 있는 RNN은 배치 크기를 알아야 한다. 따라서 첫 번쨰 층에 batch_input_shape 매개변수를 지정해야 한다. 입력은 어떤 길이도 가질 수 있으므로 두 번째 차원은 지정하지 않아도 된다.
'''

model = keras.models.Sequential([
                                 keras.layers.GRU(128, return_sequences=True, stateful=True,
                                                  dropout=0.2, recurrent_dropout=0.2,
                                                  batch_input_shape=[batch_size, None, max_id]),
                                 keras.layers.GRU(128, return_sequences=True, stateful=True,
                                                  dropout=0.2, recurrent_dropout=0.2),
                                 keras.layers.TimeDistributed(keras.layers.Dense(max_id,
                                                                                 activation="softmax"))
])

class ResetStatesCallback(keras.callbacks.Callback):
  def on_epoch_begin(self, epoch, logs):
    self.model.reset_states()
    
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
model.fit(dataset, steps_per_epoch=steps_per_epoch, epochs=50,
          callbacks=[ResetStatesCallback()])

# 감성 분석 #
import tensorflow_datasets as tfds

datasets, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)
train_size = info.splits["train"].num_examples

def preprocess(X_batch, y_batch): 
  X_batch = tf.strings.substr(X_batch, 0, 300) 
  X_batch = tf.strings.regex_replace(X_batch, b"<br\\s*/?>", b" ")
  X_batch = tf.strings.regex_replace(X_batch, b"[^a-zA-Z']", b" ")
  X_batch = tf.strings.split(X_batch) 
  return X_batch.to_tensor(default_value=b"<pad>"), y_batch

'''
리뷰 텍스트를 잘라내어 각 리뷰에서 처음 300 글자만 남긴다. 이렇게 하면 훈련 속도를 높일 수 있다. 또 일반적으로 처음 한두 문장에서 리뷰가 긍정적인지 아닌지 판단할 수 있기 때문에 성능에 크게 영향을 미치지 않는다. 그다음 정규식(regular expression)을 사용해 <br /> 태그를 공백으로 바꾼다. 문자와 작은 따옴표가 아닌 다른 모든 문자를 공백으로 바꾼다. 예를 들어, “Well, I can’t<br />”란 텍스트는 “Well I can’t”가 될 것이다. 마지막으로 preprocess() 함수는 리뷰를 공백으로 나눈다. 이때 래그드 텐서(ragged tensor)가 반환된다. 이 래그드 텐서를 밀집 텐서로 바꾸고 동일한 길이가 되도록 패딩 토큰 “<pad>”로 모든 리뷰를 패딩한다.
그다음 어휘 사전을 구축해야 한다. 전체 훈련 세트를 한 번 순회하면서 preprocess() 함수를 적용하고 Counter로 단어의 등장 횟수를 센다.
'''

from collections import Counter
vocabulary = Counter()
for X_batch, y_batch in datasets["train"].batch(32).map(preprocess):
  for review in X_batch:
    vocabulary.update(list(review.numpy()))
    
vocab_size = 10000
truncated_vocabulary = [
                        word for word, count in vocabulary.most_common()[:vocab_size]]

words = tf.constant(truncated_vocabulary)
word_ids = tf.range(len(truncated_vocabulary), dtype=tf.int64)
vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)
num_oov_buckets = 1000
table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)

'''
이제 최종 훈련 세트를 만들 준비가 되었다. 리뷰를 배치로 묶고 preprocess() 함수를 사용해 단어의 짧은 시퀀스로 바꾸겠다. 그다음 앞서 만든 테이블을 사용하는 encode_words() 함수로 단어를 인코딩한다. 마지막으로 다음 배치를 프리페치한다.
'''

def encode_words(X_batch, y_batch):
  return table.lookup(X_batch), y_batch

train_set = datasets["train"].batch(32).map(preprocess)
train_set = train_set.map(encode_words).prefetch(1)

embed_size = 129
model = keras.models.Sequential([
                                 keras.layers.Embedding(vocab_size + num_oov_buckets, embed_size,
                                                        input_shape=[None]),
                                 keras.layers.GRU(128, return_sequences=True),
                                 keras.layers.GRU(128),
                                 keras.layers.Dense(1, activation="sigmoid")
])
model.compile(loss="binary_crossentropy", optimizer="adam",
              metrics=["accuracy"])
history = model.fit(train_set, epochs=5)

'''
첫 번째 층은 단어 ID를 임베딩으로 변환하는 Embedding 층이다. 임베딩 행렬은 단어 ID당 하나의 행과 임베딩 차원당 하나의 열을 가진다. 모델의 입력은 [배치 크기, 타임 스텝 수] 크기를 가진 2D 텐서이지만 Embedding 층의 출력은 [배치 크기, 타임 스텝 수, 임베딩 크기] 크기를 가진 3D 텐서가 된다.
모델의 나머지 부분은 매우 간단하다. GRU 층 두 개로 구성되고 두 번째 층은 마지막 타임 스텝의 출력만 반환한다. 출력층은 시그모이드 활성화 함수를 사용하는 하나의 뉴런이다. 리뷰가 영화에 대한 긍정적인 감정을 표현하는지에 대한 추정 확률을 출력한다. 그다음 간단히 모델을 컴파일하고 앞서 준비한 데이터셋에서 몇 번의 에포크 동안 훈련한다.
'''

## 마스킹 ##
'''
마스킹 층과 마스크 자동 전파는 Sequential 모델에 가장 잘 맞다. Conv1D 층과 순환 층을 섞는 것과 같이 복잡한 모델에서는 항상 작동하지 않는다. 이런 경우에는 함수형 API나 서브클래싱 API를 사용해 직접 마스크를 계산하여 다음 층에 전달해야 한다. 예를 들어 다음 모델은 이전 모델과 동일하지만 함수형 API를 사용하여 직접 마스킹을 처리한다.
'''

K = keras.backend
inputs = keras.layers.Input(shape=[None])
mask = keras.layers.Lambda(lambda inputs: K.not_equal(inputs, 0))(inputs)
z = keras.layers.Embedding(vocab_size + num_oov_buckets, embed_size)(inputs)
z = keras.layers.GRU(128, return_sequences=True)(z, mask=mask)
z = keras.layers.GRU(128)(z, mask=mask)
outputs = keras.layers.Dense(1, activation="sigmoid")(z)
model = keras.Model(inputs=[inputs], outputs=[outputs])

'''
몇 번의 에포크를 훈련하고 나면 이 모델은 리뷰가 긍정적인지 아닌지 꽤 잘 판단한다. TensorBoard() 콜백을 사용하면 텐서보드에서 학습된 임베딩을 시각화할 수 있다. awesome과 amazing 같은 단어가 한쪽에 군집을 이루고 awful과 terrible 같은 단어가 다른 쪽에 군집되어 있는 것을 보면 매우 흥미롭다. good과 같은 일부 단어는 기대한 것만큼 긍정적이지 않다. 아마도 부정적인 많은 리뷰들에 not good 구절이 포함되어 있기 때문일 것이다. 이 모델이 영화 리뷰 25,000에서 유용한 단어 임베딩을 학습할 수 있다는 사실이 놀랍다. 하지만 다른 대량의 텍스트 코퍼스(corpus)에서 훈련된 단어 임베딩을 재사용할 수 있다. 일반적으로 amazing이란 단어는 영화에 관해 이야기할 때나 다른 것에 관한 이야기할 때 같은 뜻을 가진다. 또한 임베딩이 다른 작업에서 훈련되었다 하더라도 감성 분석에 유용할 수 있다. awesome과 amazing 같은 단어는 비슷한 의미를 가지니 다른 작업에서도 임베딩 공간에 군집을 이룰 가능성이 높다. 긍정적인 단어와 부정적인 단어가 모두 클러스터를 형성하면 감성 분석에 도움이 될 것이다. 그러므로 많은 파라미터를 사용해 단어 임베딩을 학습하기보다 사전훈련된 임베딩을 재사용할 수 있는지 검토해보자.
'''

## 사전훈련된 임베딩 재사용하기 ##
'''
nnlm-en-dim50 문장 임베딩 모듈 버전 1을 감성 분석 모델에 사용해보겠다.
'''

import tensorflow_hub as hub 

model = keras.Sequential([
                          hub.KerasLayer("https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1",
                                         dtype=tf.string, input_shape=[], output_shape=[50]),
                          keras.layers.Dense(128, activation="relu"),
                          keras.layers.Dense(1, activation="sigmoid")
])
model.compile(loss="binary_crossentropy", optimizer="adam",
              metrics=["accuracy"])

'''
hub.KerasLayer 층이 주어진 URL에서 모듈을 다운로드한다. 이 모듈의 이름은 문장 인코더(sentence encoder)이다. 문자열을 입력으로 받아 하나의 벡터로 인코딩한다. 내부적으로는 문자열을 파싱해서 대규모 코퍼스에서 사전훈련된 임베딩 행렬을 사용해 각 단어를 임베딩한다. 이 코퍼스는 구글 뉴스 7B 코퍼스이다. 그다음 모든 단어 임베딩의 평균을 계산한다. 이 결과가 문장 임베딩이다. 그다음 두 개의 Dense 층을 추가해 감성 분석 모델을 만든다. 기본적으로 hub.KerasLayer 층은 훈련되지 않는다. 하지만 이 층을 만들 때 trainable=True로 설정하여 작업에 맞게 미세 조정할 수 있다.
'''

datasets, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)
train_size = info.splits["train"].num_examples
batch_size = 32
train_set = datasets["train"].batch(batch_size).prefetch(1)
history = model.fit(train_set, epochs=5)

'''
TF 허브 모듈 URL의 마지막 부분은 필요한 버전을 저장한다. 버전이 있기 때문에 새로운 버전의 모듈이 릴리스되더라도 모델에 영향을 미치지 않는다. 이 URL을 웹 브라우저에 입력하면 이 모듈에 관한 문서를 볼 수 있다. 기본적으로 TF 허브는 다운로드한 파일을 로컬 시스템의 임시 디렉터리에 캐싱한다. 시스템을 정리할 때마다 다시 다운로드되는 것을 피하려면 고정 디렉터리에 다운로드할 수 있다. 이렇게 하려면 TFHUB_CACHE_DIR 환경 변수에 원하는 디렉터리를 지정한다.
'''

# 신경망 기계 번역을 위한 인코더-디코더 네트워크 #
import tensorflow_addons as tfa

encoder_inputs = keras.layers.Input(shape=[None], dtype=np.int32)
decoder_inputs = keras.layers.Input(shape=[None], dtype=np.int32)
sequence_lengths = keras.layers.Input(shape=[], dtype=np.int32)

embeddings = keras.layers.Embedding(vocab_size, embed_size)
encoder_embeddings = embeddings(encoder_inputs)
decoder_embeddings = embeddings(decoder_inputs)

encoder = keras.layers.LSTM(512, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_embeddings)
encoder_state = [state_h, state_c]

sampler = tfa.seq2seq.sampler.TrainingSampler()

decoder_cell = keras.layers.LSTMCell(512)
output_layer = keras.layers.Dense(vocab_size)
decoder = tfa.seq2seq.basic_decoder.BasicDecoder(decoder_cell, sampler,
                                                 output_layer=output_layer)
final_outputs, final_state, final_sequence_lengths = decoder(
    decoder_embeddings, initial_state=encoder_state,
    sequence_length=sequence_lengths)
Y_proba = tf.nn.softmax(final_outputs.rnn_output)

model = keras.models.Model(
    inputs=[encoder_inputs, decoder_inputs, sequence_lengths],
    outputs=[Y_proba])

'''
먼저 LSTM 층을 만들 때 최종 은닉 상태를 디코더로 보내기 위해 return_state=True로 지정했다. LSTM 셀을 사용하기 때문에 은닉 상태 두 개를 반환한다. TrainingSampler는 텐서플로 애드온에 포함되어 있는 여러 샘플러 중 하나이다. 이 샘플러는 각 스텝에서 디코더에게 이전 스텝의 출력이 무엇인지 알려준다. 추론 시에는 실제로 출력되는 토큰의 임베딩이 된다. 훈련 시에는 이전 타깃 토큰의 임베딩이 되어야 한다. 이 때문에 TrainingSampler를 사용한다. 실전에서는 이전 타임 스텝의 타깃의 임베딩을 사용해 훈련을 시작해서 이전 스텝에서 출력된 실제 토큰의 임베딩으로 점차 바꾸는 것이 좋다. 새미 벤지오(Samy Bengio) 등이 2015년 논문에서 이 아이디어를 소개했다. ScheduledEmbeddingTrainingSampler는 타깃과 실제 출력 사이에서 무작위로 선택하며 훈련하는 동안 점진적으로 확률을 바꿀 수 있다.
'''

## 양방향 RNN ##
'''
케라스에서 양방향 순환 층을 구현하려면 keras.layers.Bidirectional으로 순환 층을 감싼다. 예를 들어 다음 코드는 양방향 GRU 층을 만든다.
'''
keras.layers.Bidirectional(keras.layers.GRU(10, return_sequences=True))

## 빔 검색 ##
beam_width = 10
decoder = tfa.seq2seq.beam_search_decoder.BeamSearchDecoder(
  cell=decoder_cell, beam_width=beam_width, output_layer=output_layer)
decoder_initial_state = tfa.seq2seq.beam_search_decoder.tile_batch(
  encoder_state, multiplier=beam_width)
outputs, _, _ = decoder(
  embedding_decoder, start_tokens=start_tokens, end_token=end_token,
  initial_state=decoder_initial_state)

'''
먼저 모든 디코더 셀을 감싼 BeamSearchDecoder를 만든다. 그다음 각 디코더를 위해 인코더의 마지막 상태를 복사한다. 시작과 종료 토큰과 함께 이 상태를 디코더에게 전달한다.
'''

# 어텐션 메커니즘 #
attention_mechanism = tfa.seq2seq.attention_wrapper.LuongAttention(
  units, encoder_state, memory_sequence_length=encoder_sequence_length)
attention_decoder_cell = tfa.seq2seq.attention_wrapper.AttentionWrapper(
  decoder_cell, attention_mechanism, attention_layer_size=n_units)

'''
간단히 디코더 셀을 AttentionWrapper 클래스로 감싸고 원하는 어텐션 메커니즘을 지정한다.
'''

## 트랜스포머 구조: 어텐션이 필요한 전부다 ##
'''
텐서플로에는 PositionalEmbedding와 같은 층이 없지만 만드는 것이 어렵지 않다. 효율적인 이유로 생성자에서 위치 인코딩 행렬을 미리 계산한다. 그다음 call() 메서드에서 이 인코딩 행렬을 입력의 크기로 잘라 입력에 더한다. 위치 인코딩 행렬을 만들 때 크기가 1인 첫 번째 차원을 추가했으므로 브로드캐스팅 규칙에 의해 이 행렬이 입력의 모든 문장에 더해진다.
'''

class PositionalEncoding(keras.layers.Layer):
    def __init__(self, max_steps, max_dims, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        if max_dims % 2 == 1: max_dims += 1 # max_dims must be even
        p, i = np.meshgrid(np.arange(max_steps), np.arange(max_dims // 2))
        pos_emb = np.empty((1, max_steps, max_dims))
        pos_emb[0, :, ::2] = np.sin(p / 10000**(2 * i / max_dims)).T
        pos_emb[0, :, 1::2] = np.cos(p / 10000**(2 * i / max_dims)).T
        self.positional_embedding = tf.constant(pos_emb.astype(self.dtype))
    def call(self, inputs):
        shape = tf.shape(inputs)
        return inputs + self.positional_embedding[:, :shape[-2], :shape[-1]]
      
embed_size = 512; max_steps = 500; vocab_size = 10000
encoder_inputs = keras.layers.Input(shape=[None], dtype=np.int32)
decoder_inputs = keras.layers.Input(shape=[None], dtype=np.int32)
embeddings = keras.layers.Embedding(vocab_size, embed_size)
encoder_embeddings = embeddings(encoder_inputs)
decoder_embeddings = embeddings(decoder_inputs)
positional_encoding = PositionalEncoding(max_steps, max_dims=embed_size)
encoder_in = positional_encoding(encoder_embeddings)
decoder_in = positional_encoding(decoder_embeddings)

Z = encoder_in
for N in range(6):
    Z = keras.layers.Attention(use_scale=True)([Z, Z])

encoder_outputs = Z
Z = decoder_in
for N in range(6):
    Z = keras.layers.Attention(use_scale=True, causal=True)([Z, Z])
    Z = keras.layers.Attention(use_scale=True)([Z, encoder_outputs])

outputs = keras.layers.TimeDistributed(
    keras.layers.Dense(vocab_size, activation="softmax"))(Z)

'''
use_scale=True로 지정하면 파라미터가 추가되어 유사도 점수의 스케일을 적절히 낮추는 방법을 배운다. 항상 동일한 인자로 유사도 점수의 스케일을 낮추는 트랜스포머 모델과는 조금 다르다. 두 번째 어텐션 층을 만들 때 causal=True로 지정하면 각 출력 토큰은 미래 토큰이 아니라 이전 출력 토큰에만 주의를 기울인다.
'''
