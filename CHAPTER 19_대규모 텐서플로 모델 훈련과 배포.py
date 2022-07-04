# 텐서플로 모델 서빙 #
## 텐서플로 서빙 사용하기 ##
### SavedModel로 내보내기 ###
'''
텐서플로는 모델을 SavedModel 포맷으로 내보내기 위한 간편한 tf.saved_model.save() 함수를 제공한다. 모델과 함께 이름과 버전을 포함한 경로를 전달하면 이 함수는 이 경로에 모델의 계산 그래프와 가중치를 저장한다.
'''

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28, 1]),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(learning_rate=1e-2),
              metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))

model_version = "0001"
model_name = "my_mnist_model"
model_path = os.path.join(model_name, model_version)
tf.saved_model.save(model, model_path)

'''
또는 모델의 save() 메서드를 사용할 수 있다. 파일 확장자가 .h5가 아니면 HDF5 포맷 대신 SavedModel 포맷으로 모델을 저장한다.
tf.saved_model.load() 함수를 사용해 SavedModel을 로드할 수 있다. 하지만 반환되는 객체가 케라스 모델은 아니다. 이 객체는 계산 그래프와 변숫값을 담은 SavedModel을 나타낸다. 이를 함수처럼 사용하여 예측을 만들 수 있다.
'''

saved_model = tf.saved_model.load(model_path)
y_pred = saved_model(tf.constant(X_new, dtype=tf.float32))

'''
또는 keras.models.load_model() 함수를 사용해 SavedModel을 케라스 모델로 직접 로드할 수 있다.
'''

model = keras.models.load_model(model_path)
y_pred = model.predict(tf.constant(X_new, dtype=tf.float32))

'''
saved_model_cli 명령을 사용해 예측을 만들 수도 있다. 손글씨 숫자 이미지를 3개를 담은 넘파이 배열을 사용해 예측을 만든다고 가정해보자. 먼저 넘파이 npy 포맷으로 내보내야 한다.
'''

np.save("my_mnist_tests.npy", X_new)

'''
그다음 saved_model_cli 명령을 사용한다.
'''

### REST API로 TF 서빙에 쿼리하기 ###
import json

input_data_json = json.dumps({
    "signature_name": "serving_default",
    "instances": X_new.tolist(),
})

'''
JSON 포맷은 100% 텍스트이므로 X_new 넘파이 배열을 파이썬 리스트로 변환한 후 JSON 문자열로 만들어야 한다.
이 입력 데이터를 HTTP POST 메서드로 TF 서빙에 전송해보자. requests 라이브러리를 사용해 쉽게 처리할 수 있다.
'''

import requests

SERVER_URL = 'http://localhost:8501/v1/models/my_mnist_model:predict'
response = requests.post(SERVER_URL, data=input_data_json)
response.raise_for_status() # raise an exception in case of error
response = response.json()

'''
응답은 "predictions" 키 하나를 가진 딕셔너리이다. 이 키에 해당하는 값은 예측의 리스트이다. 이 리스트는 파이썬 리스트이므로 넘파이 배열로 변환하고 소수점 셋째 자리에서 반올림한다.
'''

### gRPC API로 TF 서빙에 쿼리하기 ###
'''
gRPC API는 직렬화된 PredictRequest 프로토콜 버퍼를 입력으로 기대한다. 직렬화된 PredictResponse 프로토콜 버퍼를 출력한다. 이 프로토콜은 tensorflow-serving-api에 포함되어 있다. 이 라이브러리는 사전에 설치해야 한다. 먼저 요청(request)을 만들어보자.
'''

from tensorflow_serving.apis.predict_pb2 import PredictRequest

request = PredictRequest()
request.model_spec.name = model_name
request.model_spec.signature_name = "serving_default"
input_name = model.input_names[0]
request.inputs[input_name].CopyFrom(tf.make_tensor_proto(X_new))

'''
이 코드는 PredictRequest 프로토콜 버퍼를 만들고 필수 필드를 채운다. 여기에는 모델 이름, 호출할 함수의 시그니처 이름, Tensor 프로토콜 버퍼 형식으로 변환한 입력 데이터가 포함된다. tf.make_tensor_proto() 함수는 주어진 텐서나 넘파이 배열을 기반으로 Tensor 프로토콜 버퍼를 만든다.
그다음 서버로 이 요청을 보내고 응답을 받는다.
'''

import grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc

channel = grpc.insecure_channel('localhost:8500')
predict_service = prediction_service_pb2_grpc.PredictionServiceStub(channel)
response = predict_service.Predict(request, timeout=10.0)

'''
이 코드는 매우 간단하다. 임포트한 다음 localhost에서 TCP 포트 8500번으로 gRPC 통신 채널을 만든다. 그다음 이 채널에 대해 gRPC 서비스를 만들고 이를 사용해 10초 타임아웃(timeout)이 설정된 요청을 보낸다. 이 예에서는 보안 채널을 사용하지 않는다. 하지만 gRPC와 텐서플로 서빙은 SSL/TLS 기반의 보안 채널도 제공한다.
그다음 PredictResponse 프로토콜 버퍼를 텐서로 바꾸어보자.
'''

output_name = model.output_names[0]
outputs_proto = response.outputs[output_name]
y_proba = tf.make_ndarray(outputs_proto)

'''
이 코드를 실행하고 y_proba.numpy().round(2)를 출력하면 동일한 클래스 추정 확률을 얻을 것이다. 이것이 전부이다. 몇 줄 코드만으로 REST나 gRPC를 사용해 텐서플로 모델을 원격에서 접속할 수 있다.
'''

### 새로운 버전의 모델 배포하기 ###
'''
새로운 버전의 모델을 만들어 my_mnist_model/0002 디렉터리에 SavedModel을 내보내겠다.
'''

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28, 1]),
    keras.layers.Dense(50, activation="relu"),
    keras.layers.Dense(50, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(learning_rate=1e-2),
              metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))

model_version = "0002"
model_name = "my_mnist_model"
model_path = os.path.join(model_name, model_version)
tf.saved_model.save(model, model_path)

## 예측 서비스 사용하기 ##
### 구글 클라우드 클라이언트 라이브러리 ###
'''
AI 플랫폼을 위한 클라이언트 라이브러리가 없기 때문에 구글 API 클라이언트 라이브러리를 사용하겠다. 이 라이브러리는 서비스 계정의 개인 키를 사용해야 한다. GOOGLE_APPLICATION_CREDENTIALS 환경 변수를 설정하여 파일 위치를 알려줄 수 있다. 스크립트를 시작하기 전이나 다음처럼 스크립트 안에서 설정할 수 있다.
'''

import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "my_service_account_private_key.json"

import googleapiclient.discovery

project_id = "onyx-smoke-242003"
model_id = "my_mnist_model"
model_path = "projects/{}/models/{}".format(project_id, model_id)
ml_resource = googleapiclient.discovery.build("ml", "v1").projects()

'''
model_path에 /versions/0001을 연결하여 쿼리할 버전을 지정할 수 있다. A/B 테스트나 공식 릴리스 전에 일부 사용자에게만 테스트하는 목적으로 사용할 수 있다. 그다음 리소스 객체를 사용해 예측 서비스를 호출하고 예측 결과를 반환하는 함수를 작성하자.
'''

def predict(X):
    input_data_json = {"signature_name": "serving_default",
                       "instances": X.tolist()}
    request = ml_resource.predict(name=model_path, body=input_data_json)
    response = request.execute()
    if "error" in response:
        raise RuntimeError(response["error"])
    return np.array([pred[output_name] for pred in response["predictions"]])

# 모바일 또는 임베디드 장치에 모델 배포하기 #
'''
모델 크기를 줄이기 위해 TFLite의 모델 변환기는 SavedModel을 받아 FlatBuffers 기반의 경량 포맷으로 압축한다. 이 라이브러리는 원래 게임을 위해 구글이 만든 플랫폼에 종속적이지 않은 직렬화(serialisation) 라이브러리이다. FlatBuffers는 어떤 전처리도 없이 바로 RAM으로 로드될 수 있다. 이를 통해 로드에 걸리는 시간과 메모리 사용을 줄인다. 모델이 모바일이나 임베디드 장치에 로드되면 TFLite 인터프리터가 이 모델을 실행하여 예측을 만든다. 다음은 SavedModel을 FlatBuffers로 변환하여 .tflite 파일로 저장하는 방법을 보여준다.
'''

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
tflite_model = converter.convert()
with open("converted_model.tflite", "wb") as f:
    f.write(tflite_model)
    
'''
변환기가 크기도 줄이고 지연 시간도 줄이기 위해 모델 최적화도 수행한다. 예측에 필요하지 않은 모든 연산을 삭제하고 가능한 연산을 최적화한다. 또한 가능하면 연산을 결합시킨다. 예를 들어 가능하면 배치 정규화 층은 이전 층에 덧셈 연산과 곱셈 연산으로 합쳐질 수 있다. TFLite가 모델을 얼마나 최적화할 수 있는지 알아보려면 사전훈련된 TFLite 모델 하나를 다운로드하여 압축을 해제하자. 그다음 멋진 그래프 시각화 도구인 Netron에 접속해서 .pb 파일을 업로드하고 원본 모델을 확인해보자. 크고 복잡한 그래프가 나올 것이다. 이번에는 최적화된 .tflite 모델을 업로드해보자.
훈련 후 양자화를 수행하려면 convert() 메서드를 호출하기 전에 OPTIMIZE_FOR_SIZE를 변환기 최적화 리스트에 추가하면 된다.
'''

converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]

# 계산 속도를 높이기 위해 GPU 사용하기 #
## GPU RAM 관리하기 ##
for gpu in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_virtual_device_configuration(
        gpu,
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])

for gpu in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

'''
이렇게 하는 다른 방법은 TF_FORCE_GPU_ALLOW_GROWTH 환경 변수를 true로 설정하는 것이다. 이렇게 하면 텐서플로는 프로그램이 종료되기 전까지는 한번 점유한 메모리를 다시 해제하지 않는다. 이 방법은 결정적인 행동을 보장하기 어렵다. 따라서 제품에 적용할 때는 이전 방법 중 하나를 선택하게 될 것이다. 하지만 이 방법이 유용한 경우가 있다. 예를 들어, 한 대의 머신에 텐서플로를 사용하는 주피터 노트북을 여러 개 실행하는 경우이다. 이것이 코랩 런타임에서 TF_FORCE_GPU_ALLOW_GROWTH 환경 변수가 true로 설정된 이유이다.
'''

physical_gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_virtual_device_configuration(
    physical_gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048),
     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])

'''
이 두 가상 장치 이름은 /gpu:0과 /gpu:1이다. 실제 두 개의 독립적인 GPU처럼 연산과 변수를 각각 할당할 수 있다.
'''

# 다중 장치에서 모델 훈련하기 #
## 분산 전략 API를 사용한 대규모 훈련 ##
'''
텐서플로는 복잡성을 모두 대신 처리해주는 매우 간단한 분산 전략 API를 제공한다. 미러드 전략으로 데이터 병렬화를 사용해 가능한 모든 GPU에서 케라스 모델을 훈련하려면 MirroredStrategy 객체를 만들고 scope() 메서드를 호출하여 분산 컨텍스트를 얻는다. 이 컨텍스트로 모델 생성과 컴파일 과정을 감싼다. 그다음 보통과 같이 모델의 fit() 메서드를 호출한다.
'''

distribution = tf.distribute.MirroredStrategy()

with distribution.scope():
    model = create_model()
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.SGD(learning_rate=1e-2),
                  metrics=["accuracy"])
    
batch_size = 100 # must be divisible by the number of workers
model.fit(X_train, y_train, epochs=10,
          validation_data=(X_valid, y_valid), batch_size=batch_size)

'''
내부적으로 tf.keras는 분산을 자동으로 인식한다. MirroredStrategy 컨텍스트 안에서 모든 변수와 연산이 가능한 모든 GPU 장치에 복제되어야 하는 것을 알고 있다. fit() 메서드는 자동으로 훈련 배치를 모든 복제 모델에 나눈다. 따라서 배치 크기가 복제 모델의 개수로 나누어 떨어져야 한다. 하나의 장치를 사용하는 것보다 일반적으로 훨씬 빠르게 훈련될 것이다. 코드 변경도 아주 적다.
모델 훈련이 끝나면 이 모델을 사용해 예측을 효율적으로 만들 수 있다. predict() 메서드를 호출하면 자동으로 모든 복제 모델에 배치(batch)를 나누어 병렬로 예측을 만들 것이다. 모델의 save() 메서드를 호출하면 복제 모델을 여러 개 가진 미러드 모델이 아니라 일반적인 모델로 저장된다. 따라서 이 모델을 다시 로드하면 하나의 장치를 가진 일반 모델처럼 실행될 것이다. 모델을 로드하여 가능한 모든 장치에서 실행하고 싶다면 분산 컨텍스트 안에서 keras.models.load_model()을 호출해야 한다.
'''

with distribution.scope():
    mirrored_model = keras.models.load_model("my_mnist_model.h5")

'''
가능한 GPU 장치 중 일부만 사용하고 싶다면 MirroredStrategy 생성자에 장치 리스트를 전달할 수 있다.
'''

distribution = tf.distribute.MirroredStrategy(["/gpu:0", "/gpu:1"])

'''
기본적으로 MirroredStrategy 클래스는 평균을 계산하는 올리듀스 연산을 위해 NCCL(NVIDIA Collective Communications Library)을 사용한다. 하지만 tf.distribute.HierarchicalCopyAllReduce 클래스의 인스턴스나 tf.distribute.ReductionToOneDevice 클래스의 인스턴스에 cross_device_ops 매개변수를 설정하여 바꿀 수 있다. 기본 NCCL 옵션은 tf.distribute.NcclAllReduce 클래스를 기반으로 한다. 일반적으로 빠르지만 GPU 개수와 종류에 의존성이 있다. 따라서 다른 옵션을 시도해볼 수 있다.
중앙 집중적인 파라미터로 데이터 병렬화를 사용한다면 MirroredStrategy를 CentralStorageStrategy로 바꾸어주자.
'''

distribution = tf.distribute.experimental.CentralStorageStrategy()

## 텐서플로 클러스터에서 모델 훈련하기 ##
cluster_spec = {
    "worker": [
        "machine-a.example.com:2222",  # /job:worker/task:0
        "machine-b.example.com:2222"   # /job:worker/task:1
    ],
    "ps": ["machine-c.example.com:2222"] # /job:ps/task:0
}

'''
태스크를 시작할 때 클러스터 명세를 전달해야 하고 이 태스크의 타입과 인덱스가 무엇인지 알려주어야 한다. 한 번에 모든 것을 지정하는 가장 간단한 방법은 텐서플로를 시작하기 전에 TF_CONFIG 환경 변수를 설정하는 것이다. 클러스터 명세와 현재 태스크의 타입과 인덱스를 담은 딕셔너리를 JSON으로 인코딩하여 입력한다. 예를 들어 다음의 TF_CONFIG 환경 변수는 정의한 클러스터를 사용하고 시작하려는 태스크가 첫 번째 워커라는 것을 지정한다.
'''

import os
import json

os.environ["TF_CONFIG"] = json.dumps({
    "cluster": cluster_spec,
    "task": {"type": "worker", "index": 1}
})

'''
이제 클러스터에서 모델을 훈련해보자. 미러드 전략을 시작해보겠다. 먼저 태스크에 맞게 TF_CONFIG 환경 변수를 적절히 설정한다. 파라미터 서버가 없고 보통 하나의 머신에 하나의 워커를 설정한다. 각 태스크에 대해 다른 태스크 인덱스를 설정해야 한다는 것을 특별히 주의하자. 마지막으로 각 워커에서 다음 훈련 코드를 실행한다.
'''

distribution = tf.distribute.experimental.MultiWorkerMirroredStrategy()

with distribution.scope():
    mirrored_model = keras.models.Sequential([...])
    mirrored_model.compile([...])
    
batch_size = 100 # 복제 모델 개수로 나누어 떨어져야 한다
history = mirrored_model.fit(X_train, y_train, epochs=0)

'''
이번에는 MultiWorkerMirroredStrategy를 사용한 것만 다르다. 첫 번째 워커에서 이 스크립트를 시작할 때 올리듀스 스텝에서 멈추게 된다. 하지만 마지막 워커가 시작하자마자 훈련이 시작되고 정확히 같은 속도로 진행되는 것을 볼 수 있다.
마지막으로 구글 클라우드에 있는 TPU를 사용할 수 있다면 다음과 같이 TPUStrategy를 만들 수 있다.
'''

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu_address)
tf.tpu.experimental.initialize_tpu_system(resolver)
distribution = tf.distribute.experimental.TPUStrategy(resolver)



