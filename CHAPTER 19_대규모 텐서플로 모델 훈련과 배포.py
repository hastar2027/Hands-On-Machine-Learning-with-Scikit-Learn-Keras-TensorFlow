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
'''

