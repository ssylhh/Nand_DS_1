############################# 2장

# import tensorflow as tf

# # 학습 데이터
# x_train = tf.constant([1, 2, 3], dtype=tf.float32)
# y_train = tf.constant([1, 2, 3], dtype=tf.float32)

# # 변수 초기화
# W = tf.Variable(tf.random.normal([1]))
# b = tf.Variable(tf.random.normal([1]))

# # 학습률
# learning_rate = 0.01

# # 학습 루프
# for step in range(2001):
#     with tf.GradientTape() as tape:
#         hypothesis = W * x_train + b
#         cost = tf.reduce_mean(tf.square(hypothesis - y_train))

#     gradients = tape.gradient(cost, [W, b])
#     W.assign_sub(learning_rate * gradients[0])
#     b.assign_sub(learning_rate * gradients[1])

#     if step % 20 == 0:
#         print(f"step: {step}, cost: {cost.numpy():.4f}, W: {W.numpy()}, b: {b.numpy()}")

################################### 3장 

# import numpy as np
# import tensorflow as tf

# # XOR 데이터셋
# x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
# y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

# # 모델 구성: 은닉층 1개 + 출력층
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(2, activation='sigmoid'),  # 은닉층 (뉴런 2개)
#     tf.keras.layers.Dense(1, activation='sigmoid')   # 출력층
# ])

# # 컴파일: 손실 함수 & 옵티마이저 설정
# model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# # 학습
# model.fit(x_data, y_data, epochs=5000, verbose=0)

# # 평가
# results = model.evaluate(x_data, y_data)
# print("정확도:", results[1])

# # 예측
# print("예측값:")
# print(model.predict(x_data))


###################### 4장

# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt

# # 1. MNIST 데이터셋 로딩
# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# # 2. 데이터 전처리 (정규화)
# x_train = x_train / 255.0
# x_test = x_test / 255.0

# # 3. 모델 구성
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),         # 입력층: 28x28 → 784
#     tf.keras.layers.Dense(128, activation='relu'),         # 은닉층: 뉴런 128개, ReLU
#     tf.keras.layers.Dropout(0.2),                          # 과적합 방지용 Dropout
#     tf.keras.layers.Dense(10, activation='softmax')        # 출력층: 클래스 10개, softmax
# ])

# # 4. 컴파일: 최적화기, 손실 함수, 평가 지표 설정
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# # 5. 모델 학습
# model.fit(x_train, y_train, epochs=5)

# # 6. 테스트셋으로 평가
# test_loss, test_acc = model.evaluate(x_test, y_test)
# print("\n테스트 정확도:", test_acc)

# # 7. 예측 수행
# predictions = model.predict(x_test)

# # 8. 예측 결과 확인 (예: 첫 번째 이미지)
# plt.imshow(x_test[0], cmap='gray')
# plt.title(f"정답: {y_test[0]}, 예측: {np.argmax(predictions[0])}")
# plt.show()

################################### 5장


# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt

# # 1. 데이터 로딩 및 정규화
# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

# # 2. 모델 구성 함수
# def create_model():
#     return tf.keras.models.Sequential([
#         tf.keras.layers.Flatten(input_shape=(28, 28)),
#         tf.keras.layers.Dense(128, activation='relu'),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(10, activation='softmax')
#     ])

# # 3. 실험용 학습률 목록
# learning_rates = [0.1, 0.01, 0.001]

# # 4. 결과 저장용 리스트
# histories = []

# # 5. 학습률에 따른 실험 반복
# for lr in learning_rates:
#     print(f"\n▶ 학습률: {lr}")
#     model = create_model()
#     model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
#     history = model.fit(x_train, y_train, epochs=20, validation_split=0.2, verbose=0)
#     histories.append((lr, history))

# # 6. 그래프로 시각화
# plt.figure(figsize=(10, 6))
# for lr, history in histories:
#     plt.plot(history.history['loss'], label=f"lr={lr}")
# plt.title("학습률에 따른 손실 감소 곡선")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.grid(True)
# plt.show()

############################# 6 장

# import tensorflow as tf
# import numpy as np

# # 1. 데이터 로딩
# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

# # 2. 심층 신경망 모델 구성
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),       # 입력층: 28x28 → 784
#     tf.keras.layers.Dense(512, activation='relu'),       # 은닉층 1: 512 뉴런
#     tf.keras.layers.Dropout(0.3),                        # Dropout: 30%
#     tf.keras.layers.Dense(256, activation='relu'),       # 은닉층 2: 256 뉴런
#     tf.keras.layers.Dropout(0.3),                        # Dropout: 30%
#     tf.keras.layers.Dense(10, activation='softmax')      # 출력층: 10 클래스
# ])

# # 3. 컴파일
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# # 4. 학습
# model.fit(x_train, y_train, epochs=15, validation_split=0.2)

# # 5. 평가
# test_loss, test_acc = model.evaluate(x_test, y_test)
# print("\n테스트 정확도:", test_acc)

################################# 7장

# import tensorflow as tf
# import numpy as np

# # 1. 데이터 불러오기
# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

# # 2. 모델 정의 함수 (Dropout 유무 선택)
# def build_model(use_dropout=True):
#     model = tf.keras.Sequential()
#     model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
#     model.add(tf.keras.layers.Dense(512, activation='relu'))
#     if use_dropout:
#         model.add(tf.keras.layers.Dropout(0.3))  # 30% 뉴런 제거
#     model.add(tf.keras.layers.Dense(10, activation='softmax'))
#     return model

# # 3. Dropout 있음/없음 모델 각각 생성
# model_with_dropout = build_model(use_dropout=True)
# model_without_dropout = build_model(use_dropout=False)

# # 4. 컴파일 및 학습 함수
# def train_model(model, name):
#     model.compile(optimizer='adam',
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
#     print(f"\n▶ 학습 시작: {name}")
#     history = model.fit(x_train, y_train, epochs=15, validation_split=0.2, verbose=0)
#     test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
#     print(f"{name} - 테스트 정확도: {test_acc:.4f}")
#     return history

# # 5. 학습 및 평가
# history_drop = train_model(model_with_dropout, "Dropout 사용")
# history_nodrop = train_model(model_without_dropout, "Dropout 미사용")


############################ 8장

# import tensorflow as tf

# # 1. 데이터 로딩 및 전처리
# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# # 2. CNN용 데이터 형태로 변형 (채널 차원 추가)
# x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
# x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# # 3. CNN 모델 구성
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
#     tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])

# # 4. 컴파일 및 학습
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=5, validation_split=0.2)

# # 5. 평가
# test_loss, test_acc = model.evaluate(x_test, y_test)
# print("\n테스트 정확도:", test_acc)


################################# 9장

# import tensorflow as tf
# import numpy as np

# # 1. 문자 데이터셋 만들기
# char_set = ['h', 'e', 'l', 'o']
# char2idx = {c: i for i, c in enumerate(char_set)}  # 문자 → 숫자 매핑
# idx2char = {i: c for i, c in enumerate(char_set)}  # 숫자 → 문자 매핑

# # 'hello' 입력과 출력 정의
# x_data = [char2idx[c] for c in 'hell']   # [0, 1, 2, 2]
# y_data = [char2idx[c] for c in 'ello']   # [1, 2, 2, 3]

# # 2. One-hot encoding
# x_one_hot = tf.keras.utils.to_categorical(x_data, num_classes=len(char_set))

# # RNN 입력 형식: (batch_size, time_steps, input_dim)
# x_one_hot = np.reshape(x_one_hot, (1, 4, 4))  # 1개 샘플, 4글자, 4차원(one-hot)
# y_data = np.reshape(y_data, (1, 4))           # 정답

# # 3. RNN 모델 정의
# model = tf.keras.Sequential([
#     tf.keras.layers.SimpleRNN(10, return_sequences=True, input_shape=(4, 4)),
#     tf.keras.layers.Dense(len(char_set), activation='softmax')
# ])

# # 4. 컴파일 및 학습
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(x_one_hot, y_data, epochs=100)

# # 5. 예측
# pred = model.predict(x_one_hot)
# pred_char = [idx2char[np.argmax(p)] for p in pred[0]]
# print("예측 결과:", ''.join(pred_char))

###################################### 10 장


# import tensorflow as tf
# from tensorflow.keras.datasets import imdb
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# # 1. 데이터셋 로딩 (상위 10,000단어만 사용)
# num_words = 10000
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

# # 2. 입력 데이터 전처리 (길이 맞추기)
# maxlen = 500
# x_train = pad_sequences(x_train, maxlen=maxlen)
# x_test = pad_sequences(x_test, maxlen=maxlen)

# # 3. 모델 구성 (RNN 기반)
# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(num_words, 32, input_length=maxlen),
#     tf.keras.layers.SimpleRNN(32),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# # 4. 컴파일 및 학습
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# model.summary()
# model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# # 5. 평가
# test_loss, test_acc = model.evaluate(x_test, y_test)
# print("\n테스트 정확도:", test_acc)


################################ CNN example.


# import matplotlib.pyplot as plt
# import numpy as np
# import tensorflow as tf

# # 입력 이미지 (MNIST 숫자 7)
# (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
# x = x_train[y_train == 6][0]  # 숫자 7 중 하나

# # 정규화 및 차원 확장
# x_input = x.astype(np.float32) / 255.0
# x_input = x_input.reshape(1, 28, 28, 1)

# # 간단한 CNN 모델 정의 (한 개의 필터만 사용하여 시각화 쉽게)
# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1))
# ])

# # 임의의 초기 가중치로 특징 맵 생성
# feature_map = model.predict(x_input)

# # 시각화
# fig, axs = plt.subplots(1, 2, figsize=(10, 4))

# axs[0].imshow(x, cmap='gray')
# axs[0].set_title("input image (.....)")
# axs[0].axis('off')

# axs[1].imshow(feature_map[0, :, :, 0], cmap='viridis')
# axs[1].set_title("Conv filter applied results (Feature Map)")
# axs[1].axis('off')

# plt.tight_layout()
# plt.show()

############################################# CNN exmaple

# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt

# # MNIST 데이터 중 숫자 7 하나 불러오기
# (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
# x_sample = x_train[y_train == 7][0]
# x_input = x_sample.astype(np.float32) / 255.0
# x_input = x_input.reshape(1, 28, 28, 1)

# # 필터 수별 CNN 모델들 생성 (1, 8, 32 filters)
# filter_counts = [1, 8, 32]
# feature_maps = []

# for count in filter_counts:
#     model = tf.keras.Sequential([
#         tf.keras.layers.Conv2D(filters=count, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1))
#     ])
#     fmap = model.predict(x_input)
#     feature_maps.append(fmap[0])

# # 시각화: 원본 + feature map들 (1, 8, 32 filters)
# fig, axes = plt.subplots(4, 8, figsize=(16, 8))
# axes = axes.flatten()

# # 0: 원본 이미지
# axes[0].imshow(x_sample, cmap='gray')
# axes[0].set_title("원본 이미지")
# axes[0].axis('off')

# # 나머지: 필터 수 별 feature map
# titles = ["1 filter", "8 filters", "32 filters"]
# idx = 1
# for i, fmap in enumerate(feature_maps):
#     for j in range(min(fmap.shape[-1], 8)):  # 최대 8개까지만 시각화
#         axes[idx].imshow(fmap[:, :, j], cmap='viridis')
#         axes[idx].set_title(f"{titles[i]} #{j+1}")
#         axes[idx].axis('off')
#         idx += 1

# # 빈 칸 비우기
# for k in range(idx, 32):
#     axes[k].axis('off')

# plt.tight_layout()
# plt.show()



######################################### CNN full example

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터셋 로딩 및 정규화
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

# CNN용 형식으로 reshape: (batch, height, width, channel)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 2. CNN 모델 정의
model = tf.keras.Sequential([
    # 첫 번째 Convolution + Pooling
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    # 두 번째 Convolution + Pooling
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Flatten + Fully Connected Layer
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')  # 숫자 0~9 분류용
])

# 3. 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. 학습
model.fit(x_train, y_train, epochs=5, validation_split=0.2)

# 5. 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print("테스트 정확도:", test_acc)