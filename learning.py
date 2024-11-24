import os
import glob
import librosa
import numpy as np
import pandas as pd
import pickle


from tensorflow.keras.utils import to_categorical

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.python.client import device_lib
# from tensorflow.python.client import device_lib
print(device_lib.list_local_devices() )

physical_devices = tf.config.list_physical_devices('GPU')

try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  print(physical_devices[0])
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass


max_pad_len = 174

def extract_feature(file_name):
    print('file name :', file_name)
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0,0), (0, pad_width)), mode='constant')
        print(mfccs.shape)
        
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        print(e)
        return None
    
#     return padded_mfccs
    return mfccs


def setUp():
    fulldatasetpath = 'UrbanSound8K/audio/'
    metadata = pd.read_csv('UrbanSound8K/metadata/UrbanSound8K.csv')
    features = []

    # Filter metadata for class_ID == 3
    # filtered_metadata = metadata[metadata["classID"]]

    # Iterate through filtered sound files and extract the features
    for index, row in metadata.iterrows():
        file_name = os.path.join(os.path.abspath(fulldatasetpath),
                                 'fold' + str(row["fold"]) + '/', str(row["slice_file_name"]))
        
        class_label = row["classID"]  # This will always be 3 in this case
        data = extract_feature(file_name)  # Extract features for the filtered data
        
        features.append([data, class_label])

    # Convert into a Panda dataframe
    featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])

    print(len(featuresdf))
    featuresdf.to_pickle("featuresdf.pkl")

setUp()


featuresdf = pd.read_pickle("featuresdf.pkl")


X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))
x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)

from tensorflow.keras.utils import to_categorical

# # y_train을 one-hot encoding으로 변환
# y_train = to_categorical(y_train, num_classes=10)
# y_test = to_categorical(y_test, num_classes=10)



# 예시로 출력
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print(y_test[:10])
# setUp()

n_columns = 174    
n_row = 40       
n_channels = 1
n_classes = 10

# input shape 조정
# cpu를 사용해서 수행한다
with tf.device('/gpu:0'):
    x_train = tf.reshape(x_train, [-1, n_row, n_columns, n_channels])
    x_test = tf.reshape(x_test, [-1, n_row, n_columns, n_channels])

model = keras.Sequential()

model.add(layers.Conv2D(input_shape=(n_row, n_columns, n_channels), filters=16, kernel_size=2, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=2))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(kernel_size=2, filters=32, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=2))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(kernel_size=2, filters=64, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=2))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(kernel_size=2, filters=128, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=2))
model.add(layers.Dropout(0.2))

model.add(layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

model.summary() 


training_epochs = 72
num_batch_size = 128

learning_rate = 0.001
opt = keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# history = model.fit(x_train, y_train, batch_size=num_batch_size, epochs=training_epochs)
history = model.fit(x_train, y_train, batch_size=num_batch_size, epochs=training_epochs, validation_split=0.2)



import matplotlib.pyplot as plt

# 시각화 함수 정의
def vis(history, key):
    plt.plot(history.history[key])  # 훈련 데이터를 플로팅
    plt.plot(history.history['val_' + key])  # 검증 데이터를 플로팅
    plt.title(f'Model {key.capitalize()}')  # 제목
    plt.xlabel('Epoch')  # x축 레이블
    plt.ylabel(key.capitalize())  # y축 레이블
    plt.legend(['Train', 'Val'], loc='best')  # 범례 표시

# 훈련 히스토리 시각화 함수
def plot_history(history):
    # history.keys()에서 'val_'로 시작하는 키를 제외한 유니크한 값들을 찾습니다
    key_value = list(set([i.split("val_")[-1] for i in list(history.history.keys())]))
    
    plt.figure(figsize=(12, 4))  # 그림 크기 설정
    for idx, key in enumerate(key_value):
        plt.subplot(1, len(key_value), idx+1)  # 서브플롯을 생성
        vis(history, key)  # 각 지표에 대해 vis 함수 호출
    
    plt.tight_layout()  # 레이아웃을 자동으로 맞춰줌
    plt.show()

# 이 함수를 사용하여 훈련 기록을 시각화
plot_history(history)

print('\n# Evaluate on test data')
results = model.evaluate(x_test, y_test, batch_size=128)
print('test loss, test acc:', results)

model.save("sound_classifier_model")
