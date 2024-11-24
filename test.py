import librosa
import numpy as np
import tensorflow as tf

# max_pad_len 변수 정의
max_pad_len = 174
sampling_rate = 22050  # 오디오 샘플링 레이트, 파일에 맞게 수정할 수 있습니다.

# 특징 추출 함수
def extract_feature(audio_segment):
    mfccs = librosa.feature.mfcc(y=audio_segment, sr=sampling_rate, n_mfcc=40)
    
    # mfccs의 두 번째 차원이 max_pad_len보다 크다면 자르고, 그렇지 않으면 패딩을 추가
    if mfccs.shape[1] > max_pad_len:
        mfccs = mfccs[:, :max_pad_len]  # 자르기
    else:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')  # 패딩하기
    
    return mfccs

# 모델 로드
model = tf.keras.models.load_model("sound_classifier_model")

# 테스트 파일 경로
file_path = 'recorded_audio.wav'

# 오디오 파일 로드
audio, _ = librosa.load(file_path, sr=sampling_rate)

# 1초씩 슬라이딩 윈도우로 오디오 자르기
segment_duration = 2  # 1초씩 자르기
segment_samples = segment_duration * sampling_rate  # 1초에 해당하는 샘플 수

# 오디오 길이
audio_length = len(audio)

# 결과를 저장할 리스트
predictions = []

# 1초씩 오디오 자르기 및 예측
for start in range(0, audio_length, segment_samples):
    end = min(start + segment_samples, audio_length)
    audio_segment = audio[start:end]
    
    # 특징 추출
    mfccs = extract_feature(audio_segment)
    
    # 모델에 맞게 형태 변경
    mfccs = np.reshape(mfccs, (-1, 40, 174, 1))
    
    # 예측 수행
    prediction = model.predict(mfccs)
    
    # 예측 결과를 저장
    predictions.append(prediction)

class_labels = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]

# 예측 결과 출력
barkcount = 0

for idx, prediction in enumerate(predictions):
    predicted_class = class_labels[np.argmax(prediction)]
    if predicted_class == "dog_bark" :
        barkcount+=1
    
    print(f"Segment {idx + 1} - Predicted class: {predicted_class}")
    # print(f"Prediction probabilities: {prediction}")

print(barkcount)
