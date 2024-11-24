
# # import librosa
# # import numpy as np
# # import tensorflow as tf
# # from tensorflow.keras.models import load_model

# # # 이미 학습된 모델을 로드
# # model = load_model("sound_classifier_model")

# # # MFCC 특성을 추출하는 함수

# # # # # 오디오 데이터를 모델에 맞게 전처리
# # def preprocess_audio(audio_path, sr=44100):
# #     # 오디오 파일 로드
# #     # audio, _ = librosa.load(audio_path, sr=sr)
# #     audio, _ = librosa.load(audio_path, sr=sr)
    
# #     # Mel-spectrogram 추출
# #     mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
    
# #     # 로그 변환 후 크기 조정
# #     log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
# #     # 입력 데이터 차원 맞추기 (모델의 입력 형태에 맞게)
# #     log_mel_spec = np.expand_dims(log_mel_spec, axis=-1)  # (height, width, channels)
# #     log_mel_spec = np.expand_dims(log_mel_spec, axis=0)  # (1, height, width, channels)
    
# #     return log_mel_spec

# # # 예측을 위한 함수
# # def predict_barks(audio_file):
# #     barks_count = 0
# #     processed_audio = preprocess_audio(audio_file)
    
# #     predictions = model.predict(processed_audio)
# #     print(predictions)

# #     return barks_count

# # # 예시로 오디오 파일에 대해 개 짖은 횟수를 예측
# # audio_file = 'extracted_audio.wav'
# # barks = predict_barks(audio_file)
# # print(f"Number of barks detected: {barks}")



# # # # import librosa
# # # # import numpy as np
# # # # import tensorflow as tf
# # # # from tensorflow.keras.models import load_model
# # # # import soundfile as sf

# # # # # 모델 로드
# # # # model = load_model('test_model')  # 이미 저장된 모델을 불러옵니다

# # # # # 오디오 데이터를 모델에 맞게 전처리
# # # # def preprocess_audio(audio_path, sr=44100):
# # # #     # 오디오 파일 로드
# # # #     audio, _ = librosa.load(audio_path, sr=sr)
    
# # # #     # Mel-spectrogram 추출
# # # #     mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
    
# # # #     # 로그 변환 후 크기 조정
# # # #     log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
# # # #     # 입력 데이터 차원 맞추기 (모델의 입력 형태에 맞게)
# # # #     log_mel_spec = np.expand_dims(log_mel_spec, axis=-1)  # (height, width, channels)
# # # #     log_mel_spec = np.expand_dims(log_mel_spec, axis=0)  # (1, height, width, channels)
    
# # # #     return log_mel_spec

# # # # # 강아지 소리 감지 함수
# # # # def detect_dog_barks(model, audio_path):
# # # #     processed_audio = preprocess_audio(audio_path)
    
# # # #     # 예측 (강아지 소리 감지)
# # # #     predictions = model.predict(processed_audio)
# # # #     bark_count = np.sum(predictions > 0.5)  # threshold > 0.5인 예측을 강아지 소리로 간주
    
# # # #     return bark_count

# # # # # 테스트: wav 파일에서 강아지 소리 감지
# # # # audio_path = 'extracted_audio.wav'  # wav 파일 경로
# # # # bark_count = detect_dog_barks(model, audio_path)

# # # # print(f"강아지 소리가 {bark_count}번 감지되었습니다.")


# # # # import librosa
# # # # import numpy as np
# # # # import tensorflow as tf
# # # # from tensorflow.keras.models import load_model
# # # # import soundfile as sf

# # # # # 모델 로드
# # # # model = load_model('test_model')  # 이미 저장된 모델을 불러옵니다

# # # # # 오디오 데이터를 모델에 맞게 전처리
# # # # def preprocess_audio(audio_path, sr=44100):
# # # #     # 오디오 파일 로드 (28초 오디오)
# # # #     audio, _ = librosa.load(audio_path, sr=sr, duration=28)  # 28초만 로드
    
# # # #     # Mel-spectrogram 추출 (n_fft=4096, hop_length=1024로 설정)
# # # #     mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000, n_fft=4096, hop_length=1024)
    
# # # #     # 로그 변환 후 크기 조정
# # # #     log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
# # # #     # 입력 데이터 차원 맞추기 (모델의 입력 형태에 맞게)
# # # #     log_mel_spec = np.expand_dims(log_mel_spec, axis=-1)  # (height, width, channels)
# # # #     log_mel_spec = np.expand_dims(log_mel_spec, axis=0)  # (1, height, width, channels)
    
# # # #     return log_mel_spec, audio

# # # # # 강아지 소리 감지 함수 (짖은 소리의 시간을 계산)
# # # # def detect_dog_barks(model, audio_path, threshold=0, sr=44100):
# # # #     processed_audio, audio = preprocess_audio(audio_path, sr)
    
# # # #     # 예측 (강아지 소리 감지)
# # # #     predictions = model.predict(processed_audio)
    
# # # #     # 예측값이 threshold 이상인 프레임 수 카운트
# # # #     bark_indices = np.where(predictions > threshold)[1]  # 짖은 소리가 감지된 프레임 인덱스
    
# # # #     # 각 짖은 소리가 발생한 시간 계산
# # # #     hop_length = 1024  # 각 프레임 간 샘플 간격
# # # #     frame_time = hop_length / sr  # 각 프레임의 시간 (초 단위)
    
# # # #     bark_times = bark_indices * frame_time  # 짖은 소리가 발생한 시간 (초 단위)
    
# # # #     return bark_times

# # # # # 테스트: wav 파일에서 강아지 소리 감지
# # # # audio_path = 'extracted_audio.wav'  # wav 파일 경로
# # # # bark_times = detect_dog_barks(model, audio_path)

# # # # print(f"강아지 소리는 {len(bark_times)}번 감지되었습니다.")
# # # # print(f"짖은 시간: {bark_times}초")


# # # # import librosa
# # # # import numpy as np
# # # # import tensorflow as tf
# # # # from tensorflow.keras.models import load_model
# # # # import soundfile as sf

# # # # # 모델 로드
# # # # test_model = load_model('test_model')  # 저장된 test_model을 불러옵니다

# # # # # 오디오 데이터를 모델에 맞게 전처리
# # # # def preprocess_audio(audio_path, sr=44100):
# # # #     # 오디오 파일 로드 (28초 오디오)
# # # #     audio, _ = librosa.load(audio_path, sr=sr, duration=25)  # 28초만 로드
    
# # # #     # Mel-spectrogram 추출 (n_fft=4096, hop_length=1024로 설정)
# # # #     mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000, n_fft=4096, hop_length=1024)
    
# # # #     # 로그 변환 후 크기 조정
# # # #     log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
# # # #     # 입력 데이터 차원 맞추기 (모델의 입력 형태에 맞게)
# # # #     log_mel_spec = np.expand_dims(log_mel_spec, axis=-1)  # (height, width, channels)
# # # #     log_mel_spec = np.expand_dims(log_mel_spec, axis=0)  # (1, height, width, channels)
    
# # # #     return log_mel_spec, audio

# # # # # 강아지 소리 감지 함수 (짖은 소리의 시간을 계산)
# # # # def detect_dog_barks(model, audio_path, threshold=0.5, sr=44100):
# # # #     processed_audio, audio = preprocess_audio(audio_path, sr)
    
# # # #     # 예측 (강아지 소리 감지)
# # # #     predictions = model.predict(processed_audio)
    
# # # #     print(f"Predictions shape: {predictions.shape}")
# # # #     frames = predictions.shape[1]  # 프레임 수 확인
# # # #     print(f"Number of frames: {frames}")
    
# # # #     # 각 프레임에 대해 예측값 출력
# # # #     hop_length = 1024
# # # #     frame_time = hop_length / sr  # 각 프레임의 시간 (초 단위)
    
# # # #     for i in range(frames):
# # # #         second = i * frame_time
# # # #         pred = predictions[0, i]  # 예측값
# # # #         print(f"Time: {second:.2f} sec, Prediction: {pred:.2f}")
        
# # # #     return predictions

# # # # # 테스트: wav 파일에서 강아지 소리 감지
# # # # audio_path = 'extracted_audio.wav'  # wav 파일 경로
# # # # predictions = detect_dog_barks(test_model, audio_path)

# # # # # 강아지 소리 감지 결과
# # # # print("강아지 소리 감지 완료!")


# # # import librosa
# # # import numpy as np
# # # import tensorflow as tf
# # # from tensorflow.keras.models import load_model

# # # # 모델 로드
# # # model = load_model('sound_classifier_model')  # 이미 저장된 모델을 불러옵니다

# # # # 오디오 데이터를 모델에 맞게 전처리
# # # def preprocess_audio(audio_path, sr=44100, frame_length=1.0, hop_length=0.5):
# # #     # 오디오 파일 로드
# # #     audio, _ = librosa.load(audio_path, sr=sr)
    
# # #     # Mel-spectrogram 추출
# # #     hop_length_samples = int(hop_length * sr)  # hop_length를 샘플 단위로 변환
# # #     mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000, hop_length=hop_length_samples)
    
# # #     # 로그 변환 후 크기 조정
# # #     log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
# # #     # 입력 데이터 차원 맞추기 (모델의 입력 형태에 맞게)
# # #     log_mel_spec = np.expand_dims(log_mel_spec, axis=-1)  # (height, width, channels)
# # #     log_mel_spec = np.expand_dims(log_mel_spec, axis=0)  # (1, height, width, channels)
    
# # #     return log_mel_spec, audio

# # # # 강아지 소리 감지 함수
# # # def detect_dog_barks(model, audio_path, threshold=0.5, sr=44100, frame_length=1.0, hop_length=0.5):
# # #     processed_audio, audio = preprocess_audio(audio_path, sr, frame_length, hop_length)
    
# # #     # 예측 (강아지 소리 감지)
# # #     predictions = model.predict(processed_audio)
# # #     print(f"Predictions shape: {predictions.shape}")
# # #     print("$$$$$$$$$$$")
# # #     print(predictions)
    
# # #     # 프레임 수
# # #     num_frames = predictions.shape[1]  # 예측 결과의 프레임 수
    
# # #     # 각 프레임에 대해 예측값 출력
# # #     hop_length_samples = int(hop_length * sr)  # 샘플 단위 hop_length 계산
# # #     frame_time = hop_length  # 각 프레임의 시간(초 단위)
    
# # #     # 오디오의 길이 확인
# # #     total_audio_time = len(audio) / sr  # 오디오 파일의 전체 시간 (초 단위)
# # #     print(f"$$$$$$$$ Total audio duration: {total_audio_time:.2f} seconds")

# # #     # 각 프레임에 대해 예측값 출력
# # #     for i in range(num_frames):
# # #         second = i * frame_time  # 시간 계산
# # #         pred = predictions[0, i]  # 예측값
# # #         print(f"Time: {second:.2f} sec, Prediction: {pred:.2f}")  # 예측값 출력
        
# # #         if pred > threshold:
# # #             print(f"Bark detected at {second:.2f} seconds.")
    
# # #     return predictions

# # # # 테스트: wav 파일에서 강아지 소리 감지
# # # audio_path = 'extracted_audio.wav'  # wav 파일 경로
# # # predictions = detect_dog_barks(model, audio_path)


# import librosa
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import soundfile as sf

# # 모델 로드
# model = load_model('sound_classifier_model')  # 이미 저장된 모델을 불러옵니다

# # 오디오 데이터를 모델에 맞게 전처리
# def preprocess_audio(audio, sr=44100):

#     # Mel-spectrogram 추출
#     mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
    
#     # 로그 변환 후 크기 조정
#     log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
#     # 입력 데이터 차원 맞추기 (모델의 입력 형태에 맞게)
#     log_mel_spec = np.expand_dims(log_mel_spec, axis=-1)  # (height, width, channels)
#     log_mel_spec = np.expand_dims(log_mel_spec, axis=0)  # (1, height, width, channels)
    
#     return log_mel_spec

# # 강아지 소리 감지 함수
# def detect_dog_barks(model, audio):
#     processed_audio = preprocess_audio(audio)
    
#     # 예측 (강아지 소리 감지)
#     predictions = model.predict(processed_audio)
#     print(predictions)
#     # bark_count = np.sum(predictions > 0.5)  # threshold > 0.5인 예측을 강아지 소리로 간주
#     return predictions[0][3]>0.5

#     # return bark_count

# # 테스트: wav 파일에서 강아지 소리 감지
# audio_path = '81722-3-0-21.wav'  # wav 파일 경로
#     # 오디오 파일 로드
# audio, _ = librosa.load(audio_path, sr=44100)
# bark_count = 0
# isbark = detect_dog_barks(model, audio)
# if isbark :
#     bark_count+=1

# print(f"강아지 소리가 {bark_count}번 감지되었습니다.")


import librosa
import numpy as np
import tensorflow as tf

# max_pad_len 변수 정의
max_pad_len = 174

# 특징 추출 함수
def extract_feature(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    
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
file_path = 'extracted_audio.wav'

# 특징 추출
mfccs = extract_feature(file_path)

# mfccs의 크기 확인
print("mfccs shape before reshape:", mfccs.shape)

# mfccs를 모델에 맞는 형태로 reshape
mfccs = np.reshape(mfccs, (-1, 40, 174, 1))

# 예측 수행
prediction = model.predict(mfccs)

# 예측 결과 확인
class_labels = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]
predicted_class = class_labels[np.argmax(prediction)]

print(f"Predicted class: {predicted_class}")
print(f"Prediction probabilities: {prediction}")
