import sounddevice as sd
import numpy as np
import librosa
import wave
from tensorflow.keras.models import load_model
import time

# 모델 로드
model = load_model('sound_classifier_model.h5')

# 디바이스 목록 출력 및 선택 함수
def select_audio_device():
    print("Available audio devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        print(f"{i}: {device['name']} (Input Channels: {device['max_input_channels']})")
    
    while True:
        try:
            device_id = int(input("Enter the device ID to use for audio capture: "))
            if 0 <= device_id < len(devices) and devices[device_id]['max_input_channels'] > 0:
                print(f"Selected Device: {devices[device_id]['name']}")
                return device_id
            else:
                print("Invalid selection. Please select a valid input device.")
        except ValueError:
            print("Please enter a valid integer.")

# 오디오 캡처 함수 (duration 동안 녹음)
def capture_audio(device_id, duration=2, sample_rate=32000):
    """
    오디오를 캡처하는 함수.
    :param device_id: 사용할 오디오 입력 디바이스 ID
    :param duration: 녹음 시간 (초)
    :param sample_rate: 샘플링 레이트 (Hz)
    :return: 녹음된 오디오 데이터 (1D NumPy 배열), 샘플링 레이트
    """
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32', device=device_id)
    sd.wait()  # 녹음 완료 대기
    print("Recording Complete.")
    return audio.flatten(), sample_rate

# 특징 추출 함수 (MFCC를 사용)
def extract_features_from_audio(audio, sample_rate):
    """
    오디오 데이터에서 MFCC 특징을 추출하는 함수.
    :param audio: 1D NumPy 배열 오디오 데이터
    :param sample_rate: 샘플링 레이트 (Hz)
    :return: MFCC 특징 (NumPy 배열)
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

# 예측 함수 (강아지 소리 감지)
# 예측 함수 수정
def predict_dog_sound(audio_features):
    """
    추출된 오디오 특징으로 강아지 소리를 감지하는 함수.
    :param audio_features: MFCC 특징 (NumPy 배열)
    :return: 예측 결과 문자열
    """
    # 모델이 예상하는 형태로 입력 데이터 차원 변경
    audio_features = np.expand_dims(audio_features, axis=-1)  # (40,) -> (40, 1)
    audio_features = np.expand_dims(audio_features, axis=0)   # (40, 1) -> (1, 40, 1)
    audio_features = np.repeat(audio_features, 174, axis=2)  # (1, 40, 1) -> (1, 40, 174, 1)
    
    predictions = model.predict(audio_features)
    print(predictions)
    class_labels = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]


    # 예측 결과를 순차적으로 확인하여 강아지 소리 감지
    for idx, prediction in enumerate(predictions):

        predicted_class = class_labels[np.argmax(prediction)]  # 가장 큰 확률 값을 가지는 클래스 선택
        if predicted_class == "dog_bark":  # 예측된 클래스가 "dog_bark"인 경우
            return "Dog Sound Detected!"
        else: return "No Dog Sound."

def save_audio_to_wav(audio_data, sample_rate, filename="recorded_audio.wav"):
    """
    오디오 데이터를 WAV 파일로 저장하는 함수.
    :param audio_data: 오디오 데이터 (1D NumPy 배열)
    :param sample_rate: 샘플링 레이트
    :param filename: 저장할 파일 이름
    """
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)  # 모노 채널
        wf.setsampwidth(2)  # 샘플 폭 (2바이트, 즉 16비트)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())  # 바이트 형태로 변환하여 저장

# 일정 주기마다 오디오 데이터 수집 및 판독
def monitor_audio(device_id, duration=2, sample_rate=32000, interval=3):
    """
    특정 duration마다 오디오 데이터를 수집하고 판독하는 함수.
    :param device_id: 사용할 오디오 입력 디바이스 ID
    :param duration: 녹음 시간 (초)
    :param sample_rate: 샘플링 레이트 (Hz)
    :param interval: 녹음 간격 (초)
    """
    print("Monitoring for dog sounds... Press Ctrl+C to stop.")
    try:
        while True:
            # Step 1: 오디오 캡처
            audio, sr = capture_audio(device_id=device_id, duration=duration, sample_rate=sample_rate)
            
            # Step 2: 특징 추출
            features = extract_features_from_audio(audio, sr)
            
            # Step 3: 예측
            result = predict_dog_sound(features)
            print(result)

            save_audio_to_wav(audio, sr, filename="captured_audio.wav")
            
            # Step 4: 다음 녹음을 위해 대기
            #time.sleep(interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

# 메인 실행
if __name__ == "__main__":
    # Step 1: 오디오 입력 디바이스 선택
    selected_device_id = 4
    
    # Step 2: 오디오 모니터링 시작
    monitor_audio(device_id=selected_device_id, duration=2, sample_rate=22050, interval=3)
