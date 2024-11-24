import pyaudio
import numpy as np
import time

# 오디오 스트림 설정
FORMAT = pyaudio.paInt16  # 16비트 정수 형식
CHANNELS = 1  # 모노 채널
RATE = 44100  # 샘플링 레이트 (44.1kHz)
CHUNK = 1024  # 한 번에 읽을 오디오 샘플의 크기
DURATION = 2  # 2초마다 소리 판단

# PyAudio 객체 생성
p = pyaudio.PyAudio()

# 오디오 스트림 열기
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("소리 분석 시작...")

# 소리 판단 함수 (소리의 볼륨 계산)
def get_volume(data):
    # numpy 배열로 변환하여 RMS (Root Mean Square) 계산
    samples = np.frombuffer(data, dtype=np.int16)
    rms = np.sqrt(np.mean(samples**2))
    return rms

# 2초마다 소리 측정
try:
    while True:
        # 오디오 데이터 읽기
        data = stream.read(CHUNK)

        # 소리의 볼륨 계산
        volume = get_volume(data)

        # 2초마다 소리 출력
        print(f"현재 소리 볼륨: {volume:.2f}")

        # 2초마다 소리 판단
        time.sleep(DURATION)

except KeyboardInterrupt:
    print("소리 분석 종료")

finally:
    # 스트림 종료 및 정리
    stream.stop_stream()
    stream.close()
    p.terminate()
