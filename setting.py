import pyaudio
import sounddevice as sd

def list_audio_devices():
    p = pyaudio.PyAudio()
    print("Available audio input devices:")
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if dev_info['maxInputChannels'] > 0:  # 입력 채널이 있는 장치만 표시
            print(f"Device {i}: {dev_info['name']}")
    p.terminate()


if __name__ == "__main__":
    # Step 1: 사용 가능한 장치 나열
    list_audio_devices()

    device_info = sd.query_devices(32)
    print(device_info)