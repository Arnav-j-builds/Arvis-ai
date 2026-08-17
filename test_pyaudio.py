import pyaudio

def test_installation():
    try:
        p = pyaudio.PyAudio()
        print(f"PyAudio version: {pyaudio.__version__}")
        print("Default input device info:")
        try:
            device_info = p.get_default_input_device_info()
            print(f" - Name: {device_info.get('name')}")
            print(f" - Max Input Channels: {device_info.get('maxInputChannels')}")
            print(f" - Default Sample Rate: {device_info.get('defaultSampleRate')}")
        except IOError:
            print(" - No default input device found (normal if no microphone is plugged in/enabled).")
        p.terminate()
        print("\nPyAudio is successfully installed and working!")
    except Exception as e:
        print(f"Failed to initialize PyAudio: {e}")

if __name__ == "__main__":
    test_installation()
