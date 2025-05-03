# app/speech.py

import whisper
import pyaudio
import wave
import os
import tempfile

# Load Whisper model (tiny/small/base/medium/large)
MODEL_NAME = "base"  # You can try "small" or "medium" for better accuracy
model = whisper.load_model(MODEL_NAME)

def record_audio(duration=5, filename=None):
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1
    fs = 16000  # Sample rate

    p = pyaudio.PyAudio()

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    print("🎤 Recording...")
    frames = []

    for _ in range(0, int(fs / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()
    print("✅ Recording complete.")

    # Save the recorded data as a WAV file
    if not filename:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        filename = temp_file.name

    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

    return filename

def recognize_voice(duration=5):
    audio_path = record_audio(duration=duration)
    result = model.transcribe(audio_path, language="ur")  # or None to auto-detect
    os.remove(audio_path)
    return result["text"]
