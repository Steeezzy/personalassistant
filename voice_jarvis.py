import os
import wave
from typing import Any, Optional

import numpy as np
import pyttsx3
import requests
import sounddevice as sd
import whisper

API_URL = os.getenv("JARVIS_API_URL", "http://127.0.0.1:8001/run")
API_KEY = os.getenv("JARVIS_API_KEY", "")
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "base")
WAKE_WORD = os.getenv("JARVIS_WAKE_WORD", "jarvis").strip().lower()

SAMPLE_RATE = 16000
RECORD_SECONDS = 5
AUDIO_FILE = "input.wav"

_model: Optional[Any] = None
_engine: Optional[Any] = None


def record_audio(filename: str = AUDIO_FILE, duration: int = RECORD_SECONDS, fs: int = SAMPLE_RATE) -> str:
    print("Listening...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()

    with wave.open(filename, "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(fs)
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        audio_file.writeframes(pcm16.tobytes())

    return filename


def get_whisper_model() -> Any:
    global _model
    if _model is None:
        _model = whisper.load_model(WHISPER_MODEL_NAME)
    return _model


def speech_to_text(file_path: str) -> str:
    model = get_whisper_model()
    result = model.transcribe(file_path)
    return result.get("text", "").strip()


def call_agent(text: str) -> Any:
    if not API_KEY:
        raise RuntimeError("JARVIS_API_KEY is not set. Export it before running voice_jarvis.py.")

    response = requests.post(
        API_URL,
        headers={
            "Content-Type": "application/json",
            "x-api-key": API_KEY,
        },
        json={"input": text},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def get_tts_engine() -> Any:
    global _engine
    if _engine is None:
        _engine = pyttsx3.init()
    return _engine


def speak(text: str) -> None:
    engine = get_tts_engine()
    engine.say(text)
    engine.runAndWait()


def response_for_speech(response: Any) -> str:
    if isinstance(response, dict):
        if "results" in response:
            return str(response["results"])
        if "error" in response:
            return f"Error: {response['error']}"
    return str(response)


def command_from_transcript(transcript: str) -> Optional[str]:
    cleaned = transcript.strip()
    if not cleaned:
        return None

    if not WAKE_WORD:
        return cleaned

    lowered = cleaned.lower()
    pos = lowered.find(WAKE_WORD)
    if pos < 0:
        return None

    after = cleaned[pos + len(WAKE_WORD):].strip(" ,:-")
    if after:
        return after

    before = cleaned[:pos].strip(" ,:-")
    return before or None


def main() -> None:
    print("Voice Jarvis started. Say 'exit' to stop.")
    if WAKE_WORD:
        print(f"Wake word enabled: {WAKE_WORD}")
    else:
        print("Wake word disabled.")

    while True:
        try:
            audio_file = record_audio()
            text = speech_to_text(audio_file)

            if not text:
                print("You: <silence>")
                continue

            print(f"You: {text}")

            if "exit" in text.lower():
                print("Stopping voice loop.")
                break

            command = command_from_transcript(text)
            if command is None:
                if WAKE_WORD:
                    print(f"Wake word '{WAKE_WORD}' not detected; ignoring.")
                continue

            response = call_agent(command)
            print(f"Jarvis: {response}")
            speak(response_for_speech(response))
        except requests.RequestException as err:
            print(f"Jarvis request failed: {err}")
            speak("I could not reach the agent endpoint.")
        except Exception as err:
            print(f"Runtime error: {err}")
            if "certificate verify failed" in str(err).lower():
                print("Hint: install Python certificates on macOS, then retry.")
            speak("An unexpected error occurred.")


if __name__ == "__main__":
    main()
