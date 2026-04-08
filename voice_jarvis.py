import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import time
import warnings
import wave
from collections import deque
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
USE_VAD = os.getenv("JARVIS_USE_VAD", "1").strip().lower() not in {"0", "false", "no", "off"}
USE_PORCUPINE = os.getenv("JARVIS_USE_PORCUPINE", "1").strip().lower() not in {"0", "false", "no", "off"}
PORCUPINE_ACCESS_KEY = os.getenv("PORCUPINE_ACCESS_KEY", "").strip()
PORCUPINE_KEYWORD_PATH = os.getenv("PORCUPINE_KEYWORD_PATH", "").strip()
PORCUPINE_BUILTIN_KEYWORD = os.getenv("PORCUPINE_BUILTIN_KEYWORD", "jarvis").strip().lower()
PORCUPINE_SENSITIVITY = float(os.getenv("PORCUPINE_SENSITIVITY", "0.55"))
PORCUPINE_WAIT_TIMEOUT_SECONDS = int(os.getenv("PORCUPINE_WAIT_TIMEOUT_SECONDS", "0"))

SAMPLE_RATE = 16000
RECORD_SECONDS = 5
AUDIO_FILE = "input.wav"
VAD_FRAME_MS = 30
VAD_AGGRESSIVENESS = int(os.getenv("JARVIS_VAD_AGGRESSIVENESS", "2"))
VAD_SILENCE_MS = int(os.getenv("JARVIS_VAD_SILENCE_MS", "1000"))
VAD_MAX_SECONDS = int(os.getenv("JARVIS_VAD_MAX_SECONDS", "15"))

_model: Optional[Any] = None
_engine: Optional[Any] = None
_vad_module: Optional[Any] = None
_vad_checked = False
_pvporcupine_module: Optional[Any] = None
_pvporcupine_checked = False


def record_audio_fixed(filename: str = AUDIO_FILE, duration: int = RECORD_SECONDS, fs: int = SAMPLE_RATE) -> str:
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


def get_vad_module() -> Optional[Any]:
    global _vad_module, _vad_checked
    if not _vad_checked:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="pkg_resources is deprecated as an API.*",
                    category=UserWarning,
                )
                import webrtcvad  # type: ignore

            _vad_module = webrtcvad
        except Exception:
            _vad_module = None
        _vad_checked = True
    return _vad_module


def record_audio_vad(
    filename: str = AUDIO_FILE,
    fs: int = SAMPLE_RATE,
    aggressiveness: int = VAD_AGGRESSIVENESS,
    silence_threshold_ms: int = VAD_SILENCE_MS,
    max_seconds: int = VAD_MAX_SECONDS,
) -> Optional[str]:
    vad_module = get_vad_module()
    if vad_module is None:
        print("VAD dependency missing (webrtcvad). Falling back to fixed-window recording.")
        return record_audio_fixed(filename=filename, duration=RECORD_SECONDS, fs=fs)

    vad = vad_module.Vad(max(0, min(3, aggressiveness)))
    frame_ms = VAD_FRAME_MS
    frame_samples = int(fs * frame_ms / 1000)
    silence_frames = max(1, silence_threshold_ms // frame_ms)
    max_frames = max(1, int(max_seconds * 1000 / frame_ms))

    print("Listening (VAD)...")
    ring: deque[bool] = deque(maxlen=silence_frames)
    frames: list[bytes] = []
    triggered = False

    # macOS Fix: Try to open the stream. If -50 occurs, it often means the blocksize or samplerate
    # is not currently supported for streaming. Fall back to a safer capture if needed.
    try:
        stream = sd.InputStream(samplerate=fs, channels=1, dtype="int16", blocksize=frame_samples)
        stream.start()
    except Exception as e:
        print(f"Streaming VAD failed ({e}). Falling back to safer capture mode...")
        # Fallback to a mode without fixed blocksize which is more compatible on Mac
        try:
            stream = sd.InputStream(samplerate=fs, channels=1, dtype="int16")
            stream.start()
        except Exception as e2:
            print(f"Recovery failed: {e2}. Please check mic permissions.")
            return record_audio_fixed(filename=filename, duration=RECORD_SECONDS, fs=fs)

    try:
        while len(frames) < max_frames:
            # Read exactly frame_samples using the stream
            audio_chunk, overflowed = stream.read(frame_samples)
            if overflowed:
                pass # print("Audio overflowed")
            
            pcm_bytes = audio_chunk.tobytes()
            is_speech = vad.is_speech(pcm_bytes, fs)

            if not triggered:
                if is_speech:
                    triggered = True
                    print("Speech detected.")
                    frames.append(pcm_bytes)
            else:
                frames.append(pcm_bytes)
                ring.append(is_speech)
                if len(ring) == silence_frames and not any(ring):
                    print("Silence detected, done.")
                    break
    finally:
        stream.stop()
        stream.close()

    if not frames:
        return None

    with wave.open(filename, "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(fs)
        audio_file.writeframes(b"".join(frames))

    return filename


def record_audio() -> Optional[str]:
    if USE_VAD:
        return record_audio_vad()
    return record_audio_fixed()


def get_pvporcupine_module() -> Optional[Any]:
    global _pvporcupine_module, _pvporcupine_checked
    if not _pvporcupine_checked:
        try:
            import pvporcupine  # type: ignore

            _pvporcupine_module = pvporcupine
        except Exception:
            _pvporcupine_module = None
        _pvporcupine_checked = True
    return _pvporcupine_module


def create_porcupine_detector() -> Optional[Any]:
    if not USE_PORCUPINE or not WAKE_WORD:
        return None

    if not PORCUPINE_ACCESS_KEY:
        print("Porcupine disabled: PORCUPINE_ACCESS_KEY is missing. Falling back to transcript wake-word filter.")
        return None

    pvporcupine = get_pvporcupine_module()
    if pvporcupine is None:
        print("Porcupine dependency missing (pvporcupine). Falling back to transcript wake-word filter.")
        return None

    sensitivity = max(0.0, min(1.0, PORCUPINE_SENSITIVITY))

    try:
        if PORCUPINE_KEYWORD_PATH:
            return pvporcupine.create(
                access_key=PORCUPINE_ACCESS_KEY,
                keyword_paths=[PORCUPINE_KEYWORD_PATH],
                sensitivities=[sensitivity],
            )

        return pvporcupine.create(
            access_key=PORCUPINE_ACCESS_KEY,
            keywords=[PORCUPINE_BUILTIN_KEYWORD],
            sensitivities=[sensitivity],
        )
    except Exception as err:
        print(f"Porcupine init failed: {err}. Falling back to transcript wake-word filter.")
        return None


def wait_for_wake_word(detector: Any, timeout_seconds: int = PORCUPINE_WAIT_TIMEOUT_SECONDS) -> bool:
    print("Listening for wake word...")
    started_at = time.time()

    with sd.RawInputStream(
        samplerate=detector.sample_rate,
        channels=1,
        dtype="int16",
        blocksize=detector.frame_length,
    ) as stream:
        while True:
            pcm_bytes, _ = stream.read(detector.frame_length)
            pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
            if len(pcm) != detector.frame_length:
                continue

            result = detector.process(pcm)
            if result >= 0:
                print("Wake word detected.")
                return True

            if timeout_seconds > 0 and (time.time() - started_at) >= timeout_seconds:
                print("Wake-word wait timed out.")
                return False


def get_whisper_model() -> Any:
    global _model
    if _model is None:
        _model = whisper.load_model(WHISPER_MODEL_NAME)
    return _model


def speech_to_text(file_path: str) -> str:
    model = get_whisper_model()
    # Force language="en" to prevent Whisper from mis-identifying the language as gibberish
    result = model.transcribe(file_path, language="en")
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
    # macOS Fix: Use the system's built-in 'say' command which is much more reliable
    # Escape single quotes so they don't break the shell command
    safe_text = text.replace("'", " ").replace('"', " ").replace("\n", " ").strip()
    if safe_text:
        os.system(f"say '{safe_text}'")


def response_for_speech(response: Any) -> str:
    """Convert raw agent results into a natural, human-friendly summary for TTS."""
    if not isinstance(response, dict):
        return str(response)

    if "error" in response:
        error_msg = response["error"]
        if error_msg == "Invalid controller output":
            return "I'm sorry, I'm having trouble understanding my own instructions right now."
        return f"I encountered an error: {error_msg}"

    # Priority 1: Use the agent's direct 'thought' or spoken response if present
    # (The server might extract this from the raw LLM output)
    if "thought" in response and response["thought"]:
        return response["thought"]

    results = response.get("results", [])
    if not results:
        return "I completed the task, but there were no specific results to report."

    # Summarize the most relevant result (usually the last one or the most 'productive' one)
    summaries = []
    for item in results:
        action = item.get("action")
        result = item.get("result")
        
        if not result:
            continue

        if action == "list_files":
            if isinstance(result, list):
                files = ", ".join(result[:5])
                extras = f" and {len(result) - 5} more" if len(result) > 5 else ""
                summaries.append(f"I found {len(result)} files, including {files}{extras}.")
            else:
                summaries.append("I looked at your files.")

        elif action == "read_file":
            summaries.append(f"I finished reading the contents of the requested file.")

        elif action == "run_command":
            result_str = str(result).strip()
            summaries.append(f"The result is: {result_str[:150]}.")

        elif action == "search_project":
            if isinstance(result, list):
                summaries.append(f"I found {len(result)} matches for your search in the project.")
            else:
                summaries.append("I searched the project for you.")

        elif action == "open_app":
            app_name = str(result).replace("Opened ", "").strip()
            summaries.append(f"Opening {app_name} for you right now!")

    if not summaries:
        return "I've processed your request."

    # Return the last relevant summary (the most recent outcome)
    return summaries[-1]


def command_from_transcript(transcript: str) -> Optional[str]:
    cleaned = transcript.strip()
    if not cleaned:
        return None

    if not WAKE_WORD:
        return cleaned

    lowered = cleaned.lower().strip(" .,!?;:")
    
    # Flexible wake-word check (handles common Whisper mis-transcriptions)
    WAKE_VARIANTS = [WAKE_WORD, "service", "jarvis.", "jarvis,", "jarvis!", "javis"]
    
    pos = -1
    found_variant = ""
    for variant in WAKE_VARIANTS:
        pos = lowered.find(variant)
        if pos >= 0:
            found_variant = variant
            break
            
    if pos < 0:
        return None

    after = cleaned[pos + len(found_variant):].strip(" ,:-.!?")
    if after:
        return after

    before = cleaned[:pos].strip(" ,:-")
    return before or None


def main() -> None:
    print("Voice Jarvis started. Say 'exit' to stop.")
    print(f"Recording mode: {'VAD' if USE_VAD else 'fixed-window'}")
    if WAKE_WORD:
        print(f"Wake word enabled: {WAKE_WORD}")
    else:
        print("Wake word disabled.")

    detector = create_porcupine_detector()
    if detector is not None:
        print("Wake gate mode: Porcupine pre-trigger (Whisper runs only after wake detection).")
    elif WAKE_WORD:
        print("Wake gate mode: transcript filter fallback.")

    try:
        while True:
            try:
                if detector is not None:
                    if not wait_for_wake_word(detector):
                        continue
                    print("Speak your command.")

                audio_file = record_audio()
                if audio_file is None:
                    print("You: <silence>")
                    continue

                text = speech_to_text(audio_file)

                if not text:
                    print("You: <silence>")
                    continue

                print(f"You: {text}")

                if "exit" in text.lower():
                    print("Stopping voice loop.")
                    break

                if detector is not None:
                    command = text.strip()
                    if WAKE_WORD and WAKE_WORD in command.lower():
                        command = command_from_transcript(command) or ""
                else:
                    command = command_from_transcript(text) or ""
                    if not command and WAKE_WORD:
                        print(f"Wake word '{WAKE_WORD}' not detected; ignoring.")
                        continue

                command = command.strip()
                if not command:
                    print("No command detected after wake word.")
                    continue

                response = call_agent(command)
                speech_text = response_for_speech(response)
                print(f"Jarvis: {speech_text}")
                speak(speech_text)
            except requests.RequestException as err:
                print(f"Jarvis request failed: {err}")
                speak("I could not reach the agent endpoint.")
            except Exception as err:
                print(f"Runtime error: {err}")
                if "certificate verify failed" in str(err).lower():
                    print("Hint: install Python certificates on macOS, then retry.")
                speak("An unexpected error occurred.")
    finally:
        if detector is not None:
            detector.delete()


if __name__ == "__main__":
    main()
