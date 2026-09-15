"""
tts_stt.py - Audio Engine Module (TTS via gTTS + pygame, STT via SpeechRecognition)

This module handles:
1. Text-to-Speech (TTS):
   - Generates an MP3 using Google Text-to-Speech (gTTS - free, no API key needed).
   - Plays the audio locally through host speakers using pygame.mixer.
   - Safely catches headless/cloud environments where host audio devices might not exist.
   - Returns the audio file path so Streamlit can also render an in-browser audio player.

2. Speech-to-Text (STT):
   - Browser Audio: Transcribes audio bytes recorded by Streamlit's in-browser audio recorder.
   - Local Hardware Microphone: Captures microphone audio using SpeechRecognition and PyAudio,
     automatically detecting the real hardware microphone device.
   - Transcribes speech to text using the free Google Web Speech API (recognize_google).
"""

import os
import io
import wave
from typing import Tuple, Optional
from gtts import gTTS
import pygame
import speech_recognition as sr
import numpy as np
from scipy.io import wavfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_CACHE_DIR = os.path.join(BASE_DIR, "temp_audio")


def ensure_audio_dir() -> str:
    """Ensures that the directory for temporary audio files exists."""
    os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)
    return AUDIO_CACHE_DIR


def speak_text(text: str, filename: str = "question.mp3") -> Tuple[bool, str]:
    """
    Converts text to speech using gTTS, saves it as an MP3 file,
    and plays it through host speakers via pygame.mixer.

    Args:
        text: The interview question text to speak aloud.
        filename: Destination MP3 file name.

    Returns:
        (success: bool, audio_filepath: str)
    """
    ensure_audio_dir()
    filepath = os.path.abspath(os.path.join(AUDIO_CACHE_DIR, filename))

    try:
        # 1. Generate speech with gTTS
        tts = gTTS(text=text, lang="en", slow=False)
        tts.save(filepath)

        # 2. Play audio locally using pygame
        try:
            # Initialize mixer if not already initialized
            if not pygame.mixer.get_init():
                pygame.mixer.pre_init(44100, -16, 2, 2048)
                pygame.mixer.init()

            pygame.mixer.music.set_volume(1.0)

            # Stop any previously playing audio before loading new track
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
            pygame.mixer.music.unload()

            pygame.mixer.music.load(filepath)
            pygame.mixer.music.play()
        except Exception as audio_device_err:
            # In headless environments (like cloud deployment), host sound cards may not exist.
            # We log this gracefully and rely on Streamlit's in-browser audio player.
            print(f"[Notice] Pygame audio playback skipped (headless or no soundcard): {audio_device_err}")

        return True, filepath

    except Exception as e:
        print(f"[Error] TTS generation failed: {e}")
        return False, ""


def get_best_microphone_index() -> Optional[int]:
    """
    Finds the index of a physical hardware microphone (e.g. Realtek, Microphone Array, Headset)
    to avoid recording from silent virtual audio drivers (e.g. Virtual Audio Cable, DroidCam).
    """
    try:
        names = sr.Microphone.list_microphone_names()
        for idx, name in enumerate(names):
            name_lower = name.lower()
            if any(k in name_lower for k in ["realtek", "microphone array", "built-in", "internal", "headset"]):
                return idx
        return None
    except Exception:
        return None


def transcribe_audio_bytes(audio_bytes: bytes) -> Tuple[bool, str]:
    """
    Transcribes raw audio bytes (such as from Streamlit's st.audio_input widget)
    using SpeechRecognition's Google Web Speech backend.
    
    Automatically converts float/stereo audio to standard 16-bit PCM WAV.
    """
    if not audio_bytes:
        return False, "No audio recorded."

    try:
        # Read WAV from bytes
        sample_rate, data = wavfile.read(io.BytesIO(audio_bytes))
        
        # Convert stereo to mono if necessary
        if len(data.shape) > 1:
            data = data.mean(axis=1)

        # Convert float audio to 16-bit signed PCM
        if data.dtype in (np.float32, np.float64):
            data = np.clip(data * 32767.0, -32768, 32767).astype(np.int16)
        elif data.dtype != np.int16:
            data = data.astype(np.int16)

        # Write clean PCM WAV to in-memory buffer
        pcm_buffer = io.BytesIO()
        with wave.open(pcm_buffer, "wb") as wav_out:
            wav_out.setnchannels(1)
            wav_out.setsampwidth(2)
            wav_out.setframerate(sample_rate)
            wav_out.writeframes(data.tobytes())
        pcm_buffer.seek(0)

        # Transcribe with SpeechRecognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(pcm_buffer) as source:
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data)
        return True, text.strip()

    except sr.UnknownValueError:
        return False, "Could not understand the audio. Please speak clearly into your microphone."
    except sr.RequestError as req_err:
        return False, f"Google Speech Recognition service unreachable: {req_err}"
    except Exception as e:
        return False, f"Audio processing error: {str(e)}"


def record_and_transcribe(timeout: int = 10, phrase_time_limit: int = 35) -> Tuple[bool, str]:
    """
    Listens to the user's microphone hardware and transcribes spoken words
    using Google Web Speech API.
    """
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 1.0

    best_idx = get_best_microphone_index()

    try:
        mic_kwargs = {"device_index": best_idx} if best_idx is not None else {}
        with sr.Microphone(**mic_kwargs) as source:
            # Calibrate for ambient room noise
            recognizer.adjust_for_ambient_noise(source, duration=0.6)
            audio_data = recognizer.listen(
                source,
                timeout=timeout,
                phrase_time_limit=phrase_time_limit
            )

        transcribed_text = recognizer.recognize_google(audio_data)
        return True, transcribed_text.strip()

    except sr.WaitTimeoutError:
        return False, "No speech detected within the time limit. Please speak directly after clicking the button."
    except sr.UnknownValueError:
        return False, "Could not understand the audio. Please speak clearly into your microphone."
    except sr.RequestError as req_err:
        return False, f"Speech recognition service error (check internet connectivity): {req_err}"
    except Exception as general_err:
        return False, f"Microphone error: {str(general_err)}. You can also type your answer."
