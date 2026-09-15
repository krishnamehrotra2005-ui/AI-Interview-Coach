"""
tts_stt.py - Audio Engine Module (TTS via gTTS + pygame, STT via SpeechRecognition)

This module handles:
1. Text-to-Speech (TTS):
   - Generates an MP3 using Google Text-to-Speech (gTTS - free, no API key needed).
   - Plays the audio locally through host speakers using pygame.mixer.
   - Safely catches headless/cloud environments where host audio devices might not exist.
   - Returns the audio file path so Streamlit can also render an in-browser audio player.

2. Speech-to-Text (STT):
   - Captures microphone audio using SpeechRecognition and PyAudio.
   - Transcribes speech to text using the free Google Web Speech API (recognize_google).
   - Returns clean transcription strings for downstream Gemini evaluation.
"""

import os
import time
from typing import Tuple
from gtts import gTTS
import pygame
import speech_recognition as sr

AUDIO_CACHE_DIR = "temp_audio"


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
    filepath = os.path.join(AUDIO_CACHE_DIR, filename)

    try:
        # 1. Generate speech with gTTS
        tts = gTTS(text=text, lang="en", slow=False)
        tts.save(filepath)

        # 2. Play audio locally using pygame
        try:
            # Initialize mixer if not already initialized
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            
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


def record_and_transcribe(timeout: int = 8, phrase_time_limit: int = 30) -> Tuple[bool, str]:
    """
    Listens to the user's microphone and transcribes spoken words using Google Web Speech API.

    Args:
        timeout: Seconds to wait for speech to begin before raising a timeout.
        phrase_time_limit: Maximum allowed speaking duration in seconds.

    Returns:
        (success: bool, result_text: str): Transcribed text on success, or error message on failure.
    """
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True

    try:
        with sr.Microphone() as source:
            # Calibrate for room ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=0.8)
            print("[STT] Listening for user speech...")
            audio_data = recognizer.listen(
                source,
                timeout=timeout,
                phrase_time_limit=phrase_time_limit
            )

        # Transcribe using Google's free Web Speech recognition backend
        transcribed_text = recognizer.recognize_google(audio_data)
        return True, transcribed_text.strip()

    except sr.WaitTimeoutError:
        return False, "No speech detected within the time limit. Please try again."
    except sr.UnknownValueError:
        return False, "Could not understand the audio. Please speak clearly into your microphone."
    except sr.RequestError as req_err:
        return False, f"Speech recognition service error (check internet connectivity): {req_err}"
    except Exception as general_err:
        return False, f"Microphone error: {str(general_err)}. You can switch to typing your answer."
