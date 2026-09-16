"""
tts_stt.py - Audio Engine Module (TTS via gTTS + pygame, STT via SpeechRecognition)

This module handles:
1. Text-to-Speech (TTS):
   - Preprocesses technical text so acronyms (EDA, LLM, RAG, SQL, API) are spoken letter-by-letter.
   - Generates natural, crisp speech using Google Text-to-Speech (gTTS) with professional pacing.
   - Plays audio locally through host speakers using pygame.mixer.
   - Returns absolute audio file paths for Streamlit in-browser playback with autoplay.

2. Speech-to-Text (STT):
   - Transcribes in-browser audio recordings from Streamlit.
   - Post-processes transcriptions to correct common technical homophones (e.g. 'mike check' -> 'mic check', 'sequel' -> 'SQL').
"""

import os
import io
import re
import time
import wave
import shutil
import subprocess
from typing import Tuple, Optional
from gtts import gTTS
import pygame
import speech_recognition as sr
import numpy as np
from scipy.io import wavfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_CACHE_DIR = os.path.join(BASE_DIR, "temp_audio")

# Dictionary of technical acronyms that TTS engines mispronounce as single words
# e.g. EDA -> 'eeda' becomes 'E-D-A' so each letter is pronounced distinctly.
COMMON_ACRONYMS = {
    "EDA": "E-D-A",
    "LLM": "L-L-M",
    "LLMS": "L-L-Ms",
    "RAG": "R-A-G",
    "NLP": "N-L-P",
    "ML": "M-L",
    "AI": "A-I",
    "SQL": "S-Q-L",
    "NOSQL": "No-S-Q-L",
    "API": "A-P-I",
    "APIS": "A-P-Is",
    "REST": "R-E-S-T",
    "CI/CD": "C-I C-D",
    "CI": "C-I",
    "CD": "C-D",
    "AWS": "A-W-S",
    "GCP": "G-C-P",
    "ETL": "E-T-L",
    "ELT": "E-L-T",
    "JD": "J-D",
    "PR": "P-R",
    "SRE": "S-R-E",
    "OOP": "O-O-P",
    "UI": "U-I",
    "UX": "U-X",
    "K8S": "Kubernetes",
    "K8s": "Kubernetes",
    "SDK": "S-D-K",
    "DB": "database",
    "QA": "Q-A",
    "CSV": "C-S-V",
    "JSON": "J-S-O-N",
    "HTML": "H-T-M-L",
    "CSS": "C-S-S",
}

# Technical homophones where general STT confuses conversational English with tech words
TECHNICAL_HOMOPHONES = {
    r"\bmike check\b": "mic check",
    r"\bmike\b": "mic",
    r"\bsequel\b": "SQL",
    r"\bno sequel\b": "NoSQL",
    r"\bpost gres\b": "PostgreSQL",
    r"\bpostgress\b": "Postgres",
    r"\bsee sharp\b": "C#",
    r"\bsee plus plus\b": "C++",
    r"\bdot net\b": ".NET",
    r"\bdock er\b": "Docker",
    r"\bpie torch\b": "PyTorch",
    r"\bsci kit\b": "Scikit",
    r"\btensor flow\b": "TensorFlow",
    r"\be d a\b": "EDA",
    r"\beeda\b": "EDA",
    r"\bkubernetees\b": "Kubernetes",
    r"\bgit hub\b": "GitHub",
    r"\bgit lab\b": "GitLab",
    r"\breact js\b": "React.js",
    r"\bnode js\b": "Node.js",
    r"\bvue js\b": "Vue.js",
    r"\bfront end\b": "frontend",
    r"\bback end\b": "backend",
    r"\bfull stack\b": "fullstack",
    r"\brest api\b": "REST API",
    r"\brest apis\b": "REST APIs",
    r"\bgraph ql\b": "GraphQL",
    r"\bci cd\b": "CI/CD",
}


def ensure_audio_dir() -> str:
    """Ensures that the directory for temporary audio files exists."""
    os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)
    return AUDIO_CACHE_DIR


def preprocess_text_for_speech(text: str) -> str:
    """
    Expands uppercase acronyms (e.g. EDA -> E-D-A, SQL -> S-Q-L) so gTTS speaks
    each letter distinctly instead of mispronouncing them as phonetic words.
    """
    if not text:
        return ""

    processed = text
    # Replace registered technical abbreviations
    for acronym, spoken_form in COMMON_ACRONYMS.items():
        processed = re.sub(r"\b" + re.escape(acronym) + r"\b", spoken_form, processed)

    return processed


def clean_transcribed_text(text: str) -> str:
    """
    Cleans raw voice transcription text, replacing common phonetic homophones
    with correct technical terminology (e.g., 'mike' -> 'mic', 'sequel' -> 'SQL').
    """
    if not text:
        return ""

    cleaned = text.strip()
    for pattern, replacement in TECHNICAL_HOMOPHONES.items():
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

    # Capitalize first character
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]

    return cleaned


def time_stretch_audio(signal: np.ndarray, speed: float = 1.28, sample_rate: int = 44100) -> np.ndarray:
    """
    Time-scale modification using WSOLA (Waveform Similarity Overlap-Add)
    to accelerate audio playback to natural conversational interview speed (~170-185 WPM)
    without altering pitch.

    Args:
        signal: 1D or 2D numpy array of audio PCM samples (int16).
        speed: Speedup factor (e.g. 1.28 for natural conversational pace).
        sample_rate: Audio sampling frequency in Hz (typically 44100).

    Returns:
        np.ndarray: Accelerated int16 audio array with original pitch preserved.
    """
    if abs(speed - 1.0) < 0.02 or len(signal) == 0:
        return signal

    is_stereo = (signal.ndim == 2 and signal.shape[1] == 2)
    mono_ref = signal.mean(axis=1).astype(np.float32) if is_stereo else signal.astype(np.float32)

    # Frame window ~30ms
    win_len = int(sample_rate * 0.030)
    if win_len % 2 != 0:
        win_len += 1
    hop_syn = win_len // 2
    hop_ana = int(hop_syn * speed)
    delta_max = win_len // 4

    window = np.hanning(win_len).astype(np.float32)
    window_expanded = window[:, np.newaxis] if is_stereo else window

    total_samples = len(signal)
    num_frames = int((total_samples - win_len - delta_max) / hop_ana)
    if num_frames <= 0:
        return signal

    out_len = int(total_samples / speed) + win_len * 2
    output = np.zeros((out_len, signal.shape[1]) if is_stereo else (out_len,), dtype=np.float32)
    weight = np.zeros(out_len, dtype=np.float32)

    # Place initial windowed frame
    output[0:win_len] += signal[0:win_len] * window_expanded
    weight[0:win_len] += window
    prev_ana = 0
    pos_syn = 0

    for _ in range(1, num_frames):
        pos_syn += hop_syn
        target_ana = prev_ana + hop_ana

        search_start = max(0, target_ana - delta_max)
        search_end = min(total_samples - win_len, target_ana + delta_max)
        if search_start >= search_end:
            break

        ref_seg = mono_ref[prev_ana + hop_syn : prev_ana + hop_syn + delta_max]
        ref_len = len(ref_seg)
        num_cands = search_end - search_start
        if num_cands <= 0 or ref_len == 0:
            break

        # Fast vectorized similarity comparison across candidate shifts
        cands = np.lib.stride_tricks.as_strided(
            mono_ref[search_start:],
            shape=(num_cands, ref_len),
            strides=(mono_ref.strides[0], mono_ref.strides[0])
        )
        diffs = np.sum(np.abs(cands - ref_seg), axis=1)
        best_cand_idx = int(np.argmin(diffs))
        actual_ana = search_start + best_cand_idx

        if actual_ana + win_len > total_samples or pos_syn + win_len > out_len:
            break

        output[pos_syn : pos_syn + win_len] += signal[actual_ana : actual_ana + win_len] * window_expanded
        weight[pos_syn : pos_syn + win_len] += window
        prev_ana = actual_ana

    valid_weight = weight > 1e-4
    if is_stereo:
        output[valid_weight] /= weight[valid_weight, np.newaxis]
    else:
        output[valid_weight] /= weight[valid_weight]

    last_valid = np.max(np.where(valid_weight)[0]) if np.any(valid_weight) else 0
    return np.clip(output[:last_valid + 1], -32768, 32767).astype(np.int16)


def cleanup_audio_cache() -> None:
    """
    Purges any leftover temporary audio files in the audio directory.
    Ensures zero disk accumulation.
    """
    if os.path.exists(AUDIO_CACHE_DIR):
        for f in os.listdir(AUDIO_CACHE_DIR):
            file_path = os.path.join(AUDIO_CACHE_DIR, f)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass


def speak_text(text: str, filename: Optional[str] = None, speed: float = 1.28) -> Tuple[bool, str]:
    """
    Converts text to speech using gTTS, time-stretches the audio to natural conversational
    interview pace (~170-185 WPM) without altering pitch, and plays it locally via pygame.mixer.

    By default, operates completely in-memory (zero disk files created and zero memory leaks).
    If an explicit filename is provided (e.g. for unit testing), saves to that path.

    Args:
        text: The interview question text to speak aloud.
        filename: Optional destination audio filename (.wav or .mp3). Defaults to None (in-memory).
        speed: Speech pacing multiplier (default: 1.28x for natural conversational human pacing).

    Returns:
        (success: bool, audio_filepath: str)
    """
    # Preprocess text so acronyms like EDA are pronounced cleanly as E-D-A
    spoken_text = preprocess_text_for_speech(text)

    try:
        # Step 1: Synthesize base speech using gTTS into an in-memory buffer (zero disk writes)
        mp3_buf = io.BytesIO()
        tts = gTTS(text=spoken_text, lang="en", tld="co.uk", slow=False)
        tts.write_to_fp(mp3_buf)
        mp3_buf.seek(0)

        # Step 2: Ensure pygame mixer is initialized
        if not pygame.mixer.get_init():
            pygame.mixer.pre_init(44100, -16, 2, 2048)
            pygame.mixer.init()

        # Step 3: Load raw audio into numpy array for speed adjustment
        raw_sound = pygame.mixer.Sound(mp3_buf)
        raw_arr = pygame.sndarray.array(raw_sound)

        # Step 4: Apply time-stretching if speed != 1.0 (defaults to 1.28x for conversational speed)
        if abs(speed - 1.0) > 0.02 and len(raw_arr) > 0:
            sped_arr = time_stretch_audio(raw_arr, speed=speed, sample_rate=44100)
        else:
            sped_arr = raw_arr

        # Step 5: Play audio locally through host speakers via pygame.mixer Sound
        try:
            pygame.mixer.stop()
            sound = pygame.sndarray.make_sound(sped_arr)
            sound.set_volume(1.0)
            sound.play()
        except Exception as audio_device_err:
            print(f"[Notice] Pygame audio playback skipped (headless or no soundcard): {audio_device_err}")

        # Step 6: If an explicit filename was requested (e.g. automated test), write it out
        final_filepath = ""
        if filename:
            ensure_audio_dir()
            base_name = os.path.basename(filename)
            filepath = os.path.abspath(os.path.join(AUDIO_CACHE_DIR, base_name))
            base_no_ext, ext = os.path.splitext(filepath)
            wav_filepath = base_no_ext + ".wav"

            wavfile.write(wav_filepath, 44100, sped_arr)
            final_filepath = wav_filepath

            if ext.lower() == ".mp3":
                ffmpeg_exe = shutil.which("ffmpeg")
                if ffmpeg_exe:
                    try:
                        subprocess.run(
                            [ffmpeg_exe, "-y", "-i", wav_filepath, "-b:a", "192k", filepath],
                            check=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                        final_filepath = filepath
                    except Exception:
                        final_filepath = wav_filepath
                else:
                    final_filepath = wav_filepath

        return True, final_filepath

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
    Transcribes raw audio bytes (from Streamlit's st.audio_input widget)
    using SpeechRecognition's Google Web Speech backend with technical post-processing.
    """
    if not audio_bytes:
        return False, "No audio recorded."

    try:
        sample_rate, data = wavfile.read(io.BytesIO(audio_bytes))

        # Convert stereo to mono
        if len(data.shape) > 1:
            data = data.mean(axis=1)

        # Convert to 16-bit signed PCM
        if data.dtype in (np.float32, np.float64):
            data = np.clip(data * 32767.0, -32768, 32767).astype(np.int16)
        elif data.dtype != np.int16:
            data = data.astype(np.int16)

        pcm_buffer = io.BytesIO()
        with wave.open(pcm_buffer, "wb") as wav_out:
            wav_out.setnchannels(1)
            wav_out.setsampwidth(2)
            wav_out.setframerate(sample_rate)
            wav_out.writeframes(data.tobytes())
        pcm_buffer.seek(0)

        recognizer = sr.Recognizer()
        with sr.AudioFile(pcm_buffer) as source:
            audio_data = recognizer.record(source)

        raw_text = recognizer.recognize_google(audio_data, language="en-US")
        cleaned_text = clean_transcribed_text(raw_text)
        return True, cleaned_text

    except sr.UnknownValueError:
        return False, "Could not understand the audio. Please speak clearly into your microphone."
    except sr.RequestError as req_err:
        return False, f"Google Speech Recognition service unreachable: {req_err}"
    except Exception as e:
        return False, f"Audio processing error: {str(e)}"


def record_and_transcribe(timeout: int = 10, phrase_time_limit: int = 35) -> Tuple[bool, str]:
    """
    Listens to the user's hardware microphone and transcribes spoken words
    using Google Web Speech API with technical post-processing.
    """
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 1.0

    best_idx = get_best_microphone_index()

    try:
        mic_kwargs = {"device_index": best_idx} if best_idx is not None else {}
        with sr.Microphone(**mic_kwargs) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.6)
            audio_data = recognizer.listen(
                source,
                timeout=timeout,
                phrase_time_limit=phrase_time_limit
            )

        raw_text = recognizer.recognize_google(audio_data, language="en-US")
        cleaned_text = clean_transcribed_text(raw_text)
        return True, cleaned_text

    except sr.WaitTimeoutError:
        return False, "No speech detected within the time limit. Please speak directly after clicking."
    except sr.UnknownValueError:
        return False, "Could not understand the audio. Please speak clearly into your microphone."
    except sr.RequestError as req_err:
        return False, f"Speech recognition service error (check internet connectivity): {req_err}"
    except Exception as general_err:
        return False, f"Microphone error: {str(general_err)}. You can also type your answer."
