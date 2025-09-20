import numpy as np
import librosa
import torch
import noisereduce as nr
from pydub import AudioSegment

# --- Params ---
SR = 16000
MIN_SPEECH_DURATION = 0.3
# ---------------

print("Loading Silero VAD...")
model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False
)
(get_speech_timestamps, *_ ) = utils
print("Model loaded.")


def has_voicing(segment, sr):
    """Check harmonic/voiced structure (rejects claps, bangs)."""
    try:
        f0, voiced_flag, _ = librosa.pyin(segment, fmin=50, fmax=400, sr=sr)
        if voiced_flag is None or np.isnan(voiced_flag).all():
            return False
        return np.nanmean(voiced_flag.astype(float)) > 0.15
    except Exception:
        return False


def detect_speech(file_path: str) -> bool:
    """
    Detect if an audio file contains human speech.
    Returns True/False.
    """

    # Load with pydub (handles webm, wav, mp3…)
    audio = AudioSegment.from_file(file_path)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples /= np.iinfo(audio.array_type).max  # normalize

    if audio.channels == 2:
        samples = samples.reshape((-1, 2))
        samples = samples.mean(axis=1)  # convert to mono

    # Resample to SR
    wav = librosa.resample(samples, orig_sr=audio.frame_rate, target_sr=SR).astype(np.float32)

    # Optional denoise
    try:
        wav = nr.reduce_noise(y=wav, sr=SR)
    except:
        pass

    # Run Silero VAD
    speech_segments = get_speech_timestamps(wav, model, sampling_rate=SR)

    for seg in speech_segments:
        start, end = seg["start"], seg["end"]
        dur = (end - start) / SR
        if dur < MIN_SPEECH_DURATION:
            continue
        segment = wav[start:end]
        if has_voicing(segment, SR):
            return True

    return False


# --- Example usage ---
if __name__ == "__main__":
    path = "uploads/20250920_182226_785995.webm"  # replace with your file
    result = detect_speech(path)
    print("Speech detected:", result)