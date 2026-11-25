"""Audio loading and caching (16 kHz mono float32).

Mirrors the lecturer's `load_audio()` (load -> resample to 16 kHz -> average
channels) but uses soundfile + soxr instead of torchaudio, because recent
torchaudio releases delegate decoding to torchcodec, which adds an FFmpeg
dependency that is awkward on Windows.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

TARGET_SR = 16_000


def to_mono_16k(data: np.ndarray, sr: int) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != TARGET_SR:
        data = soxr.resample(data, sr, TARGET_SR).astype(np.float32)
    return data


def load_audio(path: str | Path) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return to_mono_16k(data, sr)


def decode_bytes(raw: bytes) -> tuple[np.ndarray, float]:
    data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    audio = to_mono_16k(data, sr)
    return audio, len(audio) / TARGET_SR


def duration_from_file(path: str | Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate


def write_wav(audio: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, TARGET_SR, subtype="PCM_16")
    return path
