"""Test helpers. All audio here is a SYNTHETIC FIXTURE (sine tones), not speech."""

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

FIXTURE_UTTS = [
    # (utt_id, speaker, seconds, FAKE transcript)
    ("fx_a", "spk1", 1.0, "Mr. Kurz spoke in Strasbourg."),
    ("fx_b", "spk1", 2.0, "the anti-poverty strategy"),
    ("fx_c", "spk2", 3.0, "one hundred and twenty million"),
    ("fx_d", "spk2", 1.5, "European citizens"),
    ("fx_e", "spk3", 0.2, "too short"),
]


def make_kaldi_fixture(root: Path, sr: int = 22050) -> Path:
    """Write a Kaldi-style dir (wav.scp, text, utt2spk) with sine tones at 22.05 kHz,
    so resampling to 16 kHz is exercised."""
    wav_dir = root / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    scp, text, spk = [], [], []
    for utt, speaker, secs, transcript in FIXTURE_UTTS:
        t = np.arange(int(secs * sr)) / sr
        tone = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        p = wav_dir / f"{utt}.wav"
        sf.write(p, tone, sr)
        scp.append(f"{utt}\t{p}")
        text.append(f"{utt}\t{transcript}")
        spk.append(f"{utt} {speaker}")
    (root / "wav.scp").write_text("\n".join(scp) + "\n", encoding="utf-8")
    (root / "text").write_text("\n".join(text) + "\n", encoding="utf-8")
    (root / "utt2spk").write_text("\n".join(spk) + "\n", encoding="utf-8")
    return root


@pytest.fixture
def kaldi_fixture(tmp_path):
    return make_kaldi_fixture(tmp_path / "kaldi_test")
