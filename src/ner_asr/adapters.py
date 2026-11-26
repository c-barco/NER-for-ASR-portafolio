"""Load datasets as AudioRecord objects."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from .audio import decode_bytes, duration_from_file, load_audio
from .config import DatasetConfig, SelectionConfig, resolve
from .records import AudioRecord


def _read_kaldi_map(path: Path) -> dict[str, str]:
    """Read `<id><TAB or space><value>` lines (lecturer files use a TAB)."""
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if "\t" in line:
                key, value = line.split("\t", 1)
            else:
                key, _, value = line.partition(" ")
            out[key.strip()] = value.strip()
    return out


def iter_kaldi_dir(cfg: DatasetConfig, sel: SelectionConfig) -> Iterator[AudioRecord]:
    """Lecturer format, e.g. /training-1/asr/kaldi_data/mls/en/test on the cluster.

    Required files: wav.scp, text. Optional: utt2spk, utt2dur.
    Piped wav.scp entries ("sox ... |") and `segments` files are not supported.
    """
    root = resolve(cfg.path)
    wav_scp = _read_kaldi_map(root / "wav.scp")
    texts = _read_kaldi_map(root / "text")
    utt2spk = _read_kaldi_map(root / "utt2spk") if (root / "utt2spk").exists() else {}
    utt2dur = _read_kaldi_map(root / "utt2dur") if (root / "utt2dur").exists() else {}
    if (root / "segments").exists():
        raise NotImplementedError("Kaldi 'segments' files are not supported yet.")
    missing_text = sorted(set(wav_scp) - set(texts))
    if missing_text:
        raise ValueError(f"{len(missing_text)} utterances in wav.scp have no transcript, e.g. {missing_text[:3]}")
    for utt_id in sorted(wav_scp):
        wav_path = wav_scp[utt_id]
        if wav_path.endswith("|"):
            raise NotImplementedError(f"Piped wav.scp entry not supported: {utt_id}")
        p = Path(wav_path)
        if not p.is_absolute():
            p = root / p
        dur = float(utt2dur[utt_id]) if utt_id in utt2dur else duration_from_file(p)
        yield AudioRecord(
            sample_id=utt_id,
            audio_ref=str(p),
            reference_text=texts[utt_id],
            duration_seconds=dur,
            speaker_id=utt2spk.get(utt_id),
            source_split=cfg.split,
            dataset=cfg.name,
            dataset_revision=cfg.revision,
            provenance=f"kaldi_dir:{cfg.path}",
        )


def _hf_stream(repo: str, config_name: str | None, cfg: DatasetConfig):
    from datasets import Audio, load_dataset

    ds = load_dataset(
        repo,
        config_name,
        split=cfg.split,
        streaming=True,
        revision=cfg.revision,
    )
    return ds.cast_column("audio", Audio(decode=False))


def iter_hf_eurospeech(cfg: DatasetConfig, sel: SelectionConfig) -> Iterator[AudioRecord]:
    """Stream one EuroSpeech country configuration.

    EuroSpeech stores English under the ``uk`` configuration and German under
    ``germany``. Its ``key`` is the stable utterance ID and
    ``human_transcript`` is the reference. The corpus does not publish a
    speaker field; ``video_id`` identifies a parliamentary session rather than
    a speaker, so this adapter leaves ``speaker_id`` unset.

    Audio is decoded here to normalize it to 16 kHz mono and to discard source
    items that cannot be read. Empty transcripts remain visible to the common
    selection filter, which records their exclusion.
    """
    config_name = cfg.options.get("config_name")
    if not config_name:
        raise ValueError("hf_eurospeech requires dataset.options.config_name (for example 'uk' or 'germany').")
    text_field = cfg.options.get("text_field", "human_transcript")
    ds = _hf_stream("disco-eth/EuroSpeech", config_name, cfg)
    for row in ds:
        audio_value = row.get("audio") or {}
        raw_audio = audio_value.get("bytes") if isinstance(audio_value, dict) else None
        audio_path = audio_value.get("path") if isinstance(audio_value, dict) else None
        try:
            if raw_audio:
                audio, duration = decode_bytes(raw_audio)
            elif audio_path:
                audio = load_audio(audio_path)
                duration = len(audio) / 16_000
            else:
                raise ValueError("EuroSpeech row has no readable audio reference")
        except (OSError, RuntimeError, TypeError, ValueError):
            # The deliverable and baseline must not include unreadable audio.
            continue
        sample_id = str(row.get("key") or "").strip()
        if not sample_id:
            continue
        yield AudioRecord(
            sample_id=sample_id,
            audio_ref=f"hf://disco-eth/EuroSpeech@{cfg.revision}:{config_name}:{cfg.split}:{sample_id}",
            reference_text=str(row.get(text_field) or ""),
            duration_seconds=duration,
            speaker_id=None,
            source_split=cfg.split,
            dataset=cfg.name,
            dataset_revision=cfg.revision,
            provenance=(
                f"hf:disco-eth/EuroSpeech/{config_name}/{cfg.split};"
                f"text_field={text_field};source_language={row.get('language')}"
            ),
            audio_array=audio,
            sampling_rate=16_000,
        )


def iter_fixture(cfg: DatasetConfig, sel: SelectionConfig) -> Iterator[AudioRecord]:
    """Synthetic FIXTURE data (tones, fake transcripts) for technical tests only."""
    yield from iter_kaldi_dir(cfg, sel)


ADAPTERS = {
    "kaldi_dir": iter_kaldi_dir,
    "hf_eurospeech": iter_hf_eurospeech,
    "fixture": iter_fixture,
}


def iter_records(cfg: DatasetConfig, sel: SelectionConfig) -> Iterator[AudioRecord]:
    try:
        fn = ADAPTERS[cfg.adapter]
    except KeyError as exc:
        raise ValueError(f"Unknown adapter {cfg.adapter!r}; choose from {sorted(ADAPTERS)}") from exc
    return fn(cfg, sel)
