"""Dataset-specific loaders. Each adapter yields `AudioRecord`s in the common format.

Status (keep this table in sync with docs/dataset_comparison.md):

| adapter             | source                                   | tested?                                  |
|---------------------|------------------------------------------|------------------------------------------|
| kaldi_dir           | lecturer's cluster dirs (wav.scp + text) | fixture tests only (no cluster access)   |
| hf_voxpopuli        | facebook/voxpopuli on Hugging Face       | NOT run (needs `datasets`, blocked here) |
| hf_mls_en           | parler-tts/mls_eng (mirror of OpenSLR 94)| NOT run (needs `datasets`, blocked here) |
| common_voice_local  | extracted Mozilla Data Collective archive| NOT run (no data downloaded)             |
| fixture             | synthetic tones for technical tests      | yes (tests/)                             |

Streaming adapters decode audio in memory and the selection step caches only the
*selected* recordings as 16 kHz WAV files under data/audio_cache (git-ignored).
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

from .audio import decode_bytes, duration_from_file
from .config import DatasetConfig, SelectionConfig, resolve
from .records import AudioRecord


# --------------------------------------------------------------------------- kaldi
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


# ---------------------------------------------------------------- hugging face
def _hf_stream(repo: str, config_name: str | None, cfg: DatasetConfig):
    # Imported lazily: `datasets` needs pyarrow.compute, which Smart App Control
    # blocks on the project laptop. Everything else works without it.
    from datasets import Audio, load_dataset

    ds = load_dataset(
        repo,
        config_name,
        split=cfg.split,
        streaming=True,
        revision=cfg.revision,
    )
    return ds.cast_column("audio", Audio(decode=False))


def iter_hf_voxpopuli(cfg: DatasetConfig, sel: SelectionConfig) -> Iterator[AudioRecord]:
    """facebook/voxpopuli, config 'en'. Fields: audio_id, raw_text, normalized_text,
    speaker_id, gender, is_gold_transcript, accent.

    option text_field: 'raw_text' (default; cased + punctuated, better for later NER)
    or 'normalized_text' (lowercase, closer to the lecturer's cluster labels).
    """
    text_field = cfg.options.get("text_field", "raw_text")
    ds = _hf_stream("facebook/voxpopuli", cfg.options.get("config_name", "en"), cfg)
    for row in ds:
        audio, dur = decode_bytes(row["audio"]["bytes"])
        yield AudioRecord(
            sample_id=row["audio_id"],
            audio_ref=f"hf://facebook/voxpopuli@{cfg.revision}:{cfg.split}:{row['audio_id']}",
            reference_text=row[text_field],
            duration_seconds=dur,
            speaker_id=row.get("speaker_id"),
            source_split=cfg.split,
            dataset=cfg.name,
            dataset_revision=cfg.revision,
            provenance=f"hf:facebook/voxpopuli/en/{cfg.split};text_field={text_field}",
            audio_array=audio,
            sampling_rate=16_000,
        )


def iter_hf_mls_en(cfg: DatasetConfig, sel: SelectionConfig) -> Iterator[AudioRecord]:
    """English MLS. facebook/multilingual_librispeech has NO English config, so we use
    parler-tts/mls_eng, a Hugging Face re-upload of OpenSLR 94 English (community
    mirror, not the original publisher). Fields: audio, original_path, begin_time,
    end_time, transcript (lowercase, no punctuation), audio_duration, speaker_id, book_id.
    """
    ds = _hf_stream("parler-tts/mls_eng", None, cfg)
    for row in ds:
        audio, dur = decode_bytes(row["audio"]["bytes"])
        sid = f"{row['speaker_id']}_{row['book_id']}_{row['begin_time']:.2f}"
        yield AudioRecord(
            sample_id=sid,
            audio_ref=f"hf://parler-tts/mls_eng@{cfg.revision}:{cfg.split}:{sid}",
            reference_text=row["transcript"],
            duration_seconds=dur,
            speaker_id=str(row.get("speaker_id")),
            source_split=cfg.split,
            dataset=cfg.name,
            dataset_revision=cfg.revision,
            provenance=f"hf:parler-tts/mls_eng/{cfg.split};book={row.get('book_id')}",
            audio_array=audio,
            sampling_rate=16_000,
        )


# ------------------------------------------------------------- common voice
def iter_common_voice_local(cfg: DatasetConfig, sel: SelectionConfig) -> Iterator[AudioRecord]:
    """Extracted Common Voice release downloaded manually from Mozilla Data Collective
    (account + terms acceptance required; full English scripted set ~88 GB - do NOT
    download it just for this). Expects <path>/<split>.tsv and <path>/clips/.
    Uses clip_durations.tsv (milliseconds) when present. UNTESTED.
    """
    root = resolve(cfg.path)
    durations: dict[str, float] = {}
    dur_file = root / "clip_durations.tsv"
    if dur_file.exists():
        with open(dur_file, encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                durations[row["clip"]] = float(row["duration[ms]"]) / 1000.0
    with open(root / f"{cfg.split}.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            clip = root / "clips" / row["path"]
            yield AudioRecord(
                sample_id=Path(row["path"]).stem,
                audio_ref=str(clip),
                reference_text=row["sentence"],
                duration_seconds=durations.get(row["path"]),  # None -> measured at selection
                speaker_id=row.get("client_id"),
                source_split=cfg.split,
                dataset=cfg.name,
                dataset_revision=cfg.revision,
                provenance=f"common_voice_local:{cfg.path}",
            )


# ------------------------------------------------------------------ fixture
def iter_fixture(cfg: DatasetConfig, sel: SelectionConfig) -> Iterator[AudioRecord]:
    """Synthetic FIXTURE data (tones, fake transcripts) for technical tests only."""
    yield from iter_kaldi_dir(cfg, sel)


ADAPTERS = {
    "kaldi_dir": iter_kaldi_dir,
    "hf_voxpopuli": iter_hf_voxpopuli,
    "hf_mls_en": iter_hf_mls_en,
    "common_voice_local": iter_common_voice_local,
    "fixture": iter_fixture,
}


def iter_records(cfg: DatasetConfig, sel: SelectionConfig) -> Iterator[AudioRecord]:
    try:
        fn = ADAPTERS[cfg.adapter]
    except KeyError as exc:
        raise ValueError(f"Unknown adapter {cfg.adapter!r}; choose from {sorted(ADAPTERS)}") from exc
    return fn(cfg, sel)
