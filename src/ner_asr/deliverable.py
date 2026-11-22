"""Step 2 EuroSpeech parquet export.

The exam artifact is intentionally prepared outside the baseline notebook: this
module streams one official EuroSpeech split, validates the selected audio,
embeds normalized 16 kHz WAV bytes in a single parquet file, and verifies that
the saved artifact can be reloaded. It performs no model inference and no NER.
"""

from __future__ import annotations

import io
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import soundfile as sf

from .audio import TARGET_SR, load_audio, to_mono_16k
from .records import AudioRecord
from .selection import select_by_duration

DELIVERABLE_COLUMNS = [
    "sample_id",
    "audio",
    "text",
    "duration_seconds",
    "speaker_id",
    "source_split",
    "language",
]
DEFAULT_CANDIDATE_LIMIT = 1_200
DEFAULT_TARGET_ROWS = 1_100
MINIMUM_ROWS = 1_000
MIN_DURATION_S = 2.0
MAX_DURATION_S = 20.0
DURATION_BINS = (2.0, 5.0, 10.0, 15.0, 20.0)


class DeliverableValidationError(RuntimeError):
    """The requested Step 2 artifact does not meet its stated constraints."""


@dataclass
class DeliverableReport:
    output_path: str
    candidate_limit: int
    target_rows: int
    candidates_scanned: int
    eligible_before_audio_validation: int
    dropped_unreadable_during_wav_export: int
    rows: int
    audio_hours: float
    source_split: str
    language: str
    duration_seconds: dict[str, float]
    duration_bins: dict[str, int]
    entity_like_rows: int
    entity_like_share: float
    selection: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _PreparedRecord:
    record: AudioRecord
    wav_bytes: bytes
    duration_seconds: float


# This is deliberately not an NER system: EuroSpeech has no named-entity
# labels, and Steps 3--8 are out of scope. It is a transparent descriptive
# proxy for report-only use; German noun capitalization makes it especially
# conservative as an interpretation of real entity prevalence.
_ENTITY_LIKE_SEQUENCE = re.compile(
    r"\b[A-ZÄÖÜ][\wÀ-ÖØ-öø-ÿ'’-]*\s+[A-ZÄÖÜ][\wÀ-ÖØ-öø-ÿ'’-]*\b",
    flags=re.UNICODE,
)


def _duration_bin_label(lower: float, upper: float) -> str:
    return f"{lower:g}-{upper:g}s"


def _duration_bin_index(duration_seconds: float) -> int:
    for index, upper in enumerate(DURATION_BINS[1:]):
        if duration_seconds <= upper:
            return index
    raise DeliverableValidationError(f"Duration outside deliverable bins: {duration_seconds:.3f}s")


def _validate_wav_bytes(payload: bytes) -> float:
    """Return a WAV duration after checking embedded bytes are readable at 16 kHz."""
    if not payload:
        raise DeliverableValidationError("WAV payload is empty")
    info = sf.info(io.BytesIO(payload))
    if info.samplerate != TARGET_SR:
        raise DeliverableValidationError(f"Expected {TARGET_SR} Hz WAV, found {info.samplerate} Hz")
    if info.frames <= 0:
        raise DeliverableValidationError("WAV payload has no frames")
    decoded, sample_rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=False)
    if sample_rate != TARGET_SR or np.asarray(decoded).size == 0:
        raise DeliverableValidationError("WAV payload could not be decoded as non-empty 16 kHz audio")
    return info.frames / info.samplerate


def _to_embedded_wav(record: AudioRecord) -> _PreparedRecord:
    if record.audio_array is None:
        waveform = load_audio(record.audio_ref)
    else:
        waveform = to_mono_16k(np.asarray(record.audio_array), record.sampling_rate or TARGET_SR)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim != 1 or waveform.size == 0:
        raise DeliverableValidationError("Audio is empty after mono/16 kHz conversion")
    buffer = io.BytesIO()
    sf.write(buffer, waveform, TARGET_SR, format="WAV", subtype="PCM_16")
    payload = buffer.getvalue()
    duration = _validate_wav_bytes(payload)
    return _PreparedRecord(record=record, wav_bytes=payload, duration_seconds=duration)


def _mixed_duration_sample(records: list[_PreparedRecord], target_rows: int) -> list[_PreparedRecord]:
    """Keep every available duration stratum represented before filling by seed order."""
    if target_rows <= 0:
        raise ValueError("target_rows must be positive")
    bins: list[list[_PreparedRecord]] = [[] for _ in range(len(DURATION_BINS) - 1)]
    for prepared in records:
        bins[_duration_bin_index(prepared.duration_seconds)].append(prepared)
    nonempty = [group for group in bins if group]
    if len(nonempty) < 2:
        raise DeliverableValidationError(
            "Eligible candidates occupy fewer than two duration bins; cannot document a mixed length distribution."
        )

    wanted = min(target_rows, len(records))
    # At least 25 examples from every populated bin when possible, then retain
    # the common fixed-seed order established by select_by_duration.
    per_bin = min(25, max(1, wanted // len(nonempty)))
    selected: list[_PreparedRecord] = []
    selected_ids: set[str] = set()
    for group in nonempty:
        for prepared in group[:per_bin]:
            selected.append(prepared)
            selected_ids.add(prepared.record.sample_id)
    for prepared in records:
        if len(selected) >= wanted:
            break
        if prepared.record.sample_id not in selected_ids:
            selected.append(prepared)
            selected_ids.add(prepared.record.sample_id)
    return selected


def _entity_like(text: str) -> bool:
    return bool(_ENTITY_LIKE_SEQUENCE.search(text))


def _rows_to_dataset(rows: list[dict[str, Any]]):
    """Create an HF Dataset whose audio column is embedded 16 kHz WAV bytes."""
    from datasets import Audio, Dataset, Features, Value

    features = Features(
        {
            "sample_id": Value("string"),
            "audio": Audio(sampling_rate=TARGET_SR, decode=False),
            "text": Value("string"),
            "duration_seconds": Value("float32"),
            "speaker_id": Value("string"),
            "source_split": Value("string"),
            "language": Value("string"),
        }
    )
    return Dataset.from_list(rows, features=features)


def reload_and_validate_parquet(path: str | Path, *, minimum_rows: int = MINIMUM_ROWS):
    """Reload an exported file through ``datasets.load_dataset`` and validate one sample."""
    from datasets import Audio, load_dataset

    artifact = Path(path)
    dataset = load_dataset("parquet", data_files={"train": str(artifact)}, split="train")
    if dataset.column_names != DELIVERABLE_COLUMNS:
        raise DeliverableValidationError(
            f"Unexpected parquet columns: {dataset.column_names}; expected {DELIVERABLE_COLUMNS}"
        )
    if len(dataset) < minimum_rows:
        raise DeliverableValidationError(
            f"Parquet reload has {len(dataset)} rows, below the required minimum of {minimum_rows}."
        )
    dataset = dataset.cast_column("audio", Audio(sampling_rate=TARGET_SR, decode=False))
    sample_audio = dataset[0]["audio"]
    if not isinstance(sample_audio, dict) or not sample_audio.get("bytes"):
        raise DeliverableValidationError("Reloaded parquet does not contain embedded audio bytes")
    _validate_wav_bytes(sample_audio["bytes"])
    return dataset


def sample_audio_bytes(path: str | Path) -> bytes:
    """Return a reload-validated WAV payload for an optional notebook player."""
    dataset = reload_and_validate_parquet(path)
    return dataset[0]["audio"]["bytes"]


def export_step2_eurospeech(
    records,
    *,
    language: str,
    output_path: str | Path,
    seed: int,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    target_rows: int = DEFAULT_TARGET_ROWS,
    minimum_rows: int = MINIMUM_ROWS,
    overwrite: bool = False,
) -> tuple[Path, DeliverableReport]:
    """Build the one-file Step 2 artifact from one official EuroSpeech split.

    ``records`` must all originate from a single official split. The common
    selector sorts IDs and applies the supplied fixed seed before duration
    stratification. No train, validation, and test records can be mixed.
    """
    if candidate_limit < minimum_rows:
        raise ValueError("candidate_limit must be at least minimum_rows")
    if target_rows < minimum_rows:
        raise ValueError("target_rows must be at least minimum_rows")
    if target_rows > candidate_limit:
        raise ValueError("target_rows cannot exceed candidate_limit")

    artifact = Path(output_path)
    if artifact.exists() and not overwrite:
        raise FileExistsError(f"Deliverable already exists: {artifact}. Use overwrite=True to replace it.")
    artifact.parent.mkdir(parents=True, exist_ok=True)

    candidates, selection = select_by_duration(
        records,
        target_seconds=None,
        seed=seed,
        min_duration_s=MIN_DURATION_S,
        max_duration_s=MAX_DURATION_S,
        max_scan_items=candidate_limit,
        allow_insufficient=True,
    )
    prepared: list[_PreparedRecord] = []
    unreadable = 0
    for record in candidates:
        try:
            item = _to_embedded_wav(record)
        except (OSError, RuntimeError, TypeError, ValueError, DeliverableValidationError):
            unreadable += 1
            continue
        # Enforce the duration window on the normalized WAV, not only metadata.
        if MIN_DURATION_S <= item.duration_seconds <= MAX_DURATION_S:
            prepared.append(item)

    if len(prepared) < minimum_rows:
        raise DeliverableValidationError(
            f"Only {len(prepared)} valid 2-20 s rows remain after audio validation; "
            f"the Step 2 deliverable requires at least {minimum_rows}."
        )
    selected = _mixed_duration_sample(prepared, target_rows)
    if len(selected) < minimum_rows:
        raise DeliverableValidationError(
            f"Only {len(selected)} rows were selected; the Step 2 deliverable requires at least {minimum_rows}."
        )
    assert len(selected) >= minimum_rows, "Step 2 parquet must contain at least the requested minimum rows"

    source_splits = {item.record.source_split for item in selected}
    if len(source_splits) != 1:
        raise DeliverableValidationError(f"Deliverable would mix source splits: {sorted(source_splits)}")
    source_split = next(iter(source_splits)) or ""
    rows = [
        {
            "sample_id": item.record.sample_id,
            "audio": {"bytes": item.wav_bytes, "path": None},
            "text": item.record.reference_text,
            "duration_seconds": float(item.duration_seconds),
            "speaker_id": item.record.speaker_id,
            "source_split": source_split,
            "language": language,
        }
        for item in selected
    ]
    dataset = _rows_to_dataset(rows)
    temporary = artifact.with_name(f".{artifact.stem}.tmp{artifact.suffix}")
    if temporary.exists():
        raise FileExistsError(f"Temporary export path already exists: {temporary}")
    dataset.to_parquet(str(temporary))
    try:
        reload_and_validate_parquet(temporary, minimum_rows=minimum_rows)
        temporary.replace(artifact)
    except Exception:
        # Preserve the temporary file for inspection rather than hiding a failed
        # validation by deleting it.
        raise

    durations = [item.duration_seconds for item in selected]
    bin_counts = {
        _duration_bin_label(lower, upper): 0
        for lower, upper in zip(DURATION_BINS, DURATION_BINS[1:])
    }
    for duration in durations:
        label = _duration_bin_label(
            DURATION_BINS[_duration_bin_index(duration)], DURATION_BINS[_duration_bin_index(duration) + 1]
        )
        bin_counts[label] += 1
    entity_like_rows = sum(_entity_like(item.record.reference_text) for item in selected)
    report = DeliverableReport(
        output_path=str(artifact),
        candidate_limit=candidate_limit,
        target_rows=target_rows,
        candidates_scanned=selection.candidates_scanned,
        eligible_before_audio_validation=selection.eligible,
        dropped_unreadable_during_wav_export=unreadable,
        rows=len(selected),
        audio_hours=sum(durations) / 3600.0,
        source_split=source_split,
        language=language,
        duration_seconds={
            "min": min(durations),
            "median": median(durations),
            "mean": sum(durations) / len(durations),
            "max": max(durations),
        },
        duration_bins=bin_counts,
        entity_like_rows=entity_like_rows,
        entity_like_share=entity_like_rows / len(selected),
        selection=selection.to_dict(),
    )
    print(
        "Step 2 EuroSpeech parquet: "
        f"{report.rows} rows, {report.audio_hours:.2f} h, split={report.source_split}, language={report.language}"
    )
    print(
        "Duration seconds "
        f"(min/median/mean/max): {report.duration_seconds['min']:.2f} / "
        f"{report.duration_seconds['median']:.2f} / {report.duration_seconds['mean']:.2f} / "
        f"{report.duration_seconds['max']:.2f}; bins={report.duration_bins}"
    )
    print(
        "Entity-like share (capitalized multi-token heuristic; not NER or ground truth): "
        f"{report.entity_like_rows}/{report.rows} ({report.entity_like_share:.1%})"
    )
    return artifact, report

