"""Step 2 EuroSpeech parquet export.

The exam artifact is intentionally prepared outside the baseline notebook: this
module streams one official EuroSpeech split, validates selected source audio,
and writes lightweight audio references plus transcripts to one parquet file.
It performs no model inference and no NER.
"""

from __future__ import annotations

import io
import random
import re
from collections import Counter
from collections.abc import Iterable, Mapping
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
    "audio_ref",
    "text",
    "duration_seconds",
    "source_split",
    "language",
]
# The candidate window and quota apply independently to English and German.
# They deliberately leave a small quality-filtering margin while preserving an
# exact 2,000-row final artifact.
DEFAULT_CANDIDATE_LIMIT_PER_LANGUAGE = 1_200
DEFAULT_ROWS_PER_LANGUAGE = 1_000
# Backward-compatible names for the standalone, single-language helper.
DEFAULT_CANDIDATE_LIMIT = DEFAULT_CANDIDATE_LIMIT_PER_LANGUAGE
DEFAULT_TARGET_ROWS = DEFAULT_ROWS_PER_LANGUAGE
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
    dropped_unreadable_during_audio_validation: int
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
class BilingualDeliverableReport:
    """Audit report for the one bilingual Step 2 parquet artifact."""

    output_path: str
    candidate_limit_per_language: int
    rows_per_language: int
    rows: int
    rows_by_language: dict[str, int]
    audio_hours: float
    source_splits: dict[str, str]
    duration_seconds: dict[str, float]
    duration_bins: dict[str, int]
    entity_like_rows: int
    entity_like_share: float
    languages: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _PreparedRecord:
    record: AudioRecord
    duration_seconds: float


@dataclass
class _PreparedLanguage:
    language: str
    selected: list[_PreparedRecord]
    source_split: str
    candidates_scanned: int
    eligible_before_audio_validation: int
    dropped_unreadable_during_audio_validation: int
    selection: dict[str, Any]


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
    """Return a duration after checking normalized 16 kHz WAV bytes are readable."""
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


def _validate_source_audio(record: AudioRecord) -> _PreparedRecord:
    """Decode, normalize, and reopen source audio without retaining its bytes."""
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
    return _PreparedRecord(record=record, duration_seconds=duration)


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
    """Create a lightweight HF Dataset with a stable source-audio reference."""
    from datasets import Dataset, Features, Value

    features = Features(
        {
            "sample_id": Value("string"),
            "audio_ref": Value("string"),
            "text": Value("string"),
            "duration_seconds": Value("float32"),
            "speaker_id": Value("string"),
            "source_split": Value("string"),
            "language": Value("string"),
        }
    )
    return Dataset.from_list(rows, features=features)


def reload_and_validate_parquet(
    path: str | Path,
    *,
    minimum_rows: int = MINIMUM_ROWS,
    expected_rows: int | None = None,
    expected_language_counts: Mapping[str, int] | None = None,
):
    """Reload the lightweight parquet and validate its schema and references.

    Source audio is decoded and WAV-validated before writing. The parquet keeps
    only the source reference, so it is intentionally not an audio player.
    """
    from datasets import load_dataset

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
    if expected_rows is not None and len(dataset) != expected_rows:
        raise DeliverableValidationError(
            f"Parquet reload has {len(dataset)} rows; expected exactly {expected_rows}."
        )
    if expected_language_counts is not None:
        actual_counts = dict(Counter(dataset["language"]))
        expected_counts = dict(expected_language_counts)
        if actual_counts != expected_counts:
            raise DeliverableValidationError(
                f"Unexpected language counts: {actual_counts}; expected {expected_counts}."
            )
    sample_ids = dataset["sample_id"]
    if len(sample_ids) != len(set(sample_ids)):
        raise DeliverableValidationError("Reloaded parquet has duplicate sample_id values")
    if any(not str(text or "").strip() for text in dataset["text"]):
        raise DeliverableValidationError("Reloaded parquet contains an empty transcript")
    if any(not str(audio_ref or "").strip() for audio_ref in dataset["audio_ref"]):
        raise DeliverableValidationError("Reloaded parquet contains an empty audio reference")
    if any(not (MIN_DURATION_S <= float(value) <= MAX_DURATION_S) for value in dataset["duration_seconds"]):
        raise DeliverableValidationError("Reloaded parquet contains audio outside the 2-20 s window")
    return dataset


def sample_audio_reference(path: str | Path) -> str:
    """Return one reload-validated source reference from the lightweight artifact."""
    dataset = reload_and_validate_parquet(path)
    return dataset[0]["audio_ref"]


def _validate_export_request(*, candidate_limit: int, target_rows: int, minimum_rows: int) -> None:
    if minimum_rows <= 0:
        raise ValueError("minimum_rows must be positive")
    if target_rows < minimum_rows:
        raise ValueError("target_rows must be at least minimum_rows")
    if candidate_limit < target_rows:
        raise ValueError("candidate_limit must be at least target_rows")


def _prepare_language(
    records: Iterable[AudioRecord],
    *,
    language: str,
    seed: int,
    candidate_limit: int,
    target_rows: int,
    minimum_rows: int,
) -> _PreparedLanguage:
    """Select and WAV-validate one language without mixing official splits."""
    _validate_export_request(
        candidate_limit=candidate_limit,
        target_rows=target_rows,
        minimum_rows=minimum_rows,
    )
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
            item = _validate_source_audio(record)
        except (OSError, RuntimeError, TypeError, ValueError, DeliverableValidationError):
            unreadable += 1
            continue
        # Enforce the duration window on normalized WAVs, not only metadata.
        if MIN_DURATION_S <= item.duration_seconds <= MAX_DURATION_S:
            prepared.append(item)

    if len(prepared) < target_rows:
        raise DeliverableValidationError(
            f"{language}: only {len(prepared)} valid 2-20 s rows remain after audio validation; "
            f"exactly {target_rows} are required."
        )
    selected = _mixed_duration_sample(prepared, target_rows)
    if len(selected) != target_rows:
        raise DeliverableValidationError(
            f"{language}: selected {len(selected)} rows; expected exactly {target_rows}."
        )

    source_splits = {item.record.source_split or "" for item in selected}
    if len(source_splits) != 1:
        raise DeliverableValidationError(f"{language}: records from several source splits would be mixed: {sorted(source_splits)}")
    source_split = next(iter(source_splits))
    if not source_split:
        raise DeliverableValidationError(f"{language}: selected records do not declare a source split")
    if source_split.lower() in {"train", "training"}:
        raise DeliverableValidationError(f"{language}: the exam deliverable must use a held-out split, not {source_split!r}")
    return _PreparedLanguage(
        language=language,
        selected=selected,
        source_split=source_split,
        candidates_scanned=selection.candidates_scanned,
        eligible_before_audio_validation=selection.eligible,
        dropped_unreadable_during_audio_validation=unreadable,
        selection=selection.to_dict(),
    )


def _duration_statistics(selected: list[_PreparedRecord]) -> dict[str, Any]:
    if not selected:
        raise DeliverableValidationError("Cannot summarize an empty deliverable selection")
    durations = [item.duration_seconds for item in selected]
    bin_counts = {
        _duration_bin_label(lower, upper): 0
        for lower, upper in zip(DURATION_BINS, DURATION_BINS[1:])
    }
    for duration in durations:
        index = _duration_bin_index(duration)
        bin_counts[_duration_bin_label(DURATION_BINS[index], DURATION_BINS[index + 1])] += 1
    entity_like_rows = sum(_entity_like(item.record.reference_text) for item in selected)
    return {
        "rows": len(selected),
        "audio_hours": sum(durations) / 3600.0,
        "duration_seconds": {
            "min": min(durations),
            "median": median(durations),
            "mean": sum(durations) / len(durations),
            "max": max(durations),
        },
        "duration_bins": bin_counts,
        "entity_like_rows": entity_like_rows,
        "entity_like_share": entity_like_rows / len(selected),
    }


def _language_report(prepared: _PreparedLanguage) -> dict[str, Any]:
    return {
        "source_split": prepared.source_split,
        "candidates_scanned": prepared.candidates_scanned,
        "eligible_before_audio_validation": prepared.eligible_before_audio_validation,
        "dropped_unreadable_during_audio_validation": prepared.dropped_unreadable_during_audio_validation,
        "selection": prepared.selection,
        **_duration_statistics(prepared.selected),
    }


def _rows_from_prepared(
    prepared: _PreparedLanguage,
) -> list[dict[str, Any]]:
    rows = []
    for item in prepared.selected:
        source_id = item.record.sample_id
        rows.append(
            {
                "sample_id": source_id,
                "audio_ref": item.record.audio_ref,
                "text": item.record.reference_text,
                "duration_seconds": float(item.duration_seconds),
                "speaker_id": item.record.speaker_id,
                "source_split": prepared.source_split,
                "language": prepared.language,
            }
        )
    return rows


def _write_parquet(
    rows: list[dict[str, Any]],
    *,
    output_path: str | Path,
    minimum_rows: int,
    expected_language_counts: Mapping[str, int] | None,
    overwrite: bool,
) -> Path:
    artifact = Path(output_path)
    if artifact.exists() and not overwrite:
        raise FileExistsError(f"Deliverable already exists: {artifact}. Use overwrite=True to replace it.")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact.with_name(f".{artifact.stem}.tmp{artifact.suffix}")
    if temporary.exists():
        raise FileExistsError(f"Temporary export path already exists: {temporary}")
    _rows_to_dataset(rows).to_parquet(str(temporary))
    try:
        reload_and_validate_parquet(
            temporary,
            minimum_rows=minimum_rows,
            expected_rows=len(rows),
            expected_language_counts=expected_language_counts,
        )
        temporary.replace(artifact)
    except Exception:
        # Preserve a failed temporary artifact for direct inspection.
        raise
    return artifact


def _print_report(prefix: str, report: DeliverableReport | BilingualDeliverableReport) -> None:
    print(f"{prefix}: {report.rows} rows, {report.audio_hours:.2f} h")
    print(
        "Duration seconds (min/median/mean/max): "
        f"{report.duration_seconds['min']:.2f} / {report.duration_seconds['median']:.2f} / "
        f"{report.duration_seconds['mean']:.2f} / {report.duration_seconds['max']:.2f}; "
        f"bins={report.duration_bins}"
    )
    print(
        "Entity-like share (capitalized multi-token heuristic; not NER or ground truth): "
        f"{report.entity_like_rows}/{report.rows} ({report.entity_like_share:.1%})"
    )


def export_step2_eurospeech(
    records: Iterable[AudioRecord],
    *,
    language: str,
    output_path: str | Path,
    seed: int,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    target_rows: int = DEFAULT_TARGET_ROWS,
    minimum_rows: int = MINIMUM_ROWS,
    overwrite: bool = False,
) -> tuple[Path, DeliverableReport]:
    """Build a standalone one-language parquet (kept for reusable local checks)."""
    prepared = _prepare_language(
        records,
        language=language,
        seed=seed,
        candidate_limit=candidate_limit,
        target_rows=target_rows,
        minimum_rows=minimum_rows,
    )
    artifact = _write_parquet(
        _rows_from_prepared(prepared),
        output_path=output_path,
        minimum_rows=minimum_rows,
        expected_language_counts={language: target_rows},
        overwrite=overwrite,
    )
    stats = _duration_statistics(prepared.selected)
    report = DeliverableReport(
        output_path=str(artifact),
        candidate_limit=candidate_limit,
        target_rows=target_rows,
        candidates_scanned=prepared.candidates_scanned,
        eligible_before_audio_validation=prepared.eligible_before_audio_validation,
        dropped_unreadable_during_audio_validation=prepared.dropped_unreadable_during_audio_validation,
        rows=stats["rows"],
        audio_hours=stats["audio_hours"],
        source_split=prepared.source_split,
        language=language,
        duration_seconds=stats["duration_seconds"],
        duration_bins=stats["duration_bins"],
        entity_like_rows=stats["entity_like_rows"],
        entity_like_share=stats["entity_like_share"],
        selection=prepared.selection,
    )
    _print_report(
        f"Step 2 EuroSpeech parquet (split={report.source_split}, language={report.language})",
        report,
    )
    return artifact, report


def export_step2_eurospeech_bilingual(
    records_by_language: Mapping[str, Iterable[AudioRecord]],
    *,
    output_path: str | Path,
    seed: int,
    candidate_limit_per_language: int = DEFAULT_CANDIDATE_LIMIT_PER_LANGUAGE,
    rows_per_language: int = DEFAULT_ROWS_PER_LANGUAGE,
    minimum_rows_per_language: int = MINIMUM_ROWS,
    overwrite: bool = False,
) -> tuple[Path, BilingualDeliverableReport]:
    """Build the required single, balanced English/German Step 2 artifact.

    Each held-out language stream is selected, duration-stratified, and WAV
    validated independently. Only then are the exact quotas combined and
    deterministically shuffled into one parquet.
    """
    required_languages = {"en", "de"}
    if set(records_by_language) != required_languages:
        raise ValueError("records_by_language must contain exactly {'en', 'de'}")
    _validate_export_request(
        candidate_limit=candidate_limit_per_language,
        target_rows=rows_per_language,
        minimum_rows=minimum_rows_per_language,
    )
    prepared_by_language = {
        language: _prepare_language(
            records_by_language[language],
            language=language,
            seed=seed,
            candidate_limit=candidate_limit_per_language,
            target_rows=rows_per_language,
            minimum_rows=minimum_rows_per_language,
        )
        for language in ("en", "de")
    }
    rows = [
        row
        for language in ("en", "de")
        for row in _rows_from_prepared(prepared_by_language[language])
    ]
    if len(rows) != 2 * rows_per_language:
        raise DeliverableValidationError(
            f"Bilingual selection has {len(rows)} rows; expected exactly {2 * rows_per_language}."
        )
    if dict(Counter(row["language"] for row in rows)) != {"en": rows_per_language, "de": rows_per_language}:
        raise DeliverableValidationError("Bilingual selection is not balanced between English and German")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise DeliverableValidationError("Bilingual selection has duplicate source sample IDs")
    random.Random(seed).shuffle(rows)
    artifact = _write_parquet(
        rows,
        output_path=output_path,
        minimum_rows=2 * minimum_rows_per_language,
        expected_language_counts={"en": rows_per_language, "de": rows_per_language},
        overwrite=overwrite,
    )

    language_reports = {
        language: _language_report(prepared_by_language[language])
        for language in ("en", "de")
    }
    all_selected = [
        item
        for language in ("en", "de")
        for item in prepared_by_language[language].selected
    ]
    stats = _duration_statistics(all_selected)
    report = BilingualDeliverableReport(
        output_path=str(artifact),
        candidate_limit_per_language=candidate_limit_per_language,
        rows_per_language=rows_per_language,
        rows=stats["rows"],
        rows_by_language={language: len(prepared_by_language[language].selected) for language in ("en", "de")},
        audio_hours=stats["audio_hours"],
        source_splits={language: prepared_by_language[language].source_split for language in ("en", "de")},
        duration_seconds=stats["duration_seconds"],
        duration_bins=stats["duration_bins"],
        entity_like_rows=stats["entity_like_rows"],
        entity_like_share=stats["entity_like_share"],
        languages=language_reports,
    )
    _print_report(
        "Step 2 EuroSpeech bilingual parquet "
        f"(rows_by_language={report.rows_by_language}, source_splits={report.source_splits})",
        report,
    )
    return artifact, report

