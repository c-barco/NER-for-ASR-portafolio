"""Reproducible, duration-based selection and fixed manifests.

Rules enforced here:
* Records come from ONE official split; train and test are never mixed to reach
  a duration target.
* Selection is deterministic: candidates are sorted by sample_id, shuffled with
  `random.Random(seed)`, then taken until the cumulative duration reaches the target.
* If the split does not contain enough eligible audio, selection fails loudly
  (unless `allow_insufficient` is set, in which case the shortfall is recorded).
* The saved manifest is the source of truth for every later step.
"""

from __future__ import annotations

import csv
import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .audio import duration_from_file, write_wav
from .records import MANIFEST_COLUMNS, AudioRecord


class InsufficientAudioError(RuntimeError):
    pass


@dataclass
class SelectionReport:
    candidates_scanned: int
    eligible: int
    eligible_seconds: float
    selected: int
    selected_seconds: float
    target_seconds: float | None
    target_reached: bool
    excluded_too_short: int
    excluded_too_long: int
    excluded_empty_text: int
    duplicate_ids: int
    scan_limited: bool

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def select_by_duration(
    records: Iterable[AudioRecord],
    *,
    target_seconds: float | None,
    seed: int,
    min_duration_s: float = 0.0,
    max_duration_s: float = float("inf"),
    max_items: int | None = None,
    max_scan_items: int | None = None,
    allow_insufficient: bool = False,
) -> tuple[list[AudioRecord], SelectionReport]:
    eligible: list[AudioRecord] = []
    seen: set[str] = set()
    splits: set[str | None] = set()
    n = too_short = too_long = empty = dup = 0
    scan_limited = False
    for rec in records:
        if max_scan_items is not None and n >= max_scan_items:
            scan_limited = True
            break
        n += 1
        splits.add(rec.source_split)
        if rec.sample_id in seen:
            dup += 1
            continue
        seen.add(rec.sample_id)
        if rec.duration_seconds is None:
            rec.duration_seconds = duration_from_file(rec.audio_ref)
        if not (rec.reference_text or "").strip():
            empty += 1
            continue
        if rec.duration_seconds < min_duration_s:
            too_short += 1
            continue
        if rec.duration_seconds > max_duration_s:
            too_long += 1
            continue
        eligible.append(rec)

    if len(splits) > 1:
        raise ValueError(f"Records from several splits were mixed: {sorted(map(str, splits))}")

    eligible.sort(key=lambda r: r.sample_id)
    random.Random(seed).shuffle(eligible)

    selected: list[AudioRecord] = []
    total = 0.0
    for rec in eligible:
        if max_items is not None and len(selected) >= max_items:
            break
        if target_seconds is not None and total >= target_seconds:
            break
        selected.append(rec)
        total += rec.duration_seconds

    eligible_seconds = sum(r.duration_seconds for r in eligible)
    reached = target_seconds is None or total >= target_seconds or (
        max_items is not None and len(selected) >= max_items
    )
    report = SelectionReport(
        candidates_scanned=n,
        eligible=len(eligible),
        eligible_seconds=eligible_seconds,
        selected=len(selected),
        selected_seconds=total,
        target_seconds=target_seconds,
        target_reached=reached,
        excluded_too_short=too_short,
        excluded_too_long=too_long,
        excluded_empty_text=empty,
        duplicate_ids=dup,
        scan_limited=scan_limited,
    )
    if not reached and not allow_insufficient:
        raise InsufficientAudioError(
            f"Split has only {eligible_seconds / 3600:.2f} h eligible audio "
            f"({'scan limited, ' if scan_limited else ''}{len(eligible)} items) but the target is "
            f"{target_seconds / 3600:.2f} h. Do not top up from another split; clarify the requirement "
            "or set allow_insufficient: true to record the shortfall explicitly."
        )
    return selected, report


def cache_audio(records: list[AudioRecord], cache_dir: Path) -> None:
    """Write in-memory (streamed) audio to WAV so the manifest has a stable local path."""
    for rec in records:
        if rec.audio_array is None:
            continue
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in rec.sample_id)
        path = cache_dir / (rec.dataset or "dataset") / (rec.source_split or "split") / f"{safe}.wav"
        if not path.exists():
            write_wav(rec.audio_array, path)
        rec.provenance = f"{rec.provenance};origin={rec.audio_ref}"
        rec.audio_ref = str(path)
        rec.audio_array = None


def write_manifest(records: list[AudioRecord], path: Path, report: SelectionReport | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        for rec in records:
            w.writerow(rec.to_manifest_row())
    if report is not None:
        with open(path.with_suffix(".selection.json"), "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)


def read_manifest(path: Path) -> list[AudioRecord]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        r = {k: (v if v != "" else None) for k, v in r.items()}
        r["duration_seconds"] = float(r["duration_seconds"]) if r["duration_seconds"] else None
        r["reference_text"] = r["reference_text"] or ""
        out.append(AudioRecord(**r))
    return out


def check_disjoint(
    eval_records: list[AudioRecord], train_records: list[AudioRecord], *, by_speaker: bool = False
) -> dict:
    """For later fine-tuning: verify no recording (and optionally no speaker) overlaps."""
    eval_ids = {r.sample_id for r in eval_records}
    train_ids = {r.sample_id for r in train_records}
    eval_audio = {r.audio_ref for r in eval_records}
    train_audio = {r.audio_ref for r in train_records}
    result = {
        "overlapping_ids": sorted(eval_ids & train_ids),
        "overlapping_audio": sorted(eval_audio & train_audio),
    }
    if by_speaker:
        es = {r.speaker_id for r in eval_records if r.speaker_id}
        ts = {r.speaker_id for r in train_records if r.speaker_id}
        result["overlapping_speakers"] = sorted(es & ts)
    return result
