"""Incremental, resumable baseline transcription.

* One JSON line per recording is appended (and flushed + fsynced) as soon as it is done.
* On restart, recordings whose latest line has status "ok" are skipped; failed
  recordings are retried unless `retry_failed=False`.
* Failures are recorded with their error message - never silently dropped.
* The baseline passes NO prompt and NEVER sees reference transcripts: only
  `audio_ref` from the manifest reaches the transcriber.
"""

from __future__ import annotations

import json
import os
import platform
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from .records import AudioRecord

# A transcribe function takes a list of audio paths and returns a list of
# TranscriptionResult-like objects (prediction, raw_output, n_generated_tokens,
# hit_token_limit, runtime_s). Tests pass a fake one.
TranscribeFn = Callable[[list[str]], list]


def load_predictions(path: Path) -> dict[str, dict]:
    """Latest line per sample_id wins."""
    latest: dict[str, dict] = {}
    if not path.exists():
        return latest
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a partially written last line after a crash
            latest[row["sample_id"]] = row
    return latest


def _append(path: Path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run_transcription(
    records: list[AudioRecord],
    transcribe: TranscribeFn,
    predictions_path: Path,
    *,
    run_info: dict,
    batch_size: int = 1,
    retry_failed: bool = True,
    progress: bool = True,
) -> dict:
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_predictions(predictions_path)
    todo = [
        r
        for r in records
        if r.sample_id not in done or (retry_failed and done[r.sample_id].get("status") != "ok")
    ]
    stats = {"total": len(records), "already_done": len(records) - len(todo), "ok": 0, "error": 0}

    iterator = range(0, len(todo), batch_size)
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, total=(len(todo) + batch_size - 1) // batch_size, desc="transcribing")
        except ImportError:
            pass

    run_start = time.perf_counter()
    for i in iterator:
        batch = todo[i : i + batch_size]
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            results = transcribe([r.audio_ref for r in batch])
            if len(results) != len(batch):
                raise RuntimeError(f"Transcriber returned {len(results)} results for {len(batch)} inputs")
            rows = [
                {
                    "sample_id": r.sample_id,
                    "status": "ok",
                    "error_message": None,
                    "prediction": res.prediction,
                    "raw_output": res.raw_output,
                    "n_generated_tokens": res.n_generated_tokens,
                    "hit_token_limit": res.hit_token_limit,
                    "runtime_s": res.runtime_s,
                    "timestamp": stamp,
                    **run_info,
                }
                for r, res in zip(batch, results)
            ]
        except Exception as exc:  # recorded, not swallowed
            rows = [
                {
                    "sample_id": r.sample_id,
                    "status": "error",
                    "error_message": f"{type(exc).__name__}: {exc}",
                    "prediction": None,
                    "raw_output": None,
                    "n_generated_tokens": None,
                    "hit_token_limit": None,
                    "runtime_s": None,
                    "timestamp": stamp,
                    **run_info,
                }
                for r in batch
            ]
        for row in rows:
            _append(predictions_path, row)
            stats[row["status"]] += 1
    stats["wall_time_s"] = time.perf_counter() - run_start
    return stats


def environment_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform()}
    from importlib.metadata import PackageNotFoundError, version

    for pkg in ("torch", "transformers", "accelerate", "jiwer", "soundfile", "soxr"):
        try:
            info[pkg] = version(pkg)
        except PackageNotFoundError:
            info[pkg] = None
    return info
