"""Common record format shared by every dataset adapter.

Every adapter yields `AudioRecord` objects. Only `sample_id`, `audio_ref` and
`reference_text` are mandatory; other metadata is optional because not every
corpus provides it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

MANIFEST_COLUMNS = [
    "sample_id",
    "audio_ref",
    "reference_text",
    "duration_seconds",
    "speaker_id",
    "source_split",
    "dataset",
    "dataset_revision",
    "provenance",
]


@dataclass
class AudioRecord:
    sample_id: str
    # Stable audio reference: a local file path, or "<hf-dataset>@<revision>:<split>:<id>"
    # for streamed items that have not been cached to disk yet.
    audio_ref: str
    reference_text: str
    duration_seconds: float | None = None
    speaker_id: str | None = None
    source_split: str | None = None
    dataset: str | None = None
    dataset_revision: str | None = None
    provenance: str | None = None
    # Decoded audio (16 kHz mono float32) kept in memory only; never written to manifests.
    audio_array: Any = field(default=None, repr=False, compare=False)
    sampling_rate: int | None = field(default=None, repr=False, compare=False)

    def to_manifest_row(self) -> dict:
        row = asdict(self)
        row.pop("audio_array", None)
        row.pop("sampling_rate", None)
        return {k: row.get(k) for k in MANIFEST_COLUMNS}
