"""Safe, no-network examples for the compact Step 2 notebook.

These helpers intentionally show structure only. They do not open a dataset,
load Granite, or perform inference.
"""

from __future__ import annotations

from .deliverable import DELIVERABLE_COLUMNS
from .transcribe import Granite4ASRTranscriber, TranscriptionResult


def example_deliverable_row(*, language: str, source_split: str = "test") -> dict:
    """Return a JSON-displayable example of one exported Step 2 row.

    The audio value mirrors the ``datasets.Audio`` parquet structure, but uses
    text in place of binary WAV data so it can be shown safely in a notebook.
    """
    row = {
        "sample_id": "<EuroSpeech key>",
        "audio": {"bytes": "<embedded 16 kHz mono PCM WAV bytes>", "path": None},
        "text": "<human_transcript>",
        "duration_seconds": 12.5,
        "speaker_id": None,
        "source_split": source_split,
        "language": language,
    }
    assert list(row) == DELIVERABLE_COLUMNS
    return row


def granite_baseline_example(*, model_name: str) -> dict:
    """Return the no-download Granite request and result shape used by Step 2."""
    result_fields = list(TranscriptionResult.__dataclass_fields__)
    return {
        "model": model_name,
        "python_call": "transcriber.transcribe_batch([audio_path], prompt=None)",
        "audio_input": "<one mono 16 kHz waveform loaded from audio_path>",
        "granite_user_prompt": Granite4ASRTranscriber.BASELINE_USER_PROMPT,
        "chat_messages": [{"role": "user", "content": Granite4ASRTranscriber.BASELINE_USER_PROMPT}],
        "processor_call": "processor(chat_prompt, waveform, device=device, return_tensors='pt')",
        "entity_or_keyword_context": None,
        "result_fields": result_fields,
        "example_prediction": "<transcribed speech>",
    }
