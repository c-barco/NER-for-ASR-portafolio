"""Small, local checks for the Step 2 parquet exporter; no Hub access or model inference."""

from pathlib import Path
import io

import numpy as np
import pytest
import soundfile as sf

from ner_asr import adapters
from ner_asr.config import DatasetConfig, SelectionConfig
from ner_asr.deliverable import (
    export_step2_eurospeech,
    export_step2_eurospeech_bilingual,
    reload_and_validate_parquet,
    sample_audio_reference,
)
from ner_asr.records import AudioRecord


def _records():
    values = []
    # Two duration ranges exercise the mixed-length safeguard without creating
    # a heavyweight 1,000-row test fixture. The production default still
    # asserts 1,000 rows.
    for index, duration in enumerate([2.2, 2.4, 2.6, 6.2, 6.4, 6.6]):
        values.append(
            AudioRecord(
                sample_id=f"sample-{index}",
                audio_ref=f"memory-{index}",
                reference_text="Alice Smith addressed Parliament.",
                duration_seconds=duration,
                speaker_id=None,
                source_split="test",
                dataset="eurospeech_en",
                audio_array=np.zeros(round(duration * 16_000), dtype=np.float32),
                sampling_rate=16_000,
            )
        )
    return values


def _records_for_language(language: str):
    records = _records()
    for record in records:
        record.sample_id = f"{language}-{record.sample_id}"
        record.dataset = f"eurospeech_{language}"
    return records


def test_eurospeech_adapter_maps_the_common_record_contract(monkeypatch):
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(32_000, dtype=np.float32), 16_000, format="WAV", subtype="PCM_16")
    row = {
        "audio": {"bytes": buffer.getvalue(), "path": None},
        "key": "uk_123_0_2000",
        "human_transcript": "Alice Smith spoke.",
        "language": "en",
        "video_id": "123",
    }
    monkeypatch.setattr(adapters, "_hf_stream", lambda *args: [row])
    cfg = DatasetConfig(
        adapter="hf_eurospeech",
        name="eurospeech_en",
        language="en",
        split="test",
        revision="fixed-revision",
        options={"config_name": "uk", "text_field": "human_transcript"},
    )
    records = list(adapters.iter_hf_eurospeech(cfg, SelectionConfig()))
    assert len(records) == 1
    record = records[0]
    assert record.sample_id == "uk_123_0_2000"
    assert record.reference_text == "Alice Smith spoke."
    assert record.speaker_id is None and record.source_split == "test"
    assert record.sampling_rate == 16_000 and record.audio_array.shape == (32_000,)


def test_deliverable_rejects_a_candidate_window_smaller_than_its_required_rows(tmp_path):
    with pytest.raises(ValueError, match="candidate_limit"):
        export_step2_eurospeech(
            _records(),
            language="en",
            output_path=tmp_path / "out.parquet",
            seed=42,
            candidate_limit=2,
            target_rows=3,
            minimum_rows=3,
        )


def test_small_parquet_roundtrip_when_datasets_is_available(tmp_path):
    try:
        import datasets  # noqa: F401
    except Exception as exc:  # local Smart App Control can block pyarrow.compute
        pytest.skip(f"datasets is unavailable in this environment: {exc}")

    output = Path(tmp_path) / "step2_eurospeech_en.parquet"
    artifact, report = export_step2_eurospeech(
        _records(),
        language="en",
        output_path=output,
        seed=42,
        candidate_limit=6,
        target_rows=6,
        minimum_rows=3,
    )
    assert artifact == output and report.rows == 6
    reloaded = reload_and_validate_parquet(output, minimum_rows=3)
    assert reloaded.column_names == [
        "sample_id",
        "audio_ref",
        "text",
        "duration_seconds",
        "source_split",
        "language",
    ]
    assert reloaded[0]["audio_ref"].startswith("memory-")
    assert "audio" not in reloaded.column_names
    assert sample_audio_reference(output).startswith("memory-")


def test_bilingual_parquet_roundtrip_is_balanced_when_datasets_is_available(tmp_path):
    try:
        import datasets  # noqa: F401
    except Exception as exc:  # local Smart App Control can block pyarrow.compute
        pytest.skip(f"datasets is unavailable in this environment: {exc}")

    output = Path(tmp_path) / "step2_eurospeech_en_de.parquet"
    artifact, report = export_step2_eurospeech_bilingual(
        {"en": _records_for_language("en"), "de": _records_for_language("de")},
        output_path=output,
        seed=42,
        candidate_limit_per_language=6,
        rows_per_language=3,
        minimum_rows_per_language=3,
    )
    assert artifact == output
    assert report.rows == 6 and report.rows_by_language == {"en": 3, "de": 3}
    reloaded = reload_and_validate_parquet(
        output,
        minimum_rows=6,
        expected_rows=6,
        expected_language_counts={"en": 3, "de": 3},
    )
    assert set(reloaded["language"]) == {"en", "de"}
    assert len(set(reloaded["sample_id"])) == 6
    assert all(reference.startswith("memory-") for reference in reloaded["audio_ref"])


def test_overwrite_removes_a_stale_temporary_parquet_when_datasets_is_available(tmp_path):
    try:
        import datasets  # noqa: F401
    except Exception as exc:  # local Smart App Control can block pyarrow.compute
        pytest.skip(f"datasets is unavailable in this environment: {exc}")

    output = Path(tmp_path) / "step2_eurospeech_en.parquet"
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    temporary.write_bytes(b"stale failed export")
    artifact, _ = export_step2_eurospeech(
        _records(),
        language="en",
        output_path=output,
        seed=42,
        candidate_limit=6,
        target_rows=6,
        minimum_rows=3,
        overwrite=True,
    )
    assert artifact == output and not temporary.exists()
