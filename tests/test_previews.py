from ner_asr.deliverable import DELIVERABLE_COLUMNS
from ner_asr.previews import example_deliverable_row, granite_baseline_example
from ner_asr.transcribe import Granite4ASRTranscriber, TranscriptionResult


def test_dataset_preview_matches_the_export_schema():
    row = example_deliverable_row(language="de", source_split="test")
    assert list(row) == DELIVERABLE_COLUMNS
    assert row["audio_ref"].startswith("hf://disco-eth/EuroSpeech@")
    assert ":test:" in row["audio_ref"]
    assert "audio" not in row
    assert row["language"] == "de" and row["source_split"] == "test"


def test_granite_preview_matches_the_no_context_baseline_contract():
    preview = granite_baseline_example(model_name="ibm-granite/granite-4.0-1b-speech")
    assert preview["entity_or_keyword_context"] is None
    assert preview["granite_user_prompt"] == Granite4ASRTranscriber.BASELINE_USER_PROMPT
    assert preview["chat_messages"] == [{"role": "user", "content": Granite4ASRTranscriber.BASELINE_USER_PROMPT}]
    assert "prompt=None" in preview["python_call"]
    assert preview["result_fields"] == list(TranscriptionResult.__dataclass_fields__)
