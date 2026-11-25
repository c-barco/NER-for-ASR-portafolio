"""Config validation and an end-to-end select -> transcribe -> evaluate run on FIXTURE data."""

import json
from types import SimpleNamespace

import pytest
import yaml

from ner_asr import pipeline
from ner_asr.config import REPO_ROOT, load_config
from ner_asr.records import AudioRecord


@pytest.mark.parametrize(
    "name",
    ["eurospeech_en", "eurospeech_de", "mls_en", "common_voice_en", "cluster_kaldi", "pilot_fixture"],
)
def test_all_configs_load_in_pilot_mode(name):
    cfg = load_config(f"config/{name}.yaml")
    assert cfg.mode == "pilot" and cfg.context_mode == "none"
    assert cfg.model.name == "ibm-granite/granite-4.0-1b-speech"


def test_full_mode_requires_explicit_duration():
    with pytest.raises(ValueError, match="target_hours"):
        load_config("config/eurospeech_en.yaml", mode="full")


def test_bilingual_deliverable_quotas_are_explicit_and_match():
    en_cfg = load_config("config/eurospeech_en.yaml")
    de_cfg = load_config("config/eurospeech_de.yaml")
    assert en_cfg.deliverable == de_cfg.deliverable
    assert en_cfg.deliverable.candidate_limit_per_language == 1_200
    assert en_cfg.deliverable.rows_per_language == 1_000
    assert en_cfg.deliverable.total_rows == 2_000


def test_eurospeech_pilot_uses_the_250_recording_quota_without_a_duration_cap():
    for path in ("config/eurospeech_en.yaml", "config/eurospeech_de.yaml"):
        cfg = load_config(path)
        assert cfg.selection.pilot_max_items == 250
        assert cfg.target_seconds() is None
        assert cfg.max_items() == 250
        assert cfg.selection.max_scan_items == 1_200


def test_eurospeech_pilot_fails_loudly_when_it_cannot_select_250(monkeypatch, tmp_path):
    cfg = load_config("config/eurospeech_en.yaml")
    cfg.manifest_dir = str(tmp_path / "manifests")
    records = [
        AudioRecord(
            sample_id=f"short-{index}",
            audio_ref=f"memory-{index}",
            reference_text="A valid transcript.",
            duration_seconds=3.0,
            source_split="test",
        )
        for index in range(10)
    ]
    monkeypatch.setattr(pipeline, "iter_records", lambda *_: iter(records))
    with pytest.raises(pipeline.InsufficientAudioError, match="expected exactly 250"):
        pipeline.step_select(cfg)


class FakeGranite:
    def transcribe_batch(self, paths, prompt=None):
        assert prompt is None  # baseline: no context
        return [SimpleNamespace(prediction="fixture output", raw_output="fixture output",
                                n_generated_tokens=2, hit_token_limit=False, runtime_s=0.01) for _ in paths]


def test_end_to_end_fixture(kaldi_fixture, tmp_path):
    raw = yaml.safe_load((REPO_ROOT / "config/pilot_fixture.yaml").read_text(encoding="utf-8"))
    raw["dataset"]["path"] = str(kaldi_fixture)
    raw["output_dir"] = str(tmp_path / "out")
    raw["manifest_dir"] = str(tmp_path / "manifests")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(cfg_path)

    selected, report = pipeline.step_select(cfg)
    assert report.selected == 4  # fx_e (0.2 s) excluded by min_duration_s
    again, none = pipeline.step_select(cfg)  # manifest is reused, not re-drawn
    assert none is None and [r.sample_id for r in again] == [r.sample_id for r in selected]

    stats = pipeline.step_transcribe(cfg, transcriber=FakeGranite())
    assert stats["ok"] == 4
    summary = pipeline.step_evaluate(cfg)
    assert summary["n_success"] == 4 and summary["run_info"]["condition"] == "baseline_no_context"
    saved = json.loads((cfg.metrics_dir / "fixture_pilot_summary.json").read_text(encoding="utf-8"))
    assert saved["n_selected"] == 4
