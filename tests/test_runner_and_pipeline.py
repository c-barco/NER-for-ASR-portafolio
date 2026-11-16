"""Incremental saving, resume, failure recording and export - with a FAKE transcriber."""

import csv
import json
from types import SimpleNamespace

from ner_asr.adapters import iter_records
from ner_asr.config import DatasetConfig, SelectionConfig
from ner_asr.evaluate import RESULT_COLUMNS, build_results, export, summarize
from ner_asr.runner import load_predictions, run_transcription
from ner_asr.selection import read_manifest, select_by_duration, write_manifest


class FakeTranscriber:
    """FIXTURE transcriber: returns canned text and records every input it receives."""

    def __init__(self, fail_ids=(), crash_after=None):
        self.calls = []
        self.fail_ids = set(fail_ids)
        self.crash_after = crash_after

    def __call__(self, paths):
        if self.crash_after is not None and len(self.calls) >= self.crash_after:
            raise KeyboardInterrupt("simulated crash")
        self.calls.append(list(paths))
        if any(f in p for p in paths for f in self.fail_ids):
            raise RuntimeError("simulated decode failure")
        return [
            SimpleNamespace(prediction="Mr Kurz spoke", raw_output="language English<asr_text>Mr Kurz spoke",
                            n_generated_tokens=5, hit_token_limit=False, runtime_s=0.01)
            for _ in paths
        ]


def load_fixture(kaldi_fixture):
    cfg = DatasetConfig(adapter="kaldi_dir", name="fixture", split="test", path=str(kaldi_fixture))
    return list(iter_records(cfg, SelectionConfig()))


def selected(kaldi_fixture):
    recs, rep = select_by_duration(load_fixture(kaldi_fixture), target_seconds=None, seed=0, min_duration_s=0.5)
    return recs, rep


def test_kaldi_adapter_reads_lecturer_format(kaldi_fixture):
    recs = load_fixture(kaldi_fixture)
    assert len(recs) == 5
    r = {x.sample_id: x for x in recs}["fx_b"]
    assert r.reference_text == "the anti-poverty strategy"
    assert abs(r.duration_seconds - 2.0) < 0.01 and r.speaker_id == "spk1"


def test_incremental_save_resume_and_failures(kaldi_fixture, tmp_path):
    recs, _ = selected(kaldi_fixture)
    out = tmp_path / "pred.jsonl"
    info = {"condition": "baseline_no_context"}

    # 1) crash after 2 items: both results must already be on disk
    try:
        run_transcription(recs, FakeTranscriber(crash_after=2), out, run_info=info, progress=False)
    except KeyboardInterrupt:
        pass
    assert len(load_predictions(out)) == 2

    # 2) a partially written line (crash mid-write) must not break resume
    with open(out, "a", encoding="utf-8") as f:
        f.write('{"sample_id": "broken\n')

    # 3) resume: only the remaining 2 are transcribed; one of them fails
    t2 = FakeTranscriber(fail_ids={"fx_d"})
    stats = run_transcription(recs, t2, out, run_info=info, progress=False)
    assert stats["already_done"] == 2 and stats["ok"] == 1 and stats["error"] == 1
    assert len(t2.calls) == 2
    preds = load_predictions(out)
    assert preds["fx_d"]["status"] == "error"
    assert "simulated decode failure" in preds["fx_d"]["error_message"]

    # 4) the next resume retries only the failed recording
    t3 = FakeTranscriber()
    run_transcription(recs, t3, out, run_info=info, progress=False)
    assert t3.calls == [[r.audio_ref for r in recs if r.sample_id == "fx_d"]]
    assert load_predictions(out)["fx_d"]["status"] == "ok"


def test_transcriber_only_receives_audio_paths(kaldi_fixture, tmp_path):
    recs = load_fixture(kaldi_fixture)
    t = FakeTranscriber()
    run_transcription(recs, t, tmp_path / "p.jsonl", run_info={}, progress=False)
    flat = [p for call in t.calls for p in call]
    assert flat and all(p.endswith(".wav") for p in flat)
    assert not any(r.reference_text in p for r in recs for p in flat)


def test_evaluation_export_counts_failures(kaldi_fixture, tmp_path):
    recs, rep = selected(kaldi_fixture)
    write_manifest(recs, tmp_path / "m.csv", rep)
    recs = read_manifest(tmp_path / "m.csv")
    out = tmp_path / "pred.jsonl"
    run_transcription(recs, FakeTranscriber(fail_ids={"fx_c"}), out, run_info={}, progress=False)
    rows = build_results(recs, out)
    summary = summarize(rows)
    assert summary["n_selected"] == 4 and summary["n_failed"] == 1 and summary["n_success"] == 3
    assert summary["evaluated_audio_hours"] < summary["selected_audio_hours"]
    fx_a = next(r for r in rows if r["sample_id"] == "fx_a")
    # "mr kurz spoke in strasbourg" vs "mr kurz spoke" -> 2 deletions / 5 words
    assert abs(fx_a["wer"] - 0.4) < 1e-9
    assert fx_a["reference_text"] == "Mr. Kurz spoke in Strasbourg."  # raw text preserved
    assert summary["corpus_wer_failures_as_deletions"] > summary["corpus_wer"]
    paths = export(rows, summary, tmp_path / "metrics", "fixture")
    with open(paths["per_recording_csv"], encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header == RESULT_COLUMNS
    for col in ["sample_id", "audio_path", "duration_seconds", "reference_text", "prediction",
                "wer", "cer", "status", "error_message", "wav_path", "label",
                "pred_without_context", "wer_without_context", "cer_without_context"]:
        assert col in header
    json.loads(paths["summary_json"].read_text(encoding="utf-8"))
