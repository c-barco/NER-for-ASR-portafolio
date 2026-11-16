"""Generate notebooks/01_baseline_evaluation.ipynb from the cell list below.

Keeping the notebook as generated code avoids hidden state and keeps diffs readable.
Run:  python scripts/build_notebook.py
"""

import json
from pathlib import Path

CELLS = [
    ("md", """# 01 - Baseline evaluation (portfolio Steps 1-2)

Evaluate an **unchanged pretrained** audio language model (Qwen3-ASR, Transformers `-hf` checkpoint)
on the **approved** English dataset, **without any named-entity context**, and report WER/CER.

* Based on the lecturer's lab notebooks (`jwirth-iisys/mmai_06`, commit `924b10a`); see `docs/lecturer_reference.md`.
* Reusable code lives in `src/ner_asr/`; this notebook only orchestrates it.
* **Out of scope here:** entity extraction, context-conditioned runs, fine-tuning (portfolio Steps 3-8).
* Dataset approval is **pending**. Until then, results from this notebook are technical pilots, not portfolio results."""),
    ("md", "## 1. Configuration\nChoose the config file (or set the `NER_ASR_CONFIG` environment variable). `MODE` is `pilot` by default; `full` requires `selection.target_hours`, which is still pending clarification."),
    ("code", """import os, sys, json
from pathlib import Path

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO / "src"))

from ner_asr.config import load_config

CONFIG = os.environ.get("NER_ASR_CONFIG", "config/voxpopuli_en.yaml")  # change after approval
MODE = os.environ.get("NER_ASR_MODE", "pilot")                          # "pilot" | "full"
RUN_INFERENCE = os.environ.get("NER_ASR_RUN_INFERENCE", "0") == "1"      # downloads + runs the model

cfg = load_config(CONFIG, mode=MODE)
if MODE == "full" and not cfg.run_name.endswith("_full"):
    cfg.run_name += "_full"
IS_FIXTURE = cfg.dataset.adapter == "fixture"
print(json.dumps({"config": CONFIG, "run_name": cfg.run_name, "mode": cfg.mode, "dataset": cfg.dataset.name,
                  "split": cfg.dataset.split, "model": cfg.model.name, "context_mode": cfg.context_mode,
                  "target_seconds": cfg.target_seconds(), "FIXTURE": IS_FIXTURE}, indent=2))"""),
    ("md", "## 2. Environment and hardware check"),
    ("code", """import platform
from ner_asr.runner import environment_info

info = environment_info()
try:
    import torch
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
except Exception as exc:
    info["torch_error"] = repr(exc)
info["python_executable"] = sys.executable  # confirms which environment the kernel uses
print(json.dumps(info, indent=2))"""),
    ("md", "## 3. Dataset loading\nThe adapter named in the config yields records in the common format (`sample_id`, audio reference, `reference_text`, `duration_seconds`, `speaker_id`, `source_split`). If a manifest for this run already exists it is reused, so the selection stays fixed."),
    ("code", """from ner_asr import pipeline
from ner_asr.selection import read_manifest

if IS_FIXTURE:
    import subprocess
    subprocess.run([sys.executable, str(REPO / "tests" / "make_fixture.py")], check=True)  # SYNTHETIC tones
print("manifest:", cfg.manifest_path, "exists:", cfg.manifest_path.exists())"""),
    ("md", "## 4. Inspection and duration-based selection\nDeterministic selection by cumulative duration from one official split. In full mode, an insufficient split raises an error instead of silently mixing splits."),
    ("code", """records, report = pipeline.step_select(cfg)
total_h = sum(r.duration_seconds or 0 for r in records) / 3600
print(f"{len(records)} recordings, {total_h:.3f} h selected from split '{cfg.dataset.split}'")
sel_json = cfg.manifest_path.with_suffix(".selection.json")
if sel_json.exists():
    print(sel_json.read_text(encoding="utf-8"))"""),
    ("code", """# Basic data checks: transcripts and audio of a few selected items
import soundfile as sf
problems = []
for r in records:
    if not (r.reference_text or "").strip():
        problems.append((r.sample_id, "empty reference"))
    try:
        info_ = sf.info(r.audio_ref)
        if abs(info_.frames / info_.samplerate - (r.duration_seconds or 0)) > 0.1:
            problems.append((r.sample_id, "duration mismatch"))
    except Exception as exc:
        problems.append((r.sample_id, f"unreadable audio: {exc}"))
print("problems:", problems or "none")
for r in records[:3]:
    print(f"{r.sample_id} | {r.duration_seconds:.2f}s | speaker={r.speaker_id} | {r.reference_text[:100]}")"""),
    ("code", """# Listen to one item (optional)
try:
    from IPython.display import Audio, display
    from ner_asr.audio import load_audio
    display(Audio(load_audio(records[0].audio_ref), rate=16000))
except Exception as exc:
    print("audio preview unavailable:", exc)"""),
    ("md", "## 5. Model loading\nUnchanged pretrained model, loaded like in the lecturer's notebook (`AutoProcessor` + `AutoModelForMultimodalLM`). Set `NER_ASR_RUN_INFERENCE=1` (or `RUN_INFERENCE = True`) to download and run it. FIXTURE configs use a fake transcriber instead."),
    ("code", """transcriber = None
if IS_FIXTURE:
    from types import SimpleNamespace
    class FixtureTranscriber:  # FIXTURE ONLY - not a model
        def transcribe_batch(self, paths, prompt=None):
            assert prompt is None
            return [SimpleNamespace(prediction="fixture output", raw_output="fixture output",
                                    n_generated_tokens=2, hit_token_limit=False, runtime_s=0.0) for _ in paths]
    transcriber = FixtureTranscriber()
elif RUN_INFERENCE:
    from ner_asr.transcribe import QwenASRTranscriber
    transcriber = QwenASRTranscriber(cfg.model)
    print("loaded", cfg.model.name, "on", transcriber.device, transcriber.model.dtype)
else:
    print("RUN_INFERENCE is False: skipping model loading.")"""),
    ("md", "## 6. Pilot transcription\nNo prompt, no entity hints, no reference text is passed to the model. Predictions are appended to a JSONL file after every batch; re-running this cell resumes and only retries missing or failed recordings."),
    ("code", """if transcriber is not None:
    stats = pipeline.step_transcribe(cfg, transcriber=transcriber)
else:
    print("skipped; existing predictions file:", cfg.predictions_path, cfg.predictions_path.exists())"""),
    ("md", "## 7. Baseline evaluation\nReferences and predictions are normalised identically at scoring time (`src/ner_asr/normalize.py`); the raw texts stay unchanged in the exports."),
    ("code", """summary = pipeline.step_evaluate(cfg) if cfg.predictions_path.exists() else None"""),
    ("md", "## 8. Metrics\n*Corpus-level* WER/CER = total edits / total reference words (characters). *Mean per-recording* WER = average of each recording's WER (what the lecturer's notebook prints as \"Mean WER\"). They are different numbers; report both and say which is which."),
    ("code", """if summary:
    keys = ["n_selected", "n_success", "n_failed", "selected_audio_hours", "evaluated_audio_hours",
            "corpus_wer", "corpus_cer", "corpus_wer_failures_as_deletions",
            "per_recording_wer_mean", "per_recording_wer_median", "per_recording_wer_min", "per_recording_wer_max",
            "per_recording_cer_mean", "perfect_transcription_rate", "total_inference_runtime_s", "real_time_factor"]
    for k in keys:
        v = summary.get(k)
        print(f"{k:38s} {v:.4f}" if isinstance(v, float) else f"{k:38s} {v}")
    print("flags:", summary["flags"])
    print("failure reasons:", summary["failure_reasons"] or "none")"""),
    ("md", "## 9. Basic prediction checks\nSanity checks needed to trust the baseline: worst recordings, empty outputs, possible truncation, failures. Detailed named-entity error analysis is portfolio **Step 3** and intentionally not done here."),
    ("code", """import csv, math
csv_path = cfg.metrics_dir / f"{cfg.run_name}_per_recording.csv"
if csv_path.exists():
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    scored = [r for r in rows if r["wer"] != ""]
    worst = sorted(scored, key=lambda r: float(r["wer"]), reverse=True)[:10]
    for r in worst:
        print(f"WER {float(r['wer']):.2f} | {r['sample_id']}\\n  REF : {r['reference_text']}\\n  PRED: {r['prediction']}")
    flagged = [r["sample_id"] for r in rows if "True" in (r["flag_hit_token_limit"], r["flag_empty_prediction"], r["flag_short_vs_reference"])]
    print("flagged (possible truncation / empty):", flagged or "none")"""),
    ("md", "## 10. Export results\nFiles written by `step_evaluate` (large outputs are git-ignored; commit only the small summary files you want to report)."),
    ("code", """for p in [cfg.manifest_path, cfg.predictions_path,
          cfg.metrics_dir / f"{cfg.run_name}_per_recording.csv", cfg.metrics_dir / f"{cfg.run_name}_summary.json"]:
    print(("OK  " if p.exists() else "--  ") + str(p.relative_to(REPO) if p.is_relative_to(REPO) else p))"""),
]


def build() -> dict:
    cells = []
    for kind, src in CELLS:
        cell = {"id": f"cell-{len(cells):02d}", "cell_type": "markdown" if kind == "md" else "code", "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python (ner-asr)", "language": "python", "name": "ner-asr"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "notebooks" / "01_baseline_evaluation.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", out)
