"""Generate the compact Step 2 orchestration notebook.

The notebook deliberately contains orchestration only. Dataset conversion,
Granite inference, metrics, and parquet validation remain in ``src/ner_asr``.
Run ``python scripts/build_notebook.py`` after changing this template.
"""

from __future__ import annotations

import json
from pathlib import Path


CELLS = [
    (
        "md",
        """# Step 2: EuroSpeech dataset and Granite 4 baseline (EN/DE)

This compact notebook prepares the approved EuroSpeech artifact and optionally
runs a no-context Granite 4 baseline. It borrows only a simple staged layout;
all reusable work stays in `src/ner_asr/`.

* Dataset: EuroSpeech `uk` (English) and `germany` (German), official test split.
* Model: `ibm-granite/granite-4.0-1b-speech`, always called with `prompt=None` for the baseline.
* Step 2 artifact: one parquet with exactly 1,000 English and 1,000 German rows.
* Out of scope: NER, keyword-conditioned runs, error mining, and fine-tuning.
* Nothing downloads by default. Each networked action has its own explicit switch.""",
    ),
    (
        "md",
        """## 1. Configuration

Choose a language config. The default is English; rerun the notebook with
`NER_ASR_CONFIG=config/eurospeech_de.yaml` for the 250-record German pilot.
Set a switch to `1` only when you intend the action:
`NER_ASR_RUN_SELECTION=1` streams the pilot split,
`NER_ASR_RUN_INFERENCE=1` downloads/runs Granite, and
`NER_ASR_EXPORT_DELIVERABLE=1` creates the parquet artifact. Set
`NER_ASR_OVERWRITE_MANIFEST=1` when changing the selected pilot size so the
old manifest is replaced.""",
    ),
    (
        "code",
        """import json
import os
import sys
from pathlib import Path

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO / "src"))

from ner_asr import pipeline
from ner_asr.config import load_config
from ner_asr.selection import read_manifest

CONFIG = os.environ.get("NER_ASR_CONFIG", "config/eurospeech_en.yaml")
DELIVERABLE_DE_CONFIG = os.environ.get("NER_ASR_CONFIG_DE", "config/eurospeech_de.yaml")
MODE = os.environ.get("NER_ASR_MODE", "pilot")
RUN_INFERENCE = os.environ.get("NER_ASR_RUN_INFERENCE", "0") == "1"
EXPORT_STEP2_DELIVERABLE = os.environ.get("NER_ASR_EXPORT_DELIVERABLE", "0") == "1"
OVERWRITE_MANIFEST = os.environ.get("NER_ASR_OVERWRITE_MANIFEST", "0") == "1"

cfg = load_config(CONFIG, mode=MODE)
if MODE == "full" and not cfg.run_name.endswith("_full"):
    cfg.run_name += "_full"
IS_FIXTURE = cfg.dataset.adapter == "fixture"
RUN_SELECTION = IS_FIXTURE or os.environ.get("NER_ASR_RUN_SELECTION", "0") == "1"

print(json.dumps({
    "config": CONFIG,
    "deliverable_de_config": DELIVERABLE_DE_CONFIG,
    "dataset": cfg.dataset.name,
    "split": cfg.dataset.split,
    "model": cfg.model.name,
    "run_selection": RUN_SELECTION,
    "run_inference": RUN_INFERENCE,
    "export_deliverable": EXPORT_STEP2_DELIVERABLE,
    "overwrite_manifest": OVERWRITE_MANIFEST,
}, indent=2))""",
    ),
    (
        "md",
        """## 2. What one dataset row looks like

This is a safe structural preview of one final parquet row. It uses a readable
placeholder instead of real binary audio and does not open or download data.""",
    ),
    (
        "code",
        """from ner_asr.previews import example_deliverable_row

print(json.dumps(example_deliverable_row(
    language=cfg.dataset.language,
    source_split=cfg.dataset.split,
), indent=2))""",
    ),
    (
        "md",
        """## 3. What Granite receives and returns

This shows the exact no-context request shape used by the baseline. It is an
example only: no model weights are loaded and no inference is run.""",
    ),
    (
        "code",
        """from ner_asr.previews import granite_baseline_example

print(json.dumps(granite_baseline_example(model_name=cfg.model.name), indent=2))""",
    ),
    (
        "md",
        """## 4. Select or reuse a fixed split

Selection is deterministic and never mixes official splits. If no manifest is
available and selection is disabled, this cell remains local and simply reports
what needs to be enabled.""",
    ),
    (
        "code",
        """if IS_FIXTURE:
    import subprocess
    subprocess.run([sys.executable, str(REPO / "tests" / "make_fixture.py")], check=True)

if RUN_SELECTION:
    records, report = pipeline.step_select(cfg, overwrite=OVERWRITE_MANIFEST)
elif cfg.manifest_path.exists():
    records, report = read_manifest(cfg.manifest_path), None
    print("Using existing manifest:", cfg.manifest_path)
else:
    records, report = [], None
    print("Selection skipped. Set NER_ASR_RUN_SELECTION=1 to stream the selected split.")

if records:
    hours = sum(record.duration_seconds or 0 for record in records) / 3600
    first = records[0]
    print(f"{len(records)} recordings | {hours:.3f} h | first: {first.sample_id}")
    try:
        from IPython.display import Audio, display
        from ner_asr.audio import load_audio
        display(Audio(load_audio(first.audio_ref), rate=16_000))
    except Exception as exc:
        print("audio preview unavailable:", exc)""",
    ),
    (
        "md",
        """## 5. Optional Granite baseline

The baseline uses audio only—no reference text, entity list, or keyword prompt
reaches the model. Existing predictions can be evaluated without loading the
model.""",
    ),
    (
        "code",
        """transcriber = None
if IS_FIXTURE:
    from types import SimpleNamespace

    class FixtureTranscriber:
        def transcribe_batch(self, paths, prompt=None):
            assert prompt is None
            return [SimpleNamespace(
                prediction="fixture output", raw_output="fixture output",
                n_generated_tokens=2, hit_token_limit=False, runtime_s=0.0,
            ) for _ in paths]

    transcriber = FixtureTranscriber()
elif RUN_INFERENCE and records:
    from ner_asr.transcribe import Granite4ASRTranscriber
    transcriber = Granite4ASRTranscriber(cfg.model)
    print("Loaded", cfg.model.name, "on", transcriber.device)
elif RUN_INFERENCE:
    print("Inference requested, but no manifest is available. Run selection or reuse a manifest first.")
else:
    print("Inference skipped. Set NER_ASR_RUN_INFERENCE=1 to load Granite.")

if transcriber is not None and records:
    pipeline.step_transcribe(cfg, transcriber=transcriber)

summary = pipeline.step_evaluate(cfg) if cfg.predictions_path.exists() else None
if summary:
    print("Mean WER:", summary["per_recording_wer_mean"])
    print("Mean CER:", summary["per_recording_cer_mean"])
else:
    print("No predictions to evaluate yet.")""",
    ),
    (
        "md",
        """## 6. Step 2 parquet artifact (optional)

This is separate from model inference. It streams about 1,200 candidates for
each language from the official test split, writes one embedded-audio parquet
with exactly 1,000 English and 1,000 German valid 2–20 second rows, then
checks the final file immediately and displays one WAV sample. The final file is
`data/deliverable/step2_eurospeech_en_de.parquet`; every row carries its
`language` value.""",
    ),
    (
        "code",
        """if EXPORT_STEP2_DELIVERABLE:
    de_cfg = load_config(DELIVERABLE_DE_CONFIG, mode=MODE)
    artifact, report = pipeline.step_export_deliverable(cfg, de_cfg)
    from ner_asr.deliverable import reload_and_validate_parquet

    verified = reload_and_validate_parquet(
        artifact,
        minimum_rows=2_000,
        expected_rows=2_000,
        expected_language_counts={"en": 1_000, "de": 1_000},
    )
    print("Parquet checks passed:", len(verified), "rows | language counts:", {
        "en": verified["language"].count("en"),
        "de": verified["language"].count("de"),
    })
    try:
        from IPython.display import Audio, display
        display(Audio(data=verified[0]["audio"]["bytes"]))
    except Exception as exc:
        print("parquet audio preview unavailable:", exc)
else:
    print("Parquet export skipped. Set NER_ASR_EXPORT_DELIVERABLE=1 to run it.")""",
    ),
]


def build() -> dict:
    cells = []
    for kind, source in CELLS:
        cell = {
            "id": f"cell-{len(cells):02d}",
            "cell_type": "markdown" if kind == "md" else "code",
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }
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
    output = Path(__file__).resolve().parents[1] / "notebooks" / "01_baseline_evaluation.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", output)
