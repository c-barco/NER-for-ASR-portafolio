# NER-for-ASR portfolio

This repository prepares the Step 2 ASR dataset and runs a no-context baseline
for a named-entity-aware ASR portfolio. The selected data are **EuroSpeech**
(`disco-eth/EuroSpeech`) in English and German; the selected model is **IBM
Granite 4 Speech** (`ibm-granite/granite-4.0-1b-speech`).

No entity extraction, keyword-conditioned evaluation, error analysis, or
fine-tuning is implemented here. Those are later portfolio steps and must not
be inferred from this baseline.

## Study design

EuroSpeech is a 16 kHz corpus of aligned parliamentary speech. The project uses
the official `uk` configuration for English and `germany` for German, with the
official **test** split only. It does not combine test, validation, and train
records. This makes the submitted artifact held out from future training data.

The team selected EuroSpeech because it supplies substantially more than 40
hours of English and German parliamentary speech, is designated by the team as
outside Granite's training data, and avoids the VoxPopuli and WorldSpeech
datasets used by other teams. The exact dataset revision and model revision are
pinned in the two configs for reproducibility.

Granite is called with its documented no-context ASR request:

```text
<|audio|>can you transcribe the speech into a written format?
```

The baseline passes `prompt=None`, so no keyword list or entity information is
provided. Granite's documented `Keywords: ...` suffix is implemented only in
the transcriber interface for a future authorised step; it is not invoked by
the current pipeline.

## Required Step 2 parquet artifact

The exam artifact is one embedded-audio bilingual parquet file:

```text
data/deliverable/step2_eurospeech_en_de.parquet
```

It contains exactly 2,000 rows: 1,000 English (`en`) and 1,000 German (`de`).
The exporter streams about 1,200 candidates **per language** from the official
test split, applies fixed seeds, removes empty/unreadable entries, keeps only
2–20 second utterances, and checks mixed duration strata. It fails loudly if
either language cannot supply its required 1,000 valid rows. Each output row
has exactly:

```text
sample_id, audio, text, duration_seconds, speaker_id, source_split, language
```

Both EuroSpeech configuration files make the fixed quota explicit:
`candidate_limit_per_language: 1200`, `rows_per_language: 1000`, and
`total_rows: 2000`.

`audio` is embedded 16 kHz mono PCM WAV data represented through
`datasets.Audio`. EuroSpeech does not publish a speaker identity for these
rows; `video_id` is a parliamentary session rather than a speaker, so
`speaker_id` is null. The exporter reopens the parquet using
`datasets.load_dataset("parquet", ...)`, validates a sample WAV, and prints the
row count, hours, duration statistics, duration-bin counts, and a clearly
labelled entity-like proxy. It does not run NER: EuroSpeech has no entity
annotations, and the proxy is not a ground-truth entity rate.

The commands below explicitly download dataset data. They are intentionally not
run automatically by tests or the notebook.

```bash
python -m ner_asr.pipeline export-deliverable --config config/eurospeech_en.yaml --config-de config/eurospeech_de.yaml
```

Use `--overwrite-deliverable` only when deliberately replacing an existing
artifact. See [the delivery note](docs/step2_deliverable.md) for the validation
rules and an optional notebook audio player.

## Baseline evaluation

The reusable baseline flow is:

```bash
# Fixed seeded selection from exactly one official split
python -m ner_asr.pipeline select --config config/eurospeech_en.yaml

# Granite 4 transcription with prompt=None; downloads model weights and runs inference
python -m ner_asr.pipeline transcribe --config config/eurospeech_en.yaml

# WER/CER exports
python -m ner_asr.pipeline evaluate --config config/eurospeech_en.yaml
```

Evaluation prints mean WER/CER and descriptive statistics, then saves a
WER-descending, tab-separated no-context result file at
`results/metrics/<run_name>_results_without_context.tsv`. Its name deliberately
does not imply a context-conditioned experiment.

`all` runs those three baseline stages. A full evaluation additionally requires
an explicit `selection.target_hours`; it is not guessed by the code. The
notebook [`01_baseline_evaluation.ipynb`](notebooks/01_baseline_evaluation.ipynb)
only orchestrates these reusable functions. Set `NER_ASR_RUN_SELECTION=1` only
when streaming source audio is intended, and `NER_ASR_RUN_INFERENCE=1` only
when model download and inference are intended.

## Setup

Use Python 3.10+ and an environment with a current `transformers` release
(Granite 4 requires at least 4.52.1), PyTorch, PEFT, `datasets`, `soundfile`,
and `soxr`.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
python -m pytest -q
```

For real inference, use a machine with suitable GPU memory. The project laptop
may not import `datasets`/PyArrow because of local Windows security policy; the
fixture tests and source checks do not download data or model weights.

## Repository layout

```text
config/             selected EN/DE EuroSpeech configurations
src/ner_asr/         adapters, selection, parquet export, Granite wrapper, metrics
notebooks/           baseline orchestration and optional parquet preview
data/deliverable/    generated bilingual exam parquet artifact (git-ignored)
docs/                dataset decision, delivery method, scope and sources
tests/               synthetic and no-network validation
```

## Sources

* [EuroSpeech dataset card](https://huggingface.co/datasets/disco-eth/EuroSpeech)
  — configurations, schema, split structure, 16 kHz audio, and corpus scope.
* [EuroSpeech paper](https://arxiv.org/abs/2510.00514) — corpus construction and
  alignment method.
* [Granite 4 Speech model card](https://huggingface.co/ibm-granite/granite-4.0-1b-speech)
  — `AutoProcessor`, `AutoModelForSpeechSeq2Seq`, exact ASR prompt, keyword
  suffix, and English/German support.
