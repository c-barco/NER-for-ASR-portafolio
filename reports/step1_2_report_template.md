# NER for ASR - Portfolio Steps 1 and 2

*Team:* ... *Course:* Multimodal AI, WS 2026/27 *Date:* ...

> Template. Replace every `TODO`. Report only numbers produced by an approved full run; do not use
> fixture or pilot numbers as results.

## 1. Step 1 - Dataset review and choice

### 1.1 Candidates reviewed
Summarise `docs/dataset_comparison.md` (domain, size of the official splits, transcript format,
license, access limits). TODO

### 1.2 Approved dataset and justification
* Approved dataset: TODO (approved by ... on ...)
* Why it suits a NER-for-ASR study: TODO (expected entity types, casing of labels, recording conditions)
* Limitations: TODO (label noise, domain, access)
* Second dataset planned for Step 8: TODO

## 2. Step 2 - Baseline evaluation

### 2.1 Setup
| Item | Value |
|---|---|
| Model (name @ revision) | TODO |
| Dataset / version / split | TODO |
| Selection | seed TODO, target TODO h, min/max duration TODO; manifest `data/manifests/TODO` |
| Selected recordings / hours | TODO |
| Hardware | TODO |
| Software | see `summary.json` → `run_info.environment` |
| Decoding | greedy, `max_new_tokens=256`, language forced to English (TODO confirm), **no context prompt** |
| Normalisation | `normalize.py` v1: Unicode NFKC, lowercase, hyphen→space, remove punctuation; numbers not verbalised |

### 2.2 Results
| Metric | Value |
|---|---|
| Corpus-level WER | TODO |
| Corpus-level CER | TODO |
| Mean per-recording WER / CER | TODO / TODO |
| Median, min, max per-recording WER | TODO |
| Perfect-transcription rate (WER = 0) | TODO |
| Successful / failed recordings | TODO / TODO |
| Evaluated audio (h) of selected (h) | TODO |
| Corpus WER counting failures as deletions | TODO |
| Inference time, real-time factor | TODO |

Explain the difference between corpus-level and mean per-recording scores in one sentence.

### 2.3 Sanity checks
Worst recordings (with listening notes), flagged truncations/empty outputs, failures and their
reasons, label errors spotted. TODO

### 2.4 Discussion
How good is the baseline? Where does it seem to fail (only a first impression; detailed entity
analysis is Step 3)? Effect of normalisation choices (e.g. numbers). TODO

### 2.5 Reproducibility
Commands from README; commit hash TODO; files: manifest, predictions JSONL, per-recording CSV, summary JSON.

## 3. Deviations from the lecturer notebooks
Short version of `docs/lecturer_reference.md`. TODO

## 4. Open points
TODO (e.g. the duration requirement, from `docs/open_questions.md`)
