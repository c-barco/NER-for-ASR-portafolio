# Lecturer reference material and how this pipeline uses it

## Sources consulted

* Lecturer repository: https://github.com/jwirth-iisys/mmai_06, commit `924b10a`
  ("added evaluation over all datasets"). The local copy under the course Materials folder is
  byte-identical to this commit (checked 2026-11-16; read-only clone outside this repository).
  * `02_qwen3_asr_playground.ipynb` - model loading, `transcribe()`, context prompt example
  * `03_evaluation_granite.ipynb` - multi-dataset evaluation loop, `normalize()`, jiwer, TSV export
  * `04_named_entity_recognition.ipynb` - spaCy `en_core_web_trf` NER (Step 4, not used yet)
  * `01_granite_playground.ipynb` - alternative model (Granite Speech), not used
* Slides: `lab_06.pdf` (Lab 5, 10.11.2026) metrics + required columns; `lab_07.pdf` (Lab 7) fine-tuning
  data format (JSONL with `audio`, `prompt`, `text`); `MM-AI-01 - Overview_2026.pdf` pp. 20-23.

## What we reuse

| Topic | Lecturer approach | This repository |
|---|---|---|
| Model | `Qwen/Qwen3-ASR-1.7B-hf`, `AutoProcessor` + `AutoModelForMultimodalLM`, `device_map="auto"`, `transformers>=5.13.0` | Same API in `src/ner_asr/transcribe.py`; 1.7B for the GPU run, 0.6B-hf for laptop pilots; revisions pinned |
| Data layout | Kaldi dirs: `wav.scp` + `text`, TAB-separated, on `/training-1/...` | `kaldi_dir` adapter reads the same files (+ optional `utt2spk`, `utt2dur`); HF/Common Voice adapters for off-cluster use |
| Audio | `torchaudio.load` → resample to 16 kHz → mean over channels | Same steps with soundfile + soxr (`src/ner_asr/audio.py`) |
| Transcription | `processor.apply_transcription_request(audio=path, prompt=?, language=?)`, `generate(max_new_tokens=256)`, `decode(return_format="transcription_only")` | Same; `prompt` is fixed to `None` for the baseline; `language="English"` is passed (configurable) |
| Normalisation | strip ASCII punctuation, lowercase, collapse spaces | Same idea, Unicode-aware, hyphen → space (see deviations) |
| Metrics | `jiwer.wer`, `jiwer.cer` per item; mean WER/CER | Per item + corpus-level + median/min/max + perfect rate (lab_06 slide 7) |
| Export | TSV with `audio_path`, `label`, `prediction`, `wer`, `cer`; lab_06 asks for `wav_path`, `label`, `pred_/wer_/cer_with_context`, `pred_/wer_/cer_without_context` | CSV with our columns **plus** the lab_06 `*_without_context` columns so the later context run can be merged on `wav_path` |

## Deviations and why

1. **WER computed on normalised text.** The lecturer's loop computes `jiwer.wer(sample["text"], model_output)`
   on *raw* text but stores normalised text in the TSV, so punctuation/casing differences count as
   errors. We normalise both sides before scoring; the old style is still reported as
   `lecturer_style_mean_wer_unnormalized` for comparison.
2. **Normalisation details.** Unicode punctuation is removed (not only ASCII), and hyphens/dashes
   become spaces ("anti-poverty" → "anti poverty", matching VoxPopuli normalised labels). Numbers
   are not verbalised in either version.
3. **Corpus-level WER added.** The notebook's "Mean WER" is a mean over recordings; we report it
   as `per_recording_wer_mean` and add corpus-level WER as the primary number.
4. **No `[:50]` cap; fixed manifest instead.** The notebook evaluates the first 50 items. We select by
   cumulative duration with a seed and save the manifest.
5. **Robustness.** Incremental JSONL output, resume, failure records, runtime, truncation flags.
6. **Known notebook issue (not copied):** `output_statistics.append("Mean WER:", value)` passes two
   arguments to `list.append` and would raise `TypeError` at the end of the first dataset.
7. **Oracle context (later, Step 5).** The lecturer extracts entities from the *reference* text
   (`nlp(sample["text"])`). That is reference-derived (oracle) context; when we reach Step 5 it must be
   labelled as such. It is not used in the Step 2 baseline.
8. **University-specific settings.** `/training-1/...` paths appear only in `config/cluster_kaldi.yaml`;
   no credentials are needed or stored. The `!pip install` cells are replaced by `requirements.txt`.
