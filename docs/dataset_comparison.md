# Step 1 - Dataset comparison (English)

Status: **professor approval pending.** The ranking below is the preference we sent; it is
qualitative. Named-entity coverage has **not** been measured and no samples have been compared yet.

Figures marked *(viewer)* come from the Hugging Face dataset-viewer statistics API (row count ×
mean duration), retrieved 2026-11-16. Everything else is cited from the official pages listed below.

| | 1. VoxPopuli EN | 2. Multilingual LibriSpeech (MLS) EN | 3. Mozilla Common Voice EN |
|---|---|---|---|
| Domain | European Parliament plenary speeches (2009-2020) | LibriVox audiobooks (read literature) | Crowd-sourced read sentences |
| Speech style | Prepared/spontaneous political speech, reverberant hall | Read speech, clean | Short read prompts, very variable microphones |
| Lab_06 characterisation | "very noisy, faulty labels" | "little background noise, rather high recording quality" | "noisy, many speakers, varying recording quality" |
| Transcribed EN audio | 543 h total (dataset card) | train: 10,808,037 items *(viewer)*; per-split hours in the MLS paper (arXiv 2012.03411), not re-checked here | Release-dependent |
| Official test split | 1,842 items, **≈ 4.93 h** *(viewer)*; mean 9.6 s, max 69 s | 3,769 items, **≈ 15.55 h** *(viewer)*; 10-20 s each | Not measured |
| Official dev/validation | 1,753 items, ≈ 4.98 h *(viewer)* | 3,807 items, ≈ 15.75 h *(viewer)* | Not measured |
| Speakers in test | 286 IDs + 150 items with no speaker ID *(viewer)* | 42 speakers, 58 books *(viewer)* | Not measured |
| Transcript format | `raw_text` (cased, punctuated; some are empty) and `normalized_text` (lowercase) | lowercase, no punctuation | cased sentence with punctuation |
| Likely entity types (unmeasured) | MEPs, politicians, countries, EU bodies, laws | Character names, places in fiction; older/uncommon names | Mixed; prompts come from public-domain text and Wikipedia sentences |
| License | CC0 (plus European Parliament legal notice) | CC BY 4.0 | CC0 (per Data Collective listing) |
| Access | Hugging Face `facebook/voxpopuli`, not gated; Parquet per split; streaming supported | OpenSLR 94: English only as one 2.4 TB FLAC / 651 GB Opus archive (or 100 GB parts). `facebook/multilingual_librispeech` on HF has **no English config**. English test/dev are on HF as `parler-tts/mls_eng` (community re-upload) | **Mozilla Data Collective only** (since Oct 2025); needs an account and accepting the terms; English Scripted Speech 25.0 ≈ **87.84 GB**, no per-split download |
| Partial download | Yes: the EN test split is a single ~0.94 GB Parquet file; streaming reads row groups | Yes for test/dev via the HF mirror (~0.25 GB test Parquet); no for the official OpenSLR archive | No: the full archive. Do **not** download ~88 GB for this |
| On the university cluster (lecturer paths) | `/training-1/voxpopuli/dataset_with_spaces_voxpopuli_en/test` | `/training-1/asr/kaldi_data/mls/en/test` | `/training-1/asr/kaldi_data/cv19/en/test` (v19) |
| Adapter here | `hf_voxpopuli` (untested: blocked locally) / `kaldi_dir` | `hf_mls_en` (untested: blocked locally) / `kaldi_dir` | `common_voice_local` (untested) / `kaldi_dir` |

## Observations relevant to the choice

* **Duration vs. ~40 h.** No single official *test* split reaches 40 h (VoxPopuli ≈ 4.9 h,
  MLS ≈ 15.6 h). If ~40 h applies to the baseline evaluation, we would need audio from other
  splits. That would make those recordings unavailable for fine-tuning and may require custom,
  speaker-aware splits. If ~40 h applies to the whole project (incl. fine-tuning data), the test
  split is enough for Step 2. **This must be clarified with the professor.** The pipeline refuses
  to mix splits silently.
* **Speaker overlap.** VoxPopuli EN test and validation share speaker IDs (e.g. `28165`, `124698`
  appear in both *(viewer)*), and 150 test items have no speaker ID. Any custom split for
  fine-tuning must check speaker overlap (`selection.check_disjoint(..., by_speaker=True)`).
* **Casing matters for later NER (Step 4).** spaCy NER works much better on cased text. VoxPopuli
  `raw_text` and Common Voice keep casing. MLS and the cluster's VoxPopuli copy (`dataset_with_spaces`,
  e.g. "what im asking mr kurz") are lowercase. This does not affect Step 2 WER/CER (we normalise),
  but it matters for the dataset choice.
* **Label quality.** VoxPopuli segments were filtered at ≤20 % CER against ASR output and "may have
  incorrect transcriptions" (dataset card). Expect some errors to be label errors; check the worst items manually.
* **Second dataset (Step 8).** Whichever dataset is not approved as primary is a natural candidate
  for the generalisation test.

## Sources

* VoxPopuli dataset card: https://huggingface.co/datasets/facebook/voxpopuli
* MLS (OpenSLR 94): https://www.openslr.org/94/ ; HF: https://huggingface.co/datasets/facebook/multilingual_librispeech ; EN mirror: https://huggingface.co/datasets/parler-tts/mls_eng
* Common Voice on Mozilla Data Collective: https://datacollective.mozillafoundation.org/ (English Scripted Speech 25.0 listing: https://prod.datacollective.mozillafoundation.org/datasets/cmndapwry02jnmh07dyo46mot)
* Lab 5/6 slides (`lab_06.pdf`), slide 7: dataset characterisation.
