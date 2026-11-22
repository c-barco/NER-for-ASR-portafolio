# Relationship to the lecturer material

The project retains the lecturer notebooks' broad evaluation pattern—audio
loading, transcription, text normalisation, WER/CER, and per-recording
exports—but uses the fixed dataset and model selected for this project.

| Topic | Project decision |
|---|---|
| Dataset | EuroSpeech `uk` (EN) and `germany` (DE), official `test` split only |
| Model | `ibm-granite/granite-4.0-1b-speech` |
| Model API | `AutoProcessor` + `AutoModelForSpeechSeq2Seq` |
| Audio | mono 16 kHz PCM WAV for cached/exported data |
| Baseline prompt | Granite's documented transcription request with `prompt=None` |
| Metrics | normalised per-recording and corpus WER/CER |
| Submission data | one validated embedded-audio parquet per language |

The Granite implementation follows the current model-card generation example:
it applies the tokenizer chat template to
`<|audio|>can you transcribe the speech into a written format?`, processes a
mono 16 kHz waveform, uses greedy generation, and decodes only generated
tokens. The baseline supplies no keywords. The `prompt` argument is reserved
for Granite's documented ` Keywords: name1, name2` mechanism but is not used in
this submission.

Deliberate deviations from a simple notebook loop are fixed manifests, explicit
split checks, resumable JSONL predictions, source-audio validation, and a
parquet round trip. They make the evaluation and data artifact reproducible
without adding later-stage modelling work.

Sources: [lecturer repository](https://github.com/jwirth-iisys/mmai_06),
[Granite 4 Speech model card](https://huggingface.co/ibm-granite/granite-4.0-1b-speech).
