# Step 2 parquet deliverable

## Artifact

The exporter creates exactly one parquet file for each requested language:

```text
data/deliverable/step2_eurospeech_en.parquet
data/deliverable/step2_eurospeech_de.parquet
```

The artifact is a dataset for ASR, so every row contains speech and its
reference transcript rather than image/question/answer data. Its columns are:

| Column | Meaning |
|---|---|
| `sample_id` | EuroSpeech `key`, stable within the source configuration |
| `audio` | embedded 16 kHz mono PCM WAV bytes (`datasets.Audio`) |
| `text` | `human_transcript` reference |
| `duration_seconds` | duration measured from the normalized WAV |
| `speaker_id` | null; EuroSpeech does not publish a speaker identifier |
| `source_split` | the one official source split used (`test`) |
| `language` | `en` or `de` |

## Preparation rule

The command streams a deterministic window of 1,200 candidate records from one
official split. It sorts by ID, shuffles with the config seed (`42`), retains
only non-empty 2–20 second items, and selects up to 1,100 examples while
representing every populated duration stratum (2–5, 5–10, 10–15, 15–20 s).
It raises an error if fewer than 1,000 valid rows remain or if records from
different source splits would be mixed.

Source audio is decoded before acceptance; unreadable source rows are discarded.
Each retained waveform is converted to mono 16 kHz PCM WAV, reopened with `soundfile`, stored as embedded bytes,
written to a temporary parquet, and reloaded with
`datasets.load_dataset("parquet", ...)` before it becomes the final artifact.
The reload check requires embedded bytes and validates a sample WAV again.

EuroSpeech has no named-entity labels. The printed report therefore includes an
**entity-like** share based only on a capitalized multi-token text pattern. This
is a transparent descriptive proxy, not NER and not a ground-truth entity
prevalence estimate; in German it is particularly affected by ordinary noun
capitalization. No Step 3–8 processing is performed.

## Run deliberately

These commands download dataset data but do not run model inference:

```bash
python -m ner_asr.pipeline export-deliverable --config config/eurospeech_en.yaml
python -m ner_asr.pipeline export-deliverable --config config/eurospeech_de.yaml
```

The command refuses to overwrite an existing file. Add
`--overwrite-deliverable` only after confirming replacement is intended.

For an optional audible notebook check after export, set
`NER_ASR_EXPORT_DELIVERABLE=1` in `01_baseline_evaluation.ipynb`. The final
cell reloads the parquet and displays one embedded WAV in a notebook player.
