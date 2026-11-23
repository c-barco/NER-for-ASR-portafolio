# Step 2 parquet deliverable

## Artifact

The exporter creates exactly one combined bilingual parquet file:

```text
data/deliverable/step2_eurospeech_en_de.parquet
```

It contains exactly 2,000 rows: 1,000 English (`en`) records and 1,000 German
(`de`) records. The `language` column makes the source language explicit for
every row.

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

The command independently streams a deterministic window of 1,200 candidate
records for each language from its official `test` split. It sorts by ID,
shuffles with the fixed config seed (`42`), retains only non-empty 2–20 second
items, and selects exactly 1,000 examples per language while representing every
populated duration stratum (2–5, 5–10, 10–15, 15–20 s). It raises an error if
either language has fewer than 1,000 valid rows or if records from different
source splits would be mixed within a language.

Both `config/eurospeech_en.yaml` and `config/eurospeech_de.yaml` record the
same fixed quota explicitly: `candidate_limit_per_language: 1200`,
`rows_per_language: 1000`, and `total_rows: 2000`. The pipeline refuses
mismatched language configurations.

The combined artifact retains the original EuroSpeech `key` in `sample_id` and
fails loudly if a source key would be duplicated across the two language
selections; it never silently changes an identifier.

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
python -m ner_asr.pipeline export-deliverable --config config/eurospeech_en.yaml \
  --config-de config/eurospeech_de.yaml
```

The command refuses to overwrite an existing file. Add
`--overwrite-deliverable` only after confirming replacement is intended.

For an optional audible notebook check after export, set
`NER_ASR_EXPORT_DELIVERABLE=1` in `01_baseline_evaluation.ipynb`. The final
cell reloads the parquet and displays one embedded WAV in a notebook player.
