# Dataset decision: EuroSpeech

## Fixed decision

The approved project dataset is **EuroSpeech** (`disco-eth/EuroSpeech`) for
English and German. The project uses the official `uk` configuration for
English and `germany` for German. This document replaces the earlier candidate
ranking; no alternative corpus is the active dataset.

| Property | English | German |
|---|---:|---:|
| EuroSpeech configuration | `uk` | `germany` |
| Language value | `en` | `de` |
| Baseline / deliverable split | `test` | `test` |
| Reference field | `human_transcript` | `human_transcript` |
| Stable utterance identifier | `key` | `key` |
| Audio | 16 kHz source audio | 16 kHz source audio |
| Speaker identifier | not supplied | not supplied |

`video_id` is a parliamentary session identifier, not a speaker identifier. It
is therefore not copied into `speaker_id`.

## Rationale

The team selected EuroSpeech because it offers substantially more than 40 hours
of parliamentary audio for both project languages, was designated by the team
as outside Granite's training data, and avoids VoxPopuli and WorldSpeech, which
other teams use. Parliamentary proceedings provide a suitable setting for the
portfolio's later name and organisation-recognition questions while retaining
human-provided transcripts.

EuroSpeech is an aligned corpus rather than a perfect verbatim record. Its card
reports quality-filtered subsets and an alignment process designed for diverse
audio sources and non-verbatim transcripts. WER/CER results must therefore be
interpreted as agreement with `human_transcript`, not as a direct measurement of
speech-recognition truth independent of the corpus annotations.

## Split discipline

The bilingual export consumes the official `test` split independently for both
languages. It never tops up from `train` or `validation`; the combined parquet
contains exactly 1,000 `en` and 1,000 `de` rows, each labelled in `language`.
The code checks that a selected language set contains only one `source_split`.
This preserves train/dev/test separation for any later, separately authorised
training experiment.

## Provenance and sources

The configs pin EuroSpeech revision
`7a55bbcee3933f5d9817bb2d4c37399d596f543d` and preserve it in manifests and
run metadata. The primary sources are:

* [EuroSpeech dataset card](https://huggingface.co/datasets/disco-eth/EuroSpeech)
  for configurations, schema, 16 kHz release, splits, and data statement.
* [EuroSpeech paper](https://arxiv.org/abs/2510.00514) for corpus construction
  and alignment methodology.
