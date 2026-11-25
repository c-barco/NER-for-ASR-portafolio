# Granite 4 Speech baseline

The pilot evaluated 250 EuroSpeech test recordings per language with a standard
transcription request and no keywords. The 500 recordings (2.09 h) are part of
the 2,000-row Step 2 Parquet. The remaining 1,500 were not evaluated.

| Language | Corpus WER / CER | Mean WER / CER | Median WER / CER | WER = 0 |
|---|---:|---:|---:|---:|
| English | 26.87% / 18.13% | 30.48% / 22.25% | 24.23% / 15.67% | 2 / 250 |
| German | 17.13% / 12.30% | 20.95% / 16.10% | 14.61% / 9.13% | 17 / 250 |

German had lower error rates on these samples. Corpus rates divide total edits
by total reference words or characters. The mean gives each recording equal
weight, so short references with many errors can raise it. Two recordings per
language exceeded 100% WER. Maximum WER/CER were 557.14%/710.00% in English and
600.00%/736.36% in German. Both minima were zero. All 500 recordings were scored.
One German prediction hit the 200-token limit. A different German prediction
was flagged as short relative to its reference.

Scoring uses the [ASR lab](https://github.com/jwirth-iisys/mmai_06) evaluation
pattern. Both texts are lowercased and normalized before scoring. Dashes and
slashes become spaces, punctuation is removed and numbers keep their original
form. EuroSpeech references are aligned parliamentary transcripts, so some
differences may come from alignment or transcription. The scores describe
transcription agreement on this sample.

The Step 2 archive contains audio references and transcripts, 1,000 per language
(8.36 h). Its `speaker_id` column is empty. The current exporter writes six
columns without that field. Each `audio_ref` identifies the source revision,
country configuration, split and recording key.

[Per-recording results and run metadata](../results/metrics/)
