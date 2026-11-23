# Granite 4 Speech pilot results

These are **pilot** results, not the final full-baseline results. Granite 4
Speech was evaluated without entity or keyword context on 250 held-out
EuroSpeech `test` recordings per language.

| Language | Scored / failed | Audio (h) | Corpus WER / CER | Mean WER / CER | Median WER / CER | Real-time factor |
|---|---:|---:|---:|---:|---:|---:|
| English (`uk`) | 250 / 0 | 1.045 | 26.87% / 18.13% | 30.48% / 22.25% | 24.23% / 15.67% | 0.094 |
| German (`germany`) | 250 / 0 | 1.048 | 17.13% / 12.30% | 20.95% / 16.10% | 14.61% / 9.13% | 0.123 |

English resumed one previously saved prediction and generated the remaining
249; all 250 recordings were scored. German generated and scored all 250
recordings. On these separately selected pilot samples, German has lower WER
and CER than English; this is descriptive rather than a controlled
cross-language comparison.

## Step 2 parquet verification

| Check | Result |
|---|---|
| Artifact | `data/deliverable/step2_eurospeech_en_de.parquet` |
| Rows / language balance | 2,000 total: 1,000 EN and 1,000 DE |
| Source splits | `test` for both languages |
| Historical artifact check | the original embedded-audio parquet reloaded successfully and a WAV sample played |
| Audio duration | 8.359 h total; 10.02–20.00 s observed; median 15.05 s |
| Populated duration bins | 10–15 s: 987; 15–20 s: 1,013 |
| Unreadable selected items | 0 |

The entity-like text proxy was 50.5% overall (37.8% EN, 63.3% DE). It is not
an NER annotation or a ground-truth entity rate; German capitalization makes
it especially unsuitable for comparison.

> The checks above describe the original embedded-audio export. The approved
> compact `audio_ref` format must be regenerated and revalidated before it is
> submitted; it preserves the same selection rules but stores no waveform bytes.
