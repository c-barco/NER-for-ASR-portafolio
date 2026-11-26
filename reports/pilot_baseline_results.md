# Dataset and baseline

## Dataset selection

Multilingual LibriSpeech contains read audiobooks. VoxPopuli contains European
Parliament speech. Both appear in IBM's published Granite 4 ASR training data.
We chose EuroSpeech for parliamentary speech and its UK and German test splits.
IBM's training table omits EuroSpeech by name. Recording-level overlap is unknown.
[MLS](https://www.openslr.org/94/) · [VoxPopuli](https://github.com/facebookresearch/voxpopuli) · [EuroSpeech](https://huggingface.co/datasets/disco-eth/EuroSpeech) · [Granite 4](https://huggingface.co/ibm-granite/granite-4.0-1b-speech)

## Baseline results

The pilot evaluated 250 English and 250 German EuroSpeech test recordings
(2.09 h) with Granite 4.0 1B Speech and no keywords. The Step 2 Parquet contains
2,000 audio references and transcripts (8.36 h), including all 500 scored clips.

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
form. EuroSpeech references can differ from the audio because they come from
aligned parliamentary transcripts. The English clips span four sessions and
the German clips come from one session.

The 250 English baseline recordings are the input for the name-error analysis.

[Per-recording results and run metadata](../results/metrics/)
