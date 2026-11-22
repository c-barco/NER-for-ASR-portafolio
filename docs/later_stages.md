# Later stages are intentionally inactive

This repository stops at dataset preparation and no-context baseline ASR.
There is no implemented NER, context-conditioned transcription, fine-tuning,
or comparison experiment.

The Granite transcriber retains the public
`transcribe_batch(paths, prompt=None)` signature so a later authorised study can
use the model card's keyword syntax. That future work must append
` Keywords: <kw1>, <kw2>` to Granite's fixed transcription request and must
report where the keywords came from. It is not part of the current command,
notebook, output parquet, or reported result.

The Step 2 artifact preserves `sample_id`, source split, transcript, and
speaker-null metadata so that a future protocol can document disjoint training
and test sets. It must not be treated as a completed later-stage experiment.
