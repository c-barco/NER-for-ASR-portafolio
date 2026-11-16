# Later portfolio steps - NOT part of the Step 1-2 submission

Nothing here has been implemented or run. These notes only make sure the Step 2 design does not
block later work.

| Step | What | Hook prepared in Step 2 |
|---|---|---|
| 3 | Failure cases, especially names | Per-recording CSV keeps raw reference + prediction; worst items listed in the notebook |
| 4 | NER on text (spaCy, as in `04_named_entity_recognition.ipynb`) | none yet (spaCy is blocked on the laptop; works on Linux) |
| 5 | Re-evaluate with entity context | `transcribe_batch(..., prompt=...)` already accepts a prompt; the pipeline currently **refuses** `context_mode != none`. Output columns follow lab_06 (`*_without_context` now, `*_with_context` later, merged on `wav_path`). |
| 6 | Fine-tuning with context | `selection.check_disjoint()` for ID/audio/speaker overlap; lab_07 JSONL format (`audio`, `prompt`, `text`); HF docs: assistant turn `language <NAME><asr_text>...`, `processor_kwargs={"output_labels": True}` |
| 7 | Compare before/after, with/without | same manifest + same normalisation for every condition |
| 8 | Second dataset | same pipeline, different config |

**Oracle context warning (Step 5):** entities extracted from the *reference* transcript give the model
information it would not have in real use. Results with such context must be labelled
"reference-derived (oracle) context" and discussed separately from context that is independently
available (e.g. agenda, speaker lists, or entities from an earlier ASR pass).
