"""WER / CER computation with jiwer.

Two kinds of aggregate scores are reported and must not be confused:

* corpus-level (micro): total edits / total reference words (or characters) over
  all scored recordings. This is the standard ASR number.
* mean per-recording (macro): average of each recording's own WER. This is what
  the lecturer's notebook prints as "Mean WER"; short utterances weigh as much as
  long ones.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

import jiwer


@dataclass
class UtteranceScore:
    wer: float
    cer: float
    ref_words: int
    ref_chars: int
    word_edits: int  # S + D + I
    char_edits: int


def score_utterance(reference_norm: str, prediction_norm: str) -> UtteranceScore:
    """Score one already-normalised pair. Empty reference -> NaN (undefined)."""
    if not reference_norm:
        return UtteranceScore(math.nan, math.nan, 0, 0, 0, 0)
    hyp = prediction_norm if prediction_norm else ""
    w = jiwer.process_words(reference_norm, hyp)
    c = jiwer.process_characters(reference_norm, hyp)
    ref_words = w.hits + w.substitutions + w.deletions
    ref_chars = c.hits + c.substitutions + c.deletions
    word_edits = w.substitutions + w.deletions + w.insertions
    char_edits = c.substitutions + c.deletions + c.insertions
    return UtteranceScore(
        wer=word_edits / ref_words,
        cer=char_edits / ref_chars,
        ref_words=ref_words,
        ref_chars=ref_chars,
        word_edits=word_edits,
        char_edits=char_edits,
    )


def corpus_scores(scores: list[UtteranceScore]) -> dict:
    valid = [s for s in scores if s.ref_words > 0]
    if not valid:
        return {"corpus_wer": math.nan, "corpus_cer": math.nan}
    return {
        "corpus_wer": sum(s.word_edits for s in valid) / sum(s.ref_words for s in valid),
        "corpus_cer": sum(s.char_edits for s in valid) / sum(s.ref_chars for s in valid),
    }


def describe(values: list[float], prefix: str) -> dict:
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return {f"{prefix}_{k}": math.nan for k in ("mean", "median", "min", "max")}
    return {
        f"{prefix}_mean": statistics.fmean(vals),
        f"{prefix}_median": statistics.median(vals),
        f"{prefix}_min": min(vals),
        f"{prefix}_max": max(vals),
    }
