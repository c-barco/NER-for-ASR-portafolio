import math

import pytest

from ner_asr.metrics import corpus_scores, describe, score_utterance
from ner_asr.normalize import lecturer_normalize, normalize_text


def test_lab_slide_example_wer():
    # lab_06 slide 6: S=1, D=1, I=0, N=9 -> WER 0.222
    ref = "the cat sat on the mat in the sun"
    hyp = "the dog sat on the mat in sun"  # cat->dog substitution, one "the" deleted
    s = score_utterance(ref, hyp)
    assert s.ref_words == 9
    assert s.word_edits == 2
    assert s.wer == pytest.approx(2 / 9)


def test_cer_and_perfect():
    assert score_utterance("abcd", "abxd").cer == pytest.approx(0.25)
    assert score_utterance("hello world", "hello world").wer == 0


def test_empty_reference_is_undefined_and_empty_prediction_is_all_deletions():
    assert math.isnan(score_utterance("", "something").wer)
    assert score_utterance("two words", "").wer == 1.0


def test_insertions_can_push_wer_above_one():
    assert score_utterance("hi", "oh hi there you").wer == pytest.approx(3.0)


def test_corpus_vs_mean_per_recording_differ():
    short = score_utterance("a", "b")  # WER 1.0 on 1 word
    long = score_utterance("a b c d e f g h i j", "a b c d e f g h i j")  # 0.0 on 10 words
    assert corpus_scores([short, long])["corpus_wer"] == pytest.approx(1 / 11)
    assert describe([short.wer, long.wer], "wer")["wer_mean"] == pytest.approx(0.5)


def test_normalization():
    assert normalize_text("Mr. Kurz, the anti-poverty “strategy”!") == "mr kurz the anti poverty strategy"
    assert normalize_text("I'm  HERE") == "im here"
    assert normalize_text(None) == ""
    # documented difference to the lecturer's version
    assert lecturer_normalize("anti-poverty") == "antipoverty"
    # numbers are deliberately not verbalised (known limitation)
    assert normalize_text("120") == "120"
