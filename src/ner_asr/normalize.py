"""Text normalisation applied identically to references and predictions.

Based on the lecturer's `normalize()` in mmai_06/03_evaluation_granite.ipynb
(remove punctuation, lowercase, collapse whitespace), with two documented changes:

1. Unicode-aware: NFKC normalisation and removal of *all* Unicode punctuation
   (the lecturer's version only strips ASCII `string.punctuation`, so curly quotes
   or en-dashes survived and counted as errors).
2. Hyphens, dashes and slashes become spaces instead of being deleted, so
   "anti-poverty" -> "anti poverty" (which avoids scoring punctuation as a word difference)
   rather than "antipoverty".

Apostrophes are deleted, as in the lecturer's version ("I'm" -> "im").
Numbers are NOT verbalised ("120" vs "one hundred and twenty" still counts as an
error). This is a known limitation and must be mentioned in the report.

Raw texts are always stored unchanged; normalisation happens only at scoring time.
"""

from __future__ import annotations

import re
import unicodedata

NORMALIZATION_VERSION = "v1-lecturer-based"

_SPACE_LIKE = re.compile(r"[-‐-―/]")
_WS = re.compile(r"\s+")


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", text).lower()
    text = _SPACE_LIKE.sub(" ", text)
    # Drop every Unicode punctuation character (category P*), incl. apostrophes.
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))
    return _WS.sub(" ", text).strip()


def lecturer_normalize(text: str) -> str:
    """Exact copy of the lecturer's normalize() for side-by-side comparison."""
    import string

    text = text.translate(str.maketrans("", "", string.punctuation)).lower().strip()
    return re.sub(r" +", " ", text).strip()
