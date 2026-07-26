"""Shared name folding that survives Nordic and Germanic characters.

NFKD plus ascii-ignore silently DELETES characters that do not decompose, so
"Øyvind" became "yvind" while a model writing "Oyvind" produced "oyvind". The
two never matched and a correct answer scored as wrong. Measured on the Brreg
gold set, where Ø, Æ and Å are common, this understated accuracy on its own.

Å happens to work by accident (it decomposes to A + ring), which is exactly
why the bug was easy to miss: spot-checking one Scandinavian name can pass.

Transliterations follow the conventions those languages use when writing
themselves in ASCII, so they match what a model reading a web page will emit.
"""
from __future__ import annotations

import re
import unicodedata

# Characters NFKD refuses to decompose, mapped the way the source language
# romanises them.
TRANSLITERATE = {
    "ø": "o", "Ø": "o",      # Norwegian, Danish
    "æ": "ae", "Æ": "ae",    # Norwegian, Danish, Icelandic
    "å": "a", "Å": "a",      # decomposes anyway, kept for clarity
    "ð": "d", "Ð": "d",      # Icelandic
    "þ": "th", "Þ": "th",    # Icelandic
    "ß": "ss",               # German
    "ł": "l", "Ł": "l",      # Polish
    "đ": "d", "Đ": "d",      # Croatian, Vietnamese
    "ı": "i", "İ": "i",      # Turkish
    "œ": "oe", "Œ": "oe",
}

TUSSENVOEGSELS = {"van", "der", "den", "de", "ter", "ten", "te", "du", "le", "la", "het"}


def fold(text: str) -> str:
    """Lowercase ASCII form that preserves every letter."""
    out = "".join(TRANSLITERATE.get(ch, ch) for ch in str(text or ""))
    return unicodedata.normalize("NFKD", out).encode("ascii", "ignore").decode().lower()


def surname(value: str) -> str:
    """Last significant name token.

    A tussenvoegsel is only a prefix when another name FOLLOWS it. Stripping
    them unconditionally turned "Quoc Trung Le" into "trung", because the
    Vietnamese surname Le collides with the Dutch preposition le.
    """
    tokens = [t for t in re.sub(r"[^a-z ]", " ", fold(value)).split() if len(t) > 1]
    if not tokens:
        return ""
    kept = [t for i, t in enumerate(tokens)
            if not (t in TUSSENVOEGSELS and i < len(tokens) - 1)]
    return kept[-1] if kept else tokens[-1]


def first_name(value: str) -> str:
    tokens = [t for t in re.sub(r"[^a-z ]", " ", fold(value)).split() if len(t) > 1]
    return tokens[0] if tokens else ""
