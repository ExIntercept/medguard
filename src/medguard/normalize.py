"""Layer 0: canonicalisation and obfuscation features.

Undoes cheap evasions (zero-width characters, homoglyphs, leetspeak, base64 payloads)
so later detectors see what the model would see, and turns the evasion attempt itself
into a feature. Non-ASCII text is deliberately NOT treated as suspicious: Hindi, Kannada
and other Indic-script input is normal for this deployment.
"""
from __future__ import annotations

import base64
import binascii
import re
import string
import unicodedata
from dataclasses import dataclass, field

_ZW = "​‌‍‎‏⁠⁡⁢⁣⁤﻿­"
ZERO_WIDTH = dict.fromkeys(map(ord, _ZW), None)

# Latin look-alikes from Cyrillic / Greek (small, high-value subset)
CONFUSABLES = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i",
    "ј": "j", "ѕ": "s", "ԁ": "d", "ɡ": "g", "Α": "A", "Β": "B", "Ε": "E", "Ι": "I",
    "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Χ": "X", "Υ": "Y",
    "Ζ": "Z", "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "Х": "X",
}
_CONF_T = str.maketrans(CONFUSABLES)
_LEET_T = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
_B64 = re.compile(r"(?:[A-Za-z0-9+/]{4}){5,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?")
_LEET_WORD = re.compile(r"\b(?=\w*[A-Za-z])(?=\w*[013457@$])[\w@$]{3,}\b")
_PUNCT = set(string.punctuation)


@dataclass
class Normalized:
    original: str
    text: str
    views: list = field(default_factory=list)   # extra decoded views to scan
    features: dict = field(default_factory=dict)

    @property
    def obfuscation_score(self) -> float:
        f = self.features
        parts = [
            0.5 if f.get("zero_width", 0) > 0 else 0.0,
            0.5 if f.get("confusables", 0) > 0 else 0.0,
            0.6 if f.get("b64_payloads", 0) > 0 else 0.0,
            0.3 if f.get("leet_ratio", 0) > 0.3 else 0.0,
            0.5 if f.get("symbol_tail", 0) > 0.35 else 0.0,
        ]
        keep = 1.0
        for p in parts:
            keep *= 1.0 - p
        return round(1.0 - keep, 4)


def _decode_b64(blob: str) -> str | None:
    try:
        raw = base64.b64decode(blob, validate=True)
        txt = raw.decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None
    if len(txt) >= 8 and sum(c.isprintable() or c.isspace() for c in txt) / len(txt) > 0.9:
        return txt
    return None


def normalize(text: str) -> Normalized:
    original = text or ""
    t = unicodedata.normalize("NFKC", original)
    zw = sum(1 for c in t if ord(c) in ZERO_WIDTH)
    t = t.translate(ZERO_WIDTH)
    conf = sum(1 for c in t if c in CONFUSABLES)
    t = t.translate(_CONF_T)
    t = re.sub(r"[ \t]+", " ", t).strip()

    views = []
    decoded = [d for d in (_decode_b64(m) for m in _B64.findall(t)) if d]
    views.extend(decoded)

    words = re.findall(r"\S+", t)
    leet_words = _LEET_WORD.findall(t)
    leet_ratio = len(leet_words) / max(1, len(words))
    if leet_words:
        views.append(t.translate(_LEET_T))

    tail = t[-100:]
    symbol_tail = sum(1 for c in tail if c in _PUNCT) / max(1, len(tail))

    feats = {
        "zero_width": zw,
        "confusables": conf,
        "b64_payloads": len(decoded),
        "leet_ratio": round(leet_ratio, 3),
        "symbol_tail": round(symbol_tail, 3),
    }
    return Normalized(original=original, text=t, views=views, features=feats)


def chunk_text(text: str, size: int = 1200, overlap: int = 200) -> list[str]:
    """Character windows so long inputs are scanned end-to-end (classifiers truncate at 512 tokens)."""
    if len(text) <= size:
        return [text]
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out
