"""Medical-domain content safety and clinical red flags.

Separate from injection detection: a perfectly 'benign' prompt can still be dangerous
(self-harm, harming someone else), and some ordinary prompts describe emergencies that
must be escalated whatever the model says.
"""
from __future__ import annotations

import re

SAFETY_PATTERNS = {
    "self_harm": [
        r"\bkill\s+myself\b", r"\b(end|take)\s+my\s+(own\s+)?life\b", r"\bsuicid",
        r"\bwant\s+to\s+die\b", r"\bhow\s+many\s+.{0,30}\bto\s+(die|overdose)\b",
        r"\boverdose\s+on\s+purpose\b",
    ],
    "harm_others": [
        r"\bpoison\s+(someone|somebody|him|her|my\s+\w+)\b", r"\bundetectable\s+poison\b",
        r"\b(lethal|fatal|deadly)\s+dose\s+.{0,40}\b(someone|him|her|my\s+\w+|a\s+person)\b",
        r"\bkill\s+(someone|him|her|my\s+\w+)\b",
    ],
    "illicit": [
        r"\b(synthesi[sz]e|make|cook|extract)\s+.{0,20}\b(meth|fentanyl|heroin|mdma|cocaine)\b",
        r"\bfake\s+(prescription|medical\s+certificate)\b",
    ],
}
_SAFETY = {k: re.compile("|".join(v), re.I) for k, v in SAFETY_PATTERNS.items()}

RED_FLAGS = {
    "possible cardiac event": r"chest\s+(pain|tightness|pressure).{0,80}(sweat|radiat|left\s+arm|jaw|breathless|short(ness)?\s+of\s+breath)",
    "possible stroke": r"(face\s+droop|slurred\s+speech|sudden\s+(weakness|numbness)|one\s+side\s+(of\s+(my|the)\s+body\s+)?(is\s+)?(weak|numb))",
    "possible anaphylaxis": r"(throat\s+(swelling|closing)|tongue\s+swell|can'?t\s+breathe).{0,60}(allerg|sting|peanut|ate|after)",
    "severe bleeding": r"(heavy|severe|uncontrolled|won'?t\s+stop)\s+bleeding|vomiting\s+blood|coughing\s+(up\s+)?blood",
    "meningitis signs": r"(stiff\s+neck).{0,60}(fever|rash|light)|(fever).{0,60}stiff\s+neck",
    "infant fever": r"(baby|infant|newborn|\b[0-3]\s*(month|mo)s?\s*old).{0,60}fever",
}
_RED = {k: re.compile(v, re.I) for k, v in RED_FLAGS.items()}

SAFE_MESSAGE = (
    "I'm really sorry you're going through this. I can't help with that, but you don't have to "
    "face it alone. In India you can call **Tele-MANAS on 14416** (free, 24x7) or **112** in an "
    "emergency. If you are outside India, please contact your local emergency number. "
    "Would you like to talk about what's been happening?"
)
EMERGENCY_BANNER = (
    "> **Possible emergency ({flags}).** If this is happening now, call **112** (India) or go to "
    "the nearest emergency department. Do not wait for an online assessment.\n\n"
)


def content_safety(text: str) -> str | None:
    for cat, rx in _SAFETY.items():
        if rx.search(text or ""):
            return cat
    return None


def red_flags(text: str) -> list[str]:
    return [k for k, rx in _RED.items() if rx.search(text or "")]
