"""Layer 6: output guard. Runs on every LLM response before the user sees it."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .safety import EMERGENCY_BANNER

DISCLAIMER = ("\n\n---\n*Decision support only, not a diagnosis. Findings must be confirmed by a "
              "qualified clinician.*")
_CERTAIN = re.compile(r"\byou\s+(definitely|certainly|surely|clearly)\s+have\b|\b100\s*%\s*(sure|certain)\b"
                      r"|\bguaranteed\b|\bthere\s+is\s+no\s+doubt\b", re.I)
_DOSE = re.compile(r"\b\d+(\.\d+)?\s?(mg|mcg|µg|g|ml|mL|units|IU)\b(\s*/\s*(kg|day|dose))?", re.I)
_PII = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),
    "PHONE": re.compile(r"(?<!\d)(\+91[\s-]?)?[6-9]\d{9}(?!\d)"),
    "AADHAAR": re.compile(r"(?<!\d)\d{4}\s?\d{4}\s?\d{4}(?!\d)"),
}


def _ngrams(text: str, n: int = 8) -> set:
    w = re.findall(r"\w+", text.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


@dataclass
class OutputVerdict:
    text: str
    blocked: bool = False
    flags: list = field(default_factory=list)


class OutputGuard:
    def __init__(self, system_prompt: str, canary: str):
        self.canary = canary
        self.sys_ngrams = _ngrams(system_prompt)

    def check(self, text: str, red_flags: list[str] | None = None,
              quarantined: list[str] | None = None) -> OutputVerdict:
        flags = []
        # 1. canary / system-prompt leakage -> hard block
        if self.canary in text or self.canary.split("-")[-1] in text:
            return OutputVerdict("Response withheld: it exposed internal configuration.", True, ["canary_leak"])
        if len(_ngrams(text) & self.sys_ngrams) >= 2:
            return OutputVerdict("Response withheld: it reproduced internal instructions.", True, ["system_prompt_leak"])
        # 2. did the model act on quarantined image/document instructions?
        low = text.lower()
        for q in quarantined or []:
            frag = q.lower().strip()[:60]
            if len(frag) > 15 and frag in low:
                flags.append("echoed_quarantined_instruction")
        if re.search(r"\b(scan|image|x-?ray)\s+(is|appears)\s+(completely\s+)?normal\b.{0,80}\bdischarge\b", low):
            flags.append("possible_injection_compliance")
        # 3. overconfidence -> soften
        if _CERTAIN.search(text):
            text = _CERTAIN.sub("the findings are consistent with", text)
            flags.append("overconfidence_softened")
        # 4. dosing -> annotate
        if _DOSE.search(text):
            flags.append("dose_mentioned")
            text += "\n\n> Any dose above is illustrative and must be confirmed by the treating clinician."
        # 5. PII redaction
        for tag, rx in _PII.items():
            if rx.search(text):
                text = rx.sub(f"[{tag}]", text)
                flags.append(f"pii_{tag.lower()}_redacted")
        if "possible_injection_compliance" in flags or "echoed_quarantined_instruction" in flags:
            text = ("> **Warning:** the uploaded image contained embedded instructions. They were "
                    "quarantined, but this answer may still be influenced. Review the image manually.\n\n") + text
        if red_flags:
            text = EMERGENCY_BANNER.format(flags=", ".join(red_flags)) + text
        return OutputVerdict(text + DISCLAIMER, False, flags)
