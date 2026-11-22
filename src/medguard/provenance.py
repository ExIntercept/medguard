"""Provenance (taint) tracking.

Every piece of text that reaches the LLM is a Segment tagged with where it came from.
Only SYSTEM and USER segments may carry instructions. Anything derived from an image,
a retrieved document or a previous model output is DATA: instruction-like spans inside
it are quarantined instead of being passed through.

Idea borrowed from information-flow control for LLM agents (CaMeL, FIDES); applied here
to the VLM -> LLM hand-off, where the VLM's transcription of text printed inside an
image is the injection channel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Source(str, Enum):
    SYSTEM = "system"
    USER = "user"
    IMAGE = "image_derived"
    RETRIEVED = "retrieved"
    MODEL = "model_output"


INSTRUCTION_CAPABLE = {Source.SYSTEM, Source.USER}


@dataclass
class Segment:
    text: str
    source: Source
    meta: dict = field(default_factory=dict)
    quarantined_spans: list = field(default_factory=list)

    @property
    def may_instruct(self) -> bool:
        return self.source in INSTRUCTION_CAPABLE

    @property
    def tainted(self) -> bool:
        return not self.may_instruct
