"""The input-side guardrail pipeline: L0 normalise -> safety -> L1-3 detectors -> L4 fusion
-> session risk -> action. Plus span-level quarantine for untrusted (tainted) segments."""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum

from .detectors import (Embedder, InjectionClassifier, ScopeDetector, TemplateSimilarity,
                        instruction_like, smoothed_injection)
from .fusion import RiskFusion, SessionRisk
from .normalize import Normalized, normalize
from .provenance import Segment, Source
from .safety import content_safety, red_flags

log = logging.getLogger("medguard")


class Action(str, Enum):
    ALLOW = "allow"
    HARDEN = "allow_hardened"
    BLOCK = "block"
    SAFE_MESSAGE = "safe_message"
    REDIRECT = "redirect_offscope"


@dataclass
class GuardConfig:
    use_injection: bool = True
    use_template: bool = True
    use_scope: bool = True
    use_smoothing: bool = True
    injection_ids: tuple = InjectionClassifier.DEFAULT_IDS
    embedder_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    template_index: str = "artifacts/jb_index.npz"
    fusion_weights: str = "artifacts/fusion.json"
    allow_below: float = 0.40
    block_above: float = 0.75
    session_block: float = 0.55
    offscope_redirect: float = 0.85
    span_threshold: dict = field(default_factory=lambda: {
        Source.IMAGE: 0.35, Source.RETRIEVED: 0.50, Source.MODEL: 0.60})
    smoothing_k: int = 6
    smoothing_q: float = 0.08
    hardening_mode: str = "delimit"  # or "datamark"


@dataclass
class InputVerdict:
    action: Action
    risk: float
    session_risk: float
    features: dict
    contributions: dict
    normalized: Normalized
    safety: str | None
    red_flags: list
    latency_ms: float
    notes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "action": self.action.value, "risk": round(self.risk, 4),
            "session_risk": round(self.session_risk, 4),
            "features": {k: round(float(v), 4) for k, v in self.features.items()},
            "contributions": self.contributions, "obfuscation_detail": self.normalized.features,
            "safety": self.safety, "red_flags": self.red_flags,
            "latency_ms": round(self.latency_ms, 1), "notes": self.notes,
        }


class GuardrailPipeline:
    def __init__(self, cfg: GuardConfig, inj=None, tmpl=None, scope=None, fusion=None):
        self.cfg = cfg
        self.inj, self.tmpl, self.scope = inj, tmpl, scope
        self.fusion = fusion or RiskFusion.load(cfg.fusion_weights)
        self.session = SessionRisk()
        self.notes = []
        if inj is None:
            self.notes.append("injection classifier unavailable: using regex proxy")
        if tmpl is None:
            self.notes.append("template index unavailable")
        if scope is None:
            self.notes.append("scope detector unavailable")

    @classmethod
    def from_config(cls, cfg: GuardConfig | None = None) -> "GuardrailPipeline":
        cfg = cfg or GuardConfig()
        inj = tmpl = scope = emb = None
        if cfg.use_injection:
            try:
                inj = InjectionClassifier(cfg.injection_ids)
                log.info("injection classifier: %s", inj.model_id)
            except Exception as e:
                log.warning("injection classifier not loaded: %s", e)
        if cfg.use_template or cfg.use_scope:
            try:
                emb = Embedder(cfg.embedder_id)
            except Exception as e:
                log.warning("embedder not loaded: %s", e)
        if emb is not None and cfg.use_template:
            try:
                tmpl = TemplateSimilarity(emb, cfg.template_index)
            except Exception as e:
                log.warning("template index not loaded (run scripts/guard_bench.py index): %s", e)
        if emb is not None and cfg.use_scope:
            scope = ScopeDetector(emb)
        return cls(cfg, inj, tmpl, scope)

    # ------------------------------------------------------------------ features
    def features(self, norm: Normalized) -> dict:
        views = [norm.text] + norm.views
        f = {k: 0.0 for k in ("inj", "inj_smooth_mean", "inj_smooth_std", "template_sim", "offscope")}
        f["instr_regex"] = max(instruction_like(v) for v in views)
        f["obfuscation"] = norm.obfuscation_score
        if self.inj is not None:
            f["inj"] = max(self.inj.score_text(v) for v in views)
            if self.cfg.use_smoothing:
                f["inj_smooth_mean"], f["inj_smooth_std"] = smoothed_injection(
                    self.inj, norm.text, self.cfg.smoothing_k, self.cfg.smoothing_q)
        else:  # degraded mode: the regex stands in for the classifier
            f["inj"] = 0.9 * f["instr_regex"]
        if self.tmpl is not None:
            f["template_sim"] = max(self.tmpl.score(v) for v in views)
        if self.scope is not None:
            f["offscope"] = self.scope.offscope(norm.text)
        return f

    # ----------------------------------------------------------------- user input
    def inspect_user(self, text: str, session_id: str = "default") -> InputVerdict:
        t0 = time.perf_counter()
        norm = normalize(text)
        safety = content_safety(norm.text) or next((content_safety(v) for v in norm.views if content_safety(v)), None)
        flags = red_flags(norm.text)
        feats = self.features(norm)
        risk = self.fusion.predict(feats)
        srisk = self.session.update(session_id, risk)
        notes = list(self.notes)

        if safety == "self_harm":
            action = Action.SAFE_MESSAGE
        elif safety in ("harm_others", "illicit"):
            action = Action.BLOCK
            notes.append(f"content safety: {safety}")
        elif risk >= self.cfg.block_above:
            action = Action.BLOCK
        elif srisk >= self.cfg.session_block:
            action = Action.BLOCK
            notes.append("blocked on accumulated session risk (multi-turn)")
        elif risk >= self.cfg.allow_below:
            action = Action.HARDEN
        elif feats["offscope"] >= self.cfg.offscope_redirect:
            action = Action.REDIRECT
        else:
            action = Action.ALLOW
        return InputVerdict(action, risk, srisk, feats, self.fusion.contributions(feats), norm, safety,
                            flags, (time.perf_counter() - t0) * 1000, notes)

    # ------------------------------------------------------------ tainted segments
    def inspect_untrusted(self, seg: Segment) -> tuple[Segment, list[dict]]:
        """Span-level scan of image-derived / retrieved text. Instruction-like sentences are
        replaced with a placeholder; the rest of the evidence is kept, so an injected sentence
        on an X-ray does not kill the whole diagnosis."""
        if seg.may_instruct or not seg.text.strip():
            return seg, []
        thr = self.cfg.span_threshold.get(seg.source, 0.5)
        sentences = [s for s in re.split(r"(?<=[.!?\n])\s+", normalize(seg.text).text) if s.strip()]
        kept, report, quarantined = [], [], []
        for s in sentences:
            sn = normalize(s)
            views = [sn.text] + sn.views
            sc_regex = max(instruction_like(v) for v in views)
            sc_inj = max(self.inj.score_text(v) for v in views) if self.inj else 0.9 * sc_regex
            sc_t = 0.0
            if self.tmpl is not None:
                sim = max(self.tmpl.score(v) for v in views)
                sc_t = min(1.0, max(0.0, (sim - 0.55) / 0.35))
            score = max(sc_inj, sc_t, 0.9 * sc_regex)
            hit = score >= thr
            report.append({"span": s[:120], "score": round(score, 3), "quarantined": hit})
            if hit:
                quarantined.append(s)
                kept.append(f"[QUARANTINED: instruction-like text from {seg.source.value}]")
            else:
                kept.append(s)
        return Segment(" ".join(kept), seg.source, dict(seg.meta), quarantined), report
