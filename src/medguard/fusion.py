"""Layer 4: learned risk fusion + session-level (multi-turn) risk.

Instead of OR-ing detectors (high false-positive rate) we fit a logistic regression over
the detector signals on a labelled set, so weights reflect how informative each signal is
on *this* domain (medical benign text trips generic classifiers more than chit-chat does).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

FEATURES = [
    "inj",               # injection classifier, max over views/chunks
    "inj_smooth_mean",   # mean score over perturbed copies
    "inj_smooth_std",    # instability under perturbation
    "template_sim",      # max cosine to known jailbreak templates
    "obfuscation",       # normalisation-layer evasion score
    "offscope",          # out-of-domain probability
    "instr_regex",       # cheap instruction heuristic
]

DEFAULT_WEIGHTS = {
    "inj": 4.0, "inj_smooth_mean": 3.0, "inj_smooth_std": 2.0, "template_sim": 5.0,
    "obfuscation": 2.5, "offscope": 0.5, "instr_regex": 2.5,
}
DEFAULT_BIAS = -4.5


class RiskFusion:
    def __init__(self, weights: dict | None = None, bias: float = DEFAULT_BIAS, source: str = "default"):
        self.w = dict(DEFAULT_WEIGHTS if weights is None else weights)
        self.b = bias
        self.source = source

    @classmethod
    def load(cls, path: str | None) -> "RiskFusion":
        if path and os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            return cls(d["weights"], d["bias"], source=path)
        return cls()

    def predict(self, feats: dict) -> float:
        z = self.b + sum(self.w.get(k, 0.0) * float(feats.get(k, 0.0)) for k in FEATURES)
        return 1.0 / (1.0 + math.exp(-z))

    def contributions(self, feats: dict) -> dict:
        return {k: round(self.w.get(k, 0.0) * float(feats.get(k, 0.0)), 3) for k in FEATURES}

    @staticmethod
    def fit(rows: list[dict], labels: list[int], out_path: str, C: float = 1.0) -> "RiskFusion":
        from sklearn.linear_model import LogisticRegression
        X = [[r.get(k, 0.0) for k in FEATURES] for r in rows]
        lr = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(X, labels)
        w = {k: float(c) for k, c in zip(FEATURES, lr.coef_[0])}
        b = float(lr.intercept_[0])
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"weights": w, "bias": b, "features": FEATURES}, f, indent=2)
        return RiskFusion(w, b, source=out_path)


@dataclass
class SessionRisk:
    """EWMA of per-turn risk. Catches 'crescendo' attacks where each turn is mildly
    suspicious but the conversation as a whole is steering somewhere bad."""
    lam: float = 0.6
    state: dict = field(default_factory=dict)

    def update(self, session_id: str, risk: float) -> float:
        prev = self.state.get(session_id, 0.0)
        cur = self.lam * prev + (1 - self.lam) * risk
        self.state[session_id] = cur
        return cur

    def reset(self, session_id: str):
        self.state.pop(session_id, None)
