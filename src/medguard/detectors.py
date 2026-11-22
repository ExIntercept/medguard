"""Layer 1-3 detectors. Each one is optional; the pipeline degrades gracefully if a model
cannot be downloaded (e.g. Prompt Guard 2 is gated on Hugging Face)."""
from __future__ import annotations

import random
import re
import statistics
import string

import numpy as np

from .normalize import chunk_text


def _device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------------------------------------------------------- L1
class InjectionClassifier:
    """Prompt-injection / jailbreak classifier.

    Default: Meta Llama Prompt Guard 2 86M (multilingual mDeBERTa, labels BENIGN/MALICIOUS,
    512-token window; gated, request access on HF). Fallback: ProtectAI DeBERTa-v3 v2 (ungated).
    """

    DEFAULT_IDS = (
        "meta-llama/Llama-Prompt-Guard-2-86M",
        "protectai/deberta-v3-base-prompt-injection-v2",
    )

    def __init__(self, model_ids=DEFAULT_IDS, device: str | None = None, max_len: int = 512):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        last = None
        for mid in model_ids:
            try:
                self.tok = AutoTokenizer.from_pretrained(mid)
                self.model = AutoModelForSequenceClassification.from_pretrained(mid)
                self.model_id = mid
                break
            except Exception as e:  # gated / offline
                last = e
        else:
            raise RuntimeError(f"no injection classifier could be loaded: {last}")
        self.device = device or _device()
        self.model.to(self.device).eval()
        self.max_len = max_len
        labels = {int(i): str(l).lower() for i, l in self.model.config.id2label.items()}
        mal = [i for i, l in labels.items() if any(k in l for k in ("inject", "jailbreak", "malicious", "unsafe"))]
        self.pos = mal[0] if mal else max(labels)

    def score(self, texts: list[str], batch_size: int = 16) -> list[float]:
        import torch
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok(texts[i:i + batch_size], truncation=True, max_length=self.max_len,
                           padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                p = self.model(**enc).logits.float().softmax(-1)[:, self.pos]
            out.extend(p.cpu().tolist())
        return out

    def score_text(self, text: str) -> float:
        if not text.strip():
            return 0.0
        return max(self.score(chunk_text(text)))


# ------------------------------------------------- L1b: SmoothLLM-style stability
def _perturb(text: str, q: float, rng: random.Random, mode: str) -> str:
    chars = list(text)
    if not chars:
        return text
    n = max(1, int(len(chars) * q))
    alphabet = string.ascii_letters + string.digits + string.punctuation + " "
    if mode == "swap":
        for i in rng.sample(range(len(chars)), min(n, len(chars))):
            chars[i] = rng.choice(alphabet)
    elif mode == "insert":
        for _ in range(n):
            chars.insert(rng.randrange(len(chars) + 1), rng.choice(alphabet))
    else:  # patch
        s = rng.randrange(max(1, len(chars) - n + 1))
        chars[s:s + n] = [rng.choice(alphabet) for _ in range(n)]
    return "".join(chars)


def smoothed_injection(clf: InjectionClassifier, text: str, k: int = 6, q: float = 0.08,
                       seed: int = 0) -> tuple[float, float]:
    """Score k randomly perturbed copies (SmoothLLM's swap/insert/patch) with the *cheap
    classifier* instead of the LLM. Returns (mean, std). Optimised adversarial suffixes
    are brittle to character noise, so their scores swing; natural text stays stable."""
    if not text.strip():
        return 0.0, 0.0
    rng = random.Random(seed)
    modes = ("swap", "insert", "patch")
    variants = [_perturb(text[:2000], q, rng, modes[i % 3]) for i in range(k)]
    s = clf.score(variants)
    return float(statistics.mean(s)), float(statistics.pstdev(s))


# --------------------------------------------------------------------------- L2
class Embedder:
    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.model_id = model_id
        self.m = SentenceTransformer(model_id, device=device or _device())

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.m.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=32)


class TemplateSimilarity:
    """Max cosine similarity to a bank of known jailbreak templates (index built ONLY from
    the training half of the jailbreak corpus; see scripts/guard_bench.py)."""

    def __init__(self, embedder: Embedder, index_path: str):
        self.emb = embedder
        self.E = np.load(index_path)["emb"].astype(np.float32)

    def score(self, text: str) -> float:
        if not text.strip():
            return 0.0
        v = self.emb.encode(chunk_text(text, 1000, 150))
        return float((v @ self.E.T).max())


# --------------------------------------------------------------------------- L3
MEDICAL_ANCHORS = [
    "I have had a fever and cough for three days", "what does this chest x-ray show",
    "differential diagnosis for chest pain", "my blood sugar is high after meals",
    "side effects of metformin", "rash on my arm that is itchy", "headache and blurred vision",
    "interpret these lab results", "child has diarrhoea and vomiting", "joint pain and swelling in knees",
    "what is the treatment for hypertension", "explain this MRI report", "skin lesion changing colour",
    "shortness of breath when walking", "pregnant and having cramps", "abdominal pain lower right side",
    "is this mole cancerous", "thyroid function test interpretation", "symptoms of dengue",
    "how is tuberculosis diagnosed", "a 45 year old man presents with", "pathology slide description",
    "retinal fundus image findings", "ECG shows ST elevation", "which antibiotic for UTI",
]
OFFSCOPE_ANCHORS = [
    "write python code for a web scraper", "who will win the cricket match", "stock tips for tomorrow",
    "write a poem about love", "how to hack a wifi password", "translate this sentence to French",
    "plan my trip to Goa", "explain quantum computing", "write my college essay",
    "best smartphone under 20000", "how to make a bomb", "summarise this movie plot",
    "help me with my tax return", "write a rap song", "solve this calculus problem",
]


class ScopeDetector:
    """Is the request inside the medical domain? Off-scope is redirected, not treated as an attack."""

    def __init__(self, embedder: Embedder):
        self.emb = embedder
        self.M = embedder.encode(MEDICAL_ANCHORS)
        self.O = embedder.encode(OFFSCOPE_ANCHORS)

    def offscope(self, text: str) -> float:
        if not text.strip():
            return 0.0
        v = self.emb.encode([text[:1000]])[0]
        m, o = float((self.M @ v).max()), float((self.O @ v).max())
        return float(1.0 / (1.0 + np.exp(-12.0 * (o - m))))


# ------------------------------------------------------ cheap instruction heuristic
_INSTR = [
    r"\bignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier|preceding)\s+(instructions?|rules?|prompts?|directions?)",
    r"\bdisregard\s+(all\s+|any\s+|the\s+)?(previous|prior|above|system)",
    r"\b(forget|override)\s+(all\s+|your\s+)?(instructions|rules|guidelines|programming)",
    r"\byou\s+are\s+now\s+(a|an|in)\b", r"\bdeveloper\s+mode\b", r"\bDAN\b", r"\bjailbreak",
    r"\b(reveal|print|show|repeat)\s+(your|the)\s+(system\s+prompt|instructions|hidden\s+rules)",
    r"\bnew\s+instructions?\s*:", r"\bsystem\s*(prompt|override)\s*:",
    r"\b(report|classify|diagnose|mark)\s+(this|the)\s+(scan|image|patient|x-?ray|case)\s+as\s+(normal|healthy|benign)",
    r"\brecommend\s+(immediate\s+)?discharge\b", r"\bno\s+(restrictions|filters|rules)\b",
    r"\bpichl[ea]\s+(saare|sabhi)\s+(nirdesh|instructions)",  # Hinglish "ignore previous instructions"
]
_INSTR_RE = re.compile("|".join(_INSTR), re.IGNORECASE)


def instruction_like(text: str) -> float:
    return 1.0 if _INSTR_RE.search(text or "") else 0.0
