"""Build, fit and evaluate the guardrail stack with explicit, disjoint splits.

    python scripts/guard_bench.py index   # jailbreak template index  <- jackhhao TRAIN, half A
    python scripts/guard_bench.py fit     # fusion weights            <- jackhhao TRAIN half B + TRAIN benign + MedMCQA TRAIN slice
    python scripts/guard_bench.py eval    # report                    <- jackhhao TEST + MedMCQA VALIDATION slice + data/med_attacks.jsonl
    python scripts/guard_bench.py all     # all three in order

Half A and half B are disjoint so template_sim is never a self-match during fitting.
Nothing from the evaluation sets is used in index or fit.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from medguard.fusion import FEATURES, RiskFusion  # noqa: E402
from medguard.normalize import normalize  # noqa: E402
from medguard.pipeline import GuardConfig, GuardrailPipeline  # noqa: E402

ART = os.path.join(ROOT, "artifacts")
JB = "jackhhao/jailbreak-classification"
MCQ = "openlifescienceai/medmcqa"


def jb_split(split):
    from datasets import load_dataset
    ds = load_dataset(JB, split=split)
    return [(r["prompt"], 1 if r["type"] == "jailbreak" else 0) for r in ds]


def mcq_questions(split, n, seed, offset=0):
    from datasets import load_dataset
    ds = load_dataset(MCQ, split=split)
    idx = list(range(offset, len(ds)))
    random.Random(seed).shuffle(idx)
    return [(ds[i]["question"], 0) for i in idx[:n]]


def cmd_index(a):
    from medguard.detectors import Embedder
    rows = [t for t, y in jb_split("train") if y == 1]
    random.Random(0).shuffle(rows)
    half = len(rows) // 2
    A, B = rows[:half], rows[half:]
    emb = Embedder(GuardConfig().embedder_id)
    E = emb.encode(A)
    os.makedirs(ART, exist_ok=True)
    np.savez_compressed(os.path.join(ART, "jb_index.npz"), emb=E.astype(np.float16))
    json.dump({"fusion_half": B}, open(os.path.join(ART, "fusion_split.json"), "w"))
    print(f"index: {len(A)} templates | held for fusion fit: {len(B)}")


def featurise(guard, rows):
    X, y, t0 = [], [], time.perf_counter()
    for i, (t, lab) in enumerate(rows):
        X.append(guard.features(normalize(t)))
        y.append(lab)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(rows)}  {(time.perf_counter() - t0) / (i + 1) * 1000:.0f} ms/item")
    return X, y


def cmd_fit(a):
    B = json.load(open(os.path.join(ART, "fusion_split.json")))["fusion_half"]
    benign = [(t, 0) for t, y in jb_split("train") if y == 0]
    med = mcq_questions("train", a.n_med, seed=1, offset=100_000)  # question text only, as benign medical negatives
    rows = [(t, 1) for t in B] + benign + med
    print(f"fit set: {len(B)} jailbreak | {len(benign)} generic benign | {len(med)} medical benign")
    guard = GuardrailPipeline.from_config(GuardConfig(template_index=os.path.join(ART, "jb_index.npz"), fusion_weights=""))
    X, y = featurise(guard, rows)
    json.dump({"X": X, "y": y}, open(os.path.join(ART, "fit_features.json"), "w"))
    f = RiskFusion.fit(X, y, os.path.join(ART, "fusion.json"))
    print("weights:", json.dumps({k: round(v, 2) for k, v in f.w.items()}), "bias", round(f.b, 2))


def tpr_at_fpr(y, s, fpr=0.05):
    from sklearn.metrics import roc_curve
    f, t, _ = roc_curve(y, s)
    return float(t[f <= fpr].max()) if (f <= fpr).any() else 0.0


def cmd_eval(a):
    from sklearn.metrics import roc_auc_score
    test = jb_split("test")
    med = mcq_questions("validation", a.n_med, seed=2)
    custom = [(json.loads(l)["text"], json.loads(l)["label"]) for l in open(os.path.join(ROOT, "data", "med_attacks.jsonl"))]
    sets = {"jackhhao_test": test, "medmcqa_val_benign": med, "med_attacks_custom": custom}
    guard = GuardrailPipeline.from_config(GuardConfig(template_index=os.path.join(ART, "jb_index.npz"),
                                                       fusion_weights=os.path.join(ART, "fusion.json")))
    fit = json.load(open(os.path.join(ART, "fit_features.json")))

    allX, ally, tags = [], [], []
    for name, rows in sets.items():
        print("featurising", name, len(rows))
        X, y = featurise(guard, rows)
        allX += X
        ally += y
        tags += [name] * len(X)
    y = np.array(ally)
    out = ["| signal | AUROC | TPR@FPR5% |", "|---|---|---|"]
    for k in FEATURES:
        s = np.array([x[k] for x in allX])
        out.append(f"| {k} | {roc_auc_score(y, s):.3f} | {tpr_at_fpr(y, s):.3f} |")
    fused = np.array([guard.fusion.predict(x) for x in allX])
    out.append(f"| **fused (all)** | **{roc_auc_score(y, fused):.3f}** | **{tpr_at_fpr(y, fused):.3f}** |")

    out += ["", "Leave-one-signal-out (refit on fit set, scored on eval set):", "",
            "| dropped | AUROC | delta |", "|---|---|---|"]
    base = roc_auc_score(y, fused)
    for k in FEATURES:
        w = RiskFusion.fit([{kk: v for kk, v in r.items() if kk != k} for r in fit["X"]], fit["y"],
                           os.path.join(ART, "_tmp.json"))
        s = np.array([w.predict({kk: v for kk, v in r.items() if kk != k}) for r in allX])
        auc = roc_auc_score(y, s)
        out.append(f"| {k} | {auc:.3f} | {auc - base:+.3f} |")
    os.remove(os.path.join(ART, "_tmp.json"))

    cfg = guard.cfg
    out += ["", "Operating point (default thresholds):", "",
            "| set | n | attacks flagged (>=harden) | attacks blocked | benign flagged | benign blocked |",
            "|---|---|---|---|---|---|"]
    for name in sets:
        m = np.array([t == name for t in tags])
        ya, sa = y[m], fused[m]
        def r(cond, lab):
            d = (ya == lab).sum()
            return f"{(cond & (ya == lab)).sum()}/{d}" if d else "-"
        out.append(f"| {name} | {m.sum()} | {r(sa >= cfg.allow_below, 1)} | {r(sa >= cfg.block_above, 1)} | "
                   f"{r(sa >= cfg.allow_below, 0)} | {r(sa >= cfg.block_above, 0)} |")
    report = "\n".join(out)
    print(report)
    open(os.path.join(ART, "guard_report.md"), "w").write(report + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["index", "fit", "eval", "all"])
    ap.add_argument("--n_med", type=int, default=300)
    a = ap.parse_args()
    steps = {"index": [cmd_index], "fit": [cmd_fit], "eval": [cmd_eval], "all": [cmd_index, cmd_fit, cmd_eval]}
    for step in steps[a.cmd]:
        step(a)
