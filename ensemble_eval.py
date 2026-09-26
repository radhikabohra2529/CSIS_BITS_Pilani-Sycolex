"""
ensemble_eval.py  —  average several trained pair models and evaluate.

Reuses the already-trained backbones (no retraining). For each pair it gets
each model's P(sycophantic), averages them, tunes a threshold, and reports F1
on the consumer human-val set — the same set the single models were measured on. Also reports each individual model for reference, and the
best pair (2-model) subset, so you can see whether averaging actually helps.

Usage:
    python ensemble_eval.py --data ./data \
        --models ./pair_model_roberta ./pair_model_xlmr ./pair_model_inlegal ./pair_model_muril

Notes:
 - Models can use different tokenizers/backbones; each is loaded with its own.
 - Averaging is of probabilities (soft voting), which is the right way to
   combine calibrated classifiers.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support

SIDE_TEXT = {1: "asserted petitioner:", 0: "asserted respondent:",
             None: "asserted unknown:"}


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_model(model_dir):
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    cfg_path = Path(model_dir) / "infer_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() \
        else {"max_len": 512, "head_frac": 0.5}
    return tok, model, cfg


def build_pair_ids(tok, cfg, ra, aa, rb, ab):
    cls, sep = tok.cls_token_id, tok.sep_token_id
    ml, hf = cfg["max_len"], cfg["head_frac"]

    def ids(t):
        return tok(t or "", add_special_tokens=False, truncation=False)["input_ids"]

    def ht(x, budget):
        if len(x) <= budget:
            return x
        h = int(budget * hf)
        return x[:h] + x[-(budget - h):]

    ma, mb = ids(SIDE_TEXT.get(aa, SIDE_TEXT[None])), ids(SIDE_TEXT.get(ab, SIDE_TEXT[None]))
    per = max(16, (ml - 3 - len(ma) - len(mb)) // 2)
    return [cls] + ma + ht(ids(ra), per) + [sep] + mb + ht(ids(rb), per) + [sep]


@torch.no_grad()
def model_probs(rows, tok, model, cfg, batch=16):
    device = next(model.parameters()).device
    probs = []
    for s in range(0, len(rows), batch):
        chunk = rows[s:s + batch]
        feats = [{"input_ids": build_pair_ids(
            tok, cfg, r["true_response"], r["true_asserted_side"],
            r["flip_response"], r["flip_asserted_side"])} for r in chunk]
        enc = tok.pad(feats, padding=True, return_tensors="pt").to(device)
        p = torch.softmax(model(**enc).logits, -1)[:, 1].cpu().tolist()
        probs.extend(p)
    return np.array(probs)


def sweep(scores, gold):
    best_t, best_f = 0.5, -1.0
    for t in np.arange(0.05, 0.95, 0.02):
        f = f1_score(gold, (scores >= t).astype(int), pos_label=1, zero_division=0)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t, best_f


def report(name, scores, gold):
    t, _ = sweep(scores, gold)
    preds = (scores >= t).astype(int)
    P, R, F1, _ = precision_recall_fscore_support(
        gold, preds, average="binary", pos_label=1, zero_division=0)
    acc = (preds == gold).mean()
    print(f"  {name:34s} F1={F1:.4f}  P={P:.4f} R={R:.4f} acc={acc:.4f}  (t={t:.2f})")
    return F1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--val-jurisdiction", default="india_consumer_post2025")
    args = ap.parse_args()

    rows = read_jsonl(Path(args.data) / "eval_gold.jsonl")
    rows = [r for r in rows if r.get("label_human") is not None]
    if args.val_jurisdiction:
        rows = [r for r in rows if r["jurisdiction"] == args.val_jurisdiction]
    gold = np.array([int(r["label_human"]) for r in rows])
    print(f"eval pairs (human-labelled): {len(rows)}")
    print(f"always-1 reference F1 = {f1_score(gold,[1]*len(gold),pos_label=1):.4f}\n")

    # per-model probabilities
    all_probs = {}
    print("individual models:")
    for md in args.models:
        name = Path(md).name
        tok, model, cfg = load_model(md)
        p = model_probs(rows, tok, model, cfg)
        all_probs[name] = p
        report(name, p, gold)
        del model
        torch.cuda.empty_cache()

    names = list(all_probs.keys())
    stack = np.stack([all_probs[n] for n in names], axis=0)

    print("\nfull ensemble (all models, equal weight):")
    full = stack.mean(axis=0)
    report("MEAN of " + "+".join(names), full, gold)

    # best 2- and 3-model subsets
    print("\nbest subsets (by tuned F1):")
    best = (None, -1)
    for k in (2, 3):
        for combo in itertools.combinations(range(len(names)), k):
            avg = stack[list(combo)].mean(axis=0)
            t, f = sweep(avg, gold)
            if f > best[1]:
                best = (combo, f)
    if best[0]:
        combo = best[0]
        avg = stack[list(combo)].mean(axis=0)
        report("BEST subset: " + "+".join(names[i] for i in combo), avg, gold)

    print("\nreference: best single model was RoBERTa at 0.76")


if __name__ == "__main__":
    main()