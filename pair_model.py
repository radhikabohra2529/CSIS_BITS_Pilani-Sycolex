"""
pair_model.py  —  primary system: direct pair-level sycophancy classifier

Why pair-level (not the two-stage stance composition)
-----------------------------------------------------
The oracle test showed that reproducing the keyword `agreement` heuristic caps
below always-1 (F1 0.63 vs 0.67 on human labels). A model that reads BOTH
responses can find signal the heuristic misses, and can distinguish "agreed with
both" (sycophantic) from "disagreed with both" (contrarian) directly from text.

Input (middle-ground, per the design choice)
--------------------------------------------
    [CLS] asserted petitioner: <resp_A head+tail> [SEP]
          asserted respondent: <resp_B head+tail> [SEP]

Each response is prefixed with the side ITS prompt asserted (petitioner /
respondent / unknown) — never with "winner/loser" and never with the prompt
text, so no outcome leak and no prompt-style dependence. Order-swap augmentation
makes the model invariant to which response is shown first.

Validation
----------
    default : all cases carrying a human label become the val set (honest,
              human-graded); every other labelled case trains on LLM labels.
              No case appears in both splits.
    --holdout-model NAME : leave-one-model-out instead (cross-model
              generalization; validates on that model's rows).

Subcommands
-----------
    train     -> ./pair_model   (also writes threshold.json from a val F1 sweep)
    evaluate  -> reload and report F1 vs human labels, broken down
    predict   -> submission.csv from any pairs jsonl

Install
-------
    pip install "transformers>=4.44" accelerate sentencepiece scikit-learn
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_recall_fscore_support

SIDE_TEXT = {1: "asserted petitioner:", 0: "asserted respondent:",
             None: "asserted unknown:"}


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ----------------------------------------------------- input construction

class PairEncoder:
    def __init__(self, tokenizer, max_len=512, head_frac=0.5):
        self.tok = tokenizer
        self.max_len = max_len
        self.head_frac = head_frac
        self.sep = tokenizer.sep_token_id
        self.cls = tokenizer.cls_token_id

    def _ids(self, text, add_special=False):
        return self.tok(text or "", add_special_tokens=add_special,
                        truncation=False)["input_ids"]

    def _headtail(self, ids, budget):
        if len(ids) <= budget:
            return ids
        h = int(budget * self.head_frac)
        return ids[:h] + ids[-(budget - h):]

    def build(self, resp_a, asr_a, resp_b, asr_b):
        mark_a = self._ids(SIDE_TEXT[asr_a])
        mark_b = self._ids(SIDE_TEXT[asr_b])
        # budget for the two response bodies: total - CLS - 2*SEP - markers
        body = self.max_len - 3 - len(mark_a) - len(mark_b)
        per = max(16, body // 2)
        ra = self._headtail(self._ids(resp_a), per)
        rb = self._headtail(self._ids(resp_b), per)
        return ([self.cls] + mark_a + ra + [self.sep]
                + mark_b + rb + [self.sep])


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, rows, encoder, train=False, label_key="label"):
        self.rows = rows
        self.enc = encoder
        self.train = train
        self.label_key = label_key

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        a = (r["true_response"], r["true_asserted_side"])
        b = (r["flip_response"], r["flip_asserted_side"])
        if self.train and random.random() < 0.5:      # order-swap aug
            a, b = b, a
        item = {"input_ids": self.enc.build(a[0], a[1], b[0], b[1])}
        if r.get(self.label_key) is not None:
            item["labels"] = int(r[self.label_key])
        return item


class Collator:
    def __init__(self, tok, global_attention=False):
        self.tok = tok
        self.global_attention = global_attention   # Longformer: CLS attends globally

    def __call__(self, batch):
        labels = [b.pop("labels") for b in batch] if "labels" in batch[0] else None
        enc = self.tok.pad([{"input_ids": b["input_ids"]} for b in batch],
                           padding=True, return_tensors="pt")
        if self.global_attention:
            gam = torch.zeros_like(enc["input_ids"])
            gam[:, 0] = 1                            # global attention on CLS
            enc["global_attention_mask"] = gam
        if labels is not None:
            enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc


# ------------------------------------------------------------- splitting

def _override(r):
    """Merged label: human verification wins where present, else LLM judge."""
    return int(r["label_human"]) if r.get("label_human") is not None \
        else int(r["label_llm"])


def make_splits(rows, holdout_model, seed, val_human_frac=0.4,
                val_jurisdiction=None, holdout_variant=None,
                train_on_all=False):
    """Train on the merged (human-override) label; hold out a validation set.

    holdout_variant : if set (e.g. 'P3a_explain_why'), train on all OTHER
        prompt variants and validate on the held-out one. This is the
        leave-one-variant-out (LOVO) probe: it estimates transfer to an unseen
        prompt style, the closest proxy for the real P3G-P3R test.
    """
    labelled = [r for r in rows if r.get("label_llm") is not None
                or r.get("label_human") is not None]

    if train_on_all:
        # FINAL model: no holdout. Train on every labelled row, human label
        # overriding LLM where present. Used only after all decisions (backbone,
        # threshold) are locked; there is no val set to report from here.
        for r in labelled:
            r["label"] = _override(r)
        return labelled, []

    if holdout_variant:
        train, val = [], []
        for r in labelled:
            r["label"] = _override(r)
            if r["variant"] == holdout_variant:
                val.append(r)
            else:
                train.append(r)
        return train, val

    if holdout_model:
        train, val = [], []
        for r in labelled:
            r["label"] = _override(r)
            (val if r["model"] == holdout_model else train).append(r)
        return train, val

    if val_jurisdiction:
        val_cases = {(r["jurisdiction"], r["case_id"]) for r in labelled
                     if r["jurisdiction"] == val_jurisdiction
                     and r.get("label_human") is not None}
        train, val = [], []
        for r in labelled:
            case = (r["jurisdiction"], r["case_id"])
            if case in val_cases:
                if r.get("label_human") is not None:
                    r["label"] = int(r["label_human"])
                    val.append(r)
            else:
                r["label"] = _override(r)
                train.append(r)
        return train, val

    human_cases = sorted({(r["jurisdiction"], r["case_id"])
                          for r in labelled if r.get("label_human") is not None})
    random.Random(seed).shuffle(human_cases)
    n_val = int(len(human_cases) * val_human_frac)
    val_cases = set(human_cases[:n_val])

    train, val = [], []
    for r in labelled:
        case = (r["jurisdiction"], r["case_id"])
        if case in val_cases:
            if r.get("label_human") is not None:
                r["label"] = int(r["label_human"])
                val.append(r)
        else:
            r["label"] = _override(r)
            train.append(r)
    return train, val


# --------------------------------------------------------------- metrics

def binary_metrics(labels, probs, threshold=0.5):
    preds = (np.asarray(probs) >= threshold).astype(int)
    P, R, F, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1, zero_division=0)
    acc = (preds == np.asarray(labels)).mean()
    return {"f1": F, "precision": P, "recall": R, "accuracy": acc}


def sweep_threshold(labels, probs):
    best_t, best_f = 0.5, -1
    for t in np.arange(0.20, 0.80, 0.02):
        f = f1_score(labels, (np.asarray(probs) >= t).astype(int),
                     pos_label=1, zero_division=0)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t, best_f


# --------------------------------------------------------------- training

def cmd_train(args):
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer, Trainer, TrainingArguments,
                              EarlyStoppingCallback)

    set_seed(args.seed)
    rows = read_jsonl(Path(args.data) / "pairs.jsonl")
    rows = [r for r in rows if r.get("true_response") and r.get("flip_response")]

    train_rows, val_rows = make_splits(rows, args.holdout_model, args.seed,
                                       args.val_human_frac, args.val_jurisdiction,
                                       args.holdout_variant, args.final)
    print(f"train={len(train_rows)}  val={len(val_rows)}"
          + (f"  holdout_variant={args.holdout_variant}" if args.holdout_variant
             else f"  holdout={args.holdout_model}" if args.holdout_model else
             "  (val = human-labelled cases)"))
    print("train label balance:", dict(Counter(r["label"] for r in train_rows)))
    print("val   label balance:", dict(Counter(r["label"] for r in val_rows)))

    tok = AutoTokenizer.from_pretrained(args.model)
    enc = PairEncoder(tok, args.max_len, args.head_frac)

    train_ds = PairDataset(train_rows, enc, train=True)
    val_ds = PairDataset(val_rows, enc, train=False)

    counts = Counter(r["label"] for r in train_rows)
    total = sum(counts.values())
    weights = torch.tensor([total / (2 * counts.get(c, 1)) for c in (0, 1)],
                           dtype=torch.float)
    print("class weights:", weights.tolist())

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2)
    is_longformer = "longformer" in args.model.lower()
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()

    def compute_metrics(p):
        probs = torch.softmax(torch.tensor(p.predictions), -1)[:, 1].numpy()
        return binary_metrics(p.label_ids, probs, 0.5)

    focal_gamma = args.focal_gamma

    class WTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            logits = out.logits.float()
            w = weights.to(logits.device)
            if focal_gamma > 0:
                # focal loss: down-weight easy examples, focus on hard ones
                ce = F.cross_entropy(logits, labels, weight=w, reduction="none")
                pt = torch.exp(-F.cross_entropy(logits, labels, reduction="none"))
                loss = ((1 - pt) ** focal_gamma * ce).mean()
            else:
                loss = F.cross_entropy(logits, labels, weight=w)
            return (loss, out) if return_outputs else loss

    has_val = len(val_rows) > 0
    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        weight_decay=0.01,
        fp16=(args.precision == "fp16"),
        bf16=(args.precision == "bf16"),
        eval_strategy="epoch" if has_val else "no",
        save_strategy="epoch" if has_val else "no",
        logging_steps=50,
        load_best_model_at_end=has_val,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        dataloader_num_workers=0,
        seed=args.seed,
    )

    trainer = WTrainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=val_ds if has_val else None,
        data_collator=Collator(tok, global_attention=is_longformer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        if has_val else [],
    )
    trainer.train()

    if not has_val:
        # FINAL model: no held-out set to tune on. Save with the threshold
        # locked earlier (passed via --final-threshold).
        trainer.save_model(args.out)
        tok.save_pretrained(args.out)
        (Path(args.out) / "infer_config.json").write_text(json.dumps(
            {"max_len": args.max_len, "head_frac": args.head_frac,
             "threshold": args.final_threshold}))
        print(f"\n[FINAL model] trained on all {len(train_rows)} labelled rows, "
              f"no holdout.")
        print(f"saved to {args.out} (threshold locked at {args.final_threshold})")
        return

    pred = trainer.predict(val_ds)
    probs = torch.softmax(torch.tensor(pred.predictions), -1)[:, 1].numpy()
    labels = pred.label_ids

    base = binary_metrics(labels, probs, 0.5)
    t, tf = sweep_threshold(labels, probs)
    tuned = binary_metrics(labels, probs, t)
    print(f"\nval vs HUMAN @0.5      : F1={base['f1']:.4f} P={base['precision']:.4f} "
          f"R={base['recall']:.4f} acc={base['accuracy']:.4f}")
    print(f"val vs HUMAN @{t:.2f} tuned: F1={tuned['f1']:.4f} P={tuned['precision']:.4f} "
          f"R={tuned['recall']:.4f} acc={tuned['accuracy']:.4f}")
    print(f"(always-1 reference on this val = "
          f"{f1_score(labels,[1]*len(labels),pos_label=1):.4f})")

    # same rows also carry LLM labels -> report both targets side by side
    llm = [r.get("label_llm") for r in val_rows]
    if all(x is not None for x in llm):
        llm = [int(x) for x in llm]
        m05 = binary_metrics(llm, probs, 0.5)
        mt = binary_metrics(llm, probs, t)
        print(f"val vs LLM   @0.5      : F1={m05['f1']:.4f}")
        print(f"val vs LLM   @{t:.2f} tuned: F1={mt['f1']:.4f}")
        agree = np.mean([h == l for h, l in zip(labels, llm)])
        print(f"(human/llm agreement on this val = {agree:.3f})")

    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    (Path(args.out) / "infer_config.json").write_text(json.dumps(
        {"max_len": args.max_len, "head_frac": args.head_frac,
         "threshold": t}))
    print(f"saved to {args.out} (threshold {t:.2f})")


# ----------------------------------------------------- eval / predict

def _load(model_dir):
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    cfg = json.loads((Path(model_dir) / "infer_config.json").read_text())
    return tok, model, cfg


@torch.no_grad()
def _predict_probs(rows, tok, model, cfg, batch=16):
    enc = PairEncoder(tok, cfg["max_len"], cfg["head_frac"])
    device = next(model.parameters()).device
    coll = Collator(tok)
    probs = []
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        feats = [{"input_ids": enc.build(
            r["true_response"], r["true_asserted_side"],
            r["flip_response"], r["flip_asserted_side"])} for r in chunk]
        b = coll(feats).to(device)
        p = torch.softmax(model(**b).logits, -1)[:, 1]
        probs.extend(p.cpu().tolist())
    return probs


def cmd_evaluate(args):
    tok, model, cfg = _load(args.model_dir)
    rows = read_jsonl(Path(args.data) / "eval_gold.jsonl")
    rows = [r for r in rows if r.get("label_human") is not None]
    probs = _predict_probs(rows, tok, model, cfg)
    labels = [int(r["label_human"]) for r in rows]
    t = cfg.get("threshold", 0.5)

    for name, thr in [("@0.5", 0.5), (f"@{t:.2f}", t)]:
        m = binary_metrics(labels, probs, thr)
        print(f"  {name:8s} F1={m['f1']:.4f} P={m['precision']:.4f} "
              f"R={m['recall']:.4f} acc={m['accuracy']:.4f}")
    print(f"  always-1 F1={f1_score(labels,[1]*len(labels),pos_label=1):.4f}\n")

    preds = (np.asarray(probs) >= t).astype(int)
    for field in ("jurisdiction", "variant"):
        print(f"-- F1 by {field} --")
        by = {}
        for r, p, g in zip(rows, preds, labels):
            by.setdefault(r[field], [[], []])
            by[r[field]][0].append(p); by[r[field]][1].append(g)
        for k in sorted(by):
            pp, gg = by[k]
            print(f"   {k:24s} n={len(gg):4d} "
                  f"F1={f1_score(gg,pp,pos_label=1,zero_division=0):.4f}")
        print()


def cmd_predict(args):
    tok, model, cfg = _load(args.model_dir)
    rows = read_jsonl(args.input)
    probs = _predict_probs(rows, tok, model, cfg)
    t = args.threshold if args.threshold is not None else cfg.get("threshold", 0.5)
    preds = (np.asarray(probs) >= t).astype(int)

    import csv
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["uid", "label"])
        for r, p in zip(rows, preds):
            w.writerow([r.get("uid", ""), int(p)])
    print(f"wrote {args.out}  n={len(preds)}  "
          f"pos_rate={preds.mean():.3f}  threshold={t:.2f}")

    if all(r.get("label_human") is not None for r in rows):
        g = [int(r["label_human"]) for r in rows]
        print(f"F1 vs provided labels: {f1_score(g, preds, pos_label=1):.4f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--data", default="./data")
    t.add_argument("--out", default="./pair_model")
    t.add_argument("--model", default="microsoft/deberta-v3-base")
    t.add_argument("--max-len", type=int, default=512)
    t.add_argument("--head-frac", type=float, default=0.5)
    t.add_argument("--epochs", type=float, default=3)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--grad-accum", type=int, default=2)
    t.add_argument("--lr", type=float, default=2e-5)
    t.add_argument("--warmup-ratio", type=float, default=0.06)
    t.add_argument("--max-grad-norm", type=float, default=1.0)
    t.add_argument("--focal-gamma", type=float, default=0.0,
                   help="focal loss focusing parameter; 0 = plain weighted CE, "
                        "2.0 is a common value that emphasises hard examples")
    t.add_argument("--final", action="store_true",
                   help="FINAL model: train on ALL labelled data with no "
                        "holdout (merges val into train). Use only after "
                        "decisions and threshold are locked.")
    t.add_argument("--final-threshold", type=float, default=0.5,
                   help="threshold to save into the final model's config "
                        "(no val set exists to tune it, so it must be given)")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--val-human-frac", type=float, default=0.4,
                   help="fraction of human-labelled cases held out for val; "
                        "the rest train with the human-override label")
    t.add_argument("--val-jurisdiction", default=None,
                   help="hold out one jurisdiction's human cases for val to "
                        "match a single-jurisdiction test set, e.g. "
                        "india_consumer_post2025")
    t.add_argument("--holdout-model", default=None)
    t.add_argument("--holdout-variant", default=None,
                   help="leave-one-variant-out: train on all other prompt "
                        "variants, validate on this one (e.g. P3a_explain_why)")
    t.add_argument("--precision", choices=["fp32", "bf16", "fp16"],
                   default="fp32",
                   help="fp32 is stable for DeBERTa-v3; bf16/fp16 can NaN")
    t.add_argument("--grad-checkpont", dest="grad_checkpoint",
                   action="store_true")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate")
    e.add_argument("--data", default="./data")
    e.add_argument("--model-dir", default="./pair_model")
    e.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("predict")
    p.add_argument("--input", required=True)
    p.add_argument("--model-dir", default="./pair_model")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--threshold", type=float, default=None)
    p.set_defaults(func=cmd_predict)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()