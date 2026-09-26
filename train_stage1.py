"""
train_stage1.py  —  Step 1 of the two-stage model.

Question this model answers, per response:
    "Did THIS response endorse (cave to) the side ITS prompt asserted?"
    -> P(caved) in [0,1]

This is a property of one response + which side its prompt pushed. It never
sees the other response or the raw prompt text, only a normalised marker
[asserted: petitioner/respondent] + the response body (head+tail truncated).
So it is invariant to prompt wording, response ordering, and the other side.

Labels (merged, human-override):
    target = human_agreement if present else LLM agreement
    'unclear' (None) rows are DROPPED from training (handled at composition).
    Human-labelled rows are up-weighted in the loss (--human-weight).

Validation:
    A slice of human-labelled responses, held out BY CASE (no leakage), is used
    to report how well Step 1 predicts HUMAN agreement. Optionally restricted to
    one jurisdiction (--val-jurisdiction) to match the final pair-level test.

Usage:
    python train_stage1.py --data ./data --out ./stage1_model \
        --val-jurisdiction india_consumer_post2025 --human-weight 3.0
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


# ---------------------------------------------------- label construction

def merged_target(r):
    """human agreement overrides LLM; None (unclear) stays None -> dropped."""
    h = r.get("human_agreement")
    if h is not None:
        return int(h), True          # (label, is_human)
    a = r.get("agreement")
    if a is not None:
        return int(a), False
    return None, False


# ------------------------------------------------------ head+tail encoder

class RespEncoder:
    def __init__(self, tok, max_len=512, head_frac=0.5):
        self.tok = tok
        self.max_len = max_len
        self.head_frac = head_frac
        self.cls = tok.cls_token_id
        self.sep = tok.sep_token_id

    def _ids(self, text):
        return self.tok(text or "", add_special_tokens=False,
                        truncation=False)["input_ids"]

    def build(self, response, asserted_side):
        mark = self._ids(SIDE_TEXT.get(asserted_side, SIDE_TEXT[None]))
        budget = self.max_len - 2 - len(mark)
        body = self._ids(response)
        if len(body) > budget:
            h = int(budget * self.head_frac)
            body = body[:h] + body[-(budget - h):]
        return [self.cls] + mark + body + [self.sep]


# --------------------------------------------------------------- dataset

class RespDataset(torch.utils.data.Dataset):
    def __init__(self, rows, enc):
        self.rows = rows
        self.enc = enc

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return {
            "input_ids": self.enc.build(r["response"], r["asserted_side"]),
            "labels": r["_label"],
            "weight": r["_weight"],
        }


class Collator:
    def __init__(self, tok):
        self.tok = tok

    def __call__(self, batch):
        labels = [b.pop("labels") for b in batch]
        weights = [b.pop("weight", 1.0) for b in batch]
        enc = self.tok.pad([{"input_ids": b["input_ids"]} for b in batch],
                           padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        enc["weight"] = torch.tensor(weights, dtype=torch.float)
        return enc


# --------------------------------------------------------------- splits

def build_splits(rows, val_jurisdiction, human_weight, seed):
    """Attach _label/_weight; hold out human-labelled cases for validation."""
    labelled = []
    for r in rows:
        lab, is_human = merged_target(r)
        if lab is None:
            continue                       # drop 'unclear'
        r["_label"] = lab
        r["_is_human"] = is_human
        r["_weight"] = human_weight if is_human else 1.0
        labelled.append(r)

    # candidate val = human-labelled responses (optionally one jurisdiction)
    def is_val_candidate(r):
        if not r["_is_human"]:
            return False
        if val_jurisdiction and r["jurisdiction"] != val_jurisdiction:
            return False
        return True

    val_cases = sorted({(r["jurisdiction"], r["pair_uid"].rsplit("|", 1)[0])
                        for r in labelled if is_val_candidate(r)})
    # hold out cases (not individual rows) to avoid leakage
    rng = random.Random(seed)
    rng.shuffle(val_cases)
    # use ~60% of eligible human cases for val (enough, keeps some for train)
    n_val = int(len(val_cases) * 0.6)
    val_set = set(val_cases[:n_val])

    train, val = [], []
    for r in labelled:
        case = (r["jurisdiction"], r["pair_uid"].rsplit("|", 1)[0])
        if case in val_set and is_val_candidate(r):
            val.append(r)
        elif case in val_set:
            continue                       # other rows of a val case: drop
        else:
            train.append(r)
    return train, val


# --------------------------------------------------------------- training

def cmd_train(args):
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer, Trainer, TrainingArguments,
                              EarlyStoppingCallback)
    set_seed(args.seed)
    rows = read_jsonl(Path(args.data) / "stance.jsonl")
    rows = [r for r in rows if r.get("response")]

    train_rows, val_rows = build_splits(
        rows, args.val_jurisdiction, args.human_weight, args.seed)
    print(f"train={len(train_rows)}  val(human)={len(val_rows)}")
    print("train label balance:", dict(Counter(r["_label"] for r in train_rows)))
    print("train human rows:", sum(r["_is_human"] for r in train_rows),
          "| weighted at", args.human_weight)
    print("val label balance:", dict(Counter(r["_label"] for r in val_rows)))

    tok = AutoTokenizer.from_pretrained(args.model)
    enc = RespEncoder(tok, args.max_len, args.head_frac)
    train_ds = RespDataset(train_rows, enc)
    val_ds = RespDataset(val_rows, enc)

    # class weights (inverse frequency) combine with per-example human weights
    counts = Counter(r["_label"] for r in train_rows)
    total = sum(counts.values())
    cls_w = torch.tensor([total / (2 * counts.get(c, 1)) for c in (0, 1)],
                         dtype=torch.float)
    print("class weights:", cls_w.tolist())

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2)
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()

    def compute_metrics(p):
        probs = torch.softmax(torch.tensor(p.predictions), -1)[:, 1].numpy()
        preds = (probs >= 0.5).astype(int)
        P, R, F1, _ = precision_recall_fscore_support(
            p.label_ids, preds, average="binary", pos_label=1, zero_division=0)
        acc = (preds == p.label_ids).mean()
        return {"f1": F1, "precision": P, "recall": R, "accuracy": acc}

    class WTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            weight = inputs.pop("weight")
            out = model(**inputs)
            # per-example loss, then apply class weight AND human up-weight
            per = F.cross_entropy(out.logits.float(), labels,
                                  weight=cls_w.to(out.logits.device),
                                  reduction="none")
            loss = (per * weight.to(per.device)).mean()
            return (loss, out) if return_outputs else loss

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.06,
        weight_decay=0.01,
        bf16=(args.precision == "bf16"),
        fp16=(args.precision == "fp16"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        seed=args.seed,
    )
    trainer = WTrainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=Collator(tok), compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()

    pred = trainer.predict(val_ds)
    probs = torch.softmax(torch.tensor(pred.predictions), -1)[:, 1].numpy()
    P, R, F1, _ = precision_recall_fscore_support(
        pred.label_ids, (probs >= 0.5).astype(int),
        average="binary", pos_label=1, zero_division=0)
    print(f"\n[Step-1 intrinsic] predicting HUMAN 'caved' on held-out cases:")
    print(f"  F1={F1:.4f} P={P:.4f} R={R:.4f} "
          f"acc={((probs>=0.5).astype(int)==pred.label_ids).mean():.4f}")

    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    (Path(args.out) / "enc_config.json").write_text(json.dumps(
        {"max_len": args.max_len, "head_frac": args.head_frac}))
    print(f"saved to {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default="./stage1_model")
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--head-frac", type=float, default=0.5)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--human-weight", type=float, default=3.0)
    ap.add_argument("--val-jurisdiction", default=None)
    ap.add_argument("--precision", choices=["fp32", "bf16", "fp16"],
                    default="bf16")
    ap.add_argument("--grad-checkpont", dest="grad_checkpoint",
                    action="store_true")
    args = ap.parse_args()
    cmd_train(args)


if __name__ == "__main__":
    main()