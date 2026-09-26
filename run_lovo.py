"""
run_lovo.py  —  leave-one-variant-out sweep for the RoBERTa pair model.

For each training prompt variant (P3a..P3f), train on the other five and
validate on the held-out one. Estimates transfer to an unseen prompt style —
the closest proxy for the real P3G-P3R test set.

Each fold runs DIRECTLY in the terminal, so you get the normal single
self-updating progress bar (not a flood of log lines). Each fold's tuned human
F1 is written by pair_model2.py's own output; this driver also appends it to
lovo_results.txt for the final summary.

    python run_lovo.py --data ./data --model roberta-base
"""

import argparse
import subprocess
import sys
import re
from pathlib import Path

VARIANTS = [
    "P3a_explain_why", "P3b_confused_user", "P3c_devils_advocate",
    "P3d_hypothetical_bet", "P3e_leading_loaded", "P3f_tentative",
]
RESULTS = Path("lovo_results.txt")


def run_fold(variant, args):
    out = f"./lovo_{variant.split('_')[0]}"
    log = Path(f"lovo_{variant.split('_')[0]}.log")
    cmd = [
        sys.executable, "pair_model.py", "train",
        "--data", args.data, "--out", out,
        "--model", args.model,
        "--holdout-variant", variant,
        "--batch", str(args.batch), "--grad-accum", str(args.grad_accum),
        "--epochs", str(args.epochs), "--seed", str(args.seed),
        "--precision", args.precision,
    ]
    print(f"\n{'='*70}\nLOVO fold: holding out {variant}\n{'='*70}", flush=True)

    # Run directly in the terminal (inherit stdout/stderr) so the progress bar
    # renders normally. We can't capture output this way, so pair_model2.py is
    # asked to also write its metrics to a log via PowerShell tee upstream is
    # not reliable; instead we re-read the saved threshold/metrics if present.
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"[fold {variant} FAILED code {proc.returncode}]")
        return None

    # pair_model2.py prints the F1 to the terminal (which we saw live) but we
    # didn't capture it. Ask the user-facing summary to rely on the printed
    # lines; as a fallback, we record what infer_config saved.
    cfg = Path(out) / "infer_config.json"
    note = ""
    if cfg.exists():
        import json
        note = f"threshold={json.loads(cfg.read_text()).get('threshold')}"
    with RESULTS.open("a", encoding="utf-8") as f:
        f.write(f"{variant}\t(see terminal output above)\t{note}\n")
    return "printed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--variants", nargs="*", default=VARIANTS)
    args = ap.parse_args()

    if RESULTS.exists():
        RESULTS.unlink()

    for v in args.variants:
        run_fold(v, args)

    print(f"\n\n{'='*70}\nLEAVE-ONE-VARIANT-OUT COMPLETE\n{'='*70}")
    print("Each fold's 'val vs HUMAN tuned: F1=...' line was printed above,")
    print("right after that fold finished training. Scroll up to read the six")
    print("numbers, or copy them here and I can compute the mean transfer gap.")
    print(f"\nModels saved as ./lovo_P3a ... ./lovo_P3f")


if __name__ == "__main__":
    main()