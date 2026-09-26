import json
from pathlib import Path

rows = [json.loads(l) for l in open("data/stance.jsonl", encoding="utf-8") if l.strip()]

# keep only rows where BOTH labels exist
pairs = [(int(r["human_agreement"]), int(r["agreement"]))
         for r in rows
         if r.get("human_agreement") is not None and r.get("agreement") is not None]

n = len(pairs)
agree = sum(1 for h, l in pairs if h == l)
po = agree / n                                    # observed agreement

# marginal probabilities for chance agreement
h1 = sum(h for h, l in pairs) / n
l1 = sum(l for h, l in pairs) / n
pe = h1 * l1 + (1 - h1) * (1 - l1)               # expected by chance

kappa = (po - pe) / (1 - pe)

print(f"rows with both labels: {n}")
print(f"observed agreement (plain): {po:.4f}  ({po*100:.1f}%)")
print(f"expected-by-chance agreement: {pe:.4f}")
print(f"Cohen's kappa: {kappa:.4f}")