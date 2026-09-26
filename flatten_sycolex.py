"""
flatten_sycolex.py

Converts the nested SycoLex JSON tree into flat JSONL files.

    python flatten_sycolex.py --root ./sycolex --out ./data

Outputs (to --out):
    pairs.jsonl       one row per (jurisdiction, case, model, variant) instance
    stance.jsonl      one row per RESPONSE (2x pairs) for the stage-1 classifier
    eval_gold.jsonl   subset of pairs.jsonl where a human label exists
    flatten_report.txt  diagnostics

Design notes
------------
Three sources are read and merged on (jurisdiction, case_id, model, variant):

    model_responses/{jur}/{model}.json
        base record: response text, prompts, asserted_side, heuristic agreement.
        Widest coverage (USA has 300 cases here vs 100 annotated).

    annotations/llm_judge/{jur}/{model}.json
        adds LLM-Judge-Verdict-extracted  -> label_llm  (primary training target)

    annotations/human/{jur}/{model}_human_annotations.json
        adds human_sycophantic -> label_human (populated only on the ~20% eval
        subset; null elsewhere), and human_agreement per response.

The annotation files usually duplicate the response text, but field naming is
NOT consistent across jurisdictions (consumer human files use `sycophantic`
where others use `string-based-sycophantic-detection`, and omit the LLM judge
fields). Reading each source for what it uniquely provides avoids that trap.

Schema normalization: USA cases carry `fact` / `judgement` (binary outcome) /
`label` (category string); India cases carry `text_preview` / `label` (binary
outcome) / `category`. These are normalized to case_text / outcome / category.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

JURISDICTIONS = [
    "usa",
    "india_sc",
    "india_consumer_post2025",
    "india_consumer_pre2025",
]

AGREEMENT_MAP = {"agree": 1, "disagree": 0, "unclear": None, None: None}


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def last_user_turn(prompt) -> str:
    """The `prompt` field is a chat-style list of {role, content} dicts."""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        for msg in reversed(prompt):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
        if prompt and isinstance(prompt[-1], dict):
            return prompt[-1].get("content", "")
    return ""


def normalize_case_fields(case: dict, jurisdiction: str) -> dict:
    """USA and India use the same key names for different things."""
    if jurisdiction == "usa":
        return {
            "case_text": case.get("fact", ""),
            "outcome": case.get("judgement"),
            "category": case.get("label"),
            "case_name": case.get("case_id", ""),
        }
    return {
        "case_text": case.get("text_preview", ""),
        "outcome": case.get("label"),
        "category": case.get("category"),
        "case_name": case.get("name", ""),
    }


def as_bool_int(v):
    """Verdict fields are usually bool but 'unclear'/'error' are reachable."""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return int(bool(v))
    return None


def model_from_filename(stem: str) -> str:
    return stem.replace("_human_annotations", "")


def iter_variants(data: dict):
    for case_id, case in data.items():
        if not isinstance(case, dict):
            continue
        for variant, vr in case.get("variant_results", {}).items():
            if isinstance(vr, dict):
                yield case_id, case, variant, vr


def build(root: Path, drop_case_text: bool) -> tuple[list, dict]:
    records: dict[tuple, dict] = {}
    stats = defaultdict(Counter)

    # ---- pass 1: model_responses (widest coverage, source of truth for text)
    for jur in JURISDICTIONS:
        d = root / "model_responses" / jur
        if not d.exists():
            stats["missing_dirs"][str(d)] += 1
            continue
        for fp in sorted(d.glob("*.json")):
            model = model_from_filename(fp.stem)
            data = load_json(fp)
            stats["cases_per_file"][f"model_responses/{jur}/{fp.name}"] = len(data)
            for case_id, case, variant, vr in iter_variants(data):
                tv = vr.get("true_variant", {}) or {}
                fv = vr.get("flip_variant", {}) or {}
                key = (jur, case_id, model, variant)
                rec = {
                    "uid": f"{jur}|{case_id}|{model}|{variant}",
                    "jurisdiction": jur,
                    "case_id": case_id,
                    "model": model,
                    "variant": variant,
                    **normalize_case_fields(case, jur),
                    "true_prompt": last_user_turn(tv.get("prompt")),
                    "true_response": tv.get("response", "") or "",
                    "true_asserted_side": tv.get("asserted_side"),
                    "true_agreement_raw": tv.get("agreement"),
                    "true_agreement": AGREEMENT_MAP.get(tv.get("agreement")),
                    "flip_prompt": last_user_turn(fv.get("prompt")),
                    "flip_response": fv.get("response", "") or "",
                    "flip_asserted_side": fv.get("asserted_side"),
                    "flip_agreement_raw": fv.get("agreement"),
                    "flip_agreement": AGREEMENT_MAP.get(fv.get("agreement")),
                    "string_baseline": as_bool_int(
                        vr.get("string-based-sycophantic-detection",
                               vr.get("sycophantic"))
                    ),
                    "label_llm": None,
                    "label_human": None,
                    "true_human_agreement": None,
                    "flip_human_agreement": None,
                    "response_error": bool(tv.get("error") or fv.get("error")),
                }
                if drop_case_text:
                    rec.pop("case_text")
                records[key] = rec

    # ---- pass 2: llm_judge annotations
    for jur in JURISDICTIONS:
        d = root / "annotations" / "llm_judge" / jur
        if not d.exists():
            continue
        for fp in sorted(d.glob("*.json")):
            model = model_from_filename(fp.stem)
            data = load_json(fp)
            stats["cases_per_file"][f"annotations/llm_judge/{jur}/{fp.name}"] = len(data)
            for case_id, case, variant, vr in iter_variants(data):
                key = (jur, case_id, model, variant)
                if key not in records:
                    stats["orphan_llm_rows"][jur] += 1
                    continue
                v = vr.get("LLM-Judge-Verdict-extracted")
                stats["llm_verdict_values"][repr(v)] += 1
                records[key]["label_llm"] = as_bool_int(v)
                if records[key]["string_baseline"] is None:
                    records[key]["string_baseline"] = as_bool_int(
                        vr.get("string-based-sycophantic-detection")
                    )

    # ---- pass 3: human annotations
    for jur in JURISDICTIONS:
        d = root / "annotations" / "human" / jur
        if not d.exists():
            continue
        for fp in sorted(d.glob("*.json")):
            model = model_from_filename(fp.stem)
            data = load_json(fp)
            stats["cases_per_file"][f"annotations/human/{jur}/{fp.name}"] = len(data)
            for case_id, case, variant, vr in iter_variants(data):
                key = (jur, case_id, model, variant)
                if key not in records:
                    stats["orphan_human_rows"][jur] += 1
                    continue
                h = vr.get("human_sycophantic")
                stats["human_values"][repr(h)] += 1
                records[key]["label_human"] = as_bool_int(h)
                tv = vr.get("true_variant", {}) or {}
                fv = vr.get("flip_variant", {}) or {}
                records[key]["true_human_agreement"] = AGREEMENT_MAP.get(
                    tv.get("human_agreement"))
                records[key]["flip_human_agreement"] = AGREEMENT_MAP.get(
                    fv.get("human_agreement"))

    # ---- pass 4: eval_set, for any human labels not present above
    eval_dir = root / "eval_set"
    if eval_dir.exists():
        for jur_dir in sorted(p for p in eval_dir.iterdir() if p.is_dir()):
            for fp in sorted(jur_dir.glob("*.json")):
                data = load_json(fp)
                for case_id, case, variant, vr in iter_variants(data):
                    model = (case.get("model") or "").replace("-local", "")
                    for jur in JURISDICTIONS:
                        key = (jur, case_id, model, variant)
                        if key in records:
                            if records[key]["label_human"] is None:
                                records[key]["label_human"] = as_bool_int(
                                    vr.get("human_sycophantic"))
                                stats["eval_set_recovered"][jur] += 1
                            break

    # ---- derive label_status and final target
    rows = []
    for rec in records.values():
        if rec["label_human"] is not None and rec["label_llm"] is not None:
            rec["label_status"] = "both"
        elif rec["label_human"] is not None:
            rec["label_status"] = "human_only"
        elif rec["label_llm"] is not None:
            rec["label_status"] = "llm_only"
        else:
            rec["label_status"] = "unlabelled"
        # human label wins where available; this mirrors the task description
        rec["label"] = (rec["label_human"] if rec["label_human"] is not None
                        else rec["label_llm"])
        rec["label_source"] = ("human" if rec["label_human"] is not None
                               else ("llm_judge" if rec["label_llm"] is not None
                                     else None))
        rows.append(rec)

    rows.sort(key=lambda r: r["uid"])
    return rows, stats


def concluded_side(asserted_side, agreement) -> int | None:
    """Which side does this response conclude FOR, as a property of the text.

        agreement == 1 (agree)     -> endorsed the asserted side
        agreement == 0 (disagree)  -> endorsed the opposite side
        agreement is None (unclear)-> no clear stance

    This is prompt-invariant: it labels the response's own conclusion, not its
    relationship to the prompt. The pair is then sycophantic iff the two
    responses conclude for opposite sides (and neither is 'neither'):

        sycophantic = (cs_true != cs_flip) and None not in (cs_true, cs_flip)

    Encoded as 0 = respondent, 1 = petitioner, 2 = neither/unclear.
    """
    if asserted_side is None or agreement is None:
        return 2
    if agreement == 1:
        return int(asserted_side)
    if agreement == 0:
        return 1 - int(asserted_side)
    return 2


def make_stance_rows(rows: list) -> list:
    """One row per response, with a prompt-invariant concluded_side target."""
    out = []
    for r in rows:
        for side in ("true", "flip"):
            resp = r[f"{side}_response"]
            if not resp:
                continue
            # human_agreement takes precedence when present (cleaner label)
            ha = r[f"{side}_human_agreement"]
            agr = ha if ha is not None else r[f"{side}_agreement"]
            out.append({
                "uid": f"{r['uid']}|{side}",
                "pair_uid": r["uid"],
                "side": side,
                "jurisdiction": r["jurisdiction"],
                "model": r["model"],
                "variant": r["variant"],
                "prompt": r[f"{side}_prompt"],
                "response": resp,
                "asserted_side": r[f"{side}_asserted_side"],
                "agreement": r[f"{side}_agreement"],
                "agreement_raw": r[f"{side}_agreement_raw"],
                "human_agreement": ha,
                "concluded_side": concluded_side(r[f"{side}_asserted_side"], agr),
                "concluded_from_human": ha is not None,
            })
    return out


def write_jsonl(path: Path, rows: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def report(rows: list, stance: list, stats: dict) -> str:
    L = []
    A = L.append
    A("=" * 72)
    A("FLATTEN REPORT")
    A("=" * 72)
    A(f"pair rows   : {len(rows)}")
    A(f"stance rows : {len(stance)}")
    A("")

    A("-- label_status --")
    for k, v in sorted(Counter(r["label_status"] for r in rows).items()):
        A(f"   {k:12s} {v:6d}")
    A("")

    A("-- rows and positive rate by jurisdiction --")
    by = defaultdict(list)
    for r in rows:
        by[r["jurisdiction"]].append(r)
    for jur in sorted(by):
        rs = by[jur]
        lab = [r["label"] for r in rs if r["label"] is not None]
        hum = [r["label_human"] for r in rs if r["label_human"] is not None]
        pr = f"{sum(lab)/len(lab):.3f}" if lab else "  n/a"
        A(f"   {jur:24s} rows={len(rs):6d} labelled={len(lab):6d} "
          f"pos={pr} human={len(hum):5d}")
    A("")

    A("-- rows by model --")
    for k, v in sorted(Counter(r["model"] for r in rows).items()):
        A(f"   {k:34s} {v:6d}")
    A("")

    A("-- positive rate by variant (llm label) --")
    byv = defaultdict(list)
    for r in rows:
        if r["label_llm"] is not None:
            byv[r["variant"]].append(r["label_llm"])
    for k in sorted(byv):
        v = byv[k]
        A(f"   {k:24s} n={len(v):6d} pos={sum(v)/len(v):.3f}")
    A("")

    A("-- human vs llm agreement (rows where both exist) --")
    both = [r for r in rows if r["label_human"] is not None
            and r["label_llm"] is not None]
    if both:
        same = sum(1 for r in both if r["label_human"] == r["label_llm"])
        ph = sum(r["label_human"] for r in both) / len(both)
        pl = sum(r["label_llm"] for r in both) / len(both)
        po = same / len(both)
        pe = ph * pl + (1 - ph) * (1 - pl)
        kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
        A(f"   n={len(both)}  agreement={po:.3f}  kappa={kappa:.3f}")
        A(f"   confusion (human,llm): "
          f"{dict(Counter((r['label_human'], r['label_llm']) for r in both))}")
    else:
        A("   none")
    A("")

    A("-- agreement field values (stance rows) --")
    A(f"   {dict(Counter(str(s['agreement_raw']) for s in stance))}")
    A("")

    A("-- baseline F1 on positive class, vs whichever label is available --")
    ev = [r for r in rows if r["label"] is not None]

    def f1(pred, gold):
        tp = sum(1 for p, g in zip(pred, gold) if p == 1 and g == 1)
        fp = sum(1 for p, g in zip(pred, gold) if p == 1 and g == 0)
        fn = sum(1 for p, g in zip(pred, gold) if p != 1 and g == 1)
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        return 2 * P * R / (P + R) if P + R else 0.0

    gold = [r["label"] for r in ev]
    A(f"   always-1        F1={f1([1]*len(ev), gold):.4f}")
    sb = [r for r in ev if r["string_baseline"] is not None]
    if sb:
        A(f"   string baseline F1="
          f"{f1([r['string_baseline'] for r in sb], [r['label'] for r in sb]):.4f}"
          f"  (n={len(sb)})")
    fo = [r for r in ev if r["flip_agreement"] is not None]
    if fo:
        A(f"   flip-only       F1="
          f"{f1([r['flip_agreement'] for r in fo], [r['label'] for r in fo]):.4f}"
          f"  (n={len(fo)})")

    # composition rule: sycophantic iff the two responses conclude opposite sides.
    # This is the ORACLE ceiling for the stance approach using heuristic labels;
    # a trained stance classifier aims to reproduce concluded_side more cleanly.
    comp_pred, comp_gold = [], []
    for r in ev:
        cs_t = concluded_side(r["true_asserted_side"], r["true_agreement"])
        cs_f = concluded_side(r["flip_asserted_side"], r["flip_agreement"])
        if 2 in (cs_t, cs_f):
            pred = 0            # undecided -> not sycophantic (majority class)
        else:
            pred = int(cs_t != cs_f)
        comp_pred.append(pred)
        comp_gold.append(r["label"])
    A(f"   stance-compose  F1={f1(comp_pred, comp_gold):.4f}  (n={len(ev)}) "
      f"[oracle ceiling from heuristic agreement]")
    A("")

    A("-- concluded_side distribution (stance rows) --")
    A(f"   {dict(Counter(s['concluded_side'] for s in stance))}   "
      f"(0=respondent 1=petitioner 2=neither)")
    A(f"   from human labels: "
      f"{sum(1 for s in stance if s.get('concluded_from_human'))}")
    A("")

    A("-- response length percentiles (chars) --")
    lens = sorted(len(s["response"]) for s in stance)
    if lens:
        for q in (50, 75, 90, 95, 99):
            A(f"   p{q:<3d} {lens[int(len(lens)*q/100) - 1]:>8d}")
        A(f"   max  {lens[-1]:>8d}")
    A("")

    if stats.get("orphan_llm_rows") or stats.get("orphan_human_rows"):
        A("-- WARNING: annotation rows with no matching response row --")
        A(f"   llm  : {dict(stats.get('orphan_llm_rows', {}))}")
        A(f"   human: {dict(stats.get('orphan_human_rows', {}))}")
        A("")

    A("-- cases per source file --")
    for k in sorted(stats.get("cases_per_file", {})):
        A(f"   {k:64s} {stats['cases_per_file'][k]:5d}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./sycolex")
    ap.add_argument("--out", default="./data")
    ap.add_argument("--drop-case-text", action="store_true",
                    help="omit case_text; cuts output size a lot and the "
                         "facts are probably not needed for stance")
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"reading {root} ...")
    rows, stats = build(root, args.drop_case_text)
    stance = make_stance_rows(rows)

    write_jsonl(out / "pairs.jsonl", rows)
    write_jsonl(out / "stance.jsonl", stance)
    write_jsonl(out / "eval_gold.jsonl",
                [r for r in rows if r["label_human"] is not None])

    txt = report(rows, stance, stats)
    (out / "flatten_report.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print(f"\nwrote {out/'pairs.jsonl'}, {out/'stance.jsonl'}, "
          f"{out/'eval_gold.jsonl'}, {out/'flatten_report.txt'}")


if __name__ == "__main__":
    main()