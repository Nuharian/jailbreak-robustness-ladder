"""
Validate the rule-based judge against human labels.

Workflow
  1. `python -m jbladder.validate sample --n 60`
        stratified-samples model responses (from responses_raw.jsonl) across
        kind x judged-label x language and writes results/validation/sample.jsonl,
        each row carrying the raw response and the judge's label, plus an empty
        "human_label" field to fill in (refuse / comply / evade).
  2. A human (or the author) fills "human_label" -> results/validation/labeled.jsonl
  3. `python -m jbladder.validate score`
        reports accuracy, per-class precision/recall, and the confusion matrix of
        judge vs. human. This is the number quoted in the paper to justify trusting
        the automated metric.
"""

import argparse
import glob
import json
import os
import random
from collections import Counter, defaultdict

from . import REFUSE, COMPLY, EVADE
from .transforms import mechanism_of
from .translations import request_text

RESULTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
VAL_DIR = os.path.join(RESULTS, "validation")
LANGS = {"translate_hi": "hi", "translate_bn": "bn", "translate_ur": "ur", "translate_ar": "ar"}


def _lang_of(transform):
    for k, v in LANGS.items():
        if transform.startswith(k):
            return v
    return "en"


def load_raw():
    rows = []
    for path in glob.glob(os.path.join(RESULTS, "*", "responses_raw.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def sample(n=60, seed=0):
    rows = load_raw()
    if not rows:
        print("No responses_raw.jsonl found. Re-run eval with --keep-raw.")
        return
    # stratify by (kind, judged-label, language-bucket) then sample proportionally
    strata = defaultdict(list)
    for r in rows:
        strata[(r["kind"], r["label"], _lang_of(r["transform"]) == "en")].append(r)
    rng = random.Random(seed)
    picked = []
    per = max(1, n // max(1, len(strata)))
    for key, group in strata.items():
        rng.shuffle(group)
        picked.extend(group[:per])
    rng.shuffle(picked)
    picked = picked[:n]

    os.makedirs(VAL_DIR, exist_ok=True)
    out = os.path.join(VAL_DIR, "sample.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps({
                "model": r["model"], "prompt_id": r["prompt_id"], "kind": r["kind"],
                "transform": r["transform"], "lang": _lang_of(r["transform"]),
                "judge_label": r["label"], "human_label": "",
                "response": r["response"],
            }, ensure_ascii=False) + "\n")
    print(f"wrote {len(picked)} rows to {out}\nFill 'human_label' (refuse/comply/evade) -> "
          f"{os.path.join(VAL_DIR, 'labeled.jsonl')}, then run: python -m jbladder.validate score")


def score(labeled_path=None):
    path = labeled_path or os.path.join(VAL_DIR, "labeled.jsonl")
    if not os.path.exists(path):
        print(f"No labeled file at {path}.")
        return
    from .judge import classify
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("human_label")]
    if not rows:
        print("No human labels found.")
        return

    # re-run the CURRENT judge on the stored responses so validation always reflects the live
    # classifier (not a stale label saved at sample time); reconstruct the request text so the
    # echo detector can fire.
    for r in rows:
        r["judge_label"] = classify(r["response"], r.get("kind", "harmful"),
                                    prompt_text=request_text(r["prompt_id"], r["transform"]))

    labels = [REFUSE, COMPLY, EVADE]
    correct = sum(1 for r in rows if r["judge_label"] == r["human_label"])
    n = len(rows)
    acc = correct / n

    # confusion + per-class P/R (human = truth, judge = prediction)
    conf = Counter((r["human_label"], r["judge_label"]) for r in rows)
    print(f"Judge validation on {n} human-labeled responses")
    print(f"  overall accuracy: {acc:.3f} ({correct}/{n})")

    # attack-success is the load-bearing decision; report its precision/recall specifically
    tp = sum(1 for r in rows if r["human_label"] == COMPLY and r["judge_label"] == COMPLY)
    fp = sum(1 for r in rows if r["human_label"] != COMPLY and r["judge_label"] == COMPLY)
    fn = sum(1 for r in rows if r["human_label"] == COMPLY and r["judge_label"] != COMPLY)
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    print(f"  COMPLY (attack-success) precision={prec:.3f} recall={rec:.3f}")
    print("  confusion matrix (rows=human truth, cols=judge):")
    header = "           " + "".join(f"{c:>9}" for c in labels)
    print(header)
    for h in labels:
        row = "".join(f"{conf.get((h, j), 0):>9}" for j in labels)
        print(f"  {h:>8} {row}")

    os.makedirs(VAL_DIR, exist_ok=True)
    with open(os.path.join(VAL_DIR, "report.json"), "w", encoding="utf-8") as f:
        json.dump({"n": n, "accuracy": acc, "comply_precision": prec, "comply_recall": rec,
                   "confusion": {f"{h}->{j}": conf.get((h, j), 0)
                                 for h in labels for j in labels}}, f, indent=1)
    print(f"  wrote {os.path.join(VAL_DIR, 'report.json')}")


def main():
    ap = argparse.ArgumentParser(description="Judge validation against human labels")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=60)
    s.add_argument("--seed", type=int, default=0)
    sc = sub.add_parser("score")
    sc.add_argument("--labeled", default=None)
    args = ap.parse_args()
    if args.cmd == "sample":
        sample(n=args.n, seed=args.seed)
    else:
        score(labeled_path=args.labeled)


if __name__ == "__main__":
    main()
