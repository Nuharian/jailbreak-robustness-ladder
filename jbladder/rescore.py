"""
Re-judge every stored raw response with the current judge and rewrite scores.jsonl +
responses.jsonl (redacted). Lets us improve the classifier without re-running any model.

Reconstructs category / mechanism / defense from the prompt id, transform, and model tag, and
supplies the request text so the echo detector can fire.

Run:  python -m jbladder.rescore
"""

import glob
import json
import os

from . import COMPLY
from .judge import classify
from .prompts import load_prompts
from .run_eval import RESULTS_DIR, _redact
from .transforms import mechanism_of
from .translations import request_text


def rescore():
    h, b = load_prompts()
    id2cat = {p["id"]: p["category"] for p in h + b}
    raws = glob.glob(os.path.join(RESULTS_DIR, "*", "responses_raw.jsonl"))
    if not raws:
        print("No responses_raw.jsonl found (run eval with --keep-raw).")
        return
    for raw_path in raws:
        d = os.path.dirname(raw_path)
        rows = [json.loads(l) for l in open(raw_path, encoding="utf-8") if l.strip()]
        n_success = 0
        with open(os.path.join(d, "scores.jsonl"), "w", encoding="utf-8") as sf, \
             open(os.path.join(d, "responses.jsonl"), "w", encoding="utf-8") as rf:
            for r in rows:
                model, pid, kind = r["model"], r["prompt_id"], r["kind"]
                transform, resp = r["transform"], r["response"]
                defense = "system" if "[system]" in model else "none"
                label = classify(resp, kind, prompt_text=request_text(pid, transform))
                success = (kind == "harmful" and label == COMPLY)
                n_success += int(success)
                base = {"model": model, "prompt_id": pid, "category": id2cat.get(pid, "?"),
                        "kind": kind, "defense": defense, "transform": transform,
                        "mechanism": mechanism_of(transform), "label": label}
                sf.write(json.dumps(base, ensure_ascii=False) + "\n")
                rf.write(json.dumps({**base, "response": _redact(resp) if success else resp},
                                    ensure_ascii=False) + "\n")
        print(f"rescored {os.path.basename(d):46s} {len(rows):5d} rows  "
              f"harmful-comply={n_success}")


if __name__ == "__main__":
    rescore()
