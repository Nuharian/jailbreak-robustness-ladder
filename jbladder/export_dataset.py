"""
Export the full probe set (with translations + provenance) to data/prompts.jsonl for inspection.

Run:  python -m jbladder.export_dataset
"""

import json
import os

from . import translations
from .prompts import load_prompts

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OUT = os.path.join(DATA_DIR, "prompts.jsonl")


def export():
    harmful, benign = load_prompts()
    os.makedirs(DATA_DIR, exist_ok=True)
    n = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for p in harmful + benign:
            row = dict(p)
            row["translations"] = translations.MERGED.get(p["id"], {})
            row["translation_provenance"] = translations.PROVENANCE.get(p["id"], {})
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} prompts -> {OUT}")


if __name__ == "__main__":
    export()
