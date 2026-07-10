"""
Bootstrap translations of every probe into Hindi, Bangla, Urdu, Arabic using Meta's
NLLB-200 (No Language Left Behind) open translation model.

Why a dedicated MT model instead of a general LLM? NLLB-200 is purpose-built for exactly these
lower-resource languages and does not refuse to translate (a general safety-tuned LLM sometimes
refuses to translate a harmful sentence, which would silently corrupt the dataset). The output is
written to data/translations_auto.json with provenance so a native speaker can review/correct it.

The 47 hand-authored "gold" entries in translations.BY_ID always override these machine drafts.

Run:  python -m jbladder.translate_bootstrap
"""

import json
import os

from .prompts import load_prompts

MODEL = "facebook/nllb-200-distilled-600M"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OUT = os.path.join(DATA_DIR, "translations_auto.json")

# NLLB language codes. The first four are the low/medium-resource focus languages (with
# hand-authored gold overrides); zh/es/fr are high-resource controls for the resource-vs-safety
# analysis.
NLLB = {"hi": "hin_Deva", "bn": "ben_Beng", "ur": "urd_Arab", "ar": "arb_Arab",
        "zh": "zho_Hans", "es": "spa_Latn", "fr": "fra_Latn"}


def translate_all(batch_size=16):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(device)
    model.eval()

    harmful, benign = load_prompts()
    items = harmful + benign
    texts = [p["text"] for p in items]

    out = {p["id"]: {} for p in items}
    for lang, code in NLLB.items():
        bos = tok.convert_tokens_to_ids(code)
        print(f"[nllb] translating {len(texts)} prompts -> {lang} ({code})")
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=128).to(device)
            with torch.no_grad():
                gen = model.generate(**enc, forced_bos_token_id=bos, max_length=160,
                                     num_beams=4)
            dec = tok.batch_decode(gen, skip_special_tokens=True)
            for p, tr in zip(items[start:start + batch_size], dec):
                out[p["id"]][lang] = tr.strip()

    os.makedirs(DATA_DIR, exist_ok=True)
    payload = {"model": MODEL, "provenance": "machine (NLLB-200); pending native review",
               "by_id": out}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"[nllb] wrote {OUT}  ({len(out)} prompts x {len(NLLB)} languages)")
    return OUT


if __name__ == "__main__":
    translate_all()
