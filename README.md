# The Jailbreak Robustness Ladder (`jbladder`)

**Separating safety failure from comprehension failure in cross-lingual attacks on open language models.**

Most jailbreak evaluations count *"the model didn't refuse"* as *"the attack worked."* For the
languages that matter most — Hindi, Bangla, Urdu, Arabic — that is wrong: when a small model is
asked something harmful in Urdu and emits confused text, it wasn't jailbroken, it just didn't
understand. `jbladder` fixes this with a **three-way judge (REFUSE / COMPLY / EVADE)** and a
**comprehension-controlled attack-success rate (cc-ASR)** — jailbreak success *among requests the
model actually understood* — and uses it to show that **cross-lingual jailbreak success is
entangled with language competence.**

Everything runs **locally on open-weight models with no API keys**.

## What it does

- **Attack ladder.** 10 atomic transforms + 5 stacked combos, tagged by mechanism family
  (framing / persuasion / obfuscation / **translation**). Language-switching (hi/bn/ur/ar) is a
  first-class attack, so we can ask whether switching language beats base64 or roleplay.
- **Paired probes.** 100 harmful (non-operational refusal triggers) + 50 topically-matched benign
  twins across 10 harm categories. Benign twins measure over-refusal *and* estimate each model's
  per-language competence.
- **Three-way judge → cc-ASR.** Distinguishes a genuine jailbreak (COMPLY) from a model broken by
  obfuscation (EVADE). Multilingual, Unicode-aware, and validated against human labels.
- **A measured defense.** A hardened defensive system prompt, re-run over the whole ladder, with
  ASR reduction and over-refusal cost quantified.
- **Rigor.** Probe-clustered bootstrap confidence intervals; worst-case (Manski) bounds that need
  no assumption about evasions; native-vs-machine translation check; judge-error correction from
  human labels; composability (super-additivity) analysis; cross-model transfer; a defense
  transition audit (genuine refusal vs. comprehension collapse).

## Repository layout

```
jbladder/
  prompts.py            150 paired probes (100 harmful + 50 benign, 10 categories)
  translations.py       hand "gold" (47×4) merged over machine translations
  translate_bootstrap.py  NLLB-200 pipeline -> data/translations_auto.json
  transforms.py         composable attacks incl. translation family + combos
  judge.py              three-way REFUSE/COMPLY/EVADE classifier (multilingual)
  adapters.py           HF transformers backend (+ mock; API adapters stubbed)
  run_eval.py           model × prompt × transform → responses + scores (+ --defense)
  metrics.py            cc-ASR, attack lift, robustness, potency, composability, CIs, defense
  analyze.py            tables + figures (incl. the sweet-spot & masking plots)
  robustness.py         CIs, contrasts, bounds, gold-vs-auto, judge-error correction, transfer
  validate.py           judge validation against human labels
  make_docx.py          paper -> editable Word file (results/jbladder_paper.docx)
  make_pdfs.py          paper -> print HTML (results/paper.html) -> PDF via a headless browser
tests/                  unit tests (transforms, judge, metrics, robustness)
paper/jailbreak_ladder.md   the paper (source of the .docx and .pdf)
data/, results/         generated
```

## Quickstart

```bash
python -m pip install -r requirements.txt
# GPU torch (Blackwell / RTX 50xx):
#   pip install torch --index-url https://download.pytorch.org/whl/cu128

python -m jbladder.translate_bootstrap                 # -> data/translations_auto.json
python -m jbladder.run_eval --model Qwen/Qwen2.5-1.5B-Instruct --keep-raw
python -m jbladder.run_eval --model Qwen/Qwen2.5-1.5B-Instruct --defense system --keep-raw
python -m jbladder.analyze                              # -> results/tables, results/figures
python -m jbladder.robustness                           # -> CIs, bounds, robustness tables, Fig. 10

# or run the whole suite:
bash run_all.sh
```

No GPU: add `--device cpu` (slow). No model download: `--model mock` exercises the pipeline offline.
Quick subset: `--limit 5`. Reproduce paper figures: `python -m jbladder.analyze`.

## Judge validation

```bash
python -m jbladder.validate sample --n 60      # -> results/validation/sample.jsonl
#   fill the empty "human_label" field (refuse/comply/evade) -> labeled.jsonl
python -m jbladder.validate score              # accuracy, precision/recall, confusion matrix
```

## Ethics & scope

Harmful prompts are short, non-operational refusal *triggers* of the kind used by published safety
benchmarks (AdvBench, HarmBench, JailbreakBench). The harness records *whether* a model refused; it
never emits harmful content in shared outputs. Any successful harmful completion is redacted
(hash + length) in committed results; raw generations stay local and are git-ignored. Purpose:
defensive — measuring and reducing a safety gap that disadvantages non-English speakers.

## Status

Four open models evaluated (Qwen2.5-0.5B/1.5B/3B, SmolLM2-1.7B) on an 8 GB laptop GPU, no API keys.
See `paper/jailbreak_ladder.md` (or `results/jbladder_paper.pdf`) for methodology, results,
limitations, and references. The judge-error correction in `robustness.py` additionally needs the
human validation labels, which are not in this repository because they contain raw model responses;
everything else regenerates from the committed `scores.jsonl` files.

## Citation

See `CITATION.cff`. A Zenodo DOI will be listed here once the preprint is archived.
