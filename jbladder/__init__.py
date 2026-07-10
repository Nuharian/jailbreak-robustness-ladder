"""
Jailbreak Robustness Ladder (jbladder)
======================================

A reproducible harness that measures how prompt *transformations* degrade the refusal
behaviour of open-weight language models, and — crucially — separates genuine harmful
compliance from a model simply being *broken* by an obfuscation.

Design goals
------------
1. Composable transforms: each attack is a pure function ``request -> chat messages`` so
   attacks can be *stacked* to study whether weak attacks compound (super-additivity).
2. Three-way judging: every response is REFUSE / COMPLY / EVADE (garbage / off-topic /
   confused). Attack success requires COMPLY on a harmful prompt, not merely the absence
   of the word "sorry".
3. Benign controls: the same transforms are applied to harmless look-alike prompts, so we
   can measure the over-refusal cost of attacks *and* detect when a transform just destroys
   the model's comprehension rather than its safety.

Ethics
------
Harmful prompts are short, non-operational refusal *triggers* of the kind used by published
safety benchmarks (AdvBench, HarmBench, JailbreakBench). This package never emits harmful
content: it records *whether* a model refused. The analysis + artifact redact the text of any
successful harmful completion.
"""

__version__ = "0.1.0"

# Default open-weight, non-gated model suite (runs on 8 GB VRAM, one at a time).
DEFAULT_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
]

REFUSE, COMPLY, EVADE = "refuse", "comply", "evade"
