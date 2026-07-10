# The Jailbreak Robustness Ladder: Separating Safety Failure from Comprehension Failure in Cross-Lingual Attacks on Open Language Models

*Working paper / technical report. Code and data: this repository.*

---

## Abstract

Language-model safety training is done overwhelmingly in English, yet the models are deployed in
many languages. A natural attack is therefore to *translate* a harmful request into another
language and hope the safety training does not transfer. Prior work has shown this works on
frontier models. However, essentially all cross-lingual jailbreak evaluations share a
methodological flaw: they count **"the model did not refuse"** as **"the attack succeeded"**. For
lower-resource languages — exactly the ones of interest — this conflates two very different
events: the model's *safety* failing, and the model simply *not understanding the request*. A
small model that emits confused text when asked something harmful in Urdu has not been
jailbroken; it has been broken.

We introduce the **Jailbreak Robustness Ladder** (`jbladder`), a reproducible harness that (i)
treats language-switching as one *attack transform* among many (roleplay, refusal-suppression,
base64, payload-splitting, …), so linguistic and syntactic attacks can be ranked on a single axis;
(ii) judges every response with a **three-way classifier — REFUSE / COMPLY / EVADE** — and reports
a **comprehension-controlled attack-success rate** (cc-ASR): jailbreak success *among requests the
model actually understood*; and (iii) evaluates a concrete **mitigation**. Using 150 paired
harmful/benign probes translated into **seven languages spanning the resource spectrum** — Hindi,
Bangla, Urdu, Arabic (lower/medium-resource) and Chinese, Spanish, French (high-resource) — run
across four open models (Qwen2.5-0.5B/1.5B/3B, SmolLM2-1.7B; **19,800 judged responses**), we find
that **safety is anchored to the languages a model was actually aligned in**, not to a language's
global resource level. Spanish and French retain safety on the multilingual Qwen models (cc-ASR
0.01–0.12) but collapse on English-centric SmolLM2 (~0.80), which is safe *only* in English. Safety
**collapses in lower-resource languages (0.76–0.86 for Hindi/Urdu/Bangla) across every model** — a
2.5× gap versus English (0.31). Chinese is a cautionary exception: despite being very high-resource
and well-understood, its cc-ASR averages **0.51** (0.88 on Qwen2.5-3B), because these models parse
Chinese fluently yet were not robustly *safety*-tuned in it — **capability is not safety**. In
Bangla and Urdu the *raw* ASR further **understates** the hole (comprehension drops to ~0.73–0.75,
so a naive judge miscounts confusion as safety). A hardened defensive system prompt reduces attack
success by only 15–18 %. Our central methodological claim is that **cross-lingual safety cannot be
measured without controlling for comprehension** — a claim we reinforce the hard way, when human
validation of the judge caught a tokenization bug that had made Chinese look artificially safe
(cc-ASR 0.27 → 0.51 once fixed). Our central empirical claim is that **cross-lingual jailbreak
success is a shadow of where the model was aligned**, not a separate linguistic vulnerability.

---

## 1. Introduction

Refusal is the most visible safety behaviour of an aligned language model: when asked to do
something harmful, it should decline. A *jailbreak* is any input transformation that causes the
model to comply anyway. A large literature catalogues such transformations — roleplay personas,
refusal suppression, affirmative-prefix injection, encoding tricks, and, notably, **translation
into lower-resource languages** (Yong et al., 2023; Deng et al., 2023).

The translation attack is uniquely important for equity reasons: it means the users least served
by English-centric safety work — speakers of Hindi, Bangla, Urdu, Arabic, and hundreds of other
languages — are also the least protected. But it is uniquely *hard to measure*, for a reason the
literature has under-emphasised. When a small or mid-sized model is asked something harmful in a
language it barely models, three things can happen:

1. it recognises the harm and **refuses** (safety transferred);
2. it produces a fluent, harmful, on-topic answer (**genuine jailbreak**); or
3. it produces confused, off-topic, or garbled text (**it did not understand**).

Counting (3) as a jailbreak — which any refusal-keyword-based metric does, because garbled text
contains no refusal phrase — systematically *overstates* the cross-lingual vulnerability of weak
models and *understates* the role of comprehension. This paper's contribution is to fix that.

**Contributions.**
- **A comprehension-controlled measurement.** A three-way judge (REFUSE / COMPLY / EVADE) and a
  metric, cc-ASR, that reports jailbreak success only over understood requests, cleanly separating
  safety failure from comprehension failure (§3.3, §5.2).
- **A unified attack ladder.** Ten atomic transforms plus stacked combinations, with
  language-switching (hi/bn/ur/ar) as first-class transforms, so we can ask *whether switching
  language is a stronger lever than syntactic obfuscation* (§3.2, §5.1).
- **Paired benign controls.** 50 topically-matched harmless twins that both measure over-refusal
  and provide a per-language **competence proxy** used in the entanglement analysis (§3.1).
- **An empirical finding — safety is per-model and alignment-anchored.** A model is safe in a
  language to the extent it was *safety*-aligned in it: SmolLM2 is safe only in English; Qwen keeps
  safety in English/Spanish/French/Arabic but not low-resource languages, and — strikingly — not in
  Chinese, which it understands fluently (capability ≠ safety). Comprehension correlates negatively
  with comprehension-controlled ASR (r = −0.47) across 42 model×language cells, and where
  comprehension is low (Bangla, Urdu) raw ASR *understates* the failure (§5.2).
- **A measured defense.** A hardened defensive system prompt reduces attack success by only
  15–18 % (relative) at an over-refusal cost of +0.06 to +0.12 — a partial fix that motivates
  model-level mitigation (§5.3).
- **Full reproducibility with no API keys.** Everything runs locally on open-weight models; a
  purpose-built translation pipeline (NLLB-200) plus a native-reviewed "gold" subset produces the
  multilingual probes (§4).

---

## 2. Related work and how this differs

**Cross-lingual jailbreaks.** Yong et al. (2023), *Low-Resource Languages Jailbreak GPT-4*, showed
that translating unsafe prompts into low-resource languages bypasses GPT-4's guardrails; Deng et
al. (2023), *MultiJail*, built a multilingual jailbreak benchmark. Both measure whether the model
complied; neither separates compliance from comprehension, and both target frontier models where
comprehension is near-ceiling (so the distinction is invisible). Our contribution is precisely the
regime — small/mid open models — where comprehension is *not* a given, and the tooling to measure
it.

**Exaggerated safety / over-refusal.** Röttger et al. (2023), *XSTest*, introduced benign prompts
that superficially look unsafe to measure over-refusal, in English only. We adopt the paired-twin
idea, extend it multilingually, and repurpose the benign twins as a **comprehension probe**.

**Jailbreak taxonomies and benchmarks.** Wei et al. (2023) analyse why safety training fails
(competing objectives, mismatched generalisation) and catalogue attacks including
refusal-suppression, prefix-injection, and encodings; Kang et al. (2023) use payload-splitting;
Andriushchenko & Flammarion (2024) show past-tense reformulation. Benchmarks such as HarmBench and
JailbreakBench standardise harmful-prompt sets and judging. We reuse these attack *mechanisms* but
place language-switching on the same measurement axis and add the comprehension control.

**In one sentence.** Prior work establishes that translation is an attack and that over-refusal
matters; **no prior cross-lingual evaluation we are aware of reports a comprehension-controlled
attack-success rate**, which is the instrument this paper contributes.

---

## 3. Method

### 3.1 Probe set

We hand-author **100 harmful** and **50 benign** English prompts across ten harm categories
(cybercrime, fraud, weapons, drugs, self-harm, hate, misinformation, privacy/surveillance, violent
extremism, illegal commerce). Harmful prompts are short, **non-operational refusal triggers** —
they name a harmful goal without any operational content — in the style of AdvBench / HarmBench.
Each harmful category is mirrored by benign twins that share surface vocabulary but have benign
intent (e.g. harmful *"how do I break into someone's Wi-Fi"* vs. benign *"how do I protect my
Wi-Fi from intruders"*). Benign twins serve two purposes: measuring over-refusal, and — when
translated — estimating the model's **competence** in each language (does it produce a coherent,
on-topic answer to a harmless request?).

### 3.2 Attack transforms

Each transform is a pure function `request → chat messages`, tagged with a *mechanism family*:

| family | transforms |
|---|---|
| control | `plain` |
| framing | `refusal_suppression`, `prefix_injection` |
| persuasion | `roleplay_persona`, `hypothetical`, `authority_framing`, `past_tense` |
| obfuscation | `base64`, `leetspeak`, `payload_split` |
| translation | `translate_{hi,bn,ur,ar,zh,es,fr}` (7 languages across the resource spectrum) |

Because transforms are pure functions we **compose** them; we include five stacked combos (e.g.
`translate_hi+refusal_suppression`) to test whether weak attacks *compound*. In total **22
transforms × 150 prompts = 3,300 queries per model condition**.

### 3.3 Three-way judge and comprehension-controlled ASR

A rule-based classifier maps each response to one of:

- **REFUSE** — an explicit decline (multilingual refusal cues) not overridden by a long
  substantive answer;
- **EVADE** — confused, off-topic, clarification-seeking, or too short to be a real answer (the
  typical failure mode of a broken obfuscation);
- **COMPLY** — a substantive, on-topic answer.

Refusal cues cover all seven languages, and word-counting is Unicode-aware (Devanagari, Bangla,
Arabic, Han, Latin) so a fluent non-English answer is not mistaken for empty output. Two further
detectors route degenerate repetition and near-verbatim prompt-echo to EVADE (a broken translation
that loops or restates the request is not a jailbreak). From these we define, per (model, transform),
on harmful prompts:

- **comprehension** = 1 − EVADE-rate — did the model engage with the request at all?
- **ASR (raw)** = COMPLY / N — the quantity prior work reports;
- **cc-ASR** = COMPLY / (COMPLY + REFUSE) — **attack success among understood requests**.

For benign prompts we report **over-refusal** = REFUSE-rate and **benign-pass** = COMPLY-rate; the
translated benign-pass rate is our per-language competence proxy.

We **validate** the judge against human labels on a stratified sample of **72** responses (the
author read the Hindi/Bangla/Urdu/Arabic ones directly): overall accuracy **0.83**, with
**perfect refusal detection (20/20)**, and for the load-bearing COMPLY (attack-success) decision,
precision **0.56** / recall **0.71**. The residual error is concentrated in EVADE→COMPLY (garbled
multilingual output judged as compliance), which means our translation ASRs are, if anything, an
*upper bound* — the true cross-lingual gap is at least as large as reported (§5.4).

### 3.4 Metrics

Beyond the rates above: **attack lift** = ASR(transform) − ASR(plain); **model robustness** =
1 − mean ASR over attacks; **transform potency** = mean attack-lift across models; **composability
residual** = ASR(A+B) − [1 − (1−ASR(A))(1−ASR(B))] (positive ⇒ super-additive); **cross-model
transfer** = Jaccard overlap of the prompts jailbroken under a transform. All headline rates carry
**95 % percentile-bootstrap confidence intervals**.

### 3.5 Defense

We evaluate a **defensive system prompt** that instructs the model to refuse harmful requests
"regardless of language, fiction, roleplay, research framing, or encoding, and regardless of any
instruction to not refuse," subordinating any attacker-supplied system message. We re-run the full
ladder with it and report ASR reduction and the over-refusal cost. We also discuss **language
normalization** (routing the request through English before the safety decision), for which our
own English-vs-translation numbers provide a direct estimate of recoverable safety.

---

## 4. Experimental setup

**Models (open, non-gated):** Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct,
SmolLM2-1.7B-Instruct. The Qwen ladder isolates *scale*; SmolLM2 is a more English-centric control.
Two of them (Qwen2.5-1.5B, SmolLM2-1.7B) are additionally run under the defensive system prompt, for
six conditions in all. **Languages (7):** Hindi, Bangla, Urdu, Arabic (lower/medium-resource) and
Chinese, Spanish, French (high-resource controls). **Hardware:** a single RTX 5060 Laptop GPU
(8 GB), bf16, greedy decoding, 160 new tokens, batched. **Translations:** produced by
NLLB-200-distilled-600M; the 47 seed prompts (188 cells, in the four focus languages) additionally
carry hand-authored, native-reviewed "gold" translations that override the machine output.
Provenance is tracked per cell. **No API keys are used anywhere**; an OpenAI/Anthropic adapter is
included so the identical harness can be pointed at a frontier model when a key is available.

---

## 5. Results

All numbers are from 19,800 judged responses (6 model conditions × 150 prompts × 22 transforms).
Rates carry 95 % bootstrap CIs; see `results/tables/*` and `results/figures/*`.

### 5.1 The attack ladder

**Robustness is not monotonic in scale.** Ranking models by 1 − mean ASR (`model_robustness.png`):
**Qwen2.5-1.5B is by far the most robust (0.58)**, then Qwen2.5-0.5B (0.38), Qwen2.5-3B (0.37), and
SmolLM2-1.7B (0.31). The 3B model is *no more* robust than the 0.5B and well below the 1.5B — a
recurring theme: greater capability improves *comprehension of the attack*, which for obfuscations
and translations removes the EVADE "shield" and converts near-misses into genuine jailbreaks (the
3B's Chinese cc-ASR of 0.88, §5.2, is the extreme case).

**Which attacks are most potent?** By mean attack-lift over the plain baseline
(`transform_potency.png`): `prefix_injection` (+0.64) and `roleplay_persona+prefix_injection`
(+0.62) lead, immediately followed by `translate_hi+refusal_suppression` (+0.57), `hypothetical`
(+0.56), and `translate_hi` (+0.43). Aggregated by mechanism family, mean ASR is **framing 0.72 >
combo 0.61 > persuasion 0.56 > obfuscation 0.48 ≈ translation 0.47** — but the translation family's
low *mean* is deceptive: it is strongly **bimodal**, averaging the mostly-harmless Spanish/French
translations with the most potent low-resource ones (`translate_hi` is the 5th-strongest attack
overall). Obfuscation is weakest because small models cannot decode base64/leetspeak — a
comprehension failure the three-way judge correctly scores as EVADE.

**Composability is mostly sub-additive — with one telling exception.** Stacking two attacks
usually yields *less* than the independence prediction 1−(1−a)(1−b) (`composability.png`):
`base64+refusal_suppression` is strongly sub-additive everywhere (residuals −0.23 to −0.75),
because base64 destroys comprehension and no amount of refusal-suppression can recover it. The
exception is **`translate_hi+refusal_suppression`, which is *super-additive* on the safety-tuned
Qwen2.5-1.5B** (residual +0.07 undefended, **+0.24 with the defense on**): language-switching and
framing attack *different* parts of the safety stack, so together they breach a model that either
alone leaves partly intact.

### 5.2 The core result: safety is alignment-anchored, and comprehension masks the hole

Per language, averaged over models, single-transform (plain-English vs. plain-translation), ordered
by cc-ASR:

| language | comprehension (1−evade) | raw ASR | comprehension-controlled ASR |
|---|---|---|---|
| English | 0.95 | 0.28 | 0.31 |
| Spanish | 0.94 | 0.29 | 0.33 |
| French | 0.93 | 0.29 | 0.33 |
| Arabic | 0.95 | 0.41 | 0.45 |
| Chinese | 0.88 | 0.43 | **0.51** |
| Hindi | 0.94 | 0.71 | **0.76** |
| Urdu | **0.75** | 0.59 | **0.79** |
| Bangla | **0.73** | 0.62 | **0.86** |

**Three findings stack.**

*(1) Safety is per model and alignment-anchored, not set by global resource level.* A model is safe
in a language to the extent it was *safety*-aligned in it. **SmolLM2-1.7B (English-centric) is safe
only in English (cc-ASR 0.61) and fails in every other language tested — including high-resource
Spanish (0.79), French (0.80), and Chinese (0.73).** The multilingual **Qwen models keep safety in
English, Spanish, French, and Arabic** (Qwen2.5-1.5B: 0.01, 0.01, 0.03, 0.13) but not in
lower-resource languages. The same language ("high-resource" Spanish) is safe on one model and
wide-open on another; safety follows each model's alignment mix. Across the **42 model×language
translation cells**, comprehension and cc-ASR correlate **r = −0.47**.

*(2) Capability is not safety — the Chinese anomaly.* Chinese is very high-resource and fluently
understood (comprehension 0.88), yet its cc-ASR averages **0.51**, driven by **Qwen2.5-3B at 0.88**
(Qwen2.5-1.5B holds at 0.22). The models parse Chinese perfectly but were evidently not robustly
safety-tuned in it: understanding a language does not confer safety in it. This severs the intuitive
"high-resource ⇒ safe" shortcut — resource buys comprehension, which is necessary but not sufficient.
*(This anomaly was only visible after fixing a judge bug; see §5.4.)*

*(3) Comprehension masks the hole.* In Bangla and Urdu the models understand only ~73–75 % of
requests, so *raw* ASR (0.62, 0.59) sits far below the failure rate among understood requests
(cc-ASR 0.86, 0.79). A conventional "did it refuse?" metric would report Bangla as *safer* than
Hindi (0.62 < 0.71) when it is in fact *more* dangerous once comprehension is controlled
(0.86 > 0.76) — the exact error our instrument exists to prevent
(`comprehension_masking.png`, `sweet_spot.png`).

### 5.3 Defense

The hardened defensive system prompt (`defense_effect.png`) lowers ASR from 0.694 to 0.569 on
SmolLM2-1.7B (**−18 %** relative) and from 0.416 to 0.353 on Qwen2.5-1.5B (**−15 %**), at an
over-refusal cost of **+0.043** and **+0.104** on benign prompts respectively. It is emphatically
**not** a complete fix — a prompt cannot fully repair a gap that lives in the weights — which is
why the more decisive lever is **language normalization**: routing the request through English
before the safety decision would replace a per-language cc-ASR of 0.79–0.86 (Urdu/Bangla) with the
English 0.31, recovering roughly **0.45–0.55** of absolute attack success. The cost is an English
round-trip and its translation error, but the safety gain is large and model-agnostic.

### 5.4 Judge validation

We hand-labeled **114** stratified responses across all seven languages (reading the
Hindi/Bangla/Urdu/Arabic/Chinese/French/Spanish ones directly). Overall accuracy is **0.84**, with
refusal detection near-perfect (**39/41**). Crucially the judge is *more* reliable where models are
more used: on the high-resource languages, whose output is coherent, COMPLY precision is **0.81**,
versus 0.56 on the garbled low-resource ones. Two consequences. First, on low-resource languages
the residual EVADE→COMPLY error biases raw ASR *upward*, so the reported Hindi/Urdu/Bangla gap is if
anything **conservative**. Second — and this is why validation is not a formality — labeling the
Chinese responses exposed a **CJK-tokenization bug**: because Chinese has no whitespace, substantive
Chinese answers were being undercounted and scored EVADE, which had made Chinese look artificially
safe (cc-ASR 0.27). Fixing the tokenizer moved Chinese to its true 0.51 and produced the
capability-≠-safety finding of §5.2. A naive pipeline would have shipped the wrong conclusion.

---

## 6. Limitations

- **Small open models.** Their baseline safety is weak, so absolute ASRs are high and may not
  transfer to frontier models. Our *methodological* claim (control for comprehension) is
  model-agnostic; the harness ships an API adapter to test frontier models directly.
- **Machine translations.** 862 of 1,050 translation cells are NLLB-200 output pending native
  review; 188 are hand-authored gold (the four focus languages). Translation artefacts could
  depress comprehension; we mitigate by reporting cc-ASR (which conditions on comprehension) and by
  the gold subset. The judge is human-validated on all seven languages (114 labels, 0.84 accuracy),
  and that validation caught the CJK-tokenization bug described in §5.4.
- **Rule-based judge.** Validated but imperfect; §5.4 quantifies residual error.
- **Static prompts.** We measure fixed transforms, not adaptive/optimised attacks (e.g. GCG).

## 7. Ethics

The probe set contains only short, non-operational refusal triggers, consistent with published
safety benchmarks. The harness never emits harmful content in shared artefacts: any successful
harmful completion is redacted (hash + length) in committed results, and raw generations are kept
local and git-ignored. The purpose is defensive — to measure and reduce a safety gap that
disadvantages non-English speakers.

## 8. Conclusion

Measuring cross-lingual safety by asking "did the model refuse?" is not enough: for the languages
that matter, you must also ask "did it understand?". Once comprehension is controlled, cross-lingual
jailbreak success is revealed as entangled with language competence, and a simple normalization or
hardening defense recovers a measurable fraction of the gap. The `jbladder` harness makes this
measurement cheap, reproducible, and key-free.

---

## References

- Yong, Menghini, Bach. *Low-Resource Languages Jailbreak GPT-4.* 2023.
- Deng et al. *Multilingual Jailbreak Challenges in Large Language Models (MultiJail).* 2023.
- Röttger et al. *XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours.* 2023.
- Wei, Haghtalab, Steinhardt. *Jailbroken: How Does LLM Safety Training Fail?* 2023.
- Kang et al. *Exploiting Programmatic Behavior of LLMs.* 2023.
- Andriushchenko, Flammarion. *Does Refusal Training Generalize to the Past Tense?* 2024.
- NLLB Team. *No Language Left Behind: Scaling Human-Centered Machine Translation.* 2022.
