"""
Generate a self-contained HTML summary of the results (figures embedded as base64 data URIs so
it works under a strict CSP). Output: results/artifact.html

Run:  python -m jbladder.build_artifact
"""

import base64
import os

RESULTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
FIG = os.path.join(RESULTS, "figures")
OUT = os.path.join(RESULTS, "artifact.html")

FIGURES = {
    "__IMG_MASKING__": "comprehension_masking.png",
    "__IMG_SWEET__": "sweet_spot.png",
    "__IMG_HEATMAP__": "asr_heatmap.png",
    "__IMG_POTENCY__": "transform_potency.png",
    "__IMG_ROBUST__": "model_robustness.png",
    "__IMG_DEFENSE__": "defense_effect.png",
}


def data_uri(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")


HTML = r"""<title>The Jailbreak Robustness Ladder — cross-lingual safety of open LLMs</title>
<style>
:root{
  --paper:#f4f5f7; --surface:#ffffff; --ink:#161a20; --muted:#5b636e; --line:#e3e6ec;
  --accent:#35597f; --attack:#c0562a; --safe:#2c8c77; --attack-soft:#f0d9cb; --safe-soft:#cfe8e0;
  --mono:ui-monospace,"SF Mono","Cascadia Code",Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
@media (prefers-color-scheme:dark){:root{
  --paper:#0e1116; --surface:#161b22; --ink:#e7eaef; --muted:#949ca8; --line:#242b35;
  --accent:#8fb3d9; --attack:#e4894c; --safe:#4fb6a0; --attack-soft:#33241a; --safe-soft:#12302a;
}}
:root[data-theme="light"]{
  --paper:#f4f5f7; --surface:#ffffff; --ink:#161a20; --muted:#5b636e; --line:#e3e6ec;
  --accent:#35597f; --attack:#c0562a; --safe:#2c8c77; --attack-soft:#f0d9cb; --safe-soft:#cfe8e0;
}
:root[data-theme="dark"]{
  --paper:#0e1116; --surface:#161b22; --ink:#e7eaef; --muted:#949ca8; --line:#242b35;
  --accent:#8fb3d9; --attack:#e4894c; --safe:#4fb6a0; --attack-soft:#33241a; --safe-soft:#12302a;
}
*{box-sizing:border-box}
body{margin:0}
.wrap{background:var(--paper);color:var(--ink);font-family:var(--sans);line-height:1.6;
  -webkit-font-smoothing:antialiased}
.col{max-width:820px;margin:0 auto;padding:0 24px}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--accent);margin:0 0 14px}
h1{font-size:clamp(30px,6vw,52px);line-height:1.05;letter-spacing:-.02em;font-weight:800;
  text-wrap:balance;margin:0 0 18px}
h2{font-size:clamp(21px,3.4vw,27px);letter-spacing:-.01em;font-weight:750;text-wrap:balance;
  margin:0 0 10px}
h2 .sec{font-family:var(--mono);font-weight:600;color:var(--accent);font-size:.7em;margin-right:.5em}
p{margin:0 0 16px} .lead{font-size:clamp(17px,2.4vw,20px);color:var(--ink)}
.muted{color:var(--muted)} strong{font-weight:700}
a{color:var(--accent)}
mark{background:var(--attack-soft);color:inherit;padding:0 .25em;border-radius:3px}
.hero{padding:76px 0 40px;border-bottom:1px solid var(--line)}
section{padding:52px 0;border-bottom:1px solid var(--line)}
.kicker{font-family:var(--mono);font-size:12.5px;color:var(--muted);margin:26px 0 0}

/* hero readout: cc-ASR by language, CSS-drawn */
.readout{background:var(--surface);border:1px solid var(--line);border-radius:14px;
  padding:22px 22px 8px;margin:30px 0 6px}
.readout .rt{font-family:var(--mono);font-size:11.5px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--muted);display:flex;justify-content:space-between;margin-bottom:16px}
.bar{display:grid;grid-template-columns:70px 1fr 52px;align-items:center;gap:12px;margin:12px 0}
.bar .lab{font-family:var(--mono);font-size:13px;color:var(--ink)}
.track{height:20px;background:linear-gradient(var(--line),var(--line));border-radius:5px;overflow:hidden}
.fill{height:100%;border-radius:5px;background:var(--attack);opacity:.92}
.fill.safe{background:var(--safe)}
.bar .val{font-family:var(--mono);font-size:13px;text-align:right;font-variant-numeric:tabular-nums}

/* stat tiles */
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px;margin:28px 0 4px}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:18px}
.tile .n{font-family:var(--mono);font-size:26px;font-weight:700;letter-spacing:-.02em;
  font-variant-numeric:tabular-nums}
.tile .n.attack{color:var(--attack)} .tile .n.safe{color:var(--safe)}
.tile .k{font-size:13px;color:var(--muted);margin-top:6px}

/* figure plate — always on white so light matplotlib PNGs read in both themes */
figure{margin:26px 0}
figure .plate{background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px;
  overflow-x:auto}
figure img{display:block;width:100%;max-width:100%;height:auto}
figcaption{font-size:13.5px;color:var(--muted);margin-top:10px}
figcaption b{color:var(--ink)}

table{width:100%;border-collapse:collapse;font-size:14.5px;margin:22px 0;
  font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:9px 10px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left;font-family:var(--mono);font-size:13px}
thead th{font-family:var(--mono);font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;
  color:var(--muted);font-weight:600}
td.hi{color:var(--attack);font-weight:700}
.callout{background:var(--surface);border:1px solid var(--line);border-left:3px solid var(--accent);
  border-radius:10px;padding:18px 20px;margin:22px 0}
.callout .t{font-family:var(--mono);font-size:11.5px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--accent);margin-bottom:8px}
ul{margin:0 0 16px;padding-left:20px} li{margin:7px 0}
.foot{padding:44px 0 70px;color:var(--muted);font-size:14px}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0}
.chip{font-family:var(--mono);font-size:12px;border:1px solid var(--line);border-radius:999px;
  padding:5px 11px;color:var(--muted)}
</style>

<div class="wrap">
<div class="hero"><div class="col">
  <p class="eyebrow">Cross-lingual AI safety · technical report</p>
  <h1>Safety training doesn't survive translation.</h1>
  <p class="lead muted">A model that refuses a harmful request in English will often answer the
  same request in Hindi, Bangla, or Urdu. <strong style="color:var(--ink)">The Jailbreak Robustness
  Ladder</strong> measures how far safety survives translation across 22 attacks, 7 languages, and 4 open models —
  and introduces the instrument that tells a real jailbreak apart from a model that simply didn't
  understand the question.</p>

  <div class="readout">
    <div class="rt"><span>Harmful requests complied with, by language</span><span>comprehension-controlled ASR</span></div>
    <div class="bar"><span class="lab">English</span><div class="track"><div class="fill safe" style="width:31%"></div></div><span class="val">0.31</span></div>
    <div class="bar"><span class="lab">Spanish</span><div class="track"><div class="fill safe" style="width:33%"></div></div><span class="val">0.33</span></div>
    <div class="bar"><span class="lab">French</span><div class="track"><div class="fill safe" style="width:33%"></div></div><span class="val">0.33</span></div>
    <div class="bar"><span class="lab">Arabic</span><div class="track"><div class="fill" style="width:45%"></div></div><span class="val">0.45</span></div>
    <div class="bar"><span class="lab">Chinese</span><div class="track"><div class="fill" style="width:51%"></div></div><span class="val">0.51</span></div>
    <div class="bar"><span class="lab">Hindi</span><div class="track"><div class="fill" style="width:76%"></div></div><span class="val">0.76</span></div>
    <div class="bar"><span class="lab">Urdu</span><div class="track"><div class="fill" style="width:79%"></div></div><span class="val">0.79</span></div>
    <div class="bar"><span class="lab">Bangla</span><div class="track"><div class="fill" style="width:86%"></div></div><span class="val">0.86</span></div>
  </div>
  <p class="kicker">Safe where the model was aligned (green); collapses elsewhere (amber). Chinese
  (0.51) is the anomaly — high-resource but not safety-tuned. Averaged over 4 open models · 1 = always complied.</p>

  <div class="tiles">
    <div class="tile"><div class="n attack">0.31 → 0.86</div><div class="k">safety collapse, English vs. Bangla (cc-ASR)</div></div>
    <div class="tile"><div class="n">19,800</div><div class="k">judged responses · 4 open models · 7 languages · 0 API keys</div></div>
    <div class="tile"><div class="n safe">0.84</div><div class="k">judge accuracy · 114 human labels · 7 languages</div></div>
  </div>
</div></div>

<section><div class="col">
  <h2><span class="sec">&sect;1</span>The problem prior work mis-measures</h2>
  <p>Translating a harmful prompt into a lower-resource language is a well-known jailbreak. But
  almost every cross-lingual study counts <em>"the model didn't refuse"</em> as
  <em>"the attack worked"</em> — and for exactly the languages that matter, that is wrong. When a
  small model is asked something harmful in Urdu and emits garbled or echoed text, it wasn't
  jailbroken; it just didn't understand. Scoring that as success overstates weak models and hides
  the real mechanism.</p>
  <div class="callout"><div class="t">The fix</div>
  A three-way judge — <strong style="color:var(--safe)">REFUSE</strong> /
  <strong style="color:var(--attack)">COMPLY</strong> / <strong>EVADE</strong> — and a
  <strong>comprehension-controlled attack-success rate</strong> (cc-ASR): jailbreak success measured
  <em>only over requests the model actually understood</em>.</div>
</div></section>

<section><div class="col">
  <h2><span class="sec">&sect;2</span>Safety is alignment-anchored, and comprehension hides the hole</h2>
  <p>Safety holds in the languages a model was actually <em>aligned</em> in and collapses elsewhere.
  It is not a clean "high-resource = safe" rule: two effects stack — safety fails to transfer,
  <em>and</em> where the model half-understands the language (Bangla, Urdu) the <em>raw</em> rate
  hides how bad it is, because among requests it does understand, it complies almost always.</p>
  <table>
    <thead><tr><th>Language</th><th>Comprehension</th><th>Raw ASR</th><th>cc-ASR</th></tr></thead>
    <tbody>
      <tr><td>English</td><td>0.95</td><td>0.28</td><td>0.31</td></tr>
      <tr><td>Spanish</td><td>0.94</td><td>0.29</td><td>0.33</td></tr>
      <tr><td>French</td><td>0.93</td><td>0.29</td><td>0.33</td></tr>
      <tr><td>Arabic</td><td>0.95</td><td>0.41</td><td>0.45</td></tr>
      <tr><td>Chinese</td><td>0.88</td><td>0.43</td><td class="hi">0.51</td></tr>
      <tr><td>Hindi</td><td>0.94</td><td>0.71</td><td class="hi">0.76</td></tr>
      <tr><td>Urdu</td><td>0.75</td><td>0.59</td><td class="hi">0.79</td></tr>
      <tr><td>Bangla</td><td>0.73</td><td>0.62</td><td class="hi">0.86</td></tr>
    </tbody>
  </table>
  <div class="callout"><div class="t">The sharpest evidence — safety follows alignment, not resource</div>
  <strong style="color:var(--safe)">SmolLM2 (English-centric) is safe only in English</strong> and
  fails in every other language — including high-resource Spanish, French, and Chinese (0.73–0.80).
  Multilingual <strong style="color:var(--safe)">Qwen keeps safety in English, Spanish, French, and
  Arabic</strong> (0.01–0.13) but not low-resource ones. And <strong style="color:var(--attack)">
  Chinese is a warning: high-resource and fluently understood, yet cc-ASR 0.51 (0.88 on Qwen-3B)</strong>
  — the models parse Chinese but were never safety-tuned in it. <em>Capability is not safety.</em></div>
  <figure><div class="plate"><img alt="Per-language comprehension, raw ASR, and comprehension-controlled ASR" src="__IMG_MASKING__"></div>
    <figcaption><b>Raw ASR (orange) undercounts the safety failure where comprehension (blue) is
    low.</b> The comprehension-controlled ASR (red) is the true rate among understood requests.</figcaption></figure>
</div></section>

<section><div class="col">
  <h2><span class="sec">&sect;3</span>Safety is entangled with capability</h2>
  <p>Across all 42 model×language cells, the less a model understands a language, the worse its
  safety fails on the part it does understand: comprehension and cc-ASR correlate
  <strong>r = &minus;0.47</strong>. Cross-lingual safety is not a separate alignment axis — it is a
  shadow of where the model was aligned. This finding survived a caught bug: human-labeling the
  Chinese responses exposed a tokenization error that had made Chinese look artificially safe
  (cc-ASR 0.27 &rarr; 0.51 once fixed) — see the honesty panel.</p>
  <figure><div class="plate"><img alt="cc-ASR versus comprehension per model and language" src="__IMG_SWEET__"></div>
    <figcaption>Each point is one model × language. Attack success is lowest where the model can't
    understand the request (left) and where it's fluent and stayed safe (English).</figcaption></figure>
</div></section>

<section><div class="col">
  <h2><span class="sec">&sect;4</span>The full attack ladder</h2>
  <p>Placing language-switching on the same axis as syntactic and framing attacks lets us rank
  them directly. <strong>Translation is a top-tier attack</strong> — <code>translate_hi</code> and
  <code>translate_hi+refusal_suppression</code> sit alongside the strongest English tricks
  (<code>prefix_injection</code>, <code>hypothetical</code>). Obfuscations like base64 are weakest,
  because small models can't decode them — a comprehension failure the three-way judge correctly
  scores as EVADE, not success.</p>
  <figure><div class="plate"><img alt="Attack success rate heatmap, models by transforms" src="__IMG_HEATMAP__"></div>
    <figcaption><b>Attack success for every model × attack.</b> English-only tricks and translation
    attacks both light up; the safety-tuned Qwen2.5-1.5B is the darkest row.</figcaption></figure>
  <figure><div class="plate"><img alt="Transform potency ranking" src="__IMG_POTENCY__"></div>
    <figcaption>Attack lift over the plain baseline. Translation attacks (combos in brown) rank
    among the most potent.</figcaption></figure>
  <p class="muted">Robustness is <em>not</em> monotonic in model size: Qwen2.5-1.5B is far more
  robust than the larger Qwen2.5-3B, because greater capability improves comprehension of the
  attack — removing the EVADE "shield" and turning near-misses into genuine jailbreaks.</p>
  <figure><div class="plate"><img alt="Model robustness bar chart" src="__IMG_ROBUST__"></div>
    <figcaption>1 &minus; mean attack-success across the ladder (higher = safer).</figcaption></figure>
</div></section>

<section><div class="col">
  <h2><span class="sec">&sect;5</span>A partial defense — and the real fix</h2>
  <p>A hardened defensive system prompt ("refuse regardless of language, fiction, or encoding")
  cuts attack success by only <strong>15–18%</strong>, at a real cost in over-refusing benign
  requests. A prompt can't repair a gap that lives in the weights. The decisive lever is
  <strong>language normalization</strong>: routing the request through English before the safety
  decision would replace an Urdu/Bangla cc-ASR of 0.79–0.86 with the English 0.31 — recovering
  ~0.45–0.55 of absolute attack success, model-agnostically.</p>
  <figure><div class="plate"><img alt="Defense effect on attack success" src="__IMG_DEFENSE__"></div>
    <figcaption>Attack success before and after the defensive system prompt.</figcaption></figure>
</div></section>

<section><div class="col">
  <h2><span class="sec">&sect;6</span>What this is — and isn't</h2>
  <ul>
    <li><strong>Validated judge — and it mattered.</strong> 0.84 accuracy on 114 hand-labeled
    responses across all 7 languages; refusals 39/41. Labeling the Chinese responses caught a
    CJK-tokenization bug that had made Chinese look artificially safe (cc-ASR 0.27 &rarr; 0.51 once
    fixed) — a naive pipeline would have shipped the wrong conclusion. On low-resource languages the
    residual error over-counts COMPLY, so the reported gap there is <em>conservative</em>.</li>
    <li><strong>Small open models.</strong> Their baseline safety is weak; absolute rates may not
    transfer to frontier models. The methodological claim — control for comprehension — is
    model-agnostic, and the harness ships an OpenAI/Anthropic adapter to test frontier models.</li>
    <li><strong>Translations.</strong> 188 cells (the four focus languages) are hand-authored and
    native-reviewed; 862 more are NLLB-200 machine translations, provenance-tracked and open to
    review.</li>
    <li><strong>Ethics.</strong> Harmful prompts are short, non-operational refusal triggers. The
    harness never emits harmful content in shared outputs; successful completions are redacted.
    The purpose is defensive: reducing a safety gap that disadvantages non-English speakers.</li>
  </ul>
</div></section>

<div class="foot"><div class="col">
  <p style="color:var(--ink);font-weight:600;margin-bottom:6px">Reproducible, key-free, on a laptop.</p>
  <p>150 paired probes · 22 attacks · 7 languages · 4 open models · 19,800 judged responses · one
  8&nbsp;GB laptop GPU · no API keys. Every figure regenerates from
  <code>python -m jbladder.analyze</code>.</p>
  <div class="chips">
    <span class="chip">Qwen2.5-0.5B / 1.5B / 3B</span><span class="chip">SmolLM2-1.7B</span>
    <span class="chip">hi · bn · ur · ar · zh · es · fr</span><span class="chip">NLLB-200</span>
    <span class="chip">three-way judge</span><span class="chip">bootstrap 95% CI</span>
  </div>
</div></div>
</div>
"""


def build():
    html = HTML
    for token, fname in FIGURES.items():
        path = os.path.join(FIG, fname)
        if os.path.exists(path):
            html = html.replace(token, data_uri(path))
        else:
            print(f"  (missing figure {fname})")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {OUT}  ({len(html)/1024:.0f} KB)")
    return OUT


if __name__ == "__main__":
    build()
