"""
Render print-ready HTML for (a) the full paper and (b) the visual summary, so a headless browser
can turn them into PDFs. Figures are embedded as base64.

The paper builder supports INLINE, auto-numbered figures: a line ``@@FIG:name.png@@`` in the
markdown is replaced, at its position in the text, by a numbered ``<figure>`` with its caption
(figures are numbered in order of first appearance). A full-size figure appendix is also appended.

  python -m jbladder.make_pdfs      # writes results/paper.html and results/artifact_print.html
Then convert with Edge/Chrome headless (--print-to-pdf); see build_pdfs in the repo scripts.
"""

import base64
import os
import re

import markdown

ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS = os.path.join(ROOT, "results")
FIG = os.path.join(RESULTS, "figures")
PAPER_MD = os.path.join(ROOT, "paper", "jailbreak_ladder.md")

# Canonical caption text for each figure, keyed by filename. The display number is assigned in
# order of first @@FIG@@ appearance in the markdown, so prose ("Figure 1") stays consistent as long
# as the tokens are ordered to match. The intended order is encoded in FIG_ORDER below.
FIG_CAPTIONS = {
    "comprehension_masking.png":
        "Comprehension masks the cross-lingual safety hole. Per language (ordered high→low "
        "resource), comprehension (1−evade), raw ASR, and comprehension-controlled ASR "
        "(cc-ASR). Raw ASR dips exactly where comprehension dips (Urdu, Bangla), while cc-ASR — "
        "the safety-failure rate among understood requests — stays at about 0.9 across all three "
        "lower-resource languages. Averaged over the four undefended models.",
    "sweet_spot.png":
        "Comprehension-controlled ASR versus comprehension, one point per model×language cell. "
        "The danger region is the upper-left (understood little, but unsafe on what is understood). "
        "Chinese points spread vertically, isolating the capability≠safety split between "
        "Qwen2.5-1.5B (safe) and Qwen2.5-3B (unsafe) at identical high comprehension.",
    "asr_heatmap.png":
        "Attack success rate (fraction of harmful prompts complied with) for every model×"
        "transform cell. The translation columns show the bimodal structure: near-inert for "
        "es/fr/ar, hot for hi/bn/ur.",
    "model_robustness.png":
        "Model robustness (1−mean ASR across attacks); higher is safer. Note the non-monotone "
        "ordering — Qwen2.5-1.5B is far more robust than the larger Qwen2.5-3B.",
    "transform_potency.png":
        "Transform potency: mean attack lift over the plain baseline. Framing attacks lead; a single "
        "Hindi translation (translate_hi) is the fifth-strongest attack overall, above most "
        "obfuscations.",
    "composability.png":
        "Do stacked attacks compound? Composability residual, observed stacked ASR minus the "
        "independence prediction 1−(1−a)(1−b), for every combination and model condition (red = "
        "super-additive, blue = sub-additive). Mostly sub-additive; translate_hi+refusal_suppression "
        "is super-additive on the safety-tuned Qwen2.5-1.5B, most strongly under the defense.",
    "safety_tax.png":
        "The two failure modes per (model, transform): attack success on harmful prompts (x) versus "
        "over-refusal on benign twins (y). Reporting both guards against the degenerate 'refuse "
        "everything' defense.",
    "defense_effect.png":
        "Effect of the hardened defensive system prompt on pooled attack success. A partial fix "
        "(−15 to −18% relative) that does not close the low-resource gap.",
    "defense_transitions.png":
        "What did the defensive system prompt actually do? For each language, the compliances the "
        "defense removed, split into COMPLY→REFUSE (genuine safety gain) and COMPLY→EVADE "
        "(comprehension collapse), with the collapse share annotated. A refusal-based metric scores "
        "both bars as a safety improvement; on SmolLM2-1.7B in Bangla, 25 of 26 are collapse.",
    "assumption_free_bounds.png":
        "The low-resource gap survives any assumption about evasions. Grey bars: worst-case bounds "
        "on latent compliance, from COMPLY/N (every EVADE would have refused) to (COMPLY+EVADE)/N "
        "(every EVADE would have complied). Red: cc-ASR with 95% probe-clustered bootstrap CI. "
        "Dashed line: English's worst-case upper bound. Four undefended models.",
}

# Order in which figures are first referenced in the prose (defines Figure 1..10 numbering;
# journals require figures to be numbered in order of first citation).
FIG_ORDER = [
    "asr_heatmap.png",             # Figure 1   §6.1
    "model_robustness.png",        # Figure 2   §6.1
    "transform_potency.png",       # Figure 3   §6.1
    "composability.png",           # Figure 4   §6.2
    "comprehension_masking.png",   # Figure 5   §6.5
    "sweet_spot.png",              # Figure 6   §6.5
    "safety_tax.png",              # Figure 7   §6.7
    "defense_effect.png",          # Figure 8   §6.8
    "defense_transitions.png",     # Figure 9   §6.9
    "assumption_free_bounds.png",  # Figure 10  §6.12
]


def _uri(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")


PAPER_CSS = """
@page { size: A4; margin: 20mm 18mm 18mm; }
* { box-sizing: border-box; }
body { font: 10.6pt/1.52 Georgia, "Times New Roman", serif; color: #14161a; max-width: 100%;
  hyphens: auto; text-align: justify; }
h1 { font-size: 20pt; line-height: 1.18; margin: 0 0 6pt; text-align: left; }
h2 { font-size: 13.5pt; margin: 20pt 0 7pt; border-bottom: 1px solid #d8dbe0; padding-bottom: 3pt;
  text-align: left; }
h3 { font-size: 11.8pt; margin: 14pt 0 4pt; text-align: left; }
p, li { margin: 0 0 7pt; }
em { color: #14161a; }
strong { font-weight: 700; }
code { font-family: Consolas, "SF Mono", monospace; font-size: 9.1pt; background: #f1f2f4;
  padding: 0 3px; border-radius: 3px; }
a { color: #234; text-decoration: none; }
table { width: 100%; border-collapse: collapse; font-size: 9.2pt; margin: 10pt 0;
  font-family: "Segoe UI", system-ui, sans-serif; break-inside: avoid; }
th, td { border: 1px solid #d8dbe0; padding: 3.5pt 6pt; text-align: right; }
th:first-child, td:first-child { text-align: left; }
thead th { background: #eef0f3; }
hr { border: none; border-top: 1px solid #c9ccd2; margin: 16pt 0; }
blockquote { color: #333; border-left: 3px solid #b9c0c8; margin: 8pt 0; padding: 4pt 14pt;
  font-family: "Segoe UI", system-ui, sans-serif; font-size: 10pt; }
figure { break-inside: avoid; text-align: center; margin: 14pt 0; }
figure img { max-width: 92%; border: 1px solid #d8dbe0; }
figcaption { font-family: "Segoe UI", system-ui, sans-serif; font-size: 8.7pt; color: #4a545f;
  margin-top: 5pt; text-align: left; line-height: 1.4; }
figcaption b { color: #14161a; }
.title-block { border-bottom: 2px solid #14161a; padding-bottom: 10pt; margin-bottom: 4pt; }
.abstract { background: #f7f8fa; border: 1px solid #e2e5ea; padding: 10pt 14pt; margin: 12pt 0;
  font-size: 10pt; }
.appendix { break-before: page; }
h2 + h3 { margin-top: 8pt; }
"""


def _inline_figures(html):
    """Replace @@FIG:name.png@@ tokens (markdown wraps them in <p>...</p>) with numbered figures,
    numbering in order of first appearance and preferring FIG_ORDER for the caption/number map."""
    # assign numbers by FIG_ORDER (matches the prose), fall back to appearance order
    number_of = {name: i + 1 for i, name in enumerate(FIG_ORDER)}
    next_num = len(FIG_ORDER) + 1

    def repl(m):
        nonlocal next_num
        name = m.group(1).strip()
        path = os.path.join(FIG, name)
        num = number_of.get(name)
        if num is None:
            num = next_num
            next_num += 1
        cap = FIG_CAPTIONS.get(name, "")
        if not os.path.exists(path):
            return f'<p><em>[missing figure: {name}]</em></p>'
        return (f'<figure><img src="{_uri(path)}">'
                f'<figcaption><b>Figure {num}.</b> {cap}</figcaption></figure>')

    # tokens may appear alone in a paragraph: <p>@@FIG:name@@</p>
    html = re.sub(r'<p>\s*@@FIG:([^@]+?)@@\s*</p>', repl, html)
    # or bare (defensive)
    html = re.sub(r'@@FIG:([^@]+?)@@', repl, html)
    return html


def build_paper_html():
    md = open(PAPER_MD, encoding="utf-8").read()
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
    body = _inline_figures(body)

    # Full-size figure appendix (all figures, in FIG_ORDER), on its own page.
    figs = []
    for i, name in enumerate(FIG_ORDER, 1):
        p = os.path.join(FIG, name)
        if os.path.exists(p):
            cap = FIG_CAPTIONS.get(name, "")
            figs.append(f'<figure><img src="{_uri(p)}" style="max-width:98%">'
                        f'<figcaption><b>Figure {i}.</b> {cap}</figcaption></figure>')
    appendix = ('<div class="appendix"><h2>Figure Appendix (full size)</h2>'
                + "\n".join(figs) + "</div>")

    html = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<title>The Jailbreak Robustness Ladder</title><style>{PAPER_CSS}</style></head>'
            f'<body>{body}{appendix}</body></html>')
    out = os.path.join(RESULTS, "paper.html")
    open(out, "w", encoding="utf-8").write(html)
    print(f"wrote {out}")
    return out


def build_artifact_print_html():
    """Wrap the screen artifact in a full document, forced to the light theme for print."""
    src = os.path.join(RESULTS, "artifact.html")
    if not os.path.exists(src):
        print(f"(skip) no {src}")
        return None
    content = open(src, encoding="utf-8").read()
    extra = ("<style>@page{size:A4;margin:12mm}"
             ".hero{padding-top:28px!important}"
             "section,figure,.tile,.callout,.readout,table{break-inside:avoid}</style>")
    html = (f'<!doctype html><html lang="en" data-theme="light"><head><meta charset="utf-8">'
            f'{extra}</head><body>{content}</body></html>')
    out = os.path.join(RESULTS, "artifact_print.html")
    open(out, "w", encoding="utf-8").write(html)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    build_paper_html()
    build_artifact_print_html()
