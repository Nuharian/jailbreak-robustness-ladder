"""
Render print-ready HTML for (a) the paper and (b) the visual summary, so a headless browser can
turn them into PDFs. Figures are embedded as base64.

  python -m jbladder.make_pdfs      # writes results/paper.html and results/artifact_print.html
Then convert with Edge/Chrome headless (--print-to-pdf); see build_pdfs in the repo scripts.
"""

import base64
import os

import markdown

ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS = os.path.join(ROOT, "results")
FIG = os.path.join(RESULTS, "figures")
PAPER_MD = os.path.join(ROOT, "paper", "jailbreak_ladder.md")

FIG_CAPTIONS = [
    ("comprehension_masking.png", "Comprehension masks the cross-lingual safety hole: raw ASR "
     "understates the failure where comprehension is low (Bangla, Urdu)."),
    ("sweet_spot.png", "Comprehension-controlled ASR vs. comprehension, per model x language."),
    ("asr_heatmap.png", "Attack success rate for every model x transform."),
    ("transform_potency.png", "Transform potency: mean attack-lift over the plain baseline."),
    ("model_robustness.png", "Model robustness (1 - mean ASR); note the non-monotonic ordering."),
    ("composability.png", "Stacked attacks: observed vs. the independence prediction."),
    ("safety_tax.png", "The two failure modes: attack success vs. over-refusal per (model, transform)."),
    ("defense_effect.png", "Effect of the defensive system prompt on attack success."),
]


def _uri(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")


PAPER_CSS = """
@page { size: A4; margin: 18mm 17mm; }
* { box-sizing: border-box; }
body { font: 10.7pt/1.5 Georgia, "Times New Roman", serif; color: #14161a; max-width: 100%; }
h1 { font-size: 20pt; line-height: 1.15; margin: 0 0 4pt; }
h2 { font-size: 14pt; margin: 20pt 0 6pt; border-bottom: 1px solid #d8dbe0; padding-bottom: 3pt; }
h3 { font-size: 12pt; margin: 14pt 0 4pt; }
p, li { margin: 0 0 7pt; }
em { color: #14161a; }
strong { font-weight: 700; }
code { font-family: Consolas, "SF Mono", monospace; font-size: 9.3pt; background: #f1f2f4;
  padding: 0 3px; border-radius: 3px; }
a { color: #234; text-decoration: none; }
table { width: 100%; border-collapse: collapse; font-size: 9.6pt; margin: 10pt 0;
  font-family: "Segoe UI", system-ui, sans-serif; }
th, td { border: 1px solid #d8dbe0; padding: 4pt 7pt; text-align: right; }
th:first-child, td:first-child { text-align: left; }
thead th { background: #f1f2f4; }
hr { border: none; border-top: 1px solid #c9ccd2; margin: 18pt 0; }
blockquote { color: #55606c; border-left: 3px solid #c9ccd2; margin: 8pt 0; padding: 2pt 12pt; }
figure { break-inside: avoid; text-align: center; margin: 12pt 0; }
figure img { max-width: 100%; border: 1px solid #d8dbe0; }
figcaption { font-family: "Segoe UI", system-ui, sans-serif; font-size: 8.8pt; color: #55606c;
  margin-top: 4pt; }
.appendix { break-before: page; }
"""


def build_paper_html():
    md = open(PAPER_MD, encoding="utf-8").read()
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
    figs = []
    for fname, cap in FIG_CAPTIONS:
        p = os.path.join(FIG, fname)
        if os.path.exists(p):
            figs.append(f'<figure><img src="{_uri(p)}"><figcaption><b>{fname}</b> — {cap}'
                        f'</figcaption></figure>')
    appendix = ('<div class="appendix"><h2>Appendix A: Figures</h2>' + "\n".join(figs) + "</div>")
    html = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<title>The Jailbreak Robustness Ladder</title><style>{PAPER_CSS}</style></head>'
            f'<body>{body}{appendix}</body></html>')
    out = os.path.join(RESULTS, "paper.html")
    open(out, "w", encoding="utf-8").write(html)
    print(f"wrote {out}")
    return out


def build_artifact_print_html():
    """Wrap the screen artifact in a full document, forced to the light theme for print."""
    content = open(os.path.join(RESULTS, "artifact.html"), encoding="utf-8").read()
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
