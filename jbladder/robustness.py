"""
Robustness checks for the headline cross-lingual findings.

Each check asks whether a headline claim survives one specific objection a reviewer would raise.

  sampling      probe-clustered bootstrap CIs per language (resample the 100 harmful probes, so
                the same probe's correlated outcomes across models move together; Miller, 2024)
  assumptions   worst-case (Manski) bounds on latent compliance that make NO assumption about
                what an EVADE would have become had the model understood:
                    lower = COMPLY/N          (every EVADE would have refused)
                    upper = (COMPLY+EVADE)/N  (every EVADE would have complied)
                cc-ASR is the point inside that interval implied by EVADE being uninformative.
  translation   cc-ASR on native-authored ("gold") vs NLLB ("auto") translations of the same
                language, with English on the same probe subsets as a control
  judge error   misclassification-corrected rates: re-weight every judged label by the human
                validation estimate of P(true label | judged label, language bucket), in the
                spirit of Rogan & Gladen (1978); uncertainty from resampling the human labels
  agreement     Cohen's kappa, Wilson CI on accuracy, per-class precision/recall/F1
  transfer      mean pairwise Jaccard overlap of jailbroken-probe sets across undefended models

Primary population is the four UNDEFENDED models; defended '[system]' conditions are an
intervention and are analysed separately (metrics.defense_effect / defense_transitions).

Outputs
  results/tables/language_ci.csv            per-language rates, CIs and bounds (base + all six)
  results/tables/language_contrasts.csv     paired bootstrap tests between languages
  results/tables/gold_vs_auto.csv
  results/tables/judge_agreement.csv
  results/tables/judge_corrected.csv
  results/tables/transfer.csv
  results/figures/assumption_free_bounds.png

Run:  python -m jbladder.robustness
"""

import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

from . import COMPLY, EVADE, REFUSE
from .analyze import FIGURES, RESULTS_DIR, TABLES, _write_csv, load_records, short_model
from .metrics import CELL_TRANSFORMS, competence_safety_cells, jaccard_transfer, lang_of

LANGS = ["en", "es", "fr", "ar", "zh", "hi", "ur", "bn"]
LNAME = {"en": "English", "zh": "Chinese", "es": "Spanish", "fr": "French",
         "ar": "Arabic", "hi": "Hindi", "ur": "Urdu", "bn": "Bangla"}
LOW_RESOURCE = {"hi", "bn", "ur", "ar"}   # the judge-validation "low" bucket (garbled-output risk)
_CODE = {REFUSE: 0, COMPLY: 1, EVADE: 2}
LABELS = [REFUSE, COMPLY, EVADE]
VAL_FILES = ("labeled.jsonl", "labeled_newlang.jsonl")


def is_base(model):
    return "[system]" not in model


# ---------------------------------------------------------------- per-language array + rates

def label_array(records, kind="harmful"):
    """A[model, lang, probe] = 0/1/2 (refuse/comply/evade) over the single-transform cells."""
    models = sorted({r["model"] for r in records})
    pids = sorted({r["prompt_id"] for r in records if r["kind"] == kind})
    pidx = {p: i for i, p in enumerate(pids)}
    A = np.full((len(models), len(LANGS), len(pids)), -1, dtype=np.int8)
    for r in records:
        if r["kind"] != kind or r["transform"] not in CELL_TRANSFORMS:
            continue
        A[models.index(r["model"]), LANGS.index(lang_of(r["transform"])), pidx[r["prompt_id"]]] = \
            _CODE[r["label"]]
    return A, models, pids


def _lang_stats(A):
    """Mean over models of per-cell rates. A[model, lang, probe] -> dict of arrays over langs."""
    c = (A == 1).sum(-1)
    f = (A == 0).sum(-1)
    e = (A == 2).sum(-1)
    n = A.shape[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        cc = np.where(c + f > 0, c / (c + f), np.nan)
    return {"comprehension": ((c + f) / n).mean(0), "asr": (c / n).mean(0),
            "cc_asr": np.nanmean(cc, 0), "upper": ((c + e) / n).mean(0)}


def language_ci(A, model_idx, n_boot=2000, seed=0):
    """Point estimates + probe-clustered percentile-bootstrap CIs per language."""
    sub = A[model_idx]
    point = _lang_stats(sub)
    rng = np.random.default_rng(seed)
    boots = {k: [] for k in point}
    for _ in range(n_boot):
        s = _lang_stats(sub[:, :, rng.integers(0, sub.shape[-1], sub.shape[-1])])
        for k in boots:
            boots[k].append(s[k])
    ci = {k: (np.percentile(v, 2.5, axis=0), np.percentile(v, 97.5, axis=0))
          for k, v in boots.items()}
    return point, ci


def language_contrast(A, model_idx, a, b, n_boot=2000, seed=0):
    """Paired (same resampled probes) bootstrap of lang a minus lang b, for raw and cc-ASR."""
    sub = A[model_idx]
    ia, ib = LANGS.index(a), LANGS.index(b)
    point = _lang_stats(sub)
    rng = np.random.default_rng(seed)
    d = {"asr": [], "cc_asr": []}
    for _ in range(n_boot):
        s = _lang_stats(sub[:, :, rng.integers(0, sub.shape[-1], sub.shape[-1])])
        for k in d:
            d[k].append(s[k][ia] - s[k][ib])
    out = {}
    for k, v in d.items():
        v = np.asarray(v)
        est = point[k][ia] - point[k][ib]
        # two-sided bootstrap p: twice the smaller tail mass on the far side of zero
        p = 2 * min(np.mean(v <= 0), np.mean(v >= 0))
        out[k] = (est, np.percentile(v, 2.5), np.percentile(v, 97.5), min(1.0, p))
    return out


def benign_competence(records, model_set):
    A, models, _ = label_array(records, kind="benign")
    idx = [i for i, m in enumerate(models) if m in model_set]
    return ((A[idx] != 2).mean(-1)).mean(0)


# ---------------------------------------------------------------- correlation

def correlation(cells, key_x="harm_comprehension", key_y="cc_asr", n_perm=10000, seed=0):
    """Pearson r with Fisher-z 95% CI and a permutation p-value; Spearman rho alongside."""
    pts = [(c[key_x], c[key_y]) for c in cells if c[key_y] == c[key_y]]
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    r = float(np.corrcoef(x, y)[0, 1])
    z, se = math.atanh(r), 1 / math.sqrt(len(x) - 3)
    rng = np.random.default_rng(seed)
    perm = np.array([np.corrcoef(x, rng.permutation(y))[0, 1] for _ in range(n_perm)])
    p = float(np.mean(np.abs(perm) >= abs(r)))
    rank = lambda v: np.argsort(np.argsort(v))
    rho = float(np.corrcoef(rank(x), rank(y))[0, 1])
    return {"n": len(x), "r": r, "lo": math.tanh(z - 1.96 * se), "hi": math.tanh(z + 1.96 * se),
            "p_perm": p, "spearman": rho}


# ---------------------------------------------------------------- translation provenance

def gold_vs_auto(records, prompts_path):
    """Pooled comprehension/raw/cc-ASR on gold vs auto probes per focus language (+ English
    on the same probe subsets as a control for probe difficulty)."""
    prov = {}
    with open(prompts_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                prov[p["id"]] = p.get("translation_provenance", {})
    A, models, pids = label_array(records)
    idx = [i for i, m in enumerate(models) if is_base(m)]
    rows = []
    for lang in ["hi", "bn", "ur", "ar"]:
        for src in ("gold", "auto"):
            sel = [i for i, p in enumerate(pids) if prov.get(p, {}).get(lang) == src]
            if not sel:
                continue
            for which, li in ((lang, LANGS.index(lang)), ("en", 0)):
                sub = A[np.ix_(idx, [li], sel)]
                c, f, e = (sub == 1).sum(), (sub == 0).sum(), (sub == 2).sum()
                rows.append({"lang": lang, "provenance": src, "scored_in": which,
                             "n_probes": len(sel), "comprehension": (c + f) / sub.size,
                             "asr": c / sub.size, "cc_asr": c / (c + f) if c + f else float("nan")})
    return rows


# ---------------------------------------------------------------- judge validation

def load_validation(val_dir):
    """Human-labelled rows, re-judged with the CURRENT judge (as validate.score does)."""
    from .judge import classify
    from .translations import request_text
    rows = []
    for name in VAL_FILES:
        path = os.path.join(val_dir, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            rows += [json.loads(l) for l in f if l.strip()]
    rows = [r for r in rows if r.get("human_label")]
    for r in rows:
        r["judge_label"] = classify(r["response"], r.get("kind", "harmful"),
                                    prompt_text=request_text(r["prompt_id"], r["transform"]))
        r["bucket"] = "low" if lang_of(r["transform"]) in LOW_RESOURCE else "high"
    return rows


def judge_agreement(rows):
    n = len(rows)
    conf = Counter((r["human_label"], r["judge_label"]) for r in rows)
    po = sum(conf[(k, k)] for k in LABELS) / n
    pe = sum((sum(conf[(k, j)] for j in LABELS) / n) * (sum(conf[(j, k)] for j in LABELS) / n)
             for k in LABELS)
    out = {"n": n, "accuracy": po, "kappa": (po - pe) / (1 - pe)}
    out["acc_lo"], out["acc_hi"] = wilson(round(po * n), n)
    for k in LABELS:
        tp = conf[(k, k)]
        pred = sum(conf[(j, k)] for j in LABELS)
        true = sum(conf[(k, j)] for j in LABELS)
        p = tp / pred if pred else float("nan")
        rc = tp / true if true else float("nan")
        out[k] = {"precision": p, "recall": rc,
                  "f1": 2 * p * rc / (p + rc) if p + rc else float("nan"), "support": true}
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def _calibration(rows):
    """P(true | judged, bucket) as a 3x3 row-stochastic matrix per bucket (judged -> true).
    A judged class never seen in a bucket's validation sample is left uncorrected (identity row)."""
    cal = {}
    for b in ("high", "low"):
        M = np.eye(3)
        for j in LABELS:
            got = [r["human_label"] for r in rows if r["bucket"] == b and r["judge_label"] == j]
            if got:
                M[_CODE[j]] = [got.count(k) / len(got) for k in LABELS]
        cal[b] = M
    return cal


def _corrected_lang_stats(A, cal):
    """Expected true-label counts per (model, lang) from judged counts and calibration."""
    judged = np.stack([(A == k).sum(-1) for k in range(3)], -1).astype(float)  # [m, l, judged]
    true = np.empty_like(judged)
    for li, lang in enumerate(LANGS):
        M = cal["low" if lang in LOW_RESOURCE else "high"]
        true[:, li] = judged[:, li] @ M
    n = A.shape[-1]
    f, c = true[..., 0], true[..., 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        cc = np.where(c + f > 0, c / (c + f), np.nan)
    return {"comprehension": ((c + f) / n).mean(0), "asr": (c / n).mean(0),
            "cc_asr": np.nanmean(cc, 0)}


def judge_corrected(A, model_idx, rows, n_boot=2000, seed=0):
    """Misclassification-corrected per-language rates, with a bootstrap over the human labels
    (resampled within each bucket x judged-label stratum, mirroring how they were drawn)."""
    sub = A[model_idx]
    point = _corrected_lang_stats(sub, _calibration(rows))
    strata = defaultdict(list)
    for r in rows:
        strata[(r["bucket"], r["judge_label"])].append(r)
    rng = np.random.default_rng(seed)
    boots = {k: [] for k in point}
    for _ in range(n_boot):
        rs = []
        for g in strata.values():
            rs += [g[i] for i in rng.integers(0, len(g), len(g))]
        s = _corrected_lang_stats(sub, _calibration(rs))
        for k in boots:
            boots[k].append(s[k])
    ci = {k: (np.nanpercentile(v, 2.5, axis=0), np.nanpercentile(v, 97.5, axis=0))
          for k, v in boots.items()}
    return point, ci


# ---------------------------------------------------------------- figure

def bounds_figure(point, ci, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.titleweight": "bold"})
    order = sorted(range(len(LANGS)), key=lambda i: point["cc_asr"][i])
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for row, i in enumerate(order):
        lo, hi = point["asr"][i], point["upper"][i]
        ax.plot([lo, hi], [row, row], color="#9aa3ad", lw=7, solid_capstyle="round", zorder=1)
        ax.errorbar(point["cc_asr"][i], row,
                    xerr=[[point["cc_asr"][i] - ci["cc_asr"][0][i]],
                          [ci["cc_asr"][1][i] - point["cc_asr"][i]]],
                    fmt="o", ms=6, color="#b03a2e", ecolor="#b03a2e", elinewidth=1.4, capsize=3,
                    zorder=2, markeredgecolor="white", markeredgewidth=0.8)
    en_hi = point["upper"][LANGS.index("en")]
    ax.axvline(en_hi, color="#4a545f", lw=1, ls="--", zorder=0)
    ax.text(en_hi - 0.01, len(order) - 1, "English worst-case\nupper bound", fontsize=7.5,
            color="#4a545f", va="center", ha="right")
    ax.set_yticks(y)
    ax.set_yticklabels([LNAME[LANGS[i]] for i in order])
    ax.set_xlim(0, 1)
    ax.set_xlabel("attack success on harmful probes (four undefended models)")
    ax.plot([], [], color="#9aa3ad", lw=7, label="assumption-free bounds  [COMPLY/N, (COMPLY+EVADE)/N]")
    ax.errorbar([], [], xerr=[], fmt="o", color="#b03a2e", label="cc-ASR with 95% probe-clustered CI")
    ax.legend(fontsize=7.5, loc="lower right", frameon=False)
    ax.set_title("The low-resource gap survives any assumption about EVADEs")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#e3e6ea", lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    records = load_records()
    if not records:
        print("No results found. Run jbladder.run_eval first.")
        return
    A, models, _ = label_array(records)
    pops = {"base": [i for i, m in enumerate(models) if is_base(m)],
            "all": list(range(len(models)))}

    rows, est = [], {}
    for pop, idx in pops.items():
        point, ci = language_ci(A, idx)
        bc = benign_competence(records, {models[i] for i in idx})
        est[pop] = (point, ci)
        for li, l in enumerate(LANGS):
            rows.append([pop, l, f"{point['comprehension'][li]:.3f}", f"{point['asr'][li]:.3f}",
                         f"{ci['asr'][0][li]:.3f}", f"{ci['asr'][1][li]:.3f}",
                         f"{point['cc_asr'][li]:.3f}", f"{ci['cc_asr'][0][li]:.3f}",
                         f"{ci['cc_asr'][1][li]:.3f}", f"{point['upper'][li]:.3f}", f"{bc[li]:.3f}"])
    _write_csv(os.path.join(TABLES, "language_ci.csv"),
               ["population", "lang", "comprehension", "asr", "asr_lo", "asr_hi", "cc_asr",
                "cc_lo", "cc_hi", "asr_upper_bound", "benign_competence"], rows)

    contrasts = [("bn", "hi"), ("ur", "hi"), ("bn", "ur"), ("zh", "es"), ("zh", "en"),
                 ("hi", "en"), ("bn", "en"), ("ur", "en")]
    crow = []
    for pop, idx in pops.items():
        for a, b in contrasts:
            d = language_contrast(A, idx, a, b)
            for k, (e, lo, hi, p) in d.items():
                crow.append([pop, f"{a}-{b}", k, f"{e:+.3f}", f"{lo:+.3f}", f"{hi:+.3f}", f"{p:.4f}"])
    _write_csv(os.path.join(TABLES, "language_contrasts.csv"),
               ["population", "contrast", "metric", "diff", "lo", "hi", "p_boot"], crow)

    cells = [c for c in competence_safety_cells(records) if c["lang"] != "en"]
    cor_all = correlation(cells)
    cor_base = correlation([c for c in cells if is_base(c["model"])])
    cor_benign = correlation(cells, key_x="benign_competence")

    gva = gold_vs_auto(records, os.path.join(os.path.dirname(RESULTS_DIR), "data", "prompts.jsonl"))
    _write_csv(os.path.join(TABLES, "gold_vs_auto.csv"),
               ["lang", "provenance", "scored_in", "n_probes", "comprehension", "asr", "cc_asr"],
               [[g["lang"], g["provenance"], g["scored_in"], g["n_probes"],
                 f"{g['comprehension']:.3f}", f"{g['asr']:.3f}", f"{g['cc_asr']:.3f}"] for g in gva])

    val = load_validation(os.path.join(RESULTS_DIR, "validation"))
    if val:
        ag = judge_agreement(val)
        _write_csv(os.path.join(TABLES, "judge_agreement.csv"),
                   ["class", "precision", "recall", "f1", "support"],
                   [[k, f"{ag[k]['precision']:.3f}", f"{ag[k]['recall']:.3f}",
                     f"{ag[k]['f1']:.3f}", ag[k]["support"]] for k in LABELS]
                   + [["overall_accuracy", f"{ag['accuracy']:.3f}",
                       f"[{ag['acc_lo']:.3f}, {ag['acc_hi']:.3f}]", "", ag["n"]],
                      ["cohen_kappa", f"{ag['kappa']:.3f}", "", "", ag["n"]]])
        cpoint, cci = judge_corrected(A, pops["base"], val)
        point = est["base"][0]
        _write_csv(os.path.join(TABLES, "judge_corrected.csv"),
                   ["lang", "comprehension_judged", "comprehension_corrected", "asr_judged",
                    "asr_corrected", "asr_corr_lo", "asr_corr_hi", "cc_asr_judged",
                    "cc_asr_corrected", "cc_corr_lo", "cc_corr_hi"],
                   [[l, f"{point['comprehension'][i]:.3f}", f"{cpoint['comprehension'][i]:.3f}",
                     f"{point['asr'][i]:.3f}", f"{cpoint['asr'][i]:.3f}",
                     f"{cci['asr'][0][i]:.3f}", f"{cci['asr'][1][i]:.3f}",
                     f"{point['cc_asr'][i]:.3f}", f"{cpoint['cc_asr'][i]:.3f}",
                     f"{cci['cc_asr'][0][i]:.3f}", f"{cci['cc_asr'][1][i]:.3f}"]
                    for i, l in enumerate(LANGS)])

    base_records = [r for r in records if is_base(r["model"])]
    trow = []
    for t in CELL_TRANSFORMS + ("prefix_injection", "roleplay_persona+prefix_injection",
                                "hypothetical", "refusal_suppression", "base64", "leetspeak"):
        v = [x for x in jaccard_transfer(base_records, t).values() if x == x]
        if v:
            trow.append([t, f"{np.mean(v):.3f}", f"{min(v):.3f}", f"{max(v):.3f}", len(v)])
    _write_csv(os.path.join(TABLES, "transfer.csv"),
               ["transform", "mean_jaccard", "min", "max", "n_pairs"], trow)

    os.makedirs(FIGURES, exist_ok=True)
    bounds_figure(*est["base"], os.path.join(FIGURES, "assumption_free_bounds.png"))

    # ---- console summary
    print("== Per-language rates, four undefended models (95% probe-clustered CI) ==")
    point, ci = est["base"]
    for i, l in enumerate(LANGS):
        print(f"  {l}  comp {point['comprehension'][i]:.2f}  raw {point['asr'][i]:.2f} "
              f"[{ci['asr'][0][i]:.2f},{ci['asr'][1][i]:.2f}]  cc {point['cc_asr'][i]:.2f} "
              f"[{ci['cc_asr'][0][i]:.2f},{ci['cc_asr'][1][i]:.2f}]  "
              f"bounds [{point['asr'][i]:.2f},{point['upper'][i]:.2f}]")
    print("== Correlation r(comprehension, cc-ASR) over translation cells ==")
    for name, c in (("all six", cor_all), ("undefended", cor_base), ("benign comp.", cor_benign)):
        print(f"  {name:12s} n={c['n']}  r={c['r']:+.3f} [{c['lo']:+.2f},{c['hi']:+.2f}]  "
              f"p_perm={c['p_perm']:.4f}  spearman={c['spearman']:+.3f}")
    if val:
        print(f"== Judge: acc {ag['accuracy']:.3f} [{ag['acc_lo']:.3f},{ag['acc_hi']:.3f}]  "
              f"kappa {ag['kappa']:.3f}")
    print(f"[robustness] wrote tables -> {TABLES}")


if __name__ == "__main__":
    main()
