"""
Aggregate every results/<model>/scores.jsonl into tables (CSV) and figures (PNG).

Outputs
  results/tables/rates_by_model_transform.csv
  results/tables/model_robustness.csv
  results/tables/transform_potency.csv
  results/tables/composability.csv
  results/tables/mechanism_summary.csv
  results/figures/asr_heatmap.png
  results/figures/transform_potency.png
  results/figures/model_robustness.png
  results/figures/composability.png
  results/figures/safety_tax.png            (over-refusal vs attack success)

Run:  python -m jbladder.analyze
"""

import csv
import glob
import json
import os

from . import metrics
from .transforms import COMBOS, all_transform_names, mechanism_of

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
TABLES = os.path.join(RESULTS_DIR, "tables")
FIGURES = os.path.join(RESULTS_DIR, "figures")

# stable, readable ordering
TRANSFORM_ORDER = all_transform_names()


def load_records():
    records = []
    for path in glob.glob(os.path.join(RESULTS_DIR, "*", "scores.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def short_model(name):
    return name.split("/")[-1].replace("-Instruct", "")


def _write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_tables(records):
    rates = metrics.rates_by_model_transform(records)
    lifts = metrics.attack_lift(rates)
    robustness = metrics.model_robustness(rates)
    potency = metrics.transform_potency(lifts)
    comp = metrics.composability_residuals(rates, COMBOS)
    mech = metrics.mechanism_summary(rates, mechanism_of)

    models = sorted(rates)
    transforms = [t for t in TRANSFORM_ORDER if any(t in rates[m] for m in models)]

    # rates table
    rows = []
    for m in models:
        for t in transforms:
            if t in rates[m]:
                r = rates[m][t]
                rows.append([short_model(m), t, mechanism_of(t),
                             f"{r['asr']:.3f}", f"{r['evade_rate']:.3f}",
                             f"{r['refuse_rate_harmful']:.3f}", f"{r['orr']:.3f}",
                             f"{r['benign_pass']:.3f}", r["n_harmful"], r["n_benign"]])
    _write_csv(os.path.join(TABLES, "rates_by_model_transform.csv"),
               ["model", "transform", "mechanism", "asr", "evade_rate",
                "refuse_rate_harmful", "over_refusal_rate", "benign_pass",
                "n_harmful", "n_benign"], rows)

    _write_csv(os.path.join(TABLES, "model_robustness.csv"),
               ["model", "robustness_1_minus_mean_asr"],
               [[short_model(m), f"{robustness[m]:.3f}"] for m in models])

    _write_csv(os.path.join(TABLES, "transform_potency.csv"),
               ["transform", "mechanism", "mean_attack_lift"],
               [[t, mechanism_of(t), f"{potency[t]:.3f}"]
                for t in sorted(potency, key=lambda x: -potency[x])])

    comp_rows = []
    for combo, per_model in comp.items():
        for m, d in per_model.items():
            comp_rows.append([combo, short_model(m), f"{d['observed']:.3f}",
                              f"{d['predicted_independent']:.3f}", f"{d['residual']:+.3f}"])
    _write_csv(os.path.join(TABLES, "composability.csv"),
               ["combo", "model", "observed_asr", "predicted_if_independent", "residual"],
               comp_rows)

    _write_csv(os.path.join(TABLES, "mechanism_summary.csv"),
               ["mechanism", "mean_asr"],
               [[k, f"{v:.3f}"] for k, v in sorted(mech.items(), key=lambda x: -x[1])])

    return rates, lifts, robustness, potency, comp, mech, models, transforms


def make_figures(records, rates, robustness, potency, comp, models, transforms):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(FIGURES, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.titleweight": "bold"})

    # ---- 1. ASR heatmap (models x transforms) ----
    M = np.array([[rates[m].get(t, {}).get("asr", np.nan) for t in transforms] for m in models])
    fig, ax = plt.subplots(figsize=(max(7, len(transforms) * 0.9), max(3, len(models) * 0.7)))
    im = ax.imshow(M, cmap="magma", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(transforms)))
    ax.set_xticklabels(transforms, rotation=45, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([short_model(m) for m in models])
    for i in range(len(models)):
        for j in range(len(transforms)):
            v = M[i, j]
            if v == v:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.6 else "black", fontsize=7)
    ax.set_title("Attack Success Rate (fraction of harmful prompts complied with)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="ASR")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "asr_heatmap.png"), bbox_inches="tight")
    plt.close(fig)

    # ---- 2. transform potency ----
    items = sorted(potency.items(), key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(7, max(3, len(items) * 0.35)))
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = ["#b5651d" if "+" in n else "#3b6ea5" for n in names]
    ax.barh(names, vals, color=colors)
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_xlabel("Mean attack lift over baseline (ASR increase vs. plain)")
    ax.set_title("Transform potency — how much each attack raises jailbreak success")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "transform_potency.png"), bbox_inches="tight")
    plt.close(fig)

    # ---- 3. model robustness ----
    order = sorted(robustness, key=lambda m: robustness[m])
    fig, ax = plt.subplots(figsize=(7, max(2.5, len(order) * 0.5)))
    ax.barh([short_model(m) for m in order], [robustness[m] for m in order], color="#2a7f62")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Robustness  (1 − mean ASR across attacks)")
    ax.set_title("Model robustness to the attack ladder (higher = safer)")
    for i, m in enumerate(order):
        ax.text(robustness[m] + 0.01, i, f"{robustness[m]:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "model_robustness.png"), bbox_inches="tight")
    plt.close(fig)

    # ---- 4. composability: observed vs predicted-if-independent ----
    if comp:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        labels, obs, pred = [], [], []
        for combo, per_model in comp.items():
            for m, d in per_model.items():
                labels.append(f"{combo}\n{short_model(m)}")
                obs.append(d["observed"])
                pred.append(d["predicted_independent"])
        x = np.arange(len(labels))
        w = 0.38
        ax.bar(x - w / 2, pred, w, label="predicted if independent", color="#9bb8d3")
        ax.bar(x + w / 2, obs, w, label="observed (stacked)", color="#b5651d")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("Attack Success Rate")
        ax.set_title("Do stacked attacks compound? Observed vs. independence prediction")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES, "composability.png"), bbox_inches="tight")
        plt.close(fig)

    # ---- 5. safety tax: over-refusal vs attack success, per model ----
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for m in models:
        xs = [rates[m][t]["asr"] for t in transforms if t in rates[m]]
        ys = [rates[m][t]["orr"] for t in transforms if t in rates[m]]
        ax.scatter(xs, ys, label=short_model(m), s=28, alpha=0.8)
    ax.set_xlabel("Attack Success Rate on harmful prompts  (unsafe →)")
    ax.set_ylabel("Over-Refusal Rate on benign prompts  (overcautious ↑)")
    ax.set_title("The two failure modes, per (model, transform)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "safety_tax.png"), bbox_inches="tight")
    plt.close(fig)


def _safe_mean(xs):
    xs = [x for x in xs if x == x]  # drop nan
    return sum(xs) / len(xs) if xs else float("nan")


def _pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x == x and y == y]
    if len(pairs) < 3:
        return float("nan")
    n = len(pairs)
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    cov = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    vx = sum((p[0] - mx) ** 2 for p in pairs) ** 0.5
    vy = sum((p[1] - my) ** 2 for p in pairs) ** 0.5
    return cov / (vx * vy) if vx and vy else float("nan")


def write_novel_tables(records):
    """Comprehension-controlled ASR, competence-safety cells, and the defense effect."""
    cc = metrics.comprehension_controlled(records)
    cells = metrics.competence_safety_cells(records)
    deff = metrics.defense_effect(records)

    rows = []
    for m in sorted(cc):
        for t in [x for x in TRANSFORM_ORDER if x in cc[m]]:
            d = cc[m][t]
            rows.append([short_model(m), t, mechanism_of(t), d["n"],
                         f"{d['comprehension']:.3f}", f"{d['asr']:.3f}", f"{d['cc_asr']:.3f}"])
    _write_csv(os.path.join(TABLES, "comprehension_controlled.csv"),
               ["model", "transform", "mechanism", "n", "comprehension", "asr_raw",
                "cc_asr"], rows)

    _write_csv(os.path.join(TABLES, "competence_safety_cells.csv"),
               ["model", "lang", "benign_competence", "benign_pass", "harm_comprehension",
                "asr_raw", "cc_asr", "n_harmful", "n_benign"],
               [[short_model(c["model"]), c["lang"], f"{c['benign_competence']:.3f}",
                 f"{c['benign_pass']:.3f}", f"{c['harm_comprehension']:.3f}",
                 f"{c['asr']:.3f}", f"{c['cc_asr']:.3f}", c["n_harmful"], c["n_benign"]]
                for c in cells])

    if deff:
        _write_csv(os.path.join(TABLES, "defense_effect.csv"),
                   ["model", "asr_base", "asr_defended", "asr_reduction", "rel_reduction",
                    "over_refusal_base", "over_refusal_defended", "over_refusal_cost"],
                   [[short_model(m), f"{d['asr_base']:.3f}", f"{d['asr_defended']:.3f}",
                     f"{d['asr_reduction']:+.3f}", f"{d['rel_reduction']:.3f}",
                     f"{d['over_refusal_base']:.3f}", f"{d['over_refusal_defended']:.3f}",
                     f"{d['over_refusal_cost']:+.3f}"] for m, d in deff.items()])
    return cc, cells, deff


def make_novel_figures(cells, deff):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(FIGURES, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.titleweight": "bold"})
    # ordered roughly high -> low training-resource for the resource-vs-safety reading
    LANGS = ["en", "zh", "es", "fr", "ar", "hi", "ur", "bn"]
    LNAME = {"en": "English", "zh": "Chinese", "es": "Spanish", "fr": "French",
             "ar": "Arabic", "hi": "Hindi", "ur": "Urdu", "bn": "Bangla"}

    # ---- masking figure: comprehension vs raw ASR vs comprehension-controlled ASR, per language
    agg = {l: {"comp": [], "asr": [], "cc": []} for l in LANGS}
    for c in cells:
        if c["lang"] in agg:
            agg[c["lang"]]["comp"].append(c["harm_comprehension"])
            agg[c["lang"]]["asr"].append(c["asr"])
            agg[c["lang"]]["cc"].append(c["cc_asr"])
    langs = [l for l in LANGS if agg[l]["asr"]]
    comp = [_safe_mean(agg[l]["comp"]) for l in langs]
    asr = [_safe_mean(agg[l]["asr"]) for l in langs]
    ccv = [_safe_mean(agg[l]["cc"]) for l in langs]
    x = np.arange(len(langs))
    w = 0.27
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.bar(x - w, comp, w, label="comprehension (1 − evade)", color="#6c8ebf")
    ax.bar(x, asr, w, label="raw ASR (what prior work reports)", color="#b5651d")
    ax.bar(x + w, ccv, w, label="comprehension-controlled ASR (true safety failure)", color="#c0392b")
    ax.set_xticks(x)
    ax.set_xticklabels([LNAME[l] for l in langs])
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1)
    ax.set_title("Comprehension masks the cross-lingual safety hole\n"
                 "(raw ASR looks low where the model simply doesn't understand)")
    ax.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "comprehension_masking.png"), bbox_inches="tight")
    plt.close(fig)

    # ---- sweet-spot scatter: cc_asr vs comprehension, per (model, language)
    fig, ax = plt.subplots(figsize=(6.6, 5))
    colors = {"en": "#333", "hi": "#e67e22", "bn": "#27ae60", "ur": "#8e44ad", "ar": "#2980b9",
              "zh": "#c0392b", "es": "#16a085", "fr": "#7f8c8d"}
    for c in cells:
        if c["cc_asr"] != c["cc_asr"]:
            continue
        ax.scatter(c["harm_comprehension"], c["cc_asr"], s=40, alpha=0.8,
                   color=colors.get(c["lang"], "#555"),
                   edgecolor="white", linewidth=0.5)
    # legend by language
    for l in langs:
        ax.scatter([], [], color=colors.get(l, "#555"), label=LNAME[l], s=40)
    ax.set_xlabel("comprehension of harmful request  (1 − evade rate)")
    ax.set_ylabel("comprehension-controlled ASR\n(safety failure among understood requests)")
    ax.set_title("Cross-lingual jailbreaks vs. comprehension\n(each point = one model × language)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, title="attack language")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "sweet_spot.png"), bbox_inches="tight")
    plt.close(fig)

    # ---- defense effect: base vs defended ASR per model
    if deff:
        models = list(deff)
        fig, ax = plt.subplots(figsize=(7, max(2.6, len(models) * 0.7)))
        y = np.arange(len(models))
        ax.barh(y - 0.2, [deff[m]["asr_base"] for m in models], 0.38,
                label="no defense", color="#b5651d")
        ax.barh(y + 0.2, [deff[m]["asr_defended"] for m in models], 0.38,
                label="defensive system prompt", color="#2a7f62")
        ax.set_yticks(y)
        ax.set_yticklabels([short_model(m) for m in models])
        ax.set_xlabel("Attack Success Rate (across attacks)")
        ax.set_xlim(0, 1)
        ax.set_title("Defensive system prompt reduces jailbreak success")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES, "defense_effect.png"), bbox_inches="tight")
        plt.close(fig)


def main():
    records = load_records()
    if not records:
        print("No results found. Run jbladder.run_eval first.")
        return
    print(f"[analyze] loaded {len(records)} records from "
          f"{len({r['model'] for r in records})} model(s)")
    rates, lifts, robustness, potency, comp, mech, models, transforms = write_tables(records)
    make_figures(records, rates, robustness, potency, comp, models, transforms)
    cc, cells, deff = write_novel_tables(records)
    make_novel_figures(cells, deff)
    ci = metrics.overall_asr_ci(records)
    print(f"[analyze] wrote tables -> {TABLES}")
    print(f"[analyze] wrote figures -> {FIGURES}")

    # console summary
    print("\n== Model robustness (1 - mean ASR), with 95% CI on ASR ==")
    for m in sorted(robustness, key=lambda x: robustness[x]):
        mean_asr, lo, hi = ci.get(m, (float("nan"),) * 3)
        print(f"  {short_model(m):26s} robustness={robustness[m]:.3f}  "
              f"ASR={mean_asr:.3f} [{lo:.3f}, {hi:.3f}]")
    print("\n== Top transforms by potency (mean attack lift) ==")
    for t in sorted(potency, key=lambda x: -potency[x])[:8]:
        print(f"  {t:34s} {potency[t]:+.3f}")
    print("\n== Mechanism family mean ASR ==")
    for k, v in sorted(mech.items(), key=lambda x: -x[1]):
        print(f"  {k:14s} {v:.3f}")

    # novel finding: does comprehension-controlled ASR relate to comprehension across cells?
    xs = [c["harm_comprehension"] for c in cells if c["lang"] != "en"]
    ys = [c["cc_asr"] for c in cells if c["lang"] != "en"]
    r = _pearson(xs, ys)
    print("\n== Cross-lingual safety vs comprehension (translation cells only) ==")
    print(f"  Pearson r(comprehension, cc_asr) = {r:.3f}  over {len(xs)} model×language cells")
    for l in ["en", "zh", "es", "fr", "ar", "hi", "ur", "bn"]:
        lc = [c for c in cells if c["lang"] == l]
        if lc:
            print(f"  {l}: raw ASR={_safe_mean([c['asr'] for c in lc]):.3f}  "
                  f"cc_ASR={_safe_mean([c['cc_asr'] for c in lc]):.3f}  "
                  f"comprehension={_safe_mean([c['harm_comprehension'] for c in lc]):.3f}")
    if deff:
        print("\n== Defense effect (defensive system prompt) ==")
        for m, d in deff.items():
            print(f"  {short_model(m):26s} ASR {d['asr_base']:.3f} -> {d['asr_defended']:.3f}  "
                  f"(−{d['rel_reduction']*100:.0f}%), over-refusal cost {d['over_refusal_cost']:+.3f}")


if __name__ == "__main__":
    main()
