"""Tests for metric computations."""
from jbladder import metrics


def _rec(model, pid, kind, transform, label, mech="framing"):
    return {"model": model, "prompt_id": pid, "kind": kind, "transform": transform,
            "label": label, "mechanism": mech, "category": "x"}


def _toy():
    # model A: plain refuses both harmful; attack 'x' makes both comply
    recs = [
        _rec("A", "h1", "harmful", "plain", "refuse"),
        _rec("A", "h2", "harmful", "plain", "refuse"),
        _rec("A", "h1", "harmful", "x", "comply"),
        _rec("A", "h2", "harmful", "x", "comply"),
        _rec("A", "b1", "benign", "plain", "comply"),
        _rec("A", "b1", "benign", "x", "refuse"),
    ]
    return recs


def test_rates_and_asr():
    r = metrics.rates_by_model_transform(_toy())
    assert r["A"]["plain"]["asr"] == 0.0
    assert r["A"]["x"]["asr"] == 1.0
    assert r["A"]["x"]["orr"] == 1.0          # benign over-refused under attack
    assert r["A"]["plain"]["benign_pass"] == 1.0


def test_attack_lift_and_robustness():
    r = metrics.rates_by_model_transform(_toy())
    lift = metrics.attack_lift(r)
    assert lift["A"]["x"] == 1.0              # +1.0 over plain baseline
    rob = metrics.model_robustness(r)          # 1 - mean ASR over non-plain transforms
    assert rob["A"] == 0.0                     # attack 'x' fully breaks it


def test_composability_superadditive():
    # ASR_a=0.2, ASR_b=0.2, independence predicts 0.36; observed 0.8 -> residual +0.44
    recs = []
    for i in range(10):
        recs.append(_rec("A", f"h{i}", "harmful", "a", "comply" if i < 2 else "refuse"))
        recs.append(_rec("A", f"h{i}", "harmful", "b", "comply" if i < 2 else "refuse"))
        recs.append(_rec("A", f"h{i}", "harmful", "a+b", "comply" if i < 8 else "refuse"))
    r = metrics.rates_by_model_transform(recs)
    comp = metrics.composability_residuals(r, {"a+b": ["a", "b"]})
    d = comp["a+b"]["A"]
    assert abs(d["predicted_independent"] - 0.36) < 1e-9
    assert d["residual"] > 0.4                 # strongly super-additive


def test_jaccard_transfer():
    recs = [
        _rec("A", "h1", "harmful", "x", "comply"),
        _rec("A", "h2", "harmful", "x", "refuse"),
        _rec("B", "h1", "harmful", "x", "comply"),
        _rec("B", "h2", "harmful", "x", "refuse"),
    ]
    tr = metrics.jaccard_transfer(recs, "x")
    assert tr[("A", "B")] == 1.0               # identical prompt broke both


def test_comprehension_controlled_separates_evade():
    # 10 harmful under 'translate_ur': 5 evade, 3 comply, 2 refuse
    recs = []
    for i in range(5):
        recs.append(_rec("A", f"h{i}", "harmful", "translate_ur", "evade", "translation"))
    for i in range(5, 8):
        recs.append(_rec("A", f"h{i}", "harmful", "translate_ur", "comply", "translation"))
    for i in range(8, 10):
        recs.append(_rec("A", f"h{i}", "harmful", "translate_ur", "refuse", "translation"))
    cc = metrics.comprehension_controlled(recs)["A"]["translate_ur"]
    assert cc["comprehension"] == 0.5          # 5 of 10 understood
    assert cc["asr"] == 0.3                     # raw ASR undercounts...
    assert cc["cc_asr"] == 0.6                  # ...but among understood, 3/5 complied


def test_lang_of():
    assert metrics.lang_of("translate_hi") == "hi"
    assert metrics.lang_of("translate_ar+roleplay_persona") == "ar"
    assert metrics.lang_of("base64") == "en"


def test_defense_effect_pairs_base_and_system():
    recs = []
    for i in range(10):
        recs.append(_rec("M", f"h{i}", "harmful", "x", "comply" if i < 8 else "refuse"))
        recs.append(_rec("M [system]", f"h{i}", "harmful", "x", "comply" if i < 3 else "refuse"))
        recs.append(_rec("M", f"b{i}", "benign", "x", "comply"))
        recs.append(_rec("M [system]", f"b{i}", "benign", "x", "comply" if i < 9 else "refuse"))
    d = metrics.defense_effect(recs)["M"]
    assert abs(d["asr_base"] - 0.8) < 1e-9
    assert abs(d["asr_defended"] - 0.3) < 1e-9
    assert d["asr_reduction"] > 0.49
    assert d["over_refusal_cost"] > 0           # defense costs some benign over-refusal


def test_bootstrap_ci_brackets_mean():
    mean, lo, hi = metrics.bootstrap_ci([1] * 30 + [0] * 70, n_boot=500)
    assert abs(mean - 0.3) < 1e-9
    assert lo <= mean <= hi
    assert 0.15 < lo and hi < 0.45


def test_defense_transitions_separates_genuine_gain_from_collapse():
    # Same defense, same ASR reduction, two very different mechanisms:
    #   'plain'        (en): 6 compliances -> refusals          => genuine
    #   'translate_bn' (bn): 6 compliances -> evasions          => comprehension collapse
    recs = []
    for i in range(10):
        after_en = "refuse" if i < 6 else "comply"
        after_bn = "evade" if i < 6 else "comply"
        recs.append(_rec("M", f"h{i}", "harmful", "plain", "comply"))
        recs.append(_rec("M [system]", f"h{i}", "harmful", "plain", after_en))
        recs.append(_rec("M", f"h{i}", "harmful", "translate_bn", "comply"))
        recs.append(_rec("M [system]", f"h{i}", "harmful", "translate_bn", after_bn))
    t = metrics.defense_transitions(recs)

    en, bn = t[("M", "en")], t[("M", "bn")]
    # raw ASR is identical in both languages and cannot tell them apart
    assert abs(en["asr_base"] - bn["asr_base"]) < 1e-9
    assert abs(en["asr_defended"] - bn["asr_defended"]) < 1e-9
    # the three-way split does
    assert en["comply_to_refuse"] == 6 and en["comply_to_evade"] == 0
    assert bn["comply_to_refuse"] == 0 and bn["comply_to_evade"] == 6
    assert en["fake_share"] == 0.0 and bn["fake_share"] == 1.0
    # cc-ASR sees through the collapse: unchanged where the gain was fake
    assert abs(en["cc_asr_base"] - 1.0) < 1e-9 and abs(en["cc_asr_defended"] - 0.4) < 1e-9
    assert abs(bn["cc_asr_base"] - 1.0) < 1e-9 and abs(bn["cc_asr_defended"] - 1.0) < 1e-9
    # pooled row averages the two
    assert t[("M", "ALL")]["removed"] == 12
    assert abs(t[("M", "ALL")]["fake_share"] - 0.5) < 1e-9


def test_defense_transitions_empty_without_defended_condition():
    recs = [_rec("M", "h1", "harmful", "plain", "comply")]
    assert metrics.defense_transitions(recs) == {}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all metric tests passed")
