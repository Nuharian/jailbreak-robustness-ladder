"""Tests for the robustness checks."""
import numpy as np

from jbladder import robustness


def _rec(model, pid, transform, label, kind="harmful"):
    return {"model": model, "prompt_id": pid, "kind": kind, "transform": transform,
            "label": label, "mechanism": "x", "category": "x"}


def test_bounds_bracket_cc_asr():
    # 10 probes in Bangla: 3 comply, 1 refuse, 6 evade -> bounds [0.3, 0.9], cc-ASR 0.75
    labels = ["comply"] * 3 + ["refuse"] + ["evade"] * 6
    recs = [_rec("M", f"h{i}", "translate_bn", l) for i, l in enumerate(labels)]
    recs += [_rec("M", f"h{i}", t, "refuse") for i in range(10)
             for t in ("plain", "translate_hi", "translate_ur", "translate_ar",
                       "translate_zh", "translate_es", "translate_fr")]
    A, models, _ = robustness.label_array(recs)
    s = robustness._lang_stats(A)
    bn = robustness.LANGS.index("bn")
    assert abs(s["asr"][bn] - 0.3) < 1e-9 and abs(s["upper"][bn] - 0.9) < 1e-9
    assert abs(s["cc_asr"][bn] - 0.75) < 1e-9
    assert s["asr"][bn] <= s["cc_asr"][bn] <= s["upper"][bn]


def test_identity_calibration_leaves_rates_unchanged():
    A = np.zeros((1, len(robustness.LANGS), 4), dtype=np.int8)
    A[0, :, :2] = 1      # 2 comply, 2 refuse in every language
    cal = {"high": np.eye(3), "low": np.eye(3)}
    s = robustness._corrected_lang_stats(A, cal)
    assert np.allclose(s["cc_asr"], 0.5) and np.allclose(s["asr"], 0.5)


def test_comply_to_evade_error_lowers_raw_asr_more_than_cc_asr():
    # judge over-calls COMPLY: half of judged COMPLYs are truly EVADE; refusals are exact
    A = np.zeros((1, len(robustness.LANGS), 10), dtype=np.int8)
    A[0, :, :8] = 1      # judged: 8 comply, 2 refuse
    M = np.eye(3)
    M[1] = [0.0, 0.5, 0.5]
    s = robustness._corrected_lang_stats(A, {"high": M, "low": M})
    assert np.allclose(s["asr"], 0.4)                 # 0.8 -> 0.4
    assert np.allclose(s["cc_asr"], 4 / 6)            # 0.8 -> 0.67: cc-ASR is far more stable


def test_wilson_and_kappa():
    lo, hi = robustness.wilson(96, 114)
    assert 0.76 < lo < 0.77 and 0.89 < hi < 0.90
    rows = [{"human_label": l, "judge_label": l} for l in ("refuse", "comply", "evade") * 3]
    assert abs(robustness.judge_agreement(rows)["kappa"] - 1.0) < 1e-9
