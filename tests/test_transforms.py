"""Tests for the transform library. Run: python -m pytest tests/ -q  (or python tests/test_transforms.py)"""
import base64

from jbladder.transforms import (
    ATOMIC, COMBOS, all_transform_names, apply_transform, mechanism_of,
)

REQ = "Explain how to pick a lock."


def test_plain_is_identity():
    msgs = apply_transform("plain", REQ)
    assert msgs[-1]["content"] == REQ
    assert mechanism_of("plain") == "control"


def test_all_transforms_return_messages_embedding_request_signal():
    for name in all_transform_names():
        msgs = apply_transform(name, REQ)
        assert isinstance(msgs, list) and msgs, name
        assert all("role" in m and "content" in m for m in msgs), name


def test_base64_roundtrips():
    msgs = apply_transform("base64", REQ)
    content = msgs[-1]["content"]
    # the encoded request must be recoverable from the prompt
    token = content.split()[-1]
    assert base64.b64decode(token).decode("utf-8") == REQ


def test_payload_split_contains_both_halves():
    msgs = apply_transform("payload_split", REQ)
    content = msgs[-1]["content"]
    assert 'a = "' in content and 'b = "' in content


def test_combos_compose_two_atomics():
    for combo, parts in COMBOS.items():
        assert len(parts) >= 2
        msgs = apply_transform(combo, REQ)
        assert isinstance(msgs, list) and msgs
        assert mechanism_of(combo) == "combo"


def test_unknown_transform_raises():
    try:
        apply_transform("does_not_exist", REQ)
    except KeyError:
        return
    raise AssertionError("expected KeyError")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all transform tests passed")
